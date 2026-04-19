"""
主训练脚本 - 精简版
"""

import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.model.simple_model import build_model
from src.data_loader import DataLoader as TSDataLoader
from src.model.dataset import AnomalyDataset
from src.training import Trainer, Evaluator
from src.training.visualizer import plot_training_history, save_history_to_csv
from src.config.config_utils import load_full_config, get_config_for_dataset


def train(cfg):
    """训练主函数"""
    print("=" * 60)
    print(f"简单 Transformer 异常检测训练")
    print(f"数据集: {cfg.dataset}")
    print("=" * 60)
    print(f"Device: {cfg.device}")
    print(f"Seq len: {cfg.seq_len}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Num samples: {cfg.num_samples}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Num classes: {cfg.num_classes}")
    print(f"Save dir: {cfg.save_dir}")
    print("=" * 60)

    # 1. 加载原始数据
    print("\n📂 Loading data...")
    loader = TSDataLoader(use_hf=True)
    data_dict = loader.load_dataset(cfg.dataset)

    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']

    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}, anomaly rate={y_test.mean():.2%}")

    # 2. 创建训练数据集
    print("\n📦 Creating training dataset...")
    train_dataset = AnomalyDataset(
        data=X_train,
        seq_len=cfg.seq_len,
        num_samples=cfg.num_samples,
        num_classes=cfg.num_classes,
        total_epochs=cfg.epochs  # 新增
    )

    # 统计异常比例
    anomaly_count = 0
    for i in range(min(1000, len(train_dataset))):
        if train_dataset[i]['has_anomaly']:
            anomaly_count += 1
    print(f"   Injected anomaly ratio: {anomaly_count/1000:.1%}")
    print(f"   Train samples: {len(train_dataset)}")

    # 3. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    # 4. 创建模型
    print("\n🏗️ Building model...")
    input_dim = X_train.shape[1]
    model = build_model(
        input_dim=input_dim,
        model_type='simple',
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_classes=cfg.num_classes,
        num_prototypes=16,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout
    ).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # 5. 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # 6. 训练器和评估器
    evaluator = Evaluator(model, cfg.device)
    trainer = Trainer(model, optimizer, scheduler, cfg.device, cfg)

    # 7. 保存路径
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = save_dir / f"simple_best_{cfg.dataset}.pth"

    # 8. 开始训练
    trainer.train(train_loader, None, evaluator, cfg.save_dir)  # 传入目录，不是文件路径

    # 9. 保存训练历史
    save_history_to_csv(history, save_dir / f"training_history_{cfg.dataset}.csv")

    # 10. 绘制训练曲线
    plot_training_history(history, save_dir, cfg.dataset, show=True)

    return model, history



def test(cfg, model_path=None):
    """测试函数"""
    print("\n" + "=" * 60)
    print("简单模型测试评估")
    print("=" * 60)

    # 加载数据
    print("\n📂 Loading test data...")
    loader = TSDataLoader(use_hf=True)
    data_dict = loader.load_dataset(cfg.dataset)

    X_test = data_dict['X_test']
    y_test = data_dict['y_test']

    print(f"   Test shape: {X_test.shape}")
    print(f"   Labels shape: {y_test.shape}, anomaly rate={y_test.mean():.2%}")

    # 滑动窗口
    test_windows = []
    test_labels = []
    for start in range(0, len(X_test) - cfg.seq_len, cfg.seq_len // 2):
        end = start + cfg.seq_len
        test_windows.append(X_test[start:end])
        test_labels.append(y_test[start:end])

    test_windows = np.array(test_windows)
    test_labels = np.array(test_labels)
    print(f"   Test windows: {len(test_windows)}")

    # 加载模型
    print("\n📦 Loading model...")
    input_dim = X_test.shape[1]
    model = build_model(
        input_dim=input_dim,
        model_type='simple',
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_classes=cfg.num_classes,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout
    ).to(cfg.device)

    save_path = Path(cfg.save_dir)
    if model_path is None:
        model_path = save_path / f"simple_best_{cfg.dataset}.pth"
    else:
        model_path = Path(model_path)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✓ Loaded model from {model_path}")
    else:
        print(f"   ⚠️ Model not found at {model_path}")
        return None

    # 评估
    evaluator = Evaluator(model, cfg.device)
    predictions, scores = evaluator.evaluate_window(test_windows, cfg.batch_size)

    # 计算指标
    all_labels = test_labels.flatten()
    all_predictions = predictions
    all_scores = scores

    from src.training.utils import compute_metrics
    metrics = compute_metrics(all_labels, all_predictions, all_scores)

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"\n📈 点级别指标:")
    print(f"   Accuracy:  {metrics['point_level']['accuracy']:.4f}")
    print(f"   Precision: {metrics['point_level']['precision']:.4f}")
    print(f"   Recall:    {metrics['point_level']['recall']:.4f}")
    print(f"   F1-score:  {metrics['point_level']['f1']:.4f}")
    print(f"\n📈 段级别指标:")
    print(f"   Accuracy:  {metrics['segment_level']['accuracy']:.4f}")
    print(f"   Precision: {metrics['segment_level']['precision']:.4f}")
    print(f"   Recall:    {metrics['segment_level']['recall']:.4f}")
    print(f"   F1-score:  {metrics['segment_level']['f1']:.4f}")
    print(f"\n📈 AUC: {metrics['auc']:.4f}")

    return metrics

if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='SWaT',
                        choices=['SMAP', 'SMD', 'SWaT', 'MSL', 'PSM'])
    parser.add_argument('--model_path', type=str, default=None)

    # 模型参数
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=6)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=None)

    args = parser.parse_args()

    full_config = load_full_config()
    cfg = get_config_for_dataset(args.dataset, full_config)

    for key, val in vars(args).items():
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)

    print("=" * 60)
    print(f"模式: {args.mode} | 数据集: {cfg.dataset} | Device: {cfg.device}")
    print(f"Epochs: {cfg.epochs} | LR: {cfg.lr} | Batch: {cfg.batch_size} | SeqLen: {cfg.seq_len}")
    print(f"D_model: {cfg.d_model} | Encoder layers: {cfg.num_encoder_layers}")
    print("=" * 60)

    if args.mode == 'train':
        model, history = train(cfg)
        print("\n开始测试评估...")
        test(cfg, model_path=None)
    else:
        test(cfg, model_path=args.model_path)

