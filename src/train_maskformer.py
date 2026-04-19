"""
MaskFormer 训练脚本 - 端到端异常检测
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time
import yaml
from datetime import timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.model.maskformer import build_model
from src.model.competitive import competitive_matching_batch
from src.model.dataset import AnomalyDataset
from src.data_loader import DataLoader as TSDataLoader
from src.inference import MaskFormerInference


def load_full_config(config_path="configs/datasets_config.yaml"):
    """加载完整配置"""
    config_full_path = project_root / config_path
    with open(config_full_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
    return full_config


def get_config_for_dataset(dataset_name, full_config):
    """获取指定数据集的配置"""
    dataset_cfg = full_config['datasets'][dataset_name]
    output_cfg = full_config.get('output', {})

    class Config:
        pass

    cfg = Config()
    cfg.dataset = dataset_name
    cfg.seq_len = dataset_cfg.get('model', {}).get('seq_len', 512)
    cfg.batch_size = dataset_cfg.get('model', {}).get('batch_size', 32)
    cfg.num_samples = dataset_cfg.get('model', {}).get('num_samples', 10000)
    cfg.save_dir = str(project_root / output_cfg.get('save_dir', 'output/maskformer_models'))

    # 模型参数默认值（会被命令行覆盖）
    cfg.d_model = 256
    cfg.nhead = 8
    cfg.num_encoder_layers = 4
    cfg.num_decoder_layers = 4
    cfg.num_queries = 100
    cfg.dim_feedforward = 1024
    cfg.dropout = 0.1
    cfg.num_classes = 6

    # 训练参数默认值（会被命令行覆盖）
    cfg.epochs = 50
    cfg.lr = 0.0001
    cfg.weight_decay = 0.0001
    cfg.clip_grad_norm = 1.0
    cfg.val_ratio = 0.2

    # 其他
    cfg.num_workers = 4
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return cfg


def collate_fn(batch):
    """自定义 collate 函数"""
    x = torch.stack([item['x'] for item in batch])

    masks = []
    classes = []
    for item in batch:
        if item['has_anomaly']:
            masks.append([item['mask']])
            classes.append([item['cls_label']])
        else:
            masks.append([])
            classes.append([])

    return {'x': x, 'masks': masks, 'classes': classes}


def adjust_predictions(gt, pred):
    """调整预测：如果某个异常段中有一个点被预测为异常，整个段都标记为异常"""
    gt_adj = gt.copy()
    pred_adj = pred.copy()

    in_anomaly = False
    start = 0
    for i in range(len(gt)):
        if gt[i] == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif gt[i] == 0 and in_anomaly:
            if pred_adj[start:i].sum() > 0:
                pred_adj[start:i] = 1
            in_anomaly = False

    if in_anomaly:
        if pred_adj[start:].sum() > 0:
            pred_adj[start:] = 1

    return gt_adj, pred_adj


class ProgressBar:
    def __init__(self, total, desc="Training"):
        self.total = total
        self.desc = desc
        self.start_time = time.time()

    def update(self, current, loss=None):
        elapsed = time.time() - self.start_time
        percent = current / self.total
        bar_len = 30
        filled = int(bar_len * percent)
        bar = '█' * filled + '░' * (bar_len - filled)

        eta = elapsed / current * (self.total - current) if current > 0 else 0

        if loss is not None:
            print(f'\r{self.desc}: |{bar}| {current}/{self.total} [{timedelta(seconds=int(elapsed))}<{timedelta(seconds=int(eta))}, loss={loss:.4f}]', end='')
        else:
            print(f'\r{self.desc}: |{bar}| {current}/{self.total} [{timedelta(seconds=int(elapsed))}<{timedelta(seconds=int(eta))}]', end='')

        if current == self.total:
            print()


def evaluate_end_to_end(model, dataloader, cfg):
    """端到端评估：使用推理模块"""
    inferencer = MaskFormerInference(
        model=model,
        device=cfg.device,
        num_classes=cfg.num_classes,
        mask_threshold=0.5
    )

    all_predictions = []
    all_labels = []
    all_anomaly_scores = []

    for batch in dataloader:
        x = batch['x'].cpu().numpy()
        masks_batch = batch['masks']

        results = inferencer.predict_batch(x)

        for b, result in enumerate(results):
            all_predictions.extend(result['anomaly_mask'])
            all_anomaly_scores.extend(result['anomaly_score'])

            if masks_batch[b] and len(masks_batch[b]) > 0:
                label = masks_batch[b][0].cpu().numpy()
            else:
                label = np.zeros(cfg.seq_len)
            all_labels.extend(label)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_anomaly_scores = np.array(all_anomaly_scores)

    acc = accuracy_score(all_labels, all_predictions)
    prec = precision_score(all_labels, all_predictions, zero_division=0)
    rec = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_anomaly_scores)
    except:
        auc = 0.0

    gt_adj, pred_adj = adjust_predictions(all_labels, all_predictions)
    acc_adj = accuracy_score(gt_adj, pred_adj)
    prec_adj = precision_score(gt_adj, pred_adj, zero_division=0)
    rec_adj = recall_score(gt_adj, pred_adj, zero_division=0)
    f1_adj = f1_score(gt_adj, pred_adj, zero_division=0)

    results = {
        'point_level': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1},
        'segment_level': {'accuracy': acc_adj, 'precision': prec_adj, 'recall': rec_adj, 'f1': f1_adj},
        'auc': auc,
    }

    return results


def train(cfg):
    """训练模型"""
    print("=" * 60)
    print(f"MaskFormer 端到端异常检测训练")
    print(f"数据集: {cfg.dataset}")
    print("=" * 60)
    print(f"Device: {cfg.device}")
    print(f"Seq len: {cfg.seq_len}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Num samples: {cfg.num_samples}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Num queries: {cfg.num_queries}")
    print(f"Num classes: {cfg.num_classes}")
    print(f"Save dir: {cfg.save_dir}")
    print("=" * 60)

    # 加载数据
    print("\n📂 Loading data...")
    loader = TSDataLoader(use_hf=True)
    data_dict = loader.load_dataset(cfg.dataset)
    data = data_dict['X_test']
    print(f"   {cfg.dataset} shape: {data.shape}")

    # 创建数据集
    print("\n📦 Creating dataset...")
    full_dataset = AnomalyDataset(
        data,
        seq_len=cfg.seq_len,
        num_samples=cfg.num_samples,
        num_classes=cfg.num_classes
    )

    val_size = int(len(full_dataset) * cfg.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate_fn)

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # 创建模型
    print("\n🏗️ Building model...")
    input_dim = data.shape[1]
    model = build_model(
        input_dim=input_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        num_queries=cfg.num_queries,
        num_classes=cfg.num_classes,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout
    ).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_f1': []}

    save_path = Path(cfg.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = save_path / f"maskformer_best_{cfg.dataset}.pth"

    print("\n🚀 Starting training...")
    print("-" * 60)

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = ProgressBar(len(train_loader), desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for batch_idx, batch in enumerate(train_loader):
            x = batch['x'].to(cfg.device)
            masks = batch['masks']
            classes = batch['classes']

            pred_class, pred_mask = model(x)
            loss = competitive_matching_batch(pred_class, pred_mask, masks, classes, cfg.num_classes)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.update(batch_idx + 1, loss=loss.item())

        avg_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_loss)

        if (epoch + 1) % 5 == 0:
            val_results = evaluate_end_to_end(model, val_loader, cfg)
            val_f1 = val_results['segment_level']['f1']
            history['val_f1'].append(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'history': history
                }, best_model_path)
                print(f"\n   💾 Best model saved (val_f1={val_f1:.4f})")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n   📊 Epoch {epoch+1}: loss={avg_loss:.4f}, lr={current_lr:.2e}, best_val_f1={best_val_f1:.4f}")

    print("-" * 60)
    print("\n✅ Training completed!")
    print(f"   💾 Best model: {best_model_path}")

    return model, history


def test(cfg, model_path=None, dataset_name=None):
    """测试模型"""
    if dataset_name is None:
        dataset_name = cfg.dataset

    print("\n" + "=" * 60)
    print("端到端模型测试评估")
    print("=" * 60)

    print("\n📂 Loading test data...")
    loader = TSDataLoader(use_hf=True)
    data_dict = loader.load_dataset(dataset_name)

    X_test = data_dict['X_test']
    y_test = data_dict['y_test']

    print(f"   {dataset_name} test shape: {X_test.shape}")
    print(f"   Labels shape: {y_test.shape}, anomaly rate={y_test.mean():.2%}")

    # 滑动窗口切分
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
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        num_queries=cfg.num_queries,
        num_classes=cfg.num_classes,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout
    ).to(cfg.device)

    save_path = Path(cfg.save_dir)
    if model_path is None:
        model_path = save_path / f"maskformer_best_{dataset_name}.pth"
    else:
        model_path = Path(model_path)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✓ Loaded model from {model_path}")
        if 'val_f1' in checkpoint:
            print(f"   ✓ Best validation F1: {checkpoint['val_f1']:.4f}")
    else:
        print(f"   ⚠️ Model not found at {model_path}")
        return None

    inferencer = MaskFormerInference(
        model=model,
        device=cfg.device,
        num_classes=cfg.num_classes,
        mask_threshold=0.5
    )

    print(f"\n🔮 Running evaluation...")

    all_predictions = []
    all_labels = []
    all_anomaly_scores = []

    for i in range(0, len(test_windows), cfg.batch_size):
        batch_x = test_windows[i:i + cfg.batch_size]
        batch_labels = test_labels[i:i + cfg.batch_size]

        results = inferencer.predict_batch(batch_x)

        for b, result in enumerate(results):
            all_predictions.extend(result['anomaly_mask'])
            all_labels.extend(batch_labels[b])
            all_anomaly_scores.extend(result['anomaly_score'])

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_anomaly_scores = np.array(all_anomaly_scores)

    acc = accuracy_score(all_labels, all_predictions)
    prec = precision_score(all_labels, all_predictions, zero_division=0)
    rec = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_anomaly_scores)
    except:
        auc = 0.0

    gt_adj, pred_adj = adjust_predictions(all_labels, all_predictions)
    acc_adj = accuracy_score(gt_adj, pred_adj)
    prec_adj = precision_score(gt_adj, pred_adj, zero_division=0)
    rec_adj = recall_score(gt_adj, pred_adj, zero_division=0)
    f1_adj = f1_score(gt_adj, pred_adj, zero_division=0)

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)

    print(f"\n📈 点级别指标:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-score:  {f1:.4f}")

    print(f"\n📈 段级别指标（调整后）:")
    print(f"   Accuracy:  {acc_adj:.4f}")
    print(f"   Precision: {prec_adj:.4f}")
    print(f"   Recall:    {rec_adj:.4f}")
    print(f"   F1-score:  {f1_adj:.4f}")

    print(f"\n📈 其他指标:")
    print(f"   AUC: {auc:.4f}")

    results = {
        'point_level': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1},
        'segment_level': {'accuracy': acc_adj, 'precision': prec_adj, 'recall': rec_adj, 'f1': f1_adj},
        'auc': auc
    }

    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / f"test_results_{dataset_name}.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"MaskFormer 端到端异常检测测试结果 - {dataset_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Point Level Metrics:\n")
        f.write(f"  Accuracy:  {acc:.4f}\n")
        f.write(f"  Precision: {prec:.4f}\n")
        f.write(f"  Recall:    {rec:.4f}\n")
        f.write(f"  F1-score:  {f1:.4f}\n\n")
        f.write(f"Segment Level Metrics (with adjustment):\n")
        f.write(f"  Accuracy:  {acc_adj:.4f}\n")
        f.write(f"  Precision: {prec_adj:.4f}\n")
        f.write(f"  Recall:    {rec_adj:.4f}\n")
        f.write(f"  F1-score:  {f1_adj:.4f}\n\n")
        f.write(f"AUC: {auc:.4f}\n")

    print(f"\n💾 Results saved to {save_path / f'test_results_{dataset_name}.txt'}")

    return results


def plot_training_history(history, dataset_name, save_dir):
    """绘制训练曲线"""
    import matplotlib.pyplot as plt

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training Loss - {dataset_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if history.get('val_f1'):
        epochs = range(5, len(history['val_f1']) * 5 + 1, 5)
        axes[1].plot(epochs, history['val_f1'], label='Validation F1', linewidth=2, marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title(f'Validation F1 Score - {dataset_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / f'training_history_{dataset_name}.pdf', format='pdf')
    plt.savefig(save_path / f'training_history_{dataset_name}.png', dpi=150)
    plt.close()
    print(f"   📊 Training curve saved to {save_path / f'training_history_{dataset_name}.pdf'}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='SWaT',
                        choices=['SMAP', 'SMD', 'SWaT', 'MSL', 'PSM'])
    parser.add_argument('--model_path', type=str, default=None)

    # 模型参数
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=6)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--val_ratio', type=float, default=0.2)

    # 数据参数
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=None)

    args = parser.parse_args()

    # 加载配置
    full_config = load_full_config()
    cfg = get_config_for_dataset(args.dataset, full_config)

    # 命令行参数覆盖
    for key, val in vars(args).items():
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)

    # 打印配置
    print("=" * 60)
    print(f"模式: {args.mode} | 数据集: {cfg.dataset} | Device: {cfg.device}")
    print(f"Epochs: {cfg.epochs} | LR: {cfg.lr} | Batch: {cfg.batch_size} | SeqLen: {cfg.seq_len}")
    print(f"D_model: {cfg.d_model} | Queries: {cfg.num_queries}")
    print("=" * 60)

    if args.mode == 'train':
        model, history = train(cfg)
        try:
            plot_training_history(history, args.dataset, cfg.save_dir)
        except Exception as e:
            print(f"⚠️ Could not plot: {e}")
        print("\n开始测试评估...")
        test(cfg, model_path=None, dataset_name=args.dataset)
    else:
        test(cfg, model_path=args.model_path, dataset_name=args.dataset)