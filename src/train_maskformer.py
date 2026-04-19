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

sys.path.insert(0, str(Path(__file__).parent))

from src.model.maskformer import build_model
from src.model.competitive import competitive_matching_batch
from src.model.dataset import AnomalyDataset
from src.data_loader import DataLoader as TSDataLoader
from src.inference import MaskFormerInference


def load_full_config(config_path="configs/datasets_config.yaml"):
    """加载完整配置"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    config_full_path = project_root / config_path

    with open(config_full_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    return full_config


def get_config_for_dataset(dataset_name, full_config):
    """获取指定数据集的配置"""
    dataset_cfg = full_config['datasets'][dataset_name]
    global_model = full_config.get('global_model', {})
    global_training = full_config.get('global_training', {})
    output_cfg = full_config.get('output', {})

    class Config:
        pass

    cfg = Config()

    # 数据集特定配置
    cfg.dataset = dataset_name
    cfg.seq_len = dataset_cfg.get('model', {}).get('seq_len', 512)
    cfg.batch_size = dataset_cfg.get('model', {}).get('batch_size', 32)
    cfg.num_samples = dataset_cfg.get('model', {}).get('num_samples', 10000)

    # 模型配置（全局）
    cfg.d_model = global_model.get('d_model', 256)
    cfg.nhead = global_model.get('nhead', 8)
    cfg.num_encoder_layers = global_model.get('num_encoder_layers', 4)
    cfg.num_decoder_layers = global_model.get('num_decoder_layers', 4)
    cfg.num_queries = global_model.get('num_queries', 100)
    cfg.dim_feedforward = global_model.get('dim_feedforward', 1024)
    cfg.dropout = global_model.get('dropout', 0.1)
    cfg.num_classes = global_model.get('num_classes', 6)

    # 训练配置（全局）
    cfg.epochs = global_training.get('epochs', 50)
    cfg.lr = global_training.get('lr', 1e-4)
    cfg.weight_decay = global_training.get('weight_decay', 1e-4)
    cfg.clip_grad_norm = global_training.get('clip_grad_norm', 1.0)
    cfg.scheduler = global_training.get('scheduler', 'cosine')
    cfg.val_ratio = global_training.get('val_ratio', 0.2)

    # 输出配置 - 修复路径
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    cfg.save_dir = str(project_root / output_cfg.get('save_dir', 'output/maskformer_models'))

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
    """
    调整预测：如果某个异常段中有一个点被预测为异常，整个段都标记为异常
    """
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
    """训练进度条"""
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
    """
    端到端评估：使用推理模块
    """
    # 创建推理器
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
        x = batch['x'].cpu().numpy()  # 转为 numpy 给推理器
        masks_batch = batch['masks']

        # 批量推理
        results = inferencer.predict_batch(x)

        for b, result in enumerate(results):
            all_predictions.extend(result['anomaly_mask'])
            all_anomaly_scores.extend(result['anomaly_score'])

            # 真实标签
            if masks_batch[b] and len(masks_batch[b]) > 0:
                label = masks_batch[b][0].cpu().numpy()
            else:
                label = np.zeros(cfg.seq_len)
            all_labels.extend(label)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_anomaly_scores = np.array(all_anomaly_scores)

    # 计算指标
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


def train(dataset_name="SMAP"):
    """训练模型"""
    full_config = load_full_config()
    cfg = get_config_for_dataset(dataset_name, full_config)

    print("=" * 60)
    print(f"MaskFormer 端到端异常检测训练")
    print(f"数据集: {dataset_name}")
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

    print("\n📂 Loading data...")
    loader = TSDataLoader(use_hf=True)
    data_dict = loader.load_dataset(dataset_name)
    data = data_dict['X_test']
    print(f"   {dataset_name} shape: {data.shape}")

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
    history = {'train_loss': [], 'val_f1': [], 'val_class_acc': []}

    save_path = Path(cfg.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = save_path / f"maskformer_best_{dataset_name}.pth"

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


def test(model_path=None, dataset_name="SMAP"):
    """
    完整测试评估 - 使用推理模块
    """
    print("\n" + "=" * 60)
    print("端到端模型测试评估")
    print("=" * 60)

    full_config = load_full_config()
    cfg = get_config_for_dataset(dataset_name, full_config)

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
    save_path = Path(cfg.save_dir)
    if model_path is None:
        model_path = save_path / f"maskformer_best_{dataset_name}.pth"
    else:
        model_path = Path(model_path)

    model = build_model(
        input_dim=X_test.shape[1],
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        num_queries=cfg.num_queries,
        num_classes=cfg.num_classes,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout
    ).to(cfg.device)

    checkpoint = torch.load(model_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   ✓ Loaded model from {model_path}")

    # 创建推理器
    inferencer = MaskFormerInference(
        model=model,
        device=cfg.device,
        num_classes=cfg.num_classes,
        mask_threshold=0.5
    )

    # 评估
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

    # 计算指标
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


def plot_training_history(history, dataset_name, save_dir="output/maskformer_models"):
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
    parser.add_argument('--model_path', type=str, default='maskformer_best_1.pth')
    args = parser.parse_args()

    if args.mode == 'train':
        model, history = train(dataset_name=args.dataset)
        try:
            plot_training_history(history, args.dataset)
        except Exception as e:
            print(f"   ⚠️ Could not plot training curve: {e}")

        print("\n" + "=" * 60)
        print("开始测试评估...")
        print("=" * 60)
        test(model_path=None, dataset_name=args.dataset)

    else:
        test(model_path=args.model_path, dataset_name=args.dataset)