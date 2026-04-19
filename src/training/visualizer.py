"""
训练曲线可视化工具
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_history(history, save_dir, dataset_name, show=True):
    """
    绘制训练曲线

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'val_f1', 'lr'
        save_dir: 保存路径
        dataset_name: 数据集名称
        show: 是否显示图像
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training History - {dataset_name}', fontsize=14, fontweight='bold')

    # 1. 训练损失曲线
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=1.5, label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 验证 F1 曲线（如果有）
    ax = axes[0, 1]
    if history.get('val_f1') and len(history['val_f1']) > 0:
        val_epochs = range(5, len(history['val_f1']) * 5 + 1, 5)
        ax.plot(val_epochs, history['val_f1'], 'g-o', linewidth=1.5, markersize=4, label='Val F1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Validation F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Validation F1 Score (Not Available)')

    # 3. 学习率曲线
    ax = axes[1, 0]
    ax.plot(epochs, history['lr'], 'r-', linewidth=1.5, label='Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 损失下降趋势（平滑后）
    ax = axes[1, 1]
    ax.plot(epochs, history['train_loss'], 'b-', alpha=0.3, linewidth=0.8, label='Raw')
    # 平滑曲线（移动平均）
    if len(history['train_loss']) >= 10:
        window = min(10, len(history['train_loss']) // 5)
        smoothed = np.convolve(history['train_loss'], np.ones(window) / window, mode='valid')
        smooth_epochs = epochs[:len(smoothed)]
        ax.plot(smooth_epochs, smoothed, 'b-', linewidth=2, label=f'Smoothed (w={window})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path / f'training_history_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_path / f'training_history_{dataset_name}.pdf', bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    print(f"📊 Training curves saved to: {save_path}")


def plot_loss_comparison(histories, save_dir, show=True):
    """
    对比多个模型的训练曲线

    Args:
        histories: dict {model_name: history}
        save_dir: 保存路径
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for name, history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], linewidth=1.5, label=f'{name}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path / 'loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(save_path / 'loss_comparison.pdf', bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def save_history_to_csv(history, save_path):
    """保存历史记录到 CSV"""
    import pandas as pd

    df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'lr': history['lr']
    })

    if history.get('val_f1'):
        # 对齐验证指标（每5个epoch）
        val_epochs = list(range(5, len(history['val_f1']) * 5 + 1, 5))
        val_df = pd.DataFrame({
            'epoch': val_epochs,
            'val_f1': history['val_f1']
        })
        df = pd.merge(df, val_df, on='epoch', how='left')

    df.to_csv(save_path, index=False)
    print(f"📊 History saved to: {save_path}")