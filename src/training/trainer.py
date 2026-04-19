"""
训练器
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .utils import ProgressBar, simple_loss


class Trainer:
    """模型训练器"""

    def __init__(self, model, optimizer, scheduler, device, cfg):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg

        self.best_val_f1 = 0.0
        self.history = {
            'train_loss': [],
            'batch_losses': [],
            'val_loss': [],
            'val_f1': [],
            'lr': []
        }

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch，返回平均loss和batch loss列表"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        batch_losses = []

        pbar = ProgressBar(len(train_loader), desc=f"Epoch {epoch+1}/{self.cfg.epochs}")

        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            x = batch['x'].to(self.device)
            masks = batch['mask'].to(self.device)
            cls_labels = batch['cls_label']

            pred_class, pred_mask = self.model(x)

            # 计算损失（使用 utils.py 中的 simple_loss，内部已包含常量超参数）
            loss = 0
            for b in range(x.shape[0]):
                target_mask = masks[b]
                target_cls = cls_labels[b]
                loss += simple_loss(pred_class[b], pred_mask[b], target_mask, target_cls, self.device)
            loss = loss / x.shape[0]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            batch_losses.append(loss.item())
            pbar.update(batch_idx + 1, loss=loss.item())

        avg_loss = epoch_loss / num_batches
        return avg_loss, batch_losses

    def save_batch_loss_plot(self, batch_losses, epoch, save_dir):
        """保存每个 batch 的 loss 变化图"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(14, 6))
        batch_indices = list(range(1, len(batch_losses) + 1))

        plt.plot(batch_indices, batch_losses, 'b-o', linewidth=1, markersize=3, alpha=0.7)
        plt.xlabel('Batch Index', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Epoch {epoch+1} - Batch-wise Loss (Total: {len(batch_losses)} batches, Avg: {np.mean(batch_losses):.4f})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=np.mean(batch_losses), color='r', linestyle='--',
                   label=f'Mean Loss: {np.mean(batch_losses):.4f}')

        if len(batch_losses) >= 10:
            window = min(10, len(batch_losses) // 5)
            moving_avg = np.convolve(batch_losses, np.ones(window)/window, mode='valid')
            avg_indices = list(range(window, len(batch_losses) + 1))
            plt.plot(avg_indices, moving_avg, 'g-', linewidth=2,
                    label=f'Moving Avg (w={window})')

        plt.legend()
        plt.savefig(save_path / f'batch_loss_epoch_{epoch+1:03d}.png', dpi=150, bbox_inches='tight')
        plt.savefig(save_path / f'batch_loss_epoch_{epoch+1:03d}.pdf', bbox_inches='tight')
        plt.close()
        print(f"   📊 Batch loss plot saved: epoch_{epoch+1:03d}")

    def train(self, train_loader, val_loader, evaluator, save_dir):
        """完整训练流程"""
        print("\n🚀 Starting training...")
        print("-" * 60)

        save_path = Path(save_dir)

        for epoch in range(self.cfg.epochs):
            avg_loss, batch_losses = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(avg_loss)
            self.history['batch_losses'].append(batch_losses)
            self.history['lr'].append(self.scheduler.get_last_lr()[0])

            # 每个 epoch 保存 batch loss 图
            self.save_batch_loss_plot(batch_losses, epoch, save_dir)

            # 验证（每5个epoch）
            if val_loader is not None and (epoch + 1) % 5 == 0:
                val_f1 = self.validate(val_loader, evaluator)
                self.history['val_f1'].append(val_f1)

                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.save_checkpoint(epoch, save_path / f"best_model_epoch_{epoch+1}.pth", val_f1)
                    print(f"\n   💾 Best model saved (val_f1={val_f1:.4f})")

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"\n   📊 Epoch {epoch+1}: avg_loss={avg_loss:.4f}, lr={current_lr:.2e}, best_val_f1={self.best_val_f1:.4f}")

        self.save_history(save_dir)
        print("-" * 60)
        print("\n✅ Training completed!")
        return self.history

    def save_history(self, save_dir):
        """保存训练历史到 CSV"""
        import pandas as pd
        save_path = Path(save_dir)
        epoch_stats = []
        for epoch, batch_losses in enumerate(self.history['batch_losses']):
            epoch_stats.append({
                'epoch': epoch + 1,
                'avg_loss': self.history['train_loss'][epoch],
                'min_loss': min(batch_losses),
                'max_loss': max(batch_losses),
                'std_loss': np.std(batch_losses),
                'lr': self.history['lr'][epoch] if epoch < len(self.history['lr']) else 0
            })
        df = pd.DataFrame(epoch_stats)
        df.to_csv(save_path / 'training_stats.csv', index=False)
        print(f"📊 Training stats saved to: {save_path / 'training_stats.csv'}")

    def validate(self, val_loader, evaluator):
        results = evaluator.evaluate(val_loader)
        return results['segment_level']['f1']

    def save_checkpoint(self, epoch, save_path, val_f1):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_f1': val_f1,
            'history': self.history
        }, save_path)