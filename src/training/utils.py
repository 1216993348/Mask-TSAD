"""
训练工具函数
"""

import time
from datetime import timedelta
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ========== 损失函数超参数（常量）==========
NO_OBJECT_WEIGHT = 0.1      # 背景类分类损失降权系数
MASK_POS_WEIGHT = 2.0       # 异常区域 BCE 正样本权重（原10过大）
CLS_LOSS_WEIGHT = 1.0       # 分类损失总权重
MASK_LOSS_WEIGHT = 0.5      # 掩码损失总权重
# ==========================================


class ProgressBar:
    """进度条"""
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


def simple_loss(pred_class, pred_mask, target_mask, target_cls, device):
    """
    简单损失函数（使用常量超参数）
    Args:
        pred_class: (C+1,) 类别预测
        pred_mask: (T,) mask预测
        target_mask: (T,) 真实mask
        target_cls: int 真实类别，-1表示无异常
        device: 设备
    """
    bg_cls = pred_class.shape[-1] - 1

    if target_cls < 0:
        # 无异常：只学背景类别（降权）
        cls_loss = F.cross_entropy(
            pred_class.unsqueeze(0),
            torch.tensor([bg_cls], device=device)
        )
        cls_loss = cls_loss * NO_OBJECT_WEIGHT
        # mask 全为 0
        mask_loss = F.binary_cross_entropy_with_logits(
            pred_mask,
            torch.zeros_like(pred_mask)
        )
        return CLS_LOSS_WEIGHT * cls_loss + MASK_LOSS_WEIGHT * mask_loss

    # 有异常
    cls_loss = F.cross_entropy(
        pred_class.unsqueeze(0),
        torch.tensor([target_cls], device=device)
    )

    # 掩码损失：异常区域权重 MASK_POS_WEIGHT
    pos_weight = torch.ones_like(target_mask)
    pos_weight[target_mask.bool()] = MASK_POS_WEIGHT
    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask,
        target_mask,
        pos_weight=pos_weight
    )

    return CLS_LOSS_WEIGHT * cls_loss + MASK_LOSS_WEIGHT * mask_loss


def adjust_predictions(gt, pred):
    """点调整：如果一个异常段内有任何点被检出，整个段标记为异常"""
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


def compute_metrics(labels, predictions, scores=None):
    """计算评估指标"""
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, zero_division=0)
    rec = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    auc = 0.0
    if scores is not None:
        try:
            auc = roc_auc_score(labels, scores)
        except:
            auc = 0.0

    # 段级别指标
    gt_adj, pred_adj = adjust_predictions(labels, predictions)
    acc_adj = accuracy_score(gt_adj, pred_adj)
    prec_adj = precision_score(gt_adj, pred_adj, zero_division=0)
    rec_adj = recall_score(gt_adj, pred_adj, zero_division=0)
    f1_adj = f1_score(gt_adj, pred_adj, zero_division=0)

    return {
        'point_level': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1},
        'segment_level': {'accuracy': acc_adj, 'precision': prec_adj, 'recall': rec_adj, 'f1': f1_adj},
        'auc': auc
    }