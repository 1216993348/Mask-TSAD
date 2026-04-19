"""
竞争匹配（Competitive Matching）
多个 query 竞争预测同一个异常段
"""

import torch
import torch.nn.functional as F


def competitive_matching(pred_class, pred_mask, target_mask, target_cls, num_classes):
    """
    多个 query 竞争预测同一个异常段
    训练和推理一致：选整体代价最低的 query
    """
    Q, T = pred_mask.shape
    device = pred_class.device
    bg_cls = num_classes

    # ========== 情况1：无异常 ==========
    if target_cls < 0 or target_mask.sum() == 0:
        total_loss = 0.0
        for q in range(Q):
            cls_loss = F.cross_entropy(
                pred_class[q, :bg_cls + 1].unsqueeze(0),
                torch.tensor([bg_cls], device=device)
            )
            mask_loss = F.binary_cross_entropy(
                torch.sigmoid(pred_mask[q]),
                torch.zeros(T, device=device)
            )
            total_loss += cls_loss + mask_loss
        return total_loss / Q, None

    # ========== 情况2：有异常 ==========
    # 计算每个 query 的整体代价（与推理一致）
    costs = torch.zeros(Q, device=device)

    for q in range(Q):
        # 1. 类别代价（负对数似然）
        cls_cost = -pred_class[q, target_cls]

        # 2. Mask 代价（BCE + Dice）
        pred = torch.sigmoid(pred_mask[q])
        target = target_mask

        bce = F.binary_cross_entropy(pred, target, reduction='sum')
        intersection = (pred * target).sum()
        dice = 1 - (2 * intersection + 1) / (pred.sum() + target.sum() + 1)

        costs[q] = cls_cost + bce + dice

    # 选代价最低的 query
    best_q = torch.argmin(costs)

    # 赢家损失
    cls_loss = F.cross_entropy(
        pred_class[best_q, :bg_cls + 1].unsqueeze(0),
        torch.tensor([target_cls], device=device)
    )
    pred = torch.sigmoid(pred_mask[best_q])
    target = target_mask
    bce_loss = F.binary_cross_entropy(pred, target)
    intersection = (pred * target).sum()
    dice_loss = 1 - (2 * intersection + 1) / (pred.sum() + target.sum() + 1)
    total_loss = cls_loss + bce_loss + dice_loss

    # 输家损失（预测背景）
    for q in range(Q):
        if q != best_q:
            cls_loss_q = F.cross_entropy(
                pred_class[q, :bg_cls + 1].unsqueeze(0),
                torch.tensor([bg_cls], device=device)
            )
            mask_loss_q = F.binary_cross_entropy(
                torch.sigmoid(pred_mask[q]),
                torch.zeros(T, device=device)
            )
            total_loss += 0.1 * (cls_loss_q + mask_loss_q)

    return total_loss, best_q


def competitive_matching_batch(pred_class, pred_mask, target_masks, target_classes, num_classes):
    """批处理版本"""
    B = pred_class.shape[0]
    total_loss = 0.0

    for b in range(B):
        masks = target_masks[b]
        classes = target_classes[b]

        if masks and len(masks) > 0:
            target_mask = masks[0].to(pred_class.device)
            target_cls = classes[0] if classes else -1
        else:
            target_mask = torch.zeros(pred_class.shape[2], device=pred_class.device)
            target_cls = -1

        loss, _ = competitive_matching(
            pred_class[b], pred_mask[b], target_mask, target_cls, num_classes
        )
        total_loss += loss

    return total_loss / B
