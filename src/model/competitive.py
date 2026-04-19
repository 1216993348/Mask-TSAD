"""
竞争匹配（Competitive Matching）
多个 query 竞争预测同一个异常段
"""

import torch
import torch.nn.functional as F


def competitive_matching(pred_class, pred_mask, target_mask, target_cls, num_classes):
    """
    多个 query 竞争预测同一个异常段

    Args:
        pred_class: (Q, C+1) 类别预测
        pred_mask: (Q, T) mask 预测（logits）
        target_mask: (T,) 目标 mask，全0表示无异常
        target_cls: int 目标类别，-1 表示无异常
        num_classes: int 类别数
    Returns:
        loss: scalar
        best_q: int or None
    """
    Q, T = pred_mask.shape
    device = pred_class.device
    bg_cls = num_classes

    # ========== 情况1：无异常 ==========
    if target_cls < 0 or target_mask.sum() == 0:
        total_loss = 0.0
        for q in range(Q):
            # 类别损失：预测背景类
            cls_loss = F.cross_entropy(
                pred_class[q, :bg_cls+1].unsqueeze(0),
                torch.tensor([bg_cls], device=device)
            )
            # Mask 损失：使用 BCEWithLogitsLoss（不需要手动 sigmoid）
            mask_loss = F.binary_cross_entropy_with_logits(
                pred_mask[q],
                torch.zeros(T, device=device)
            )
            total_loss += cls_loss + mask_loss
        return total_loss / Q, None

    # ========== 情况2：有异常 ==========
    # 计算每个 query 的综合得分（使用 logits，不需要 sigmoid）
    scores = torch.zeros(Q, device=device)
    for q in range(Q):
        # 类别得分
        cls_score = pred_class[q, target_cls]
        # Dice 得分（用 sigmoid 转换后计算）
        pred_prob = torch.sigmoid(pred_mask[q])
        target = target_mask
        intersection = (pred_prob * target).sum()
        dice = (2 * intersection + 1) / (pred_prob.sum() + target.sum() + 1)
        cls_prob = F.softmax(pred_class[q], dim=-1)[target_cls]
        scores[q] = cls_prob + dice
        #scores[q] = cls_score + dice

    best_q = torch.argmax(scores)

    # 赢家损失
    cls_loss = F.cross_entropy(
        pred_class[best_q, :bg_cls+1].unsqueeze(0),
        torch.tensor([target_cls], device=device)
    )

    # 使用 BCEWithLogitsLoss（输入 logits，不需要 sigmoid）
    bce_loss = F.binary_cross_entropy_with_logits(
        pred_mask[best_q],
        target_mask
    )

    pred_prob = torch.sigmoid(pred_mask[best_q])
    intersection = (pred_prob * target_mask).sum()
    dice_loss = 1 - (2 * intersection + 1) / (pred_prob.sum() + target_mask.sum() + 1)
    total_loss = cls_loss + bce_loss + dice_loss

    # 输家损失（预测背景）
    for q in range(Q):
        if q != best_q:
            cls_loss_q = F.cross_entropy(
                pred_class[q, :bg_cls+1].unsqueeze(0),
                torch.tensor([bg_cls], device=device)
            )
            mask_loss_q = F.binary_cross_entropy_with_logits(
                pred_mask[q],
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