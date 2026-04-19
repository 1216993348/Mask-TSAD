"""
竞争匹配（Competitive Matching）- 最终加强版
包含所有可能的稳定化方案
"""

import torch
import torch.nn.functional as F


def competitive_matching(pred_class, pred_mask, target_mask, target_cls, num_classes):
    """
    多个 query 竞争预测同一个异常段 - 加强稳定版
    """
    Q, T = pred_mask.shape
    device = pred_class.device
    bg_cls = num_classes

    # ========== 情况1：无异常 ==========
    if target_cls < 0 or target_mask.sum() == 0:
        total_loss = 0.0
        # 只随机选一部分 query 来学习背景（降低计算量）
        num_bg_queries = max(10, Q // 4)  # 只学25%的query做背景
        bg_queries = torch.randperm(Q)[:num_bg_queries]

        for q in bg_queries:
            cls_loss = F.cross_entropy(
                pred_class[q, :bg_cls+1].unsqueeze(0),
                torch.tensor([bg_cls], device=device)
            )
            # 无异常时损失权重降低到0.1
            total_loss += 0.1 * cls_loss
        return total_loss / num_bg_queries, None

    # ========== 情况2：有异常 ==========
    # 计算每个 query 的得分（使用多种指标组合）
    scores = torch.zeros(Q, device=device)
    probs = F.softmax(pred_class, dim=-1)

    for q in range(Q):
        # 1. 类别得分：目标类别的概率
        cls_prob = probs[q, target_cls]

        # 2. 背景抑制：背景概率越低越好
        bg_prob = probs[q, bg_cls]

        # 3. Dice 得分
        pred = torch.sigmoid(pred_mask[q])
        intersection = (pred * target_mask).sum()
        dice = (2 * intersection + 1) / (pred.sum() + target_mask.sum() + 1)

        # 4. 异常区域激活率（鼓励mask集中在异常区域）
        anomaly_ratio = (pred * target_mask).sum() / (target_mask.sum() + 1e-8)

        # 综合得分：类别高 + 背景低 + Dice高 + 激活率高
        scores[q] = 2.0 * cls_prob - 0.5 * bg_prob + 1.5 * dice + 1.0 * anomaly_ratio

    best_q = torch.argmax(scores)

    # ========== 赢家损失 ==========
    # 类别损失
    cls_loss = F.cross_entropy(
        pred_class[best_q, :bg_cls+1].unsqueeze(0),
        torch.tensor([target_cls], device=device)
    )

    # Mask 损失：使用组合损失
    pred = torch.sigmoid(pred_mask[best_q])

    # 1. Dice Loss（主要）
    intersection = (pred * target_mask).sum()
    dice_loss = 1 - (2 * intersection + 1) / (pred.sum() + target_mask.sum() + 1)

    # 2. 加权 BCE（异常区域权重高）
    pos_weight = torch.ones(T, device=device)
    pos_weight[target_mask.bool()] = 20.0  # 异常区域权重20倍
    bce_loss = F.binary_cross_entropy_with_logits(
        pred_mask[best_q],
        target_mask,
        pos_weight=pos_weight,
        reduction='mean'
    )

    # 3. 异常区域激活损失（鼓励mask在异常区域激活）
    activation_loss = -torch.log(pred[target_mask.bool()].mean() + 1e-8)

    # 组合 Mask 损失
    mask_loss = 2.0 * dice_loss + 1.0 * bce_loss + 0.5 * activation_loss

    total_loss = cls_loss + mask_loss

    # ========== 输家损失（选择性惩罚）==========
    # 只惩罚那些预测异常但明显错误的 query
    for q in range(Q):
        if q != best_q:
            # 检查这个 query 是否在异常区域有高激活
            pred_q = torch.sigmoid(pred_mask[q])
            anomaly_activation = pred_q[target_mask.bool()].mean()

            # 如果它在异常区域激活很高，说明它在干扰，需要惩罚
            if anomaly_activation > 0.3:
                cls_loss_q = F.cross_entropy(
                    pred_class[q, :bg_cls+1].unsqueeze(0),
                    torch.tensor([bg_cls], device=device)
                )
                # 惩罚它的异常激活
                penalty = pred_q[target_mask.bool()].mean()
                total_loss += 0.5 * cls_loss_q + 0.5 * penalty

    # ========== 多样性损失（鼓励不同 query 分工）==========
    # 计算所有 query 的类别分布熵
    class_dist = probs.mean(dim=0)  # (C+1,)
    class_entropy = -(class_dist * torch.log(class_dist + 1e-8)).sum()
    # 最大化熵，鼓励均匀分布
    total_loss -= 0.05 * class_entropy

    # 计算 mask 的多样性：不同 query 的 mask 应该不同
    mask_probs = torch.sigmoid(pred_mask)  # (Q, T)
    mask_similarity = torch.mm(mask_probs, mask_probs.t()) / (T + 1e-8)
    # 惩罚 mask 过于相似
    diversity_penalty = (mask_similarity.sum() - mask_similarity.diag().sum()) / (Q * (Q - 1) + 1e-8)
    total_loss += 0.1 * diversity_penalty

    # ========== 梯度稳定化 ==========
    # 限制 loss 范围，防止梯度爆炸
    total_loss = torch.clamp(total_loss, min=0.01, max=10.0)

    return total_loss, best_q


def competitive_matching_batch(pred_class, pred_mask, target_masks, target_classes, num_classes):
    """批处理版本"""
    B = pred_class.shape[0]
    total_loss = 0.0
    valid_batches = 0

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

        # 过滤无效 loss
        if torch.isfinite(loss) and loss < 100:
            total_loss += loss
            valid_batches += 1

    if valid_batches == 0:
        return torch.tensor(1.0, device=pred_class.device, requires_grad=True)

    return total_loss / valid_batches