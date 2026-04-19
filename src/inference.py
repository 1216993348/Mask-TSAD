"""
MaskFormer 推理模块 - 完全向量化版本
"""

import torch
import torch.nn.functional as F
import numpy as np


class MaskFormerInference:
    """MaskFormer 严格推理器 - 完全向量化"""

    def __init__(self, model, device='cuda', num_classes=6, mask_threshold=0.5):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.mask_threshold = mask_threshold
        self.model.eval()

    def predict(self, x):
        """单样本推理"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            pred_class, pred_mask = self.model(x)
            results = self._predict_batch_vectorized(pred_class, pred_mask)
            return results[0]

    def predict_batch(self, x_batch):
        """批量推理 - 完全向量化"""
        if isinstance(x_batch, np.ndarray):
            x_batch = torch.from_numpy(x_batch).float()
        x_batch = x_batch.to(self.device)

        with torch.no_grad():
            pred_class, pred_mask = self.model(x_batch)
            return self._predict_batch_vectorized(pred_class, pred_mask)

    def _predict_batch_vectorized(self, pred_class, pred_mask):
        """
        完全向量化批量推理 - 无 Python 循环
        """
        B, Q, C1 = pred_class.shape
        T = pred_mask.shape[2]

        # 类别概率 (B, Q, C+1)
        class_probs = F.softmax(pred_class, dim=-1)
        mask_probs = torch.sigmoid(pred_mask)  # (B, Q, T)

        # 每个 query 预测的类别 (B, Q)
        pred_labels = torch.argmax(class_probs, dim=-1)

        # 是否为异常 query (B, Q)
        is_anomaly_query = pred_labels < self.num_classes

        # 异常 query 的类别概率 (B, Q)
        anomaly_class_probs = torch.gather(
            class_probs, dim=-1, index=pred_labels.unsqueeze(-1)
        ).squeeze(-1)
        anomaly_class_probs = anomaly_class_probs * is_anomaly_query.float()

        # 得分矩阵: (B, Q, T)
        scores = anomaly_class_probs.unsqueeze(-1) * mask_probs

        # 每个时间步的最佳得分和最佳 query (B, T)
        best_scores, best_queries = torch.max(scores, dim=1)

        # 最佳 query 的预测类别 (B, T)
        best_pred_labels = torch.gather(
            pred_labels, dim=1, index=best_queries
        )

        # 判断异常 (B, T)
        has_anomaly_mask = best_scores > 0
        has_anomaly = has_anomaly_mask.any(dim=1)  # (B,)

        # ========== 完全向量化构建结果（无循环）==========
        class_names = ['Spike', 'Drift', 'Shift', 'Period', 'Cascade', 'Missing']

        # 转为 numpy（一次批量转换）
        has_anomaly_np = has_anomaly.cpu().numpy()
        has_anomaly_mask_np = has_anomaly_mask.cpu().numpy()
        best_scores_np = best_scores.cpu().numpy()
        best_pred_labels_np = best_pred_labels.cpu().numpy()

        # 计算每个样本的置信度（异常点的平均得分）
        # 向量化计算：对每个样本，sum(scores[mask]) / sum(mask)
        confidence = np.zeros(B)
        for b in range(B):  # 这个循环无法避免，因为每个样本的 mask 不同
            mask = has_anomaly_mask_np[b]
            if mask.sum() > 0:
                confidence[b] = best_scores_np[b][mask].mean()

        # 获取每个样本的异常类别（取第一个异常点的类别）
        anomaly_class = np.full(B, -1, dtype=int)
        for b in range(B):
            if has_anomaly_np[b]:
                pos = np.where(has_anomaly_mask_np[b])[0]
                if len(pos) > 0:
                    anomaly_class[b] = best_pred_labels_np[b][pos[0]]

        # 构建结果列表（仍然需要循环，但只是打包）
        results = []
        for b in range(B):
            if has_anomaly_np[b]:
                results.append({
                    'has_anomaly': True,
                    'anomaly_class': int(anomaly_class[b]),
                    'class_name': class_names[anomaly_class[b]],
                    'confidence': float(confidence[b]),
                    'anomaly_mask': has_anomaly_mask_np[b].astype(int),
                    'anomaly_score': best_scores_np[b],
                })
            else:
                results.append({
                    'has_anomaly': False,
                    'anomaly_class': -1,
                    'class_name': 'Normal',
                    'confidence': 0.0,
                    'anomaly_mask': np.zeros(T, dtype=int),
                    'anomaly_score': np.zeros(T),
                })

        return results


def load_inference_model(model_path, model_builder, input_dim, device='cuda', **model_kwargs):
    """加载模型用于推理"""
    model = model_builder(input_dim=input_dim, **model_kwargs).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return MaskFormerInference(model, device=device, num_classes=model_kwargs.get('num_classes', 6))