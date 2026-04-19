"""
评估器
"""

import torch
import torch.nn.functional as F
import numpy as np
from .utils import compute_metrics


class Evaluator:
    """模型评估器"""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                masks = batch['mask']

                _, pred_mask = self.model(x)
                pred_prob = torch.sigmoid(pred_mask)

                for b in range(x.shape[0]):
                    pred_binary = (pred_prob[b] > 0.5).cpu().numpy().astype(int)
                    all_predictions.extend(pred_binary)
                    all_labels.extend(masks[b].cpu().numpy())
                    all_scores.extend(pred_prob[b].cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)

        return compute_metrics(all_labels, all_predictions, all_scores)

    def evaluate_window(self, windows, batch_size=32):
        """滑动窗口评估（用于测试）"""
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch_x = torch.from_numpy(windows[i:i + batch_size]).float().to(self.device)

                _, pred_mask = self.model(batch_x)
                pred_prob = torch.sigmoid(pred_mask)

                for b in range(len(batch_x)):
                    pred_binary = (pred_prob[b] > 0.5).cpu().numpy().astype(int)
                    all_predictions.extend(pred_binary)
                    all_scores.extend(pred_prob[b].cpu().numpy())

        return np.array(all_predictions), np.array(all_scores)