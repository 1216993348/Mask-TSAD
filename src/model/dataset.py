"""
异常检测数据集
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inject.injector import UniversalAnomalyInjector
from src.inject.curriculum_injector import CurriculumInjector


class AnomalyDataset(Dataset):
    def __init__(self, data, seq_len=512, num_samples=5000, num_classes=6, total_epochs=50):
        self.data = data
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.total_epochs = total_epochs

        # 使用课程学习注入器
        self.injector = CurriculumInjector(total_epochs=total_epochs)

        # 当前 epoch（由外部设置）
        self.current_epoch = 0

        # 预采样起始位置
        T = len(data)
        self.starts = []
        for _ in range(num_samples):
            start = np.random.randint(0, max(1, T - seq_len))
            self.starts.append(start)

        # 算子到标签的映射（保持与原代码一致）
        self.op_to_label = {
            "Spike": 0,
            "Drift": 1,
            "Shift": 2,
            "Period": 3,
            "Cascade": 4,
            "Missing": 5,
        }

    def set_epoch(self, epoch):
        """训练时每轮调用，更新当前 epoch"""
        self.current_epoch = epoch

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.seq_len
        seq = self.data[start:end].copy()

        # 注入异常（传入当前 epoch）
        injected, mask, used_ops, affected_dims = self.injector.inject(seq, self.current_epoch)

        # 确定类别标签（取第一个算子的标签）
        if used_ops:
            cls_label = self.op_to_label.get(used_ops[0], -1)
            has_anomaly = True
        else:
            cls_label = -1
            has_anomaly = False

        return {
            'x': torch.from_numpy(injected).float(),
            'mask': torch.from_numpy(mask).float(),
            'cls_label': cls_label,
            'has_anomaly': has_anomaly
        }

    def __len__(self):
        """返回数据集大小"""
        return self.num_samples