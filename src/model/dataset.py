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


class AnomalyDataset(Dataset):
    """注入异常的数据集"""

    def __init__(self, data, seq_len=512, num_samples=5000, num_classes=6):
        """
        Args:
            data: (T, D) 原始时间序列
            seq_len: 序列长度
            num_samples: 样本数量
            num_classes: 异常类别数
        """
        self.data = data
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.injector = UniversalAnomalyInjector(num_classes=num_classes)

        # 算子名称到标签的映射
        self.op_to_label = {
            "Spike": 0,
            "Drift": 1,
            "Shift": 2,
            "Period": 3,
            "Cascade": 4,
            "Missing": 5,
        }

        # 预先采样起始位置
        T = len(data)
        self.starts = []
        for _ in range(num_samples):
            start = np.random.randint(0, max(1, T - seq_len))
            self.starts.append(start)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.seq_len
        seq = self.data[start:end].copy()

        # 注入异常
        injected, instance = self.injector.inject(seq)

        # 构建标签
        mask = np.zeros(self.seq_len, dtype=np.float32)
        cls_label = -1  # 无异常

        if instance is not None:
            s = instance.start
            e = instance.end
            mask[s:e] = 1.0
            op_name = instance.class_name
            cls_label = self.op_to_label.get(op_name, -1)

        return {
            'x': torch.from_numpy(injected).float(),
            'mask': torch.from_numpy(mask).float(),
            'cls_label': cls_label,
            'has_anomaly': instance is not None
        }