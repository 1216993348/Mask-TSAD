"""
课程学习注入器 - 根据 epoch 渐进增加异常复杂度
"""

import numpy as np
from typing import Tuple, Optional, List
from src.inject.operators import OPERATOR_MAP


def get_operator_by_name(op_name: str):
    """根据算子名称获取算子类"""
    for op_id, op_class in OPERATOR_MAP.items():
        if op_class.__name__ == op_name:
            return op_class
    raise ValueError(f"Unknown operator: {op_name}")


class CurriculumInjector:
    """
    课程学习注入器
    - 阶段1 (0-8): 仅 Spike，固定位置，单变量
    - 阶段2 (9-16): Spike + Shift，随机位置，单变量
    - 阶段3 (17-24): + Drift，随机位置，单变量
    - 阶段4 (25-32): + Missing，随机位置，1-2变量
    - 阶段5 (33-40): + Period，随机位置，1-2变量，允许2算子叠加
    - 阶段6 (41-50): + Cascade，随机位置，1-3变量，允许2-3算子叠加
    """

    def __init__(self, total_epochs: int = 50):
        self.total_epochs = total_epochs

        # 阶段定义
        self.stage_ops = {
            1: ["Spike"],
            2: ["Spike", "Shift"],
            3: ["Spike", "Shift", "Drift"],
            4: ["Spike", "Shift", "Drift", "Missing"],
            5: ["Spike", "Shift", "Drift", "Missing", "Period"],
            6: ["Spike", "Shift", "Drift", "Missing", "Period", "Cascade"],
        }

        # 每个阶段的参数
        self.stage_params = {
            1: {
                'length': (8, 12),
                'intensity': (0.5, 0.8),
                'n_dims': 1,
                'position': 'fixed',  # fixed: 中间位置
                'n_ops': 1,  # 只能选1个算子
            },
            2: {
                'length': (8, 15),
                'intensity': (0.6, 1.0),
                'n_dims': 1,
                'position': 'random',
                'n_ops': 1,
            },
            3: {
                'length': (10, 20),
                'intensity': (0.7, 1.2),
                'n_dims': 1,
                'position': 'random',
                'n_ops': 1,
            },
            4: {
                'length': (10, 25),
                'intensity': (0.8, 1.3),
                'n_dims': (1, 2),
                'position': 'random',
                'n_ops': 1,
            },
            5: {
                'length': (12, 30),
                'intensity': (0.9, 1.5),
                'n_dims': (1, 2),
                'position': 'random',
                'n_ops': (1, 2),  # 允许1-2个算子叠加
            },
            6: {
                'length': (15, 40),
                'intensity': (1.0, 2.0),
                'n_dims': (1, 3),
                'position': 'random',
                'n_ops': (1, 3),  # 允许1-3个算子叠加
            },
        }

    def get_stage(self, epoch: int) -> int:
        """根据 epoch 返回当前阶段 1-6"""
        if epoch < 8:
            return 1
        elif epoch < 16:
            return 2
        elif epoch < 24:
            return 3
        elif epoch < 32:
            return 4
        elif epoch < 40:
            return 5
        else:
            return 6

    def _sample_ops(self, ops_list: List[str], stage: int) -> List[str]:
        """采样算子"""
        params = self.stage_params[stage]
        n_ops = params['n_ops']

        if isinstance(n_ops, tuple):
            k = np.random.randint(n_ops[0], n_ops[1] + 1)
        else:
            k = n_ops

        k = min(k, len(ops_list))
        return np.random.choice(ops_list, size=k, replace=False).tolist()

    def inject(self, ts_data: np.ndarray, epoch: int) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """
        注入异常

        Args:
            ts_data: (T, D) 原始时间序列
            epoch: 当前 epoch

        Returns:
            injected_data: (T, D) 注入后的数据
            mask: (T,) 异常掩码
            used_ops: 使用的算子列表
            affected_dims: 受影响的维度列表
        """
        T, D = ts_data.shape
        stage = self.get_stage(epoch)

        ops_list = self.stage_ops[stage]
        params = self.stage_params[stage]

        # 1. 选择算子
        selected_ops = self._sample_ops(ops_list, stage)

        # 2. 采样长度
        if isinstance(params['length'], tuple):
            length = np.random.randint(params['length'][0], params['length'][1] + 1)
        else:
            length = params['length']
        length = min(length, T // 2)  # 不超过序列一半

        # 3. 采样位置
        if params['position'] == 'fixed':
            start = T // 2 - length // 2
        else:
            start = np.random.randint(0, max(1, T - length))
        end = min(start + length, T)
        length = end - start

        if length <= 0:
            return ts_data.copy(), np.zeros(T, dtype=np.float32), [], []

        # 4. 采样维度
        if isinstance(params['n_dims'], tuple):
            n_dims = np.random.randint(params['n_dims'][0], params['n_dims'][1] + 1)
        else:
            n_dims = params['n_dims']
        n_dims = min(n_dims, D)
        affected_dims = np.random.choice(D, size=n_dims, replace=False).tolist()

        # 5. 采样强度
        intensity = np.random.uniform(params['intensity'][0], params['intensity'][1])

        # 6. 注入异常
        injected = ts_data.copy()

        for op_name in selected_ops:
            op = get_operator_by_name(op_name)
            for dim in affected_dims:
                try:
                    injected = op.apply(injected, dim, intensity, start, end)
                except Exception as e:
                    # 如果算子出错，跳过
                    print(f"Warning: {op_name} failed on dim {dim}: {e}")
                    continue

        # 7. 构建掩码
        mask = np.zeros(T, dtype=np.float32)
        mask[start:end] = 1.0

        return injected, mask, selected_ops, affected_dims