"""
配置加载工具
"""

import yaml
import torch
from pathlib import Path
from types import SimpleNamespace


def load_full_config(config_path="configs/datasets_config.yaml"):
    """加载完整配置"""
    project_root = Path(__file__).parent.parent.parent
    config_full_path = project_root / config_path

    with open(config_full_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    return full_config


def get_config_for_dataset(dataset_name, full_config):
    """获取数据集的配置"""
    dataset_cfg = full_config['datasets'][dataset_name]
    output_cfg = full_config.get('output', {})

    cfg = SimpleNamespace()
    cfg.dataset = dataset_name
    cfg.seq_len = dataset_cfg.get('model', {}).get('seq_len', 512)
    cfg.batch_size = dataset_cfg.get('model', {}).get('batch_size', 32)
    cfg.num_samples = dataset_cfg.get('model', {}).get('num_samples', 10000)
    cfg.save_dir = str(Path(__file__).parent.parent.parent / output_cfg.get('save_dir', 'output/simple_models'))

    # 模型参数默认值
    cfg.d_model = 256
    cfg.nhead = 8
    cfg.num_encoder_layers = 4
    cfg.dim_feedforward = 1024
    cfg.dropout = 0.1
    cfg.num_classes = 6

    # 训练参数默认值
    cfg.epochs = 50
    cfg.lr = 0.0001
    cfg.weight_decay = 0.0001
    cfg.clip_grad_norm = 1.0
    cfg.val_ratio = 0.2

    cfg.num_workers = 4
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return cfg