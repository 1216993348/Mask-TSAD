"""
数据加载器 - 支持本地和 Hugging Face
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import glob
import re
import shutil
from tempfile import mkdtemp


class DataLoader:
    def __init__(self, config_path="configs/datasets_config.yaml", use_hf=True, hf_repo_id=None):
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        config_full_path = project_root / config_path

        with open(config_full_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.project_root = project_root
        self.use_hf = use_hf
        self.hf_repo_id = hf_repo_id or "x1216993348/tsad-benchmark"

    def load_dataset(self, name):
        cfg = self.config['datasets'][name]
        local_data_dir = self.project_root / cfg['data_dir']

        if self._check_local_data_exists(name, cfg, local_data_dir):
            print(f"📁 Loading {name} from local: {local_data_dir}")
            return self._load_from_local(name)
        elif self.use_hf:
            print(f"📥 Local data not found, downloading {name} from Hugging Face...")
            return self._download_and_load_from_hf(name)
        else:
            raise FileNotFoundError(f"Local data not found at {local_data_dir} and use_hf=False")

    def _check_local_data_exists(self, name, cfg, local_data_dir):
        if not local_data_dir.exists():
            return False

        if name == 'SMD':
            train_dir = local_data_dir / cfg['train_dir']
            test_dir = local_data_dir / cfg['test_dir']
            return train_dir.exists() and test_dir.exists()

        elif name in ['SMAP', 'MSL']:
            train_file = local_data_dir / cfg['files']['train']
            test_file = local_data_dir / cfg['files']['test']
            label_file = local_data_dir / cfg['files']['label']
            return train_file.exists() and test_file.exists() and label_file.exists()

        elif name == 'SWaT':
            test_file = local_data_dir / cfg['files']['test']
            label_file = local_data_dir / cfg['files'].get('label', 'SWaT_test_label.npy')
            return test_file.exists() and label_file.exists()

        elif name == 'PSM':
            train_file = local_data_dir / cfg['files']['train']
            test_file = local_data_dir / cfg['files']['test']
            label_file = local_data_dir / cfg['files']['label']
            return train_file.exists() and test_file.exists() and label_file.exists()

        return False

    def _download_and_load_from_hf(self, name):
        from huggingface_hub import hf_hub_download, snapshot_download

        cfg = self.config['datasets'][name]
        local_data_dir = self.project_root / cfg['data_dir']
        print(f"   Downloading to: {local_data_dir}")
        local_data_dir.mkdir(parents=True, exist_ok=True)

        if name == 'SMAP' or name == 'MSL':
            return self._download_smap_msl(name, cfg, local_data_dir)
        elif name == 'SMD':
            return self._download_smd(name, cfg, local_data_dir)
        elif name == 'SWaT':
            return self._download_swat(name, cfg, local_data_dir)
        elif name == 'PSM':
            return self._download_psm(name, cfg, local_data_dir)
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def _download_smap_msl(self, name, cfg, local_data_dir):
        from huggingface_hub import hf_hub_download

        tmp_dir = mkdtemp()
        try:
            hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=f"{name}/{cfg['files']['train']}",
                repo_type="dataset",
                local_dir=tmp_dir,
                local_dir_use_symlinks=False
            )
            hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=f"{name}/{cfg['files']['test']}",
                repo_type="dataset",
                local_dir=tmp_dir,
                local_dir_use_symlinks=False
            )
            hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=f"{name}/{cfg['files']['label']}",
                repo_type="dataset",
                local_dir=tmp_dir,
                local_dir_use_symlinks=False
            )

            downloaded_dir = Path(tmp_dir) / name
            if downloaded_dir.exists():
                for f in downloaded_dir.iterdir():
                    target = local_data_dir / f.name
                    shutil.move(str(f), str(target))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        X_train = np.load(local_data_dir / cfg['files']['train'])
        X_test = np.load(local_data_dir / cfg['files']['test'])
        y_test = np.load(local_data_dir / cfg['files']['label'])

        if cfg['shape_check']['transpose_if_dims_smaller']:
            if X_train.shape[0] < X_train.shape[1]:
                X_train = X_train.T
                np.save(local_data_dir / cfg['files']['train'], X_train)
            if X_test.shape[0] < X_test.shape[1]:
                X_test = X_test.T
                np.save(local_data_dir / cfg['files']['test'], X_test)

        y_test = (y_test == 1).astype(int)

        print(f"   ✓ Downloaded and saved to {local_data_dir}")
        print(f"   ✓ Train: {X_train.shape}")
        print(f"   ✓ Test: {X_test.shape}, anomaly rate={y_test.mean():.2%}")

        return {
            'name': name,
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test,
            'config': cfg
        }

    def _download_smd(self, name, cfg, local_data_dir):
        from huggingface_hub import snapshot_download

        tmp_dir = mkdtemp()
        try:
            snapshot_download(
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                local_dir=tmp_dir,
                allow_patterns=[f"{name}/*"],
                local_dir_use_symlinks=False
            )

            downloaded_dir = Path(tmp_dir) / name
            if downloaded_dir.exists():
                for sub_dir in ['train', 'test', 'labels', 'interpretation_label']:
                    src = downloaded_dir / sub_dir
                    if src.exists():
                        dst = local_data_dir / sub_dir
                        if dst.exists():
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return self._load_smd_custom(name, cfg)

    def _download_swat(self, name, cfg, local_data_dir):
        from huggingface_hub import hf_hub_download

        tmp_dir = mkdtemp()
        try:
            hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=f"{name}/{cfg['files']['test']}",
                repo_type="dataset",
                local_dir=tmp_dir,
                local_dir_use_symlinks=False
            )
            if 'label' in cfg['files']:
                hf_hub_download(
                    repo_id=self.hf_repo_id,
                    filename=f"{name}/{cfg['files']['label']}",
                    repo_type="dataset",
                    local_dir=tmp_dir,
                    local_dir_use_symlinks=False
                )
            if 'train' in cfg['files']:
                try:
                    hf_hub_download(
                        repo_id=self.hf_repo_id,
                        filename=f"{name}/{cfg['files']['train']}",
                        repo_type="dataset",
                        local_dir=tmp_dir,
                        local_dir_use_symlinks=False
                    )
                except:
                    pass

            downloaded_dir = Path(tmp_dir) / name
            if downloaded_dir.exists():
                for f in downloaded_dir.iterdir():
                    target = local_data_dir / f.name
                    shutil.move(str(f), str(target))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        X_test = np.load(local_data_dir / cfg['files']['test'])
        label_path = local_data_dir / cfg['files'].get('label', 'SWaT_test_label.npy')
        if label_path.exists():
            y_test = np.load(label_path)
            y_test = (y_test == 1).astype(int)
        else:
            raise FileNotFoundError(f"Label file not found: {label_path}")

        train_path = local_data_dir / cfg['files'].get('train', 'SWaT_train.npy')
        if train_path.exists():
            X_train = np.load(train_path)
        else:
            X_train = X_test[y_test == 0]
            np.save(local_data_dir / "SWaT_train.npy", X_train)

        print(f"   ✓ Downloaded and saved to {local_data_dir}")
        return {
            'name': name,
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test,
            'config': cfg
        }

    def _download_psm(self, name, cfg, local_data_dir):
        from huggingface_hub import hf_hub_download

        tmp_dir = mkdtemp()
        try:
            hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=f"{name}/{cfg['files']['train']}",
                repo_type="dataset",
                local_dir=tmp_dir,
                local_dir_use_symlinks=False
            )
            hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=f"{name}/{cfg['files']['test']}",
                repo_type="dataset",
                local_dir=tmp_dir,
                local_dir_use_symlinks=False
            )
            hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=f"{name}/{cfg['files']['label']}",
                repo_type="dataset",
                local_dir=tmp_dir,
                local_dir_use_symlinks=False
            )

            downloaded_dir = Path(tmp_dir) / name
            if downloaded_dir.exists():
                for f in downloaded_dir.iterdir():
                    target = local_data_dir / f.name
                    shutil.move(str(f), str(target))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        X_train = pd.read_csv(local_data_dir / cfg['files']['train']).values
        X_test = pd.read_csv(local_data_dir / cfg['files']['test']).values
        y_test = pd.read_csv(local_data_dir / cfg['files']['label']).values.flatten()
        y_test = (y_test == 1).astype(int)

        return {
            'name': name,
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test,
            'config': cfg
        }

    # ========== 本地加载方法 ==========

    def _load_from_local(self, name):
        cfg = self.config['datasets'][name]
        if name == 'SMD':
            return self._load_smd_custom(name, cfg)
        elif name == 'SWaT':
            return self._load_swat_custom(name, cfg)
        elif name in ['SMAP', 'MSL']:
            return self._load_nasa(name, cfg)
        elif name == 'PSM':
            return self._load_csv(name, cfg)
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def _load_smd_custom(self, name, cfg):
        data_dir = self.project_root / cfg['data_dir']
        train_files = sorted(glob.glob(str(data_dir / 'train' / 'machine-*.txt')))
        print(f"  找到 {len(train_files)} 个机器实体")

        all_X_train, all_X_test, all_y_test = [], [], []
        machine_names = []
        machine_info = {}

        for train_file in train_files:
            machine_name = Path(train_file).stem
            machine_names.append(machine_name)

            test_file = data_dir / 'test' / f'{machine_name}.txt'
            label_file = data_dir / 'labels' / f'{machine_name}.txt'
            interp_file = data_dir / 'interpretation_label' / f'{machine_name}.txt'

            if not test_file.exists() or not label_file.exists():
                print(f"  ⚠️ 跳过 {machine_name}: 缺少文件")
                continue

            X_train = self._load_smd_txt(train_file)
            X_test = self._load_smd_txt(test_file)
            y_test = self._load_smd_label(label_file)

            if interp_file.exists():
                anomaly_info = self._parse_interpretation(interp_file)
                machine_info[machine_name] = anomaly_info

            if X_train.shape[1] != X_test.shape[1]:
                min_dim = min(X_train.shape[1], X_test.shape[1])
                X_train = X_train[:, :min_dim]
                X_test = X_test[:, :min_dim]

            all_X_train.append(X_train)
            all_X_test.append(X_test)
            all_y_test.append(y_test.flatten())

        X_train = np.concatenate(all_X_train, axis=0)
        X_test = np.concatenate(all_X_test, axis=0)
        y_test = np.concatenate(all_y_test, axis=0)

        print(f"  ✓ SMD加载完成")
        print(f"    - 训练集: {X_train.shape}")
        print(f"    - 测试集: {X_test.shape}, 异常率={y_test.mean():.2%}")
        print(f"    - 实体数: {len(machine_names)}")

        return {
            'name': name,
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test,
            'config': cfg,
            'n_entities': len(machine_names),
            'machine_names': machine_names,
            'machine_info': machine_info
        }

    def _load_smd_txt(self, filepath):
        df = pd.read_csv(filepath, header=None, encoding='utf-8')
        return df.values.astype(np.float32)

    def _load_smd_label(self, filepath):
        labels = np.loadtxt(filepath, delimiter=',')
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        return (labels > 0).astype(int)

    def _parse_interpretation(self, filepath):
        info = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'(\d+)-(\d+):(.+)', line)
                if match:
                    start = int(match.group(1))
                    end = int(match.group(2))
                    dims = [int(x) for x in match.group(3).split(',')]
                    info.append({
                        'start': start,
                        'end': end,
                        'affected_dims': dims
                    })
        return info

    def _load_swat_custom(self, name, cfg):
        data_dir = self.project_root / cfg['data_dir']
        X_test = np.load(data_dir / cfg['files']['test'])
        label_file = data_dir / cfg['files']['label']
        if not label_file.exists():
            raise FileNotFoundError(f"找不到标签文件 {label_file}")
        y_test = np.load(label_file)
        y_test = (y_test == 1).astype(int)
        print(f"  ✓ 测试集: {X_test.shape}, 异常率={y_test.mean():.2%}")

        train_file = data_dir / cfg['files'].get('train', 'SWaT_train.npy')
        if train_file.exists():
            X_train = np.load(train_file)
        else:
            X_train = X_test[y_test == 0]
            print(f"  ⚠️ 使用测试集中的正常数据作为训练集")

        print(f"  ✓ 训练集: {X_train.shape}")
        return {
            'name': name,
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test,
            'config': cfg
        }

    def _load_nasa(self, name, cfg):
        data_dir = self.project_root / cfg['data_dir']
        X_train = np.load(data_dir / cfg['files']['train'])
        X_test = np.load(data_dir / cfg['files']['test'])
        y_test = np.load(data_dir / cfg['files']['label'])

        if cfg['shape_check']['transpose_if_dims_smaller']:
            if X_train.shape[0] < X_train.shape[1]:
                X_train = X_train.T
            if X_test.shape[0] < X_test.shape[1]:
                X_test = X_test.T

        y_test = (y_test == 1).astype(int)

        print(f"  ✓ 训练集: {X_train.shape}")
        print(f"  ✓ 测试集: {X_test.shape}, 异常率={y_test.mean():.2%}")
        return {
            'name': name,
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test,
            'config': cfg
        }

    def _load_csv(self, name, cfg):
        data_dir = self.project_root / cfg['data_dir']
        X_train = pd.read_csv(data_dir / cfg['files']['train'], **cfg['csv_format']).values
        X_test = pd.read_csv(data_dir / cfg['files']['test'], **cfg['csv_format']).values
        y_test = pd.read_csv(data_dir / cfg['files']['label'], **cfg['csv_format']).values.flatten()
        y_test = (y_test == 1).astype(int)

        print(f"  ✓ 训练集: {X_train.shape}")
        print(f"  ✓ 测试集: {X_test.shape}, 异常率={y_test.mean():.2%}")
        return {
            'name': name,
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test,
            'config': cfg
        }

    def get_segments(self, y):
        segs = []
        start = None
        for i in range(len(y)):
            if y[i] == 1 and start is None:
                start = i
            elif y[i] == 0 and start is not None:
                segs.append((start, i - 1))
                start = None
        if start is not None:
            segs.append((start, len(y) - 1))
        return segs