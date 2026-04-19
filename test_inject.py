"""
测试脚本 - 遍历所有算子
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.inject.injector import UniversalAnomalyInjector
from src.inject.anomaly_types import AnomalyClass
from src.visualizer import Visualizer
from src.data_loader import DataLoader
from src.inject.operators import OPERATOR_MAP


def load_datasets():
    print("Loading datasets...")
    loader = DataLoader()
    datasets = {}
    for name in ['SMD', 'SWaT', 'MSL', 'SMAP']:
        try:
            data = loader.load_dataset(name)
            datasets[name] = data['X_test']
            print(f"  Loaded {name}: {data['X_test'].shape}")
        except Exception as e:
            print(f"  Failed {name}: {e}")
    return datasets


def create_single_operator_injector(op_id, op_class):
    """创建只包含一个算子的注入器"""

    class SimpleGen:
        def sample_class_id(self):
            return op_id

    injector = UniversalAnomalyInjector(num_classes=7)
    injector.generator = SimpleGen()

    injector.anomaly_classes = {
        op_id: AnomalyClass(
            id=op_id,
            name=op_class.__name__,
            operators=[op_id],
            intensities=[0.0],
            length_ratio=0.05,
        )
    }
    injector.num_classes = 1
    return injector


def create_analysis_results(original_data, injected_data, instance, name):
    """创建分析结果字典"""
    T = len(original_data)

    if instance is None:
        return {
            'name': name,
            'operator_1_value': {'score': 0.0, 'qualitative': 'No injection'},
            'operator_2_trend': {'score': 0.0, 'qualitative': 'No injection'},
            'operator_3_temporal': {'score': 0.0, 'qualitative': 'No injection'},
            'operator_4_dependency': {'score': 0.0, 'qualitative': 'No injection'},
            'summary': {'dominant_operator': 'none', 'severity_score': 0.0}
        }

    from src.operator_analyzer import OperatorAnalyzer

    y_test = np.zeros(T)
    y_test[instance.start:instance.end] = 1

    data_dict = {
        'name': name,
        'X_train': original_data,
        'X_test': injected_data,
        'y_test': y_test,
        'config': {'analysis': {'max_dims_for_corr': 50}}
    }

    analyzer = OperatorAnalyzer(data_dict)
    return analyzer.analyze_all()

def run_test(save_dir="output/operator_tests"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    viz = Visualizer(output_dir=save_dir)
    datasets = load_datasets()

    for ds_name, data in datasets.items():
        print(f"\n{'='*50}\nDataset: {ds_name}\n{'='*50}")
        sample = data[:min(5000, data.shape[0])].copy()

        for op_id, op_class in OPERATOR_MAP.items():
            op_name = op_class.__name__
            print(f"\n  Testing: {op_name} (ID={op_id})")

            for i in range(2):
                injector = create_single_operator_injector(op_id, op_class)
                injected, inst = injector.inject(sample.copy())

                if inst is None:
                    print(f"    inj{i+1}: FAILED")
                    continue

                print(f"    inj{i+1}: OK, pos=[{inst.start},{inst.end}), dims={inst.affected_dims[:3]}")

                y = np.zeros(len(sample))
                y[inst.start:inst.end] = 1

                results = {
                    'name': f"{ds_name}_{op_name}_{i}",
                    'X_train': sample,
                    'X_test': injected,
                    'y_test': y,
                    'config': {'analysis': {'max_dims_for_corr': 50}}
                }

                analysis = create_analysis_results(sample, injected, inst, f"{ds_name}_{op_name}_{i}")

                # ===== 新增：构建 injection_details =====
                op_display = op_name  # 直接使用算子名
                injection_details = {}
                for dim in inst.affected_dims:
                    injection_details[dim] = {
                        'operators': inst.operators,
                        'class_name': inst.class_name,
                        'intensities': inst.intensities,
                        'display': op_display
                    }

                out_dir = save_path / ds_name / op_name / f"inj{i+1}"
                out_dir.mkdir(parents=True, exist_ok=True)

                old = viz.output_dir
                viz.output_dir = out_dir

                try:
                    # ===== 传入 injection_details =====
                    viz.create_paper_figures(results, analysis,
                                             affected_dims=inst.affected_dims,
                                             injection_details=injection_details)
                    print(f"      ✓ Figures saved")
                except Exception as e:
                    print(f"      ✗ Figure error: {e}")

                viz.output_dir = old

    print(f"\nDone! Results in {save_path}")


if __name__ == "__main__":
    run_test()