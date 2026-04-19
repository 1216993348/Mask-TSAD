import json
import numpy as np
from pathlib import Path
from datetime import datetime

from data_loader import DataLoader
from operator_analyzer import OperatorAnalyzer
from visualizer import Visualizer


class TSAnalyzer:
    def __init__(self):
        self.loader = DataLoader()
        self.viz = Visualizer()
        self.all_results = {}

    def analyze_all_datasets(self, dataset_names=None):
        """分析所有数据集"""
        if dataset_names is None:
            dataset_names = ['SMAP', 'MSL', 'SMD', 'SWaT']

        all_analysis_summary = {}

        for name in dataset_names:
            print(f"\n{'=' * 60}")
            print(f"Analyzing {name}...")
            print(f"{'=' * 60}")

            try:
                # 1. 加载数据
                data = self.loader.load_dataset(name)
                print(f"✓ Data loaded: {data['X_test'].shape}, Anomaly ratio: {data['y_test'].mean():.2%}")

                # 2. 算子分析
                analyzer = OperatorAnalyzer(data)
                results = analyzer.analyze_all()
                self.all_results[name] = results

                # 3. 打印结果
                self._print_results(results)

                # 4. 生成图表
                fig_dir = self.viz.create_paper_figures(data, results)
                print(f"✓ Figures saved to: {fig_dir}")

                # 5. 保存结果（传入 analyzer 参数）
                self._save_results(name, results, data, analyzer)  # ← 修复：添加 analyzer 参数

                # 6. 收集到汇总
                all_analysis_summary[name] = {
                    'operator_scores': {
                        'value': results['operator_1_value'].get('score', 0),
                        'trend': results['operator_2_trend'].get('score', 0),
                        'temporal': results['operator_3_temporal'].get('score', 0),
                        'dependency': results['operator_4_dependency'].get('score', 0)
                    },
                    'dominant': results['summary']['dominant_operator'],
                    'severity': results['summary']['severity_score'],
                    'qualitative': {
                        'value': results['operator_1_value'].get('qualitative', 'N/A'),
                        'temporal': results['operator_3_temporal'].get('qualitative', 'N/A'),
                        'dependency': results['operator_4_dependency'].get('qualitative', 'N/A')
                    }
                }

            except Exception as e:
                print(f"✗ Error analyzing {name}: {e}")
                import traceback
                traceback.print_exc()

        # 保存汇总报告
        self._save_summary_report(all_analysis_summary)

        return all_analysis_summary

    def _print_results(self, results):
        """打印分析结果"""
        print("\n" + "-" * 40)
        print("OPERATOR ANALYSIS RESULTS")
        print("-" * 40)

        op1 = results['operator_1_value']
        if isinstance(op1, dict):
            print(f"\n[Operator 1: Value Perturbation]")
            print(f"  Score: {op1.get('score', 0):.4f}")
            print(f"  Qualitative: {op1.get('qualitative', 'N/A')}")
            print(f"  Mean Shift: {op1.get('mean_shift', 0):.4f}")

        op2 = results['operator_2_trend']
        if isinstance(op2, dict):
            print(f"\n[Operator 2: Trend/Drift]")
            print(f"  Score: {op2.get('score', 0):.4f}")
            print(f"  Qualitative: {op2.get('qualitative', 'N/A')}")

        op3 = results['operator_3_temporal']
        if isinstance(op3, dict):
            print(f"\n[Operator 3: Temporal Warping]")
            print(f"  Score: {op3.get('score', 0):.4f}")
            print(f"  Qualitative: {op3.get('qualitative', 'N/A')}")

        op4 = results['operator_4_dependency']
        if isinstance(op4, dict):
            print(f"\n[Operator 4: Dependency Break]")
            print(f"  Score: {op4.get('score', 0):.4f}")
            print(f"  Qualitative: {op4.get('qualitative', 'N/A')}")
            print(f"  Corr Diff: {op4.get('corr_diff', 0):.4f}")

        print("\n" + "-" * 40)
        print("SUMMARY")
        print("-" * 40)
        print(f"Dominant Operator: {results['summary']['dominant_operator']}")
        print(f"Severity Score: {results['summary']['severity_score']:.4f}")

    def _save_results(self, name, results, data, analyzer):
        """保存结果到文件 - 包括所有5种图表的源数据"""

        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        output_dir = project_root / "output/results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存主结果
        serializable_results = self._make_serializable(results)
        with open(output_dir / f"{name}_results.json", 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)

        # 2. 保存 per-dimension 数据（fig4）
        per_dim = analyzer.get_per_dimension_for_fig4()
        if per_dim:
            with open(output_dir / f"{name}_per_dimension.json", 'w', encoding='utf-8') as f:
                json.dump(per_dim, f, indent=2)

        # 3. 保存 correlation 数据（fig3）
        corr_data = analyzer.get_correlation_for_fig3()
        if corr_data:
            with open(output_dir / f"{name}_correlation.json", 'w', encoding='utf-8') as f:
                json.dump(corr_data, f, indent=2)

        # 4. 保存 temporal ACF 数据（fig3的时间部分）
        acf_data = analyzer.get_acf_for_fig3_temporal()
        if acf_data:
            with open(output_dir / f"{name}_temporal_acf.json", 'w', encoding='utf-8') as f:
                json.dump(acf_data, f, indent=2)

        # 5. 保存时序演变数据（fig5）
        evolution_data = analyzer.get_temporal_evolution_for_fig5()
        if evolution_data and evolution_data['windows']:
            with open(output_dir / f"{name}_temporal_evolution.json", 'w', encoding='utf-8') as f:
                json.dump(evolution_data, f, indent=2)

        # 6. 保存异常段数据（fig1）
        segments_data = analyzer.get_anomaly_segments_for_fig1(max_segments=3)
        if segments_data:
            with open(output_dir / f"{name}_anomaly_segments.json", 'w', encoding='utf-8') as f:
                json.dump(segments_data, f, indent=2)

        print(f"✓ All source data saved to: {output_dir}/{name}_*.json")

    def _save_summary_report(self, all_analysis_summary):
        """保存汇总报告"""
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        output_dir = project_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成汇总报告
        report = []
        report.append("# TSAD 4-Operator Analysis Summary Report")
        report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Overview")
        report.append("\n| Dataset | Value | Trend | Temporal | Dependency | Dominant | Severity |")
        report.append("|---------|-------|-------|----------|------------|----------|----------|")

        for name, summary in all_analysis_summary.items():
            scores = summary['operator_scores']
            report.append(
                f"| {name} | {scores['value']:.3f} | {scores['trend']:.3f} | {scores['temporal']:.3f} | {scores['dependency']:.3f} | {summary['dominant']} | {summary['severity']:.3f} |")

        report.append("\n## Detailed Analysis")

        for name, summary in all_analysis_summary.items():
            report.append(f"\n### {name}")
            report.append(f"\n**Dominant Anomaly Type**: {summary['dominant']}")
            report.append(f"\n**Operator Scores**:")
            report.append(
                f"- Value Perturbation: {summary['operator_scores']['value']:.4f} ({summary['qualitative']['value']})")
            report.append(
                f"- Temporal Warping: {summary['operator_scores']['temporal']:.4f} ({summary['qualitative']['temporal']})")
            report.append(
                f"- Dependency Break: {summary['operator_scores']['dependency']:.4f} ({summary['qualitative']['dependency']})")

        report.append("\n---")
        report.append("*Report automatically generated by TSAD 4-Operator Analysis System*")

        with open(output_dir / "analysis_summary.md", 'w', encoding='utf-8') as f:
            f.write("\n".join(report))

        # 同时保存JSON格式的汇总
        with open(output_dir / "analysis_summary.json", 'w', encoding='utf-8') as f:
            json.dump(all_analysis_summary, f, indent=2)

        print(f"\n✓ Summary report saved to: {output_dir}/analysis_summary.*")

    def _make_serializable(self, obj):
        """将numpy类型转换为Python原生类型"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj


if __name__ == "__main__":
    print("=" * 60)
    print("TSAD 4-Operator Anomaly Analysis System")
    print("=" * 60)

    # 运行分析
    analyzer = TSAnalyzer()

    # 分析所有数据集
    summary = analyzer.analyze_all_datasets(['SMD', 'SWaT', 'SMAP', 'MSL'])

    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")
    print("=" * 60)
    print("\nOutput files:")
    print("  - output/analysis_summary.json  (汇总数据)")
    print("  - output/analysis_summary.md    (汇总报告)")
    print("  - output/results/*.json         (各数据集详细数据)")
    print("  - output/figures/*/             (可视化图表)")