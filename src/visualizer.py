import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置矢量图输出
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150


class Visualizer:
    def __init__(self, output_dir="output/figures"):
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        self.output_dir = project_root / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_paper_figures(self, results, analysis_results, affected_dims=None, injection_details=None):
        """生成论文级图表"""
        name = results['name']
        fig_dir = self.output_dir / name
        fig_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Generating figures for {name}...")

        self._fig_anomaly_waveform(results, fig_dir, affected_dims, injection_details)
        self._fig_operator_scores(analysis_results, fig_dir)
        self._fig_correlation_heatmap(results, fig_dir)
        self._fig_per_dimension_analysis(results, fig_dir)
        self._fig_temporal_mechanism_evolution(results, fig_dir)

        print(f"  ✓ All figures saved to {fig_dir}")
        return fig_dir

    def _get_segments(self, y):
        """提取异常段 - 独立实现，不依赖 DataLoader"""
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
    # ==================== Figure 1 ====================
    def _fig_anomaly_waveform(self, results, fig_dir, affected_dims=None, injection_details=None):
        """图1: 异常段 + 前后正常段对比 - 标注注入维度和机制类型"""
        X_test = results['X_test']
        y_test = results['y_test']

        if affected_dims is None:
            affected_dims = []

        if injection_details is None:
            injection_details = {}

        segments = self._get_segments(y_test)

        if len(segments) == 0:
            print("    No anomaly segments found, skipping fig1")
            return

        s, e = segments[len(segments) // 2]
        context = 200
        start = max(0, s - context)
        end = min(len(X_test), e + context)

        n_dims = X_test.shape[1]
        n_cols = 4
        n_rows = (n_dims + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
        axes = axes.flatten()

        # 动态获取算子名称
        from src.inject.operators import OPERATOR_MAP
        id_to_name = {op_id: op.__name__ for op_id, op in OPERATOR_MAP.items()}

        for d in range(n_dims):
            ax = axes[d]
            x_range = range(start, end)

            is_injected = d in affected_dims

            # 获取异常类型显示
            anomaly_display = ""
            if is_injected and d in injection_details:
                anomaly_display = injection_details[d].get('display', '')
                if not anomaly_display:
                    anomaly_display = injection_details[d].get('class_name', '')
                if not anomaly_display:
                    ops = injection_details[d].get('operators', [])
                    if ops:
                        anomaly_display = '+'.join([id_to_name.get(op, f'op{op}') for op in ops])

            if is_injected:
                ax.plot(x_range, X_test[start:end, d], 'r-', linewidth=1.2, alpha=0.9)
            else:
                ax.plot(x_range, X_test[start:end, d], 'b-', linewidth=0.8, alpha=0.7)

            ax.axvspan(s, e, alpha=0.3, color='red')
            ax.axvspan(start, s - 1, alpha=0.1, color='green')
            ax.axvspan(e + 1, end, alpha=0.1, color='green')

            if is_injected and anomaly_display:
                ax.set_ylabel(f'Dim {d}\n⚡{anomaly_display}', fontsize=7, color='red')
            elif is_injected:
                ax.set_ylabel(f'Dim {d} ⚡', fontsize=8, color='red')
            else:
                ax.set_ylabel(f'Dim {d}', fontsize=8)

            ax.grid(alpha=0.3)

            if d == 0:
                title = f'{results["name"]} - All Dimensions'
                if affected_dims:
                    title += f'\n⚡ = injected dimensions'
                title += f'\nAnomaly period: [{s}:{e}], length={e - s + 1}'
                ax.set_title(title, fontsize=10)
                ax.legend(['Injected' if is_injected else 'Normal', 'Anomaly', 'Normal Context'],
                          loc='upper right', fontsize=7)

        for d in range(n_dims, len(axes)):
            axes[d].set_visible(False)

        plt.tight_layout()
        plt.savefig(fig_dir / 'fig1_anomaly_waveform.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(fig_dir / 'fig1_anomaly_waveform.svg', format='svg')
        plt.close()
        print(f"    - fig1_anomaly_waveform.pdf (all dims, injected: {affected_dims if affected_dims else 'none'})")


    # ==================== Figure 2 ====================
    def _fig_operator_scores(self, analysis_results, fig_dir):
        """图2: 4算子得分柱状图"""
        operators = ['Value\nPerturbation', 'Trend\nDrift', 'Temporal\nWarping', 'Dependency\nBreak']

        scores = [
            analysis_results['operator_1_value'].get('score', 0),
            analysis_results['operator_2_trend'].get('score', 0),
            analysis_results['operator_3_temporal'].get('score', 0),
            analysis_results['operator_4_dependency'].get('score', 0)
        ]
        scores = [0 if np.isnan(s) else s for s in scores]

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(operators, scores, color=colors, edgecolor='black', linewidth=1.5)

        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Moderate (0.3)')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='High (0.5)')

        ax.set_ylim(0, max(scores) * 1.2 if max(scores) > 0 else 1.0)
        ax.set_ylabel('Anomaly Score (0=normal, 1=strong)', fontsize=12)
        ax.set_title(f'{analysis_results["name"]} - 4-Operator Anomaly Mechanism Analysis', fontsize=14,
                     fontweight='bold')
        ax.set_xticklabels(operators, fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        explanation = (
            "Interpretation:\n"
            "• Value: spike / drift / measurement error\n"
            "• Trend: calibration drift / slow degradation\n"
            "• Temporal: delay / acceleration / warping\n"
            "• Dependency: multi-sensor inconsistency"
        )
        ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        plt.tight_layout()
        plt.savefig(fig_dir / 'fig2_operator_scores.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(fig_dir / 'fig2_operator_scores.svg', format='svg')
        plt.close()
        print("    - fig2_operator_scores.pdf")

    # ==================== Figure 3 ====================
    def _fig_correlation_heatmap(self, results, fig_dir):
        """图3: 相关矩阵对比"""
        X_test = results['X_test']
        y_test = results['y_test']

        X_normal = X_test[y_test == 0]
        X_anomaly = X_test[y_test == 1]

        if len(X_normal) == 0 or len(X_anomaly) == 0:
            print("    No normal or anomaly samples, skipping fig3")
            return

        max_dims = min(20, X_normal.shape[1])
        from sklearn.decomposition import PCA

        try:
            pca = PCA(n_components=max_dims)
            X_n_pca = pca.fit_transform(X_normal[:min(5000, len(X_normal))])
            X_a_pca = pca.transform(X_anomaly[:min(5000, len(X_anomaly))])

            corr_n = np.corrcoef(X_n_pca.T)
            corr_a = np.corrcoef(X_a_pca.T)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            im1 = axes[0].imshow(corr_n, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0].set_title('Normal Period\n(stable correlations)', fontsize=12)
            axes[0].set_xlabel('PC Index')
            axes[0].set_ylabel('PC Index')

            im2 = axes[1].imshow(corr_a, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1].set_title('Anomaly Period\n(changed correlations)', fontsize=12)
            axes[1].set_xlabel('PC Index')

            diff = np.abs(corr_n - corr_a)
            im3 = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
            axes[2].set_title('Correlation Difference\n(0=no change, 0.5=large change)', fontsize=12)
            axes[2].set_xlabel('PC Index')

            plt.colorbar(im1, ax=axes[0], fraction=0.046, label='Correlation')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, label='Correlation')
            plt.colorbar(im3, ax=axes[2], fraction=0.046, label='Absolute Difference')

            total_change = np.mean(diff)
            axes[2].text(0.02, 0.98, f'Mean Change: {total_change:.3f}',
                         transform=axes[2].transAxes, fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.suptitle(f'{results["name"]} - Dependency Structure Analysis\n'
                         f'(PC={max_dims}, explained variance={pca.explained_variance_ratio_.sum():.1%})',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(fig_dir / 'fig3_correlation_heatmap.pdf', format='pdf', bbox_inches='tight')
            plt.savefig(fig_dir / 'fig3_correlation_heatmap.svg', format='svg')
            plt.close()
            print("    - fig3_correlation_heatmap.pdf")
        except Exception as e:
            print(f"    Warning: fig3 failed - {e}")

    # ==================== Figure 4 (修复版：无特殊处理) ====================
    def _fig_per_dimension_analysis(self, results, fig_dir):
        """图4: 逐维度分析 - 客观版本，无特殊处理"""
        X_test = results['X_test']
        y_test = results['y_test']

        X_normal = X_test[y_test == 0]
        X_anomaly = X_test[y_test == 1]

        if len(X_normal) == 0 or len(X_anomaly) == 0:
            print("    No normal or anomaly samples, skipping fig4")
            return

        # 计算
        mean_n = X_normal.mean(axis=0)
        mean_a = X_anomaly.mean(axis=0)
        std_n = X_normal.std(axis=0)
        std_a = X_anomaly.std(axis=0)

        # 客观计算，不特殊处理
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_shift = np.abs(mean_a - mean_n) / (std_n + 1e-10)
            var_ratio = std_a / (std_n + 1e-10)

        # 只保留正常时期有波动的维度 (std_n > 0)
        valid = std_n > 1e-10
        valid_indices = np.where(valid)[0]

        if len(valid_indices) == 0:
            print("    No valid dimensions with normal variance, skipping fig4")
            return

        mean_shift_valid = mean_shift[valid]
        var_ratio_valid = var_ratio[valid]
        valid_dims = valid_indices

        # 找出变化最大的维度（各取前15）
        top_n = min(15, len(valid_dims))

        # 左图：按 mean_shift 排序
        left_order = np.argsort(mean_shift_valid)[-top_n:][::-1]
        left_indices = valid_dims[left_order]
        left_values = mean_shift_valid[left_order]

        # 右图：按 var_ratio 排序
        right_order = np.argsort(var_ratio_valid)[-top_n:][::-1]
        right_indices = valid_dims[right_order]
        right_values = var_ratio_valid[right_order]

        print(f"    Left top dims: {left_indices[:5].tolist()}")
        print(f"    Left top values: {[round(v, 4) for v in left_values[:5]]}")
        print(f"    Right top dims: {right_indices[:5].tolist()}")
        print(f"    Right top values: {[round(v, 1) for v in right_values[:5]]}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图
        colors_left = plt.cm.Blues(np.linspace(0.4, 0.9, len(left_values)))
        axes[0].barh(range(len(left_values)), left_values, color=colors_left)
        axes[0].set_yticks(range(len(left_values)))
        axes[0].set_yticklabels([f'Dim {int(d)}' for d in left_indices])
        axes[0].set_xlabel('Mean Shift (standard deviations)', fontsize=11)
        axes[0].set_title(f'Top {top_n} Dimensions by Mean Shift\n(Value offset during anomaly)', fontsize=12)
        axes[0].axvline(x=1, color='red', linestyle='--', alpha=0.7, label='1 std threshold')
        axes[0].legend(loc='lower right')
        axes[0].grid(alpha=0.3, axis='x')

        for i, val in enumerate(left_values):
            axes[0].text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

        # 右图
        colors_right = plt.cm.Oranges(np.linspace(0.4, 0.9, len(right_values)))
        axes[1].barh(range(len(right_values)), right_values, color=colors_right)
        axes[1].set_yticks(range(len(right_values)))
        axes[1].set_yticklabels([f'Dim {int(d)}' for d in right_indices])
        axes[1].set_xlabel('Variance Ratio (anomaly_std / normal_std)', fontsize=11)
        axes[1].set_title(f'Top {top_n} Dimensions by Variance Ratio\n(Volatility change during anomaly)', fontsize=12)
        axes[1].axvline(x=1, color='red', linestyle='--', alpha=0.7, label='1x threshold')
        axes[1].legend(loc='lower right')
        axes[1].grid(alpha=0.3, axis='x')

        for i, val in enumerate(right_values):
            if val > 1000:
                label = f'{val:.1e}'
            elif val > 10:
                label = f'{val:.1f}'
            else:
                label = f'{val:.3f}'
            axes[1].text(val + 0.02, i, label, va='center', fontsize=9)

        plt.suptitle(f'{results["name"]} - Most Affected Dimensions During Anomaly\n'
                     f'(Left: value offset | Right: volatility change)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fig_dir / 'fig4_per_dimension_analysis.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(fig_dir / 'fig4_per_dimension_analysis.svg', format='svg')
        plt.close()
        print("    - fig4_per_dimension_analysis.pdf")

    # ==================== Figure 5 ====================
    def _fig_temporal_mechanism_evolution(self, results, fig_dir):
        """图5: 时序机制演变"""
        # 尝试从 results 中获取预计算的时序数据
        # 如果没有，跳过
        print("    - fig5_temporal_evolution.pdf (skipped - requires precomputed data)")
        # 实际实现需要从 analyzer 获取 temporal_evolution 数据