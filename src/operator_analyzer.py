import numpy as np
from scipy.fft import fft
from scipy.signal import correlate
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings


class OperatorAnalyzer:
    """4种异常算子的定性和定量分析"""

    def __init__(self, data_dict):
        self.name = data_dict['name']
        self.X_train = data_dict['X_train']
        self.X_test = data_dict['X_test']
        self.y_test = data_dict['y_test']
        self.config = data_dict['config']

        # 分离 normal 和 anomaly
        self.X_normal = self.X_test[self.y_test == 0]
        self.X_anomaly = self.X_test[self.y_test == 1]

        # 也可以用训练集作为正常参考
        if len(self.X_normal) < 1000 and len(self.X_train) > 0:
            self.X_normal = np.vstack([self.X_normal, self.X_train])

        # 检查是否有足够的异常样本
        self.has_anomaly = len(self.X_anomaly) > 100

        # 标准化
        self.scaler = StandardScaler()
        self.X_normal_scaled = self.scaler.fit_transform(self.X_normal)
        if self.has_anomaly and len(self.X_anomaly) > 0:
            self.X_anomaly_scaled = self.scaler.transform(self.X_anomaly)
        else:
            self.X_anomaly_scaled = np.array([])

        # 初始化存储源数据的变量
        self.per_dim_mean_shift = None
        self.per_dim_var_change = None
        self.corr_normal = None
        self.corr_anomaly = None
        self.acf_normal = None
        self.acf_anomaly = None
        self.temporal_evolution_windows = []
        self.temporal_evolution_value_scores = []
        self.temporal_evolution_dep_scores = []
        self.temporal_evolution_anomaly_ratios = []

    def analyze_all(self):
        """执行全部4种算子分析，并保存所有中间数据"""

        op1_result = self.operator_value_perturbation()
        op2_result = self.operator_trend_drift()
        op3_result = self.operator_temporal_warping()
        op4_result = self.operator_dependency_break()

        # 计算时序演变数据（用于fig5）
        self._compute_temporal_evolution()

        results = {
            'name': self.name,
            'operator_1_value': op1_result,
            'operator_2_trend': op2_result,
            'operator_3_temporal': op3_result,
            'operator_4_dependency': op4_result,
            'summary': {}
        }

        # 生成summary
        scores = {
            'operator_1_value': results['operator_1_value'].get('score', 0) if isinstance(results['operator_1_value'],
                                                                                          dict) else 0,
            'operator_2_trend': results['operator_2_trend'].get('score', 0) if isinstance(results['operator_2_trend'],
                                                                                          dict) else 0,
            'operator_3_temporal': results['operator_3_temporal'].get('score', 0) if isinstance(
                results['operator_3_temporal'], dict) else 0,
            'operator_4_dependency': results['operator_4_dependency'].get('score', 0) if isinstance(
                results['operator_4_dependency'], dict) else 0
        }

        dominant = max(scores, key=scores.get)
        severity = np.nanmean(list(scores.values()))
        if np.isnan(severity):
            severity = 0

        results['summary'] = {
            'dominant_operator': dominant,
            'severity_score': float(severity),
            'recommended_injection': self._recommend_injection(results)
        }

        return results

    def operator_value_perturbation(self):
        """Operator 1: Value Perturbation (spike/drift)"""
        if not self.has_anomaly:
            return {'score': 0, 'type': 'no_anomaly', 'qualitative': 'No anomaly'}

        try:
            mean_n = self.X_normal_scaled.mean(axis=0)
            mean_a = self.X_anomaly_scaled.mean(axis=0)
            std_n = self.X_normal_scaled.std(axis=0)
            std_a = self.X_anomaly_scaled.std(axis=0)

            # 保存 per-dimension 数据（用于fig4）
            self.per_dim_mean_shift = np.abs(mean_a - mean_n)
            self.per_dim_var_change = np.abs(std_a - std_n) / (std_n + 1e-6)

            # 均值偏移得分
            mean_shift = np.mean(self.per_dim_mean_shift)

            # 方差变化得分
            var_change = np.mean(self.per_dim_var_change)

            # 分布差异 (KL散度)
            hist_n, bins = np.histogram(self.X_normal_scaled.flatten(), bins=50, density=True)
            hist_a, _ = np.histogram(self.X_anomaly_scaled.flatten(), bins=bins, density=True)
            hist_n = hist_n + 1e-10
            hist_a = hist_a + 1e-10
            kl_div = entropy(hist_n, hist_a)

            score = np.clip(mean_shift + 0.5 * var_change + 0.1 * kl_div, 0, 1)

            # 定性判断
            if mean_shift > 0.5:
                q_type = "Strong spike/drift"
            elif mean_shift > 0.2:
                q_type = "Moderate value shift"
            else:
                q_type = "Weak value perturbation"

            return {
                'score': float(score),
                'mean_shift': float(mean_shift),
                'var_change': float(var_change),
                'kl_divergence': float(kl_div),
                'qualitative': q_type,
                'recommended_alpha': min(0.3 + mean_shift, 0.9)
            }
        except Exception as e:
            print(f"  Warning in value perturbation: {e}")
            return {'score': 0, 'qualitative': 'Computation failed'}

    def operator_trend_drift(self):
        """Operator 2: Trend/Drift Shift"""
        if not self.has_anomaly or len(self.X_anomaly) < 100:
            return {'score': 0, 'type': 'insufficient_data', 'qualitative': 'Insufficient data'}

        try:
            def compute_slope_series(X, window=100):
                if len(X) < window:
                    return np.array([0])
                slopes = []
                for i in range(0, len(X) - window, window // 2):
                    segment = X[i:i + window]
                    t = np.arange(len(segment))
                    slopes.append(np.polyfit(t, segment, 1)[0])
                return np.array(slopes) if slopes else np.array([0])

            slopes_n = []
            for d in range(min(self.X_normal_scaled.shape[1], 20)):
                slopes_n.extend(compute_slope_series(self.X_normal_scaled[:, d]))

            slopes_a = []
            for d in range(min(self.X_anomaly_scaled.shape[1], 20)):
                slopes_a.extend(compute_slope_series(self.X_anomaly_scaled[:, d]))

            slope_change = abs(np.mean(slopes_a) - np.mean(slopes_n))

            pos_ratio_n = np.mean(np.array(slopes_n) > 0) if slopes_n else 0
            pos_ratio_a = np.mean(np.array(slopes_a) > 0) if slopes_a else 0
            monotonic_change = abs(pos_ratio_a - pos_ratio_n)

            score = np.clip(slope_change + 0.5 * monotonic_change, 0, 1)

            if slope_change > 0.3:
                q_type = "Strong systematic drift"
            elif slope_change > 0.1:
                q_type = "Moderate trend shift"
            else:
                q_type = "Stable trend"

            return {
                'score': float(score),
                'slope_change': float(slope_change),
                'monotonic_change': float(monotonic_change),
                'qualitative': q_type,
                'recommended_beta': min(slope_change * 2, 0.5)
            }
        except Exception as e:
            print(f"  Warning in trend drift: {e}")
            return {'score': 0, 'qualitative': 'Computation failed'}

    def operator_temporal_warping(self):
        """Operator 3: Temporal Warping"""
        if not self.has_anomaly or len(self.X_anomaly) < 500:
            return {'score': 0, 'type': 'insufficient_data', 'qualitative': 'Insufficient data'}

        try:
            def compute_autocorr_avg(X, max_lag=100, n_dims=10):
                if len(X) < max_lag + 10:
                    return np.zeros(max_lag)

                dims = min(X.shape[1], n_dims)
                all_ac = []

                for d in range(dims):
                    signal = X[:min(5000, len(X)), d]
                    if np.std(signal) == 0:
                        continue
                    ac = correlate(signal, signal, mode='full')
                    ac = ac[len(ac) // 2: len(ac) // 2 + max_lag]
                    if ac[0] != 0:
                        ac = ac / ac[0]
                    all_ac.append(ac)

                if not all_ac:
                    return np.zeros(max_lag)
                return np.mean(all_ac, axis=0)

            self.acf_normal = compute_autocorr_avg(self.X_normal_scaled)
            self.acf_anomaly = compute_autocorr_avg(self.X_anomaly_scaled)

            ac_diff = np.mean(np.abs(self.acf_anomaly - self.acf_normal))

            zero_n = np.argmax(self.acf_normal < 0.1) if np.any(self.acf_normal < 0.1) else len(self.acf_normal)
            zero_a = np.argmax(self.acf_anomaly < 0.1) if np.any(self.acf_anomaly < 0.1) else len(self.acf_anomaly)
            timescale_change = abs(zero_a - zero_n) / len(self.acf_normal) if len(self.acf_normal) > 0 else 0

            score = np.clip(ac_diff + timescale_change, 0, 1)

            if timescale_change > 0.3:
                q_type = "Strong time warping"
            elif ac_diff > 0.2:
                q_type = "Moderate temporal distortion"
            else:
                q_type = "Normal temporal structure"

            return {
                'score': float(score),
                'acf_diff': float(ac_diff),
                'timescale_change': float(timescale_change),
                'qualitative': q_type,
                'recommended_warp_factor': 1.0 + timescale_change
            }
        except Exception as e:
            print(f"  Warning in temporal warping: {e}")
            return {'score': 0, 'qualitative': 'Computation failed'}

    def operator_dependency_break(self):
        """Operator 4: Dependency Break"""
        if not self.has_anomaly or len(self.X_anomaly) < 500:
            return {'score': 0, 'type': 'insufficient_data', 'qualitative': 'Insufficient data'}

        try:
            max_dims = min(self.config.get('analysis', {}).get('max_dims_for_corr', 50),
                           self.X_normal_scaled.shape[1], 30)

            vars_n = self.X_normal_scaled.var(axis=0)
            top_dims = np.argsort(vars_n)[-max_dims:] if max_dims > 0 else np.arange(
                min(10, self.X_normal_scaled.shape[1]))

            X_n_subset = self.X_normal_scaled[:, top_dims]
            X_a_subset = self.X_anomaly_scaled[:, top_dims]

            sample_size = min(5000, len(X_n_subset), len(X_a_subset))
            if sample_size < 10:
                return {'score': 0, 'qualitative': 'Too few samples'}

            idx_n = np.random.choice(len(X_n_subset), sample_size, replace=False)
            idx_a = np.random.choice(len(X_a_subset), sample_size, replace=False)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.corr_normal = np.corrcoef(X_n_subset[idx_n].T)
                self.corr_anomaly = np.corrcoef(X_a_subset[idx_a].T)

            corr_diff = np.mean(np.abs(self.corr_anomaly - self.corr_normal))

            reg = LinearRegression()
            pred_errors_n = []
            pred_errors_a = []

            n_dims = min(max_dims, 10, X_n_subset.shape[1])
            for d in range(n_dims):
                try:
                    X_other_n = np.delete(X_n_subset[idx_n], d, axis=1)
                    y_n = X_n_subset[idx_n, d]
                    reg.fit(X_other_n, y_n)
                    pred_errors_n.append(np.mean((y_n - reg.predict(X_other_n)) ** 2))

                    X_other_a = np.delete(X_a_subset[idx_a], d, axis=1)
                    y_a = X_a_subset[idx_a, d]
                    pred_errors_a.append(np.mean((y_a - reg.predict(X_other_a)) ** 2))
                except:
                    pass

            pred_error_change = np.mean(pred_errors_a) / (np.mean(pred_errors_n) + 1e-6) if pred_errors_n else 1.0

            score = np.clip(corr_diff + 0.3 * min(abs(pred_error_change - 1), 1), 0, 1)
            if np.isnan(score):
                score = 0

            if corr_diff > 0.3:
                q_type = "Severe dependency break"
            elif corr_diff > 0.15:
                q_type = "Moderate correlation change"
            else:
                q_type = "Stable dependency structure"

            return {
                'score': float(score),
                'corr_diff': float(corr_diff),
                'pred_error_ratio': float(pred_error_change),
                'qualitative': q_type,
                'recommended_gamma': min(0.3 + corr_diff, 0.8)
            }
        except Exception as e:
            print(f"  Warning in dependency break: {e}")
            return {'score': 0, 'qualitative': 'Computation failed'}

    def _compute_temporal_evolution(self):
        """计算时序演变数据（用于fig5）- 修复版"""
        if not self.has_anomaly or len(self.X_anomaly) < 100:
            return

        try:
            window_size = min(2000, len(self.X_test) // 10)
            step = max(window_size // 4, 100)

            # 预计算全局正常统计
            global_norm = self.X_test[self.y_test == 0]
            if len(global_norm) > 0:
                global_mean = global_norm.mean(axis=0)
                global_std = global_norm.std(axis=0) + 1e-6
            else:
                global_mean = self.X_test.mean(axis=0)
                global_std = self.X_test.std(axis=0) + 1e-6

            self.temporal_evolution_windows = []
            self.temporal_evolution_value_scores = []
            self.temporal_evolution_dep_scores = []
            self.temporal_evolution_anomaly_ratios = []

            for start in range(0, len(self.X_test) - window_size, step):
                end = start + window_size
                window_y = self.y_test[start:end]

                self.temporal_evolution_windows.append(int(start))
                self.temporal_evolution_anomaly_ratios.append(float(window_y.mean()))

                X_window = self.X_test[start:end]
                X_norm = X_window[window_y == 0]
                X_anom = X_window[window_y == 1]

                # ===== Value Score =====
                if len(X_norm) > 10 and len(X_anom) > 5:
                    mean_n = X_norm.mean(axis=0)
                    mean_a = X_anom.mean(axis=0)
                    std_n = X_norm.std(axis=0) + 1e-6
                    value_score = np.mean(np.abs(mean_a - mean_n) / std_n)
                elif len(X_anom) > 0:
                    mean_a = X_anom.mean(axis=0)
                    value_score = np.mean(np.abs(mean_a - global_mean) / global_std)
                else:
                    value_score = 0
                self.temporal_evolution_value_scores.append(float(min(value_score, 1.0)))

                # ===== Dependency Score - 修复 nan 问题 =====
                dep_score = 0.0  # 默认值
                if len(X_norm) > 50 and len(X_anom) > 20:
                    try:
                        n_dims = min(15, X_window.shape[1])
                        # 选择方差最大的维度
                        vars_all = X_window.var(axis=0)
                        top_dims = np.argsort(vars_all)[-n_dims:]

                        X_norm_sub = X_norm[:, top_dims]
                        X_anom_sub = X_anom[:, top_dims]

                        sample_n = min(800, len(X_norm_sub))
                        sample_a = min(800, len(X_anom_sub))

                        # 确保采样大小不超过样本数
                        sample_n = min(sample_n, len(X_norm_sub))
                        sample_a = min(sample_a, len(X_anom_sub))

                        if sample_n > 1 and sample_a > 1:
                            idx_n = np.random.choice(len(X_norm_sub), sample_n, replace=False)
                            idx_a = np.random.choice(len(X_anom_sub), sample_a, replace=False)

                            corr_n = np.corrcoef(X_norm_sub[idx_n].T)
                            corr_a = np.corrcoef(X_anom_sub[idx_a].T)

                            # 检查是否有 nan
                            if not np.isnan(corr_n).any() and not np.isnan(corr_a).any():
                                dep_score = np.mean(np.abs(corr_n - corr_a))
                                dep_score = min(dep_score, 1.0)
                            else:
                                dep_score = 0.0
                        else:
                            dep_score = 0.0
                    except Exception as e:
                        dep_score = 0.0
                else:
                    # 样本不足时，用前一个有效值或0
                    if self.temporal_evolution_dep_scores:
                        last_valid = [s for s in self.temporal_evolution_dep_scores if not np.isnan(s)]
                        dep_score = last_valid[-1] if last_valid else 0.0
                    else:
                        dep_score = 0.0

                # 确保不是 nan
                if np.isnan(dep_score):
                    dep_score = 0.0
                self.temporal_evolution_dep_scores.append(float(dep_score))

            print(f"    Temporal evolution: {len(self.temporal_evolution_windows)} windows, "
                  f"Value scores: [{min(self.temporal_evolution_value_scores):.3f}, {max(self.temporal_evolution_value_scores):.3f}], "
                  f"Dep scores: [{min(self.temporal_evolution_dep_scores):.3f}, {max(self.temporal_evolution_dep_scores):.3f}]")

        except Exception as e:
            print(f"  Warning in temporal evolution: {e}")

    def get_anomaly_segments_for_fig1(self, max_segments=5):
        """获取 fig1 需要的异常段波形数据"""
        from data_loader import DataLoader
        loader = DataLoader()
        segments = loader.get_segments(self.y_test)

        result = []
        for i, (s, e) in enumerate(segments[:max_segments]):
            if e - s > 50:  # 只取长度足够的段
                result.append({
                    'start': int(s),
                    'end': int(e),
                    'length': int(e - s + 1),
                    'data': self.X_test[s:e + 1].tolist() if len(self.X_test[s:e + 1]) < 5000 else None
                })
        return result

    def get_temporal_evolution_for_fig5(self):
        """获取 fig5 需要的时序演变数据"""
        return {
            'windows': self.temporal_evolution_windows,
            'value_scores': self.temporal_evolution_value_scores,
            'dep_scores': self.temporal_evolution_dep_scores,
            'anomaly_ratios': self.temporal_evolution_anomaly_ratios
        }

    def get_per_dimension_for_fig4(self):
        """获取 fig4 需要的 per-dimension 数据"""
        if self.per_dim_mean_shift is not None:
            return {
                'mean_shift': self.per_dim_mean_shift.tolist() if hasattr(self.per_dim_mean_shift,
                                                                          'tolist') else self.per_dim_mean_shift,
                'var_change': self.per_dim_var_change.tolist() if hasattr(self.per_dim_var_change,
                                                                          'tolist') else self.per_dim_var_change,
                'dimensions': list(range(len(self.per_dim_mean_shift)))
            }
        return None

    def get_correlation_for_fig3(self):
        """获取 fig3 需要的相关矩阵数据"""
        if self.corr_normal is not None:
            return {
                'correlation_normal': self.corr_normal.tolist() if hasattr(self.corr_normal,
                                                                           'tolist') else self.corr_normal,
                'correlation_anomaly': self.corr_anomaly.tolist() if hasattr(self.corr_anomaly,
                                                                             'tolist') else self.corr_anomaly,
                'correlation_diff': (np.abs(self.corr_anomaly - self.corr_normal)).tolist()
            }
        return None

    def get_acf_for_fig3_temporal(self):
        """获取时间自相关数据"""
        if self.acf_normal is not None:
            return {
                'acf_normal': self.acf_normal.tolist() if hasattr(self.acf_normal, 'tolist') else self.acf_normal,
                'acf_anomaly': self.acf_anomaly.tolist() if hasattr(self.acf_anomaly, 'tolist') else self.acf_anomaly
            }
        return None

    def _recommend_injection(self, results):
        """基于分析结果，推荐异常注入参数"""
        recommendations = {}

        op1 = results['operator_1_value']
        if isinstance(op1, dict) and 'recommended_alpha' in op1 and op1.get('score', 0) > 0.1:
            recommendations['value_perturbation'] = {
                'alpha': op1['recommended_alpha'],
                'noise_type': 'gaussian'
            }

        op2 = results['operator_2_trend']
        if isinstance(op2, dict) and 'recommended_beta' in op2 and op2.get('score', 0) > 0.1:
            recommendations['trend_drift'] = {
                'beta': op2['recommended_beta']
            }

        op3 = results['operator_3_temporal']
        if isinstance(op3, dict) and 'recommended_warp_factor' in op3 and op3.get('score', 0) > 0.1:
            recommendations['temporal_warping'] = {
                'warp_factor': op3['recommended_warp_factor']
            }

        op4 = results['operator_4_dependency']
        if isinstance(op4, dict) and 'recommended_gamma' in op4 and op4.get('score', 0) > 0.1:
            recommendations['dependency_break'] = {
                'gamma': op4['recommended_gamma'],
                'tau': 1
            }

        return recommendations