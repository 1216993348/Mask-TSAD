# diagnose_figures.py
import json
import numpy as np
from pathlib import Path

print("="*60)
print("图表源数据诊断")
print("="*60)

# 1. 检查 per_dimension.json
print("\n1. MSL_per_dimension.json 数据检查:")
with open('output/results/MSL_per_dimension.json', 'r') as f:
    per_dim = json.load(f)

mean_shift = per_dim['mean_shift']
var_change = per_dim['var_change']

# 找 top 5 mean_shift
mean_top5 = sorted([(i, v) for i, v in enumerate(mean_shift)], key=lambda x: -x[1])[:5]
print(f"   Top 5 Mean Shift: {[(f'dim{d}', round(v,4)) for d,v in mean_top5]}")

# 找 top 5 var_change
var_top5 = sorted([(i, v) for i, v in enumerate(var_change)], key=lambda x: -x[1])[:5]
print(f"   Top 5 Var Change: {[(f'dim{d}', f'{v:.1e}' if v>1000 else round(v,4)) for d,v in var_top5]}")

# 2. 检查 correlation.json
print("\n2. MSL_correlation.json 数据检查:")
with open('output/results/MSL_correlation.json', 'r') as f:
    corr = json.load(f)

corr_diff = np.array(corr['correlation_diff'])
max_diff = np.max(corr_diff)
mean_diff = np.mean(corr_diff)
print(f"   Max correlation diff: {max_diff:.4f}")
print(f"   Mean correlation diff: {mean_diff:.4f}")

# 找变化最大的维度对
flat_idx = np.argmax(corr_diff)
d1, d2 = divmod(flat_idx, corr_diff.shape[0])
print(f"   Largest change: dim {d1} <-> dim {d2} = {corr_diff[d1,d2]:.4f}")

# 3. 检查 temporal_acf.json
print("\n3. MSL_temporal_acf.json 数据检查:")
with open('output/results/MSL_temporal_acf.json', 'r') as f:
    acf = json.load(f)

acf_n = np.array(acf['acf_normal'])
acf_a = np.array(acf['acf_anomaly'])
acf_diff = np.mean(np.abs(acf_a - acf_n))
print(f"   Mean ACF difference: {acf_diff:.4f}")
print(f"   ACF normal first 5: {acf_n[:5].tolist()}")
print(f"   ACF anomaly first 5: {acf_a[:5].tolist()}")

# 4. 检查 temporal_evolution.json
print("\n4. MSL_temporal_evolution.json 数据检查:")
evo_file = Path('output/results/MSL_temporal_evolution.json')
if evo_file.exists():
    with open(evo_file, 'r') as f:
        evo = json.load(f)
    print(f"   Windows count: {len(evo.get('windows', []))}")
    print(f"   Value scores range: [{min(evo.get('value_scores', [0])):.3f}, {max(evo.get('value_scores', [0])):.3f}]")
    print(f"   Dep scores range: [{min(evo.get('dep_scores', [0])):.3f}, {max(evo.get('dep_scores', [0])):.3f}]")
else:
    print("   File not found!")

# 5. 检查 anomaly_segments.json
print("\n5. MSL_anomaly_segments.json 数据检查:")
seg_file = Path('output/results/MSL_anomaly_segments.json')
if seg_file.exists():
    with open(seg_file, 'r') as f:
        segs = json.load(f)
    print(f"   Number of segments: {len(segs)}")
    if segs:
        print(f"   First segment: dims={len(segs[0].get('data', [[]])[0]) if segs[0].get('data') else 'N/A'}")
else:
    print("   File not found!")

print("\n" + "="*60)
print("诊断完成")
print("="*60)

# 判断哪些图表可能有问题
print("\n判断结果:")
if mean_top5[0][0] != 0:
    print("   ⚠️ Fig4左图: mean_shift 最高不是 dim 0，检查数据")
else:
    print("   ✓ Fig4左图: mean_shift 数据正确")

if var_top5[0][0] != 23:
    print(f"   ⚠️ Fig4右图: var_change 最高是 dim {var_top5[0][0]}，不是 dim 23")
else:
    print("   ✓ Fig4右图: var_change 数据正确")

if max_diff < 0.1:
    print("   ⚠️ Fig3: correlation diff 很小 (<0.1)，图可能不明显")
else:
    print("   ✓ Fig3: correlation diff 足够大")

if acf_diff < 0.05:
    print("   ⚠️ Fig3时间部分: ACF diff 很小 (<0.05)")
else:
    print("   ✓ Fig3时间部分: ACF diff 正常")