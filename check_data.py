"""快速验证数据加载"""
import sys
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

# 添加src目录到Python路径
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 现在可以导入了
from data_loader import DataLoader

def main():
    # 初始化
    loader = DataLoader(config_path="configs/datasets_config.yaml")

    # 测试每个数据集
    datasets = ['SMD', 'SWaT', 'SMAP', 'MSL']

    for name in datasets:
        print(f"\n{'='*60}")
        print(f"Testing {name}...")
        print('='*60)

        try:
            data = loader.load_dataset(name)

            print(f"\n✓ {name} loaded successfully!")
            print(f"  - X_train shape: {data['X_train'].shape}")
            print(f"  - X_test shape: {data['X_test'].shape}")
            print(f"  - y_test shape: {data['y_test'].shape}")
            print(f"  - Anomaly ratio: {data['y_test'].mean():.4f}")

            # 显示第一个异常段
            segments = loader.get_segments(data['y_test'])
            if segments:
                print(f"  - Anomaly segments: {len(segments)}")
                print(f"    First segment: {segments[0]}")
            else:
                print(f"  - No anomaly segments found")

        except Exception as e:
            print(f"✗ Error loading {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()