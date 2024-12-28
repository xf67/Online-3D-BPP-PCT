import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_log(log_content):
    # 提取所有记录
    pattern = r"'container_index': (-?\d+), 'container_size': (\([^)]+\)|None), 'utilization': ([0-9.]+)"
    matches = re.findall(pattern, log_content)
    
    # 转换为DataFrame
    data = []
    for match in matches:
        container_index = int(match[0])
        container_size = eval(match[1]) if match[1] != 'None' else None
        utilization = float(match[2])
        data.append({
            'container_index': container_index,
            'container_size': container_size,
            'utilization': utilization
        })
    
    df = pd.DataFrame(data)
    
    # 1. 统计成功/失败数
    total = len(df)
    failed = len(df[df['container_index'] == -1])
    success_rate = (total - failed) / total * 100
    
    # 2. 计算每种container的使用情况
    success_df = df[df['container_index'] != -1]
    container_stats = success_df.groupby('container_index').agg({
        'utilization': ['count', 'mean']
    }).round(4)
    
    return {
        'total_orders': total,
        'success_rate': success_rate,
        'container_stats': container_stats,
        'utilization_data': success_df['utilization']
    }

def plot_analysis(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 箱型选择分布（包括失败案例）
    container_counts = results['container_stats']['utilization']['count']
    failed_count = results['total_orders'] - container_counts.sum()
    
    # 创建包含失败案例的新索引和数据，将失败案例放在最左边
    all_indices = [-1] + list(container_counts.index)
    all_counts = [failed_count] + list(container_counts.values)
    
    bars = ax1.bar(range(len(all_indices)), all_counts)
    ax1.set_title('Container Type Distribution')
    ax1.set_xlabel('Container Type (-1 represents failed cases)')
    ax1.set_ylabel('Usage Count')
    
    # 设置x轴刻度，显示所有数字
    ax1.set_xticks(range(len(all_indices)))
    ax1.set_xticklabels(all_indices)
    
    # 为失败案例的柱状图添加不同颜色
    bars[0].set_color('red')
    
    # 在柱状图上添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # 2. 利用率分布（成功案例）
    sns.histplot(results['utilization_data'], bins=30, ax=ax2)
    
    # 计算并添加平均值和中位数
    mean_val = results['utilization_data'].mean()
    median_val = results['utilization_data'].median()
    
    # 添加垂直线
    ax2.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
    ax2.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
    
    ax2.legend()
    ax2.set_title('Space Utilization Distribution (Successful Cases)')
    ax2.set_xlabel('Utilization')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('draw/container_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_summary(results):
    print(f"订单总数: {results['total_orders']}")
    print(f"成功率: {results['success_rate']:.2f}%")
    print("\n集装箱使用统计:")
    print("集装箱类型 | 使用次数 | 平均利用率")
    print("-" * 40)
    
    stats = results['container_stats']
    for idx in stats.index:
        count = stats.loc[idx, ('utilization', 'count')]
        mean = stats.loc[idx, ('utilization', 'mean')]
        print(f"类型 {idx:2d} | {count:8d} | {mean:.4f}")

if __name__ == "__main__":
    with open('logger/ans.txt', 'r') as f:
        content = f.read()
    results = analyze_log(content)
    plot_analysis(results)
    print_summary(results)