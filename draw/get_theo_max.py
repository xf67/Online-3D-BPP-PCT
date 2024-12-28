import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

bins = [
    (35,23,13),
    (37,26,13),
    (38,26,13),
    (40,28,16),
    (42,30,18),
    (42,30,40),
    (52,40,17),
    (54,45,36),
]

def read_and_group_boxes():
    # 读取CSV文件，指定列名
    df = pd.read_csv('dataset/task3.csv')
    
    # 创建一个字典来存储分组后的盒子数据
    boxes_by_station = {}
    
    # 遍历DataFrame的每一行
    for _, row in df.iterrows():
        sta_code = row['sta_code']
        length = row['长(CM)']
        width = row['宽(CM)']
        height = row['高(CM)']
        qty = row['qty']
        
        # 如果sta_code不在字典中，创建一个新的列表
        if sta_code not in boxes_by_station:
            boxes_by_station[sta_code] = []
        
        # 根据qty复制盒子数据并添加到对应的sta_code列表中
        box_data = [length, width, height]
        boxes_by_station[sta_code].extend([box_data] * int(qty))
    
    return boxes_by_station

def get_theo_max(boxes_by_station):
    total_count = 0
    failed_count = 0
    theo_max_list = []
    bin_counts = {}
    for bin in bins:
        bin_counts[bin] = 0
    bin_counts[(-1,-1,-1)] = 0
    
    for sta_code, boxes in boxes_by_station.items():
        V = 0
        for box in boxes:
            V += box[0] * box[1] * box[2]
        total_count += 1
        max_len = max(box[0] for box in boxes)
        V_bin = 0
        current_bin = (-1,-1,-1)
        
        for bin in bins:
            V_bin = bin[0] * bin[1] * bin[2]
            if bin[0] >= max_len and V <= V_bin:
                current_bin = bin
                break
        
        if current_bin == (-1,-1,-1):
            failed_count += 1
            bin_counts[(-1,-1,-1)] += 1  # 确保失败案例被计入
            continue
            
        theo_max = V / V_bin
        theo_max_list.append(theo_max)
        bin_counts[current_bin] += 1
    
    print(f"Failed count: {failed_count}")
    print(f"Selected bin: {bin_counts}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Container type distribution
    bin_indices = range(len(bins) + 1)  # +1 for failed cases
    counts = [bin_counts[(-1,-1,-1)]] + [bin_counts[bin] for bin in bins]
    
    bars = ax1.bar(bin_indices, counts)
    ax1.set_title('Theoretical Container Type Distribution')
    ax1.set_xlabel('Container Type (-1 represents failed cases)')
    ax1.set_ylabel('Usage Count')
    
    # Set x-axis ticks
    ax1.set_xticks(bin_indices)
    ax1.set_xticklabels([-1] + list(range(len(bins))))
    
    # Different color for failed cases
    bars[0].set_color('red')
    
    # Add value labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # 2. Theoretical utilization distribution
    sns.histplot(theo_max_list, bins=30, ax=ax2)
    
    # Calculate and add mean and median
    mean_val = sum(theo_max_list) / len(theo_max_list)
    median_val = sorted(theo_max_list)[len(theo_max_list)//2]
    
    ax2.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
    ax2.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
    
    ax2.legend()
    ax2.set_title('Theoretical Space Utilization Distribution')
    ax2.set_xlabel('Utilization')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('draw/theo_max_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    boxes_by_station = read_and_group_boxes()
    get_theo_max(boxes_by_station)