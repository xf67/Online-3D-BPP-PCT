import pandas as pd
import numpy as np
from binCreator import BinPackingGenerator

def generate_test_orders(num_orders=100, min_items=10, max_items=50):
    generator = BinPackingGenerator((100, 100, 100))
    orders = []
    
    for order_idx in range(num_orders):
        # 生成一个订单的箱子
        boxes = generator.generate_items(min_items, max_items)
        # 重置生成器以准备下一个订单
        generator.reset()
        
        # 为每个箱子创建记录
        for box_idx, box in enumerate(boxes):
            orders.append({
                'sta_code': f'ORDER_{order_idx:04d}',  # 订单编号
                'sku_code': f'BOX_{order_idx:04d}_{box_idx:04d}',  # 箱子编号
                '长(CM)': box[0],
                '宽(CM)': box[1],
                '高(CM)': box[2],
                'qty': 1  # 每个箱子数量设为1
            })
    
    return pd.DataFrame(orders)

if __name__ == "__main__":
    # 生成100个订单的测试数据
    df = generate_test_orders(num_orders=100)
    
    # 保存为CSV文件
    output_path = "dataset/generated_test_data.csv"
    df.to_csv(output_path, index=False)
    print(f"已生成测试数据并保存至: {output_path}")
    
    # 打印数据统计信息
    print("\n数据统计:")
    print(f"总订单数: {df['sta_code'].nunique()}")
    print(f"总箱子数: {len(df)}")
    print("\n前5行数据预览:")
    print(df.head())