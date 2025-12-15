import csv
import os
import numpy as np

def write_dict_to_csv(file_path, data_dict):
    """
    将字典写入CSV。
    1. 如果文件不存在/为空：写入键作为表头，写入值。
    2. 如果文件存在：检查键是否匹配，匹配则追加，不匹配则报错。
    """
    
    # --- 1. 数据预处理 (清洗 numpy 类型和 NaN) ---
    clean_data = {}
    for k, v in data_dict.items():
        # 处理 numpy 数值 (如 np.float32) 转为 python 原生类型
        if hasattr(v, 'item'):
            v = v.item()
        
        # 处理 NaN，转为空字符串 (可选，看你需要空还是'nan')
        if isinstance(v, float) and np.isnan(v):
            v = "" 
            
        clean_data[k] = v

    # --- 2. 检查文件状态 ---
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0

    if not file_exists:
        # === 场景 A: 新建文件 ===
        try:
            # 自动创建父目录
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=clean_data.keys())
                writer.writeheader()
                writer.writerow(clean_data)
        except Exception as e:
            print(f"[Error] 创建CSV失败: {e}")

    else:
        # === 场景 B: 追加文件 ===
        try:
            # 步骤 2.1: 读取现有的表头，确保追加时的列顺序一致
            with open(file_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)

            if not existing_header:
                print(f"[Error] 文件 {file_path} 存在但表头无法读取。")
                return

            # 步骤 2.2: 校验键是否匹配 (Set 比较忽略顺序)
            input_keys = set(clean_data.keys())
            file_keys = set(existing_header)
            
            # 这里的逻辑是：传入的字典必须包含CSV所有的列，或者CSV包含字典所有的键
            # 最严格的检查是：set(existing_header) == set(clean_data.keys())
            # 宽松一点（允许字典缺某些列，自动填空）：
            if not input_keys.issubset(file_keys):
                diff = input_keys - file_keys
                print(f"[Error] 字典包含未知的键，无法写入: {diff}")
                return

            # 步骤 2.3: 追加写入
            with open(file_path, mode='a', newline='', encoding='utf-8') as f:
                # 关键：必须使用 existing_header 作为 fieldnames，保证列对齐
                writer = csv.DictWriter(f, fieldnames=existing_header)
                writer.writerow(clean_data)
                
        except Exception as e:
            print(f"[Error] 追加CSV失败: {e}")