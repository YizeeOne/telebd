import pandas as pd
import numpy as np
import os
import glob



# 创建输出目录
output_dir = 'processed'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建输出目录: {output_dir}")

# 获取所有原始数据文件
batch_files = sorted(glob.glob('original/cell_data_b*.csv'), 
                     key=lambda x: int(os.path.basename(x).replace('cell_data_b', '').replace('.csv', '')))

if len(batch_files) == 0:
    print("未找到数据文件（cell_data_b*.csv）！")
    exit()

print(f"找到 {len(batch_files)} 个数据文件")

# 检查待处理文件
pending_files = []
for file_path in batch_files:
    base_name = os.path.basename(file_path)  # cell_data_bX.csv
    output_file = os.path.join(output_dir, base_name)
    if not os.path.exists(output_file):
        pending_files.append(file_path)

if len(pending_files) == 0:
    print("所有文件已处理完成！")
    exit()

print(f"待处理: {len(pending_files)} 个文件\n")

# 自定义聚合函数：如果所有值都是NaN则返回NaN，否则返回sum
def sum_with_nan(x):
    if x.isna().all():
        return np.nan
    return x.sum()

for idx, file_path in enumerate(pending_files, 1):
    base_name = os.path.basename(file_path)  # cell_data_bX.csv
    batch_num = int(base_name.replace('cell_data_b', '').replace('.csv', ''))
    output_file = os.path.join(output_dir, base_name)
    
    print(f"[{idx}/{len(pending_files)}] {base_name}... ", end='', flush=True)
    
    try:
        # 读取并转换数据
        df = pd.read_csv(file_path)
        original_records = len(df)
        
        # 统计原始缺失
        flow_missing_orig = df['FLOW_SUM'].isna().sum()
        user_missing_orig = df['USER_COUNT'].isna().sum()
        
        df['DATETIME_KEY'] = pd.to_datetime(df['DATETIME_KEY'])
        df['HOUR_KEY'] = df['DATETIME_KEY'].dt.floor('h')
        
        # 按小区和小时分组聚合，使用自定义函数处理NaN
        hourly_df = df.groupby(['CELL_ID', 'HOUR_KEY']).agg({
            'FLOW_SUM': sum_with_nan,
            'USER_COUNT': sum_with_nan
        }).reset_index()
        
        hourly_df.rename(columns={'HOUR_KEY': 'DATETIME_KEY'}, inplace=True)
        hourly_df = hourly_df.sort_values(['CELL_ID', 'DATETIME_KEY']).reset_index(drop=True)
        
        # 统计聚合后的缺失
        flow_missing_new = hourly_df['FLOW_SUM'].isna().sum()
        user_missing_new = hourly_df['USER_COUNT'].isna().sum()
        
        # 保存结果
        hourly_df.to_csv(output_file, index=False)
        
        # 显示结果
        flow_rate_orig = flow_missing_orig / original_records * 100
        user_rate_orig = user_missing_orig / original_records * 100
        flow_rate_new = flow_missing_new / len(hourly_df) * 100
        user_rate_new = user_missing_new / len(hourly_df) * 100
        
        print(f"OK (15min: F={flow_rate_orig:.2f}% U={user_rate_orig:.2f}% -> 1h: F={flow_rate_new:.2f}% U={user_rate_new:.2f}%)")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")

print(f"\n完成！结果已保存至: {output_dir}/")
