# -*- coding: utf-8 -*-
"""
节假日效应分析：对比节假日与平时的流量特征变化
图表类型：
- 箱线图：对比节假日和平时的流量分布
- 条形图：对比节假日和平时的平均流量
- 折线图：查看节假日和平时流量的时间趋势
- 热力图：对比节假日和平时在小时层级的流量分布
"""
import os
import logging
import gc
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# =========================
# 1) 配置参数
# =========================
CONFIG = {
    'data_path': r"d:\dev\code\py\class\B\report_assets\section3\section3_cell_agg.csv",
    'attr_path': r"d:\dev\code\py\class\B\report_assets\section3\section3_cell_agg.csv",
    'output_fig': r"d:\dev\code\py\class\B\flow_sum_and_type.png",
    'chunksize': 500_000,
    'agg_to_hour': True,
    'main_metric': "avg_flow_per_cell_per_hour",  # 或 "avg_flow_per_user"
    'core_cum_contrib': 0.80
}

# =========================
# 2) 数据加载模块
# =========================
def load_attributes(attr_path: str) -> tuple[dict, dict]:
    """加载小区属性映射"""
    logger.info("加载 attributes 映射...")
    try:
        attr = pd.read_csv(attr_path)
        
        # 检查必要列
        required_cols = ["CELL_ID", "TYPE"]
        if not all(col in attr.columns for col in required_cols):
            logger.warning("attributes 文件缺少必要列，将使用空映射")
            return {}, {}
            
        attr["CELL_ID"] = attr["CELL_ID"].astype(str)
        type_map = dict(zip(attr["CELL_ID"], attr["TYPE"]))
        n_cells_by_type = attr.groupby("TYPE")["CELL_ID"].nunique().to_dict()
        
        logger.info("attributes 映射加载完成")
        return type_map, n_cells_by_type
        
    except Exception as e:
        logger.error(f"加载 attributes 失败: {e}")
        return {}, {}

# =========================
# 3) 数据处理模块
# =========================
def process_data(data_path: str, type_map: dict) -> pd.DataFrame:
    """流式聚合处理数据"""
    logger.info("开始处理数据...")
    
    # 累加器初始化
    accumulators = {
        'total_flow': defaultdict(float),
        'total_users': defaultdict(float),
        'n_cell_hours': defaultdict(int)
    }
    
    # 读取数据
    try:
        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
        # 读取数据（分块处理）
        for chunk in pd.read_csv(data_path, chunksize=CONFIG['chunksize']):
            # 数据清洗
            chunk = clean_data(chunk, type_map)
            
            # 聚合到小时
            if CONFIG['agg_to_hour']:
                chunk = aggregate_to_hour(chunk)
                
            # 按 TYPE 聚合
            aggregate_by_type(chunk, accumulators)
            
            # 清理内存
            del chunk
            gc.collect()
            
        # 创建结果 DataFrame
        result = create_result_dataframe(accumulators)
        logger.info("数据处理完成")
        return result
        
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        raise

def clean_data(chunk: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """数据清洗"""
    # 转换 CELL_ID 为字符串
    if 'CELL_ID' in chunk.columns:
        chunk['CELL_ID'] = chunk['CELL_ID'].astype(str)
    
    # 处理 TYPE 列
    if 'TYPE' not in chunk.columns or chunk['TYPE'].isna().all():
        if type_map:
            chunk['TYPE'] = chunk['CELL_ID'].map(type_map)
            chunk = chunk.dropna(subset=['TYPE'])
        else:
            chunk['TYPE'] = 0
            logger.warning("未找到 TYPE 列，默认 TYPE 为 0")
    
    # 处理时间列
    if 'DATETIME_KEY' in chunk.columns:
        chunk['DATETIME_KEY'] = pd.to_datetime(chunk['DATETIME_KEY'], errors='coerce')
        chunk = chunk.dropna(subset=['DATETIME_KEY'])
    
    # 处理异常值
    for col in ['FLOW_SUM', 'USER_COUNT']:
        if col in chunk.columns:
            chunk.loc[chunk[col] < 0, col] = np.nan
    
    return chunk

def aggregate_to_hour(chunk: pd.DataFrame) -> pd.DataFrame:
    """聚合到小时级别"""
    chunk['HOUR_KEY'] = chunk['DATETIME_KEY'].dt.floor('H')
    return chunk.groupby(['TYPE', 'CELL_ID', 'HOUR_KEY'], as_index=False).agg(
        FLOW_SUM=('FLOW_SUM', 'sum', min_count=1),
        USER_COUNT=('USER_COUNT', 'sum', min_count=1)
    )

def aggregate_by_type(chunk: pd.DataFrame, accumulators: dict):
    """按 TYPE 聚合数据"""
    gb = chunk.groupby('TYPE', as_index=True)
    
    # 聚合流量和用户数
    flow_s = gb['FLOW_SUM'].sum(min_count=1)
    user_s = gb['USER_COUNT'].sum(min_count=1)
    cnt_s = gb['FLOW_SUM'].count()
    
    # 更新累加器
    for t, v in flow_s.items():
        accumulators['total_flow'][t] += float(v) if pd.notna(v) else 0.0
    
    for t, v in user_s.items():
        accumulators['total_users'][t] += float(v) if pd.notna(v) else 0.0
    
    for t, v in cnt_s.items():
        accumulators['n_cell_hours'][t] += int(v)

def create_result_dataframe(accumulators: dict) -> pd.DataFrame:
    """创建结果 DataFrame"""
    result = pd.DataFrame({
        'TYPE': list(accumulators['total_flow'].keys()),
        'total_flow': list(accumulators['total_flow'].values()),
        'total_users': list(accumulators['total_users'].values()),
        'n_cell_hours': list(accumulators['n_cell_hours'].values()),
        'avg_flow_per_cell_per_hour': [
            accumulators['total_flow'][t] / accumulators['n_cell_hours'][t] 
            for t in accumulators['total_flow'].keys()
        ],
        'avg_flow_per_user': [
            accumulators['total_flow'][t] / accumulators['total_users'][t] 
            for t in accumulators['total_flow'].keys()
        ]
    })
    return result

# =========================
# 4) 可视化模块
# =========================
def plot_type_lollipop(type_df: pd.DataFrame, out_png: str):
    """绘制 TYPE 流量密度画像"""
    logger.info("开始绘制图表...")
    
    df = type_df.copy()
    total = df["total_flow"].sum()
    df["flow_contrib"] = df["total_flow"] / total if total else np.nan
    df = df.sort_values("total_flow", ascending=False)
    df["flow_contrib_cum"] = df["flow_contrib"].cumsum()
    df["is_core"] = df["flow_contrib_cum"] <= CONFIG['core_cum_contrib']
    
    # 按主指标排序
    df = df.sort_values(CONFIG['main_metric'], ascending=True).reset_index(drop=True)
    
    # 创建图表
    fig_h = max(5, 0.6 * len(df) + 2)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    
    # 绘制 Lollipop 图
    y = np.arange(len(df))
    x = df[CONFIG['main_metric']].values
    colors = np.where(df["is_core"].values, "tab:red", "tab:blue")
    
    ax.hlines(y=y, xmin=0, xmax=x, linewidth=2, alpha=0.35)
    ax.scatter(x, y, s=90, c=colors)
    
    # 设置标签
    ax.set_yticks(y)
    ax.set_yticklabels([f"TYPE {int(v)}" for v in df["TYPE"].values])
    
    metric_labels = {
        "avg_flow_per_cell_per_hour": "小区均流量（MB / 小时）",
        "avg_flow_per_user": "人均流量（MB / 人）"
    }
    ax.set_xlabel(metric_labels.get(CONFIG['main_metric'], CONFIG['main_metric']))
    
    # 设置标题
    ax.set_title(
        f"TYPE 流量密度画像（Lollipop，流式聚合）\n"
        f"主指标={CONFIG['main_metric']} | 核心区=累计贡献≤{int(CONFIG['core_cum_contrib']*100)}%",
        pad=12
    )
    
    # 标注贡献度
    for i, row in df.iterrows():
        if pd.notna(row["flow_contrib"]):
            ax.text(row[CONFIG['main_metric']] * 1.01, i, 
                   f"流量占比 {row['flow_contrib']*100:.1f}%", va="center", fontsize=9)
    
    # 美化图表
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # 添加图例
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker='o', color='w', label='核心价值区（累计贡献≤阈值）', 
               markerfacecolor='tab:red', markersize=9),
        Line2D([0],[0], marker='o', color='w', label='非核心区', 
               markerfacecolor='tab:blue', markersize=9),
    ], loc="lower right", frameon=False)
    
    # 保存图表
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    logger.info(f"图表已保存到: {out_png}")

# =========================
# 5) 主程序
# =========================
def main():
    """主程序入口"""
    logger.info("程序启动")
    
    try:
        # 加载属性映射
        type_map, n_cells_by_type = load_attributes(CONFIG['attr_path'])
        
        # 处理数据
        type_df = process_data(CONFIG['data_path'], type_map)
        
        # 绘制图表
        plot_type_lollipop(type_df, CONFIG['output_fig'])
        
        logger.info("程序执行完成")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()