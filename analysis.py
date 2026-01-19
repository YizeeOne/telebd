import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATHS = [
    p for p in glob.glob(os.path.join(BASE_DIR, '*.xlsx'))
    if not os.path.basename(p).startswith('~$')
]
if not DATA_PATHS:
    raise SystemExit('No .xlsx file found in working directory')
DATA_PATH = DATA_PATHS[0]

OUT_DIR = os.path.join(BASE_DIR, 'output')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SCENE2_MAP = {
    '停车场': '停车场',
    '补给站': '补给站',
    '步行道': '步行道',
    '骑行道': '骑行道',
    '咖啡厅': '咖啡厅',
    '城市道路': '城市道路',
    '休息点': '休息点',
}

SCENE1_MAP = {
    'DT': '驶测',
    'CQT': '定点',
}

OP_MAP = {
    1: '移动',
    2: '电信',
    3: '联通',
}

COVER_TOKEN = '覆盖率1'
METRIC_NOTES = [
    '覆盖率1：满足(RSRP >= -105dBm 且 SINR >= -3)的采样点占比',
    '下行速率达标率：应用层下行速率 >= 2Mbps 的占比',
    '上行速率达标率：应用层上行速率 >= 0.5Mbps 的占比',
    '单位：吞吐率为 Mbps，测试总里程为 KM，总时长为 h',
    'delta_rsrp：x1_avg_rsrp - x2_avg_rsrp（空间不均匀）',
    'delta_cov2：x1_cov2 - x2_cov2（空间不均匀）',
    'effective_dl/ul：平均吞吐率 * (1 - 掉线率)',
    'dl/ul_tail_drag：前10%峰值 - 平均吞吐率（尾部拖累）',
    'dl/ul_drops_per_hour：掉线次数 / 测试总时长',
    'dl/ul_drops_per_km：掉线次数 / 测试总里程',
    'availability：RedCap 驻留比 * 5G覆盖率（优先覆盖率3/4）',
    'stay_eff：驻留时长 / 测试总里程',
    'mobility_penalty：KPI_DT - KPI_CQT（移动惩罚）',
]

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def rename_base_cols(df):
    cols = list(df.columns)
    if len(cols) < 8:
        raise ValueError('Unexpected column count')
    base_map = {
        cols[0]: 'vendor',
        cols[1]: 'city',
        cols[2]: 'scene_l1',
        cols[3]: 'scene_l2',
        cols[4]: 'operator',
        cols[5]: 'test_distance_km',
        cols[6]: 'test_duration_h',
        cols[7]: 'avg_speed_kmh',
    }
    return df.rename(columns=base_map)


def clean_df(df):
    df = rename_base_cols(df)

    for col in ['vendor', 'city', 'scene_l1', 'scene_l2', 'operator']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(['nan', 'None', 'NaT']), col] = np.nan

    df['operator'] = pd.to_numeric(df['operator'], errors='coerce').map(OP_MAP).fillna(df['operator'])
    df['scene_l1'] = df['scene_l1'].map(SCENE1_MAP).fillna(df['scene_l1'])
    df['scene_l2'] = df['scene_l2'].map(SCENE2_MAP).fillna(df['scene_l2'])

    for col in df.columns:
        if col in ['vendor', 'city', 'scene_l1', 'scene_l2', 'operator']:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def pick_metrics(df):
    rsrp = next((c for c in df.columns if 'RSRP' in c), None)
    sinr = next((c for c in df.columns if 'SINR' in c), None)
    cov1 = next((c for c in df.columns if COVER_TOKEN in c), None)
    dl = next((c for c in df.columns if c.startswith('y1_')), None)
    ul = next((c for c in df.columns if c.startswith('z1_')), None)
    return rsrp, sinr, cov1, dl, ul


def metric_label(col):
    if not col:
        return ''
    if 'RSRP' in col:
        return 'RSRP'
    if 'SINR' in col:
        return 'SINR'
    if col.startswith('y1_'):
        return '下行吞吐率'
    if col.startswith('z1_'):
        return '上行吞吐率'
    if COVER_TOKEN in col:
        return '覆盖率1'
    if col == 'effective_dl':
        return '有效下行吞吐率'
    if col == 'effective_ul':
        return '有效上行吞吐率'
    if col == 'dl_drops_per_hour':
        return '下载掉线强度(次/小时)'
    if col == 'ul_drops_per_hour':
        return '上传掉线强度(次/小时)'
    if col == 'abs_delta_rsrp':
        return 'RSRP空间不均匀度'
    if col == 'abs_delta_cov2':
        return '覆盖率2空间不均匀度'
    if col == 'availability':
        return '5G可用性'
    if col == 'dl_tail_drag':
        return '下行尾部拖累'
    if col == 'ul_tail_drag':
        return '上行尾部拖累'
    return '指标'


def find_col(df, prefix=None, contains=None, contains_any=None):
    for col in df.columns:
        if prefix and not col.startswith(prefix):
            continue
        if contains and any(token not in col for token in contains):
            continue
        if contains_any and not any(token in col for token in contains_any):
            continue
        return col
    return None


def safe_divide(numer, denom):
    result = numer / denom
    return result.replace([np.inf, -np.inf], np.nan)


def calc_drop_rate(df, count_col, attempt_col, rate_col):
    if count_col and attempt_col:
        rate = safe_divide(df[count_col], df[attempt_col])
    elif rate_col:
        rate = df[rate_col]
    else:
        return None
    return rate.clip(lower=0, upper=1)


def minmax_norm(series):
    s = series.astype(float)
    min_v = s.min()
    max_v = s.max()
    if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
        return s * 0 + 0.5
    return (s - min_v) / (max_v - min_v)


def choose_weight_col(df):
    for col in ['test_distance_km', 'test_duration_h']:
        if col in df.columns and df[col].notna().sum() > 0:
            return col
    return None


def weighted_mean(values, weights):
    if values is None or weights is None:
        return np.nan
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def weighted_group_mean(df, group_cols, metric_cols, weight_col):
    if not weight_col:
        return pd.DataFrame()
    rows = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in metric_cols:
            row[f'{col}_wmean'] = weighted_mean(group[col], group[weight_col])
            row[f'{col}_n'] = int(group[col].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def missing_summary(df, metric_cols):
    rows = []
    for col in metric_cols:
        if not col:
            continue
        total = df.shape[0]
        valid = int(df[col].notna().sum())
        miss_rate = 0 if total == 0 else 1 - valid / total
        rows.append((col, valid, miss_rate))
    return rows


def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return None, None
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def winsorize_series(series):
    if series.empty:
        return series, 0.0
    lower, upper = iqr_bounds(series)
    if lower is None or upper is None:
        return series, 0.0
    clipped = series.clip(lower=lower, upper=upper)
    outlier_rate = float(((series < lower) | (series > upper)).mean())
    return clipped, outlier_rate


def robust_group_stats(df, group_col, metric_cols):
    rows = []
    for key, group in df.groupby(group_col):
        row = {group_col: key}
        for col in metric_cols:
            s = group[col].dropna()
            if s.empty:
                continue
            clipped, outlier_rate = winsorize_series(s)
            row[f'{col}_median'] = float(s.median())
            row[f'{col}_winsor_mean'] = float(clipped.mean())
            row[f'{col}_outlier_rate'] = outlier_rate
            row[f'{col}_n'] = int(s.shape[0])
        rows.append(row)
    return pd.DataFrame(rows)


def composite_scores(df, group_col, metric_cols):
    if not metric_cols:
        return pd.DataFrame()
    grouped = df.groupby(group_col)[metric_cols].mean()
    if grouped.empty:
        return pd.DataFrame()
    normed = grouped.copy()
    for col in metric_cols:
        normed[col] = minmax_norm(grouped[col])
    normed['composite_score'] = normed.mean(axis=1, skipna=True)
    return normed.reset_index()


def pairwise_ttests(df, group_col, metric_cols):
    if not HAS_SCIPY:
        return pd.DataFrame()
    groups = [g for g in df[group_col].dropna().unique()]
    results = []
    for col in metric_cols:
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a = df.loc[df[group_col] == groups[i], col].dropna()
                b = df.loc[df[group_col] == groups[j], col].dropna()
                if a.shape[0] < 5 or b.shape[0] < 5:
                    continue
                t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
                results.append({
                    'metric': col,
                    'operator_a': groups[i],
                    'operator_b': groups[j],
                    'n_a': int(a.shape[0]),
                    'n_b': int(b.shape[0]),
                    'mean_a': float(a.mean()),
                    'mean_b': float(b.mean()),
                    't_stat': float(t_stat),
                    'p_value': float(p_val),
                })
    return pd.DataFrame(results)


def compute_derived_metrics(df):
    derived = []

    x1_rsrp = find_col(df, prefix='x1_', contains_any=['RSRP'])
    x2_rsrp = find_col(df, prefix='x2_', contains_any=['RSRP'])
    if x1_rsrp and x2_rsrp:
        df['delta_rsrp'] = df[x1_rsrp] - df[x2_rsrp]
        df['abs_delta_rsrp'] = df['delta_rsrp'].abs()
        derived.extend(['delta_rsrp', 'abs_delta_rsrp'])

    x1_cov2 = find_col(df, prefix='x1_', contains=['覆盖率2'])
    x2_cov2 = find_col(df, prefix='x2_', contains=['覆盖率2'])
    if x1_cov2 and x2_cov2:
        df['delta_cov2'] = df[x1_cov2] - df[x2_cov2]
        df['abs_delta_cov2'] = df['delta_cov2'].abs()
        derived.extend(['delta_cov2', 'abs_delta_cov2'])

    avg_dl = find_col(df, prefix='y1_', contains=['应用层下行平均吞吐率'])
    peak_dl = find_col(df, prefix='y1_', contains=['下行前10%峰值速率'])
    dl_drop_rate_col = find_col(df, prefix='y1_', contains=['FTP下载掉线率'])
    dl_drop_count = find_col(df, prefix='y1_', contains=['FTP下载掉线次数'])
    dl_attempts = find_col(df, prefix='y1_', contains=['FTP下载尝试次数'])

    dl_drop_rate = calc_drop_rate(df, dl_drop_count, dl_attempts, dl_drop_rate_col)
    if dl_drop_rate is not None:
        df['dl_drop_rate_calc'] = dl_drop_rate
        derived.append('dl_drop_rate_calc')

    if avg_dl and dl_drop_rate is not None:
        df['effective_dl'] = df[avg_dl] * (1 - dl_drop_rate)
        derived.append('effective_dl')

    if avg_dl and peak_dl:
        df['dl_tail_drag'] = df[peak_dl] - df[avg_dl]
        derived.append('dl_tail_drag')

    if dl_drop_count and 'test_duration_h' in df.columns:
        df['dl_drops_per_hour'] = safe_divide(df[dl_drop_count], df['test_duration_h'])
        derived.append('dl_drops_per_hour')

    if dl_drop_count and 'test_distance_km' in df.columns:
        df['dl_drops_per_km'] = safe_divide(df[dl_drop_count], df['test_distance_km'])
        derived.append('dl_drops_per_km')

    avg_ul = find_col(df, prefix='z1_', contains=['应用层上行平均吞吐率'])
    peak_ul = find_col(df, prefix='z1_', contains_any=['上行前10%峰值速率', '上行前10%峰值率'])
    ul_drop_rate_col = find_col(df, prefix='z1_', contains=['FTP上传掉线率'])
    ul_drop_count = find_col(df, prefix='z1_', contains=['FTP上传掉线次数'])
    ul_attempts = find_col(df, prefix='z1_', contains=['FTP上传尝试次数'])

    ul_drop_rate = calc_drop_rate(df, ul_drop_count, ul_attempts, ul_drop_rate_col)
    if ul_drop_rate is not None:
        df['ul_drop_rate_calc'] = ul_drop_rate
        derived.append('ul_drop_rate_calc')

    if avg_ul and ul_drop_rate is not None:
        df['effective_ul'] = df[avg_ul] * (1 - ul_drop_rate)
        derived.append('effective_ul')

    if avg_ul and peak_ul:
        df['ul_tail_drag'] = df[peak_ul] - df[avg_ul]
        derived.append('ul_tail_drag')

    if ul_drop_count and 'test_duration_h' in df.columns:
        df['ul_drops_per_hour'] = safe_divide(df[ul_drop_count], df['test_duration_h'])
        derived.append('ul_drops_per_hour')

    if ul_drop_count and 'test_distance_km' in df.columns:
        df['ul_drops_per_km'] = safe_divide(df[ul_drop_count], df['test_distance_km'])
        derived.append('ul_drops_per_km')

    stay_ratio = find_col(df, contains=['时长驻留比'])
    stay_duration = find_col(df, contains=['驻留时长'])
    x1_cov5g = find_col(df, prefix='x1_', contains=['覆盖率3']) or find_col(df, prefix='x1_', contains=['覆盖率4'])
    if stay_ratio and x1_cov5g:
        df['availability'] = df[stay_ratio] * df[x1_cov5g]
        derived.append('availability')

    if stay_duration and 'test_distance_km' in df.columns:
        df['stay_eff'] = safe_divide(df[stay_duration], df['test_distance_km'])
        derived.append('stay_eff')

    return derived


def mobility_penalty(df, group_cols, metric_cols):
    if 'scene_l1' not in df.columns:
        return pd.DataFrame()
    rows = []
    for keys, group in df.groupby(group_cols):
        dt = group[group['scene_l1'] == '驶测']
        cqt = group[group['scene_l1'] == '定点']
        if dt.empty or cqt.empty:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in metric_cols:
            row[f'{col}_dt_mean'] = float(dt[col].mean())
            row[f'{col}_cqt_mean'] = float(cqt[col].mean())
            row[f'{col}_penalty'] = row[f'{col}_dt_mean'] - row[f'{col}_cqt_mean']
            row[f'{col}_n_dt'] = int(dt[col].notna().sum())
            row[f'{col}_n_cqt'] = int(cqt[col].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_city_operator_bar(df, metric_col, out_path, title):
    if not metric_col:
        return False
    data = df.groupby(['city', 'operator'])[metric_col].mean().reset_index()
    if data.empty:
        return False
    pivot = data.pivot(index='city', columns='operator', values=metric_col)
    ax = pivot.plot(kind='bar', figsize=(10, 5))
    label = metric_label(metric_col)
    ax.set_title(f'{title}（{label}）' if label else title)
    ax.set_xlabel('城市')
    ax.set_ylabel(label or '指标')
    ax.legend(title='运营商')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def plot_scene_trend(df, metric_col, out_path, title):
    if not metric_col:
        return False
    data = df.groupby(['scene_l2', 'operator'])[metric_col].mean().reset_index()
    if data.empty:
        return False
    order = data.groupby('scene_l2')[metric_col].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    for op, sub in data.groupby('operator'):
        sub = sub.set_index('scene_l2').reindex(order)
        ax.plot(order, sub[metric_col], marker='o', label=str(op))
    label = metric_label(metric_col)
    ax.set_title(f'{title}（{label}）' if label else title)
    ax.set_xlabel('二级场景')
    ax.set_ylabel(label or '指标')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.legend(title='运营商')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def plot_city_operator_heatmap(df, metric_col, out_path, title):
    if not metric_col:
        return False
    pivot = df.pivot_table(index='city', columns='operator', values=metric_col, aggfunc='mean')
    if pivot.empty:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
    label = metric_label(metric_col)
    ax.set_title(f'{title}（{label}）' if label else title)
    ax.set_xlabel('运营商')
    ax.set_ylabel('城市')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def plot_scene1_operator_bar(df, metric_col, out_path, title):
    if not metric_col:
        return False
    data = df.groupby(['scene_l1', 'operator'])[metric_col].mean().reset_index()
    if data.empty:
        return False
    pivot = data.pivot(index='scene_l1', columns='operator', values=metric_col)
    ax = pivot.plot(kind='bar', figsize=(8, 4))
    label = metric_label(metric_col)
    ax.set_title(f'{title}（{label}）' if label else title)
    ax.set_xlabel('一级场景')
    ax.set_ylabel(label or '指标')
    ax.legend(title='运营商')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def plot_operator_radar(df, metric_cols, labels, out_path, title):
    if not metric_cols:
        return False
    grouped = df.groupby('operator')[metric_cols].mean()
    if grouped.empty:
        return False
    normed = grouped.copy()
    for col in metric_cols:
        normed[col] = minmax_norm(grouped[col])

    angles = np.linspace(0, 2 * math.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for op in normed.index:
        values = normed.loc[op].tolist()
        values += values[:1]
        ax.plot(angles, values, label=str(op))
        ax.fill(angles, values, alpha=0.12)

    ax.set_title(title)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), title='运营商')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def top_entity(df, group_col, metric_col, mode='max'):
    if not metric_col:
        return None
    grouped = df.groupby(group_col)[metric_col].mean().dropna()
    if grouped.empty:
        return None
    if mode == 'min':
        key = grouped.idxmin()
    else:
        key = grouped.idxmax()
    return key, grouped.loc[key]


xl = pd.ExcelFile(DATA_PATH)
summary_lines = []

notes_path = os.path.join(OUT_DIR, 'metrics_notes.md')
with open(notes_path, 'w', encoding='utf-8-sig') as f:
    f.write('# 指标与阈值说明\n')
    for line in METRIC_NOTES:
        f.write(f'- {line}\n')
    f.write('\n')
    f.write(f'- 显著性检验可用：{("是" if HAS_SCIPY else "否")}\n')

for sheet in xl.sheet_names:
    raw = xl.parse(sheet)
    df = clean_df(raw)

    out_clean = os.path.join(OUT_DIR, f'cleaned_{sheet}.csv')
    df.to_csv(out_clean, index=False, encoding='utf-8-sig')

    rsrp, sinr, cov1, dl, ul = pick_metrics(df)
    base_metrics = [m for m in [rsrp, sinr, cov1, dl, ul] if m]

    derived_cols = compute_derived_metrics(df)
    weight_metrics = base_metrics + [c for c in ['effective_dl', 'effective_ul'] if c in df.columns]
    test_metrics = base_metrics + [c for c in ['effective_dl', 'effective_ul'] if c in df.columns]
    summary_metrics = base_metrics + [c for c in [
        'effective_dl', 'effective_ul', 'dl_drops_per_hour', 'ul_drops_per_hour',
        'abs_delta_rsrp', 'abs_delta_cov2', 'availability'
    ] if c in df.columns]

    weight_col = choose_weight_col(df)

    # Charts
    plot_city_operator_bar(df, rsrp, os.path.join(FIG_DIR, f'{sheet}_city_operator_bar.png'), f'{sheet} 城市-运营商对比柱状图')
    plot_scene_trend(df, dl or rsrp, os.path.join(FIG_DIR, f'{sheet}_scene_trend.png'), f'{sheet} 场景趋势图')
    plot_city_operator_heatmap(df, sinr, os.path.join(FIG_DIR, f'{sheet}_city_operator_heatmap.png'), f'{sheet} 城市-运营商热力图')
    plot_scene1_operator_bar(df, dl or rsrp, os.path.join(FIG_DIR, f'{sheet}_scene1_operator_bar.png'), f'{sheet} 一级场景对比')

    radar_metrics = []
    radar_labels = []
    for col in [rsrp, sinr, dl, ul, cov1]:
        if col:
            radar_metrics.append(col)
            radar_labels.append(metric_label(col))

    plot_operator_radar(df, radar_metrics, radar_labels, os.path.join(FIG_DIR, f'{sheet}_operator_radar.png'), f'{sheet} 运营商雷达图（归一化）')

    # Aggregated tables
    df.groupby(['city', 'operator']).mean(numeric_only=True).to_csv(
        os.path.join(OUT_DIR, f'{sheet}_city_operator_mean.csv'),
        encoding='utf-8-sig',
    )
    df.groupby(['scene_l2', 'operator']).mean(numeric_only=True).to_csv(
        os.path.join(OUT_DIR, f'{sheet}_scene_operator_mean.csv'),
        encoding='utf-8-sig',
    )
    df.groupby(['scene_l1', 'operator']).mean(numeric_only=True).to_csv(
        os.path.join(OUT_DIR, f'{sheet}_scene1_operator_mean.csv'),
        encoding='utf-8-sig',
    )
    df.groupby(['operator']).mean(numeric_only=True).to_csv(
        os.path.join(OUT_DIR, f'{sheet}_operator_mean.csv'),
        encoding='utf-8-sig',
    )

    # Weighted means
    weighted_df = weighted_group_mean(df, ['operator'], weight_metrics, weight_col)
    if not weighted_df.empty:
        weighted_df.to_csv(os.path.join(OUT_DIR, f'{sheet}_operator_weighted.csv'), index=False, encoding='utf-8-sig')

    # Robust stats
    robust_df = robust_group_stats(df, 'operator', base_metrics)
    if not robust_df.empty:
        robust_df.to_csv(os.path.join(OUT_DIR, f'{sheet}_operator_robust.csv'), index=False, encoding='utf-8-sig')

    # Composite scores
    comp_df = composite_scores(df, 'operator', base_metrics)
    if not comp_df.empty:
        comp_df.to_csv(os.path.join(OUT_DIR, f'{sheet}_operator_composite.csv'), index=False, encoding='utf-8-sig')

    # Significance tests
    ttest_df = pairwise_ttests(df, 'operator', test_metrics)
    if not ttest_df.empty:
        ttest_df.to_csv(os.path.join(OUT_DIR, f'{sheet}_significance.csv'), index=False, encoding='utf-8-sig')

    # Mobility penalty
    penalty_metrics = [m for m in [dl, ul, rsrp, sinr, 'effective_dl', 'effective_ul'] if m in df.columns]
    mobility_df = mobility_penalty(df, ['operator'], penalty_metrics)
    if not mobility_df.empty:
        mobility_df.to_csv(os.path.join(OUT_DIR, f'{sheet}_mobility_penalty.csv'), index=False, encoding='utf-8-sig')

    # Summary
    summary_lines.append(f'## {sheet}')
    summary_lines.append(f'- 数据量：{df.shape[0]} 行，字段 {df.shape[1]} 列')
    summary_lines.append('- 缺失值处理：不做填补，统计按有效样本计算')
    summary_lines.append(f'- 一级场景输出：{sheet}_scene1_operator_mean.csv、{sheet}_scene1_operator_bar.png')

    missing_items = []
    for col, valid, miss_rate in missing_summary(df, summary_metrics):
        missing_items.append(f'{metric_label(col)} {miss_rate * 100:.1f}% (n={valid})')
    if missing_items:
        summary_lines.append(f'- 关键指标缺失率：' + '，'.join(missing_items))

    if rsrp:
        top_op = top_entity(df, 'operator', rsrp, 'max')
        top_city = top_entity(df, 'city', rsrp, 'max')
        if top_op:
            summary_lines.append(f'- RSRP最优运营商：{top_op[0]}（{top_op[1]:.3f}）')
        if top_city:
            summary_lines.append(f'- RSRP最优城市：{top_city[0]}（{top_city[1]:.3f}）')
    if sinr:
        top_op = top_entity(df, 'operator', sinr, 'max')
        top_city = top_entity(df, 'city', sinr, 'max')
        if top_op:
            summary_lines.append(f'- SINR最优运营商：{top_op[0]}（{top_op[1]:.3f}）')
        if top_city:
            summary_lines.append(f'- SINR最优城市：{top_city[0]}（{top_city[1]:.3f}）')
    if dl:
        top_op = top_entity(df, 'operator', dl, 'max')
        if top_op:
            summary_lines.append(f'- 下行吞吐率最优运营商：{top_op[0]}（{top_op[1]:.3f}）')
    if ul:
        top_op = top_entity(df, 'operator', ul, 'max')
        if top_op:
            summary_lines.append(f'- 上行吞吐率最优运营商：{top_op[0]}（{top_op[1]:.3f}）')

    if 'effective_dl' in df.columns:
        top_op = top_entity(df, 'operator', 'effective_dl', 'max')
        if top_op:
            summary_lines.append(f'- 有效下行吞吐率最优运营商：{top_op[0]}（{top_op[1]:.3f}）')
    if 'effective_ul' in df.columns:
        top_op = top_entity(df, 'operator', 'effective_ul', 'max')
        if top_op:
            summary_lines.append(f'- 有效上行吞吐率最优运营商：{top_op[0]}（{top_op[1]:.3f}）')
    if 'dl_drops_per_hour' in df.columns:
        top_op = top_entity(df, 'operator', 'dl_drops_per_hour', 'min')
        if top_op:
            summary_lines.append(f'- 下载掉线强度(次/小时)最优运营商：{top_op[0]}（{top_op[1]:.3f}）')
    if 'ul_drops_per_hour' in df.columns:
        top_op = top_entity(df, 'operator', 'ul_drops_per_hour', 'min')
        if top_op:
            summary_lines.append(f'- 上传掉线强度(次/小时)最优运营商：{top_op[0]}（{top_op[1]:.3f}）')
    if 'abs_delta_rsrp' in df.columns:
        top_op = top_entity(df, 'operator', 'abs_delta_rsrp', 'min')
        if top_op:
            summary_lines.append(f'- RSRP空间不均匀度最优运营商：{top_op[0]}（{top_op[1]:.3f}）')
    if 'abs_delta_cov2' in df.columns:
        top_op = top_entity(df, 'operator', 'abs_delta_cov2', 'min')
        if top_op:
            summary_lines.append(f'- 覆盖率2空间不均匀度最优运营商：{top_op[0]}（{top_op[1]:.3f}）')
    if 'availability' in df.columns:
        top_op = top_entity(df, 'operator', 'availability', 'max')
        if top_op:
            summary_lines.append(f'- 5G可用性最优运营商：{top_op[0]}（{top_op[1]:.3f}）')

    if weight_col and not weighted_df.empty and rsrp:
        w_col = f'{rsrp}_wmean'
        if w_col in weighted_df.columns:
            top_w = weighted_df.set_index('operator')[w_col].dropna()
            if not top_w.empty:
                best = top_w.idxmax()
                summary_lines.append(f'- 加权口径：{weight_col}，RSRP加权最优运营商：{best}（{top_w.loc[best]:.3f}）')

    if not robust_df.empty and rsrp:
        r_col = f'{rsrp}_winsor_mean'
        if r_col in robust_df.columns:
            top_r = robust_df.set_index('operator')[r_col].dropna()
            if not top_r.empty:
                best = top_r.idxmax()
                summary_lines.append(f'- IQR稳健均值(RSRP)最优运营商：{best}（{top_r.loc[best]:.3f}）')

    if not comp_df.empty:
        comp_sorted = comp_df.sort_values('composite_score', ascending=False)
        best = comp_sorted.iloc[0]
        summary_lines.append(f'- 综合评分最优运营商：{best["operator"]}（{best["composite_score"]:.3f}）')

    if not mobility_df.empty:
        penalty_metric = None
        for candidate in ['effective_dl', dl, rsrp]:
            if candidate and f'{candidate}_penalty' in mobility_df.columns:
                penalty_metric = candidate
                break
        if penalty_metric:
            pcol = f'{penalty_metric}_penalty'
            subset = mobility_df[['operator', pcol]].dropna()
            if not subset.empty:
                worst = subset.loc[subset[pcol].idxmin()]
                summary_lines.append(
                    f'- DT相对CQT移动惩罚（{metric_label(penalty_metric)}）：{worst["operator"]}（{worst[pcol]:.3f}）'
                )

    if not ttest_df.empty:
        sig = ttest_df.loc[ttest_df['p_value'] < 0.05]
        summary_lines.append(f'- 显著性检验结果：显著差异 {sig.shape[0]} 条（详见 {sheet}_significance.csv）')
    elif HAS_SCIPY:
        summary_lines.append('- 显著性检验结果：无满足样本条件的可检验项')
    else:
        summary_lines.append('- 显著性检验结果：未安装 SciPy，已跳过')

summary_path = os.path.join(OUT_DIR, 'summary.md')
with open(summary_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(summary_lines))

print('Done. Outputs in', OUT_DIR)
