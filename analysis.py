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
DATA_PATHS = glob.glob(os.path.join(BASE_DIR, '*.xlsx'))
if not DATA_PATHS:
    raise SystemExit('No .xlsx file found in working directory')
DATA_PATH = DATA_PATHS[0]

OUT_DIR = os.path.join(BASE_DIR, 'output')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SCENE2_MAP = {
    '\u505c\u8f66\u573a': '\u505c\u8f66\u573a',
    '\u8865\u7ed9\u7ad9': '\u8865\u7ed9\u7ad9',
    '\u6b65\u884c\u9053': '\u6b65\u884c\u9053',
    '\u9a91\u884c\u9053': '\u9a91\u884c\u9053',
    '\u5496\u5561\u5385': '\u5496\u5561\u5385',
    '\u57ce\u5e02\u9053\u8def': '\u57ce\u5e02\u9053\u8def',
    '\u4f11\u606f\u70b9': '\u4f11\u606f\u70b9',
}

SCENE1_MAP = {
    'DT': '\u9a71\u6d4b',
    'CQT': '\u5b9a\u70b9',
}

OP_MAP = {
    1: '\u79fb\u52a8',
    2: '\u7535\u4fe1',
    3: '\u8054\u901a',
}

COVER_TOKEN = '\u8986\u76d6\u73871'
METRIC_NOTES = [
    '\u8986\u76d6\u73871\uff1a\u6ee1\u8db3(RSRP >= -105dBm \u4e14 SINR >= -3)\u7684\u91c7\u6837\u70b9\u5360\u6bd4',
    '\u4e0b\u884c\u901f\u7387\u8fbe\u6807\u7387\uff1a\u5e94\u7528\u5c42\u4e0b\u884c\u901f\u7387 >= 2Mbps \u7684\u5360\u6bd4',
    '\u4e0a\u884c\u901f\u7387\u8fbe\u6807\u7387\uff1a\u5e94\u7528\u5c42\u4e0a\u884c\u901f\u7387 >= 0.5Mbps \u7684\u5360\u6bd4',
    '\u5355\u4f4d\uff1a\u541e\u5410\u7387\u4e3a Mbps\uff0c\u6d4b\u8bd5\u603b\u91cc\u7a0b\u4e3a KM\uff0c\u603b\u65f6\u957f\u4e3a h',
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

    # Normalize category columns
    for col in ['vendor', 'city', 'scene_l1', 'scene_l2', 'operator']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(['nan', 'None', 'NaT']), col] = np.nan

    # Map operator codes
    df['operator'] = pd.to_numeric(df['operator'], errors='coerce').map(OP_MAP).fillna(df['operator'])

    # Map scenes
    df['scene_l1'] = df['scene_l1'].map(SCENE1_MAP).fillna(df['scene_l1'])
    df['scene_l2'] = df['scene_l2'].map(SCENE2_MAP).fillna(df['scene_l2'])

    # Numeric conversion for non-category columns
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
        return '\u4e0b\u884c\u541e\u5410\u7387'
    if col.startswith('z1_'):
        return '\u4e0a\u884c\u541e\u5410\u7387'
    if COVER_TOKEN in col:
        return '\u8986\u76d6\u73871'
    return '\u6307\u6807'


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
    ax.set_xlabel('\u57ce\u5e02')
    ax.set_ylabel(label or '\u6307\u6807')
    ax.legend(title='\u8fd0\u8425\u5546')
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
    ax.set_xlabel('\u4e8c\u7ea7\u573a\u666f')
    ax.set_ylabel(label or '\u6307\u6807')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.legend(title='\u8fd0\u8425\u5546')
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
    ax.set_xlabel('\u8fd0\u8425\u5546')
    ax.set_ylabel('\u57ce\u5e02')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, shrink=0.8)
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
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), title='\u8fd0\u8425\u5546')
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

for sheet in xl.sheet_names:
    raw = xl.parse(sheet)
    df = clean_df(raw)

    out_clean = os.path.join(OUT_DIR, f'cleaned_{sheet}.csv')
    df.to_csv(out_clean, index=False, encoding='utf-8-sig')

    rsrp, sinr, cov1, dl, ul = pick_metrics(df)

    # Charts
    plot_city_operator_bar(df, rsrp, os.path.join(FIG_DIR, f'{sheet}_city_operator_bar.png'), f'{sheet} \u57ce\u5e02-\u8fd0\u8425\u5546\u5bf9\u6bd4\u67f1\u72b6\u56fe')
    plot_scene_trend(df, dl or rsrp, os.path.join(FIG_DIR, f'{sheet}_scene_trend.png'), f'{sheet} \u573a\u666f\u8d8b\u52bf\u56fe')
    plot_city_operator_heatmap(df, sinr, os.path.join(FIG_DIR, f'{sheet}_city_operator_heatmap.png'), f'{sheet} \u57ce\u5e02-\u8fd0\u8425\u5546\u70ed\u529b\u56fe')

    radar_metrics = []
    radar_labels = []
    if rsrp:
        radar_metrics.append(rsrp)
        radar_labels.append('RSRP')
    if sinr:
        radar_metrics.append(sinr)
        radar_labels.append('SINR')
    if dl:
        radar_metrics.append(dl)
        radar_labels.append('DL Throughput')
    if ul:
        radar_metrics.append(ul)
        radar_labels.append('UL Throughput')
    if cov1:
        radar_metrics.append(cov1)
        radar_labels.append('Coverage')

    plot_operator_radar(df, radar_metrics, radar_labels, os.path.join(FIG_DIR, f'{sheet}_operator_radar.png'), f'{sheet} \u8fd0\u8425\u5546\u96f7\u8fbe\u56fe\uff08\u5f52\u4e00\u5316\uff09')

    # Aggregated tables
    df.groupby(['city', 'operator']).mean(numeric_only=True).to_csv(
        os.path.join(OUT_DIR, f'{sheet}_city_operator_mean.csv'),
        encoding='utf-8-sig',
    )
    df.groupby(['scene_l2', 'operator']).mean(numeric_only=True).to_csv(
        os.path.join(OUT_DIR, f'{sheet}_scene_operator_mean.csv'),
        encoding='utf-8-sig',
    )
    df.groupby(['operator']).mean(numeric_only=True).to_csv(
        os.path.join(OUT_DIR, f'{sheet}_operator_mean.csv'),
        encoding='utf-8-sig',
    )

    # Summary
    summary_lines.append(f'## {sheet}')
    if rsrp:
        top_op = top_entity(df, 'operator', rsrp, 'max')
        top_city = top_entity(df, 'city', rsrp, 'max')
        if top_op:
            summary_lines.append(f'- RSRP\u6700\u4f18\u8fd0\u8425\u5546\uff1a{top_op[0]}\uff08{top_op[1]:.3f}\uff09')
        if top_city:
            summary_lines.append(f'- RSRP\u6700\u4f18\u57ce\u5e02\uff1a{top_city[0]}\uff08{top_city[1]:.3f}\uff09')
    if sinr:
        top_op = top_entity(df, 'operator', sinr, 'max')
        top_city = top_entity(df, 'city', sinr, 'max')
        if top_op:
            summary_lines.append(f'- SINR\u6700\u4f18\u8fd0\u8425\u5546\uff1a{top_op[0]}\uff08{top_op[1]:.3f}\uff09')
        if top_city:
            summary_lines.append(f'- SINR\u6700\u4f18\u57ce\u5e02\uff1a{top_city[0]}\uff08{top_city[1]:.3f}\uff09')
    if dl:
        top_op = top_entity(df, 'operator', dl, 'max')
        if top_op:
            summary_lines.append(f'- \u4e0b\u884c\u541e\u5410\u7387\u6700\u4f18\u8fd0\u8425\u5546\uff1a{top_op[0]}\uff08{top_op[1]:.3f}\uff09')
    if ul:
        top_op = top_entity(df, 'operator', ul, 'max')
        if top_op:
            summary_lines.append(f'- \u4e0a\u884c\u541e\u5410\u7387\u6700\u4f18\u8fd0\u8425\u5546\uff1a{top_op[0]}\uff08{top_op[1]:.3f}\uff09')

summary_path = os.path.join(OUT_DIR, 'summary.md')
with open(summary_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(summary_lines))

print('Done. Outputs in', OUT_DIR)
