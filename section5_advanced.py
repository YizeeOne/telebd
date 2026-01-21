# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("all_final_data_with_attributes.csv")
CELL_AGG_PATH = Path("report_assets/section3/section3_cell_agg.csv")
OUT_DIR = Path("report_assets/section5")

CHUNK_SIZE = 1_000_000
TOP_CELL_COUNT = None
TEST_DAYS = 7


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_cn_style() -> None:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans SC",
        "SimSun",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=150)
    plt.close()


def safe_mean(sum_vals: np.ndarray, cnt_vals: np.ndarray) -> np.ndarray:
    denom = np.where(cnt_vals == 0, np.nan, cnt_vals)
    return sum_vals / denom


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.nanstd(y_true) == 0:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def add_metric_box(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.01,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#666666", "alpha": 0.8},
    )


def weekday_labels(values: list[int]) -> list[str]:
    mapping = {
        0: "\u5468\u4e00",
        1: "\u5468\u4e8c",
        2: "\u5468\u4e09",
        3: "\u5468\u56db",
        4: "\u5468\u4e94",
        5: "\u5468\u516d",
        6: "\u5468\u65e5",
    }
    return [mapping.get(int(v), str(v)) for v in values]


def build_radar(categories: list[str], values: np.ndarray, labels: list[str]) -> None:
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    for idx, row in enumerate(values):
        data = row.tolist()
        data += data[:1]
        ax.plot(angles, data, linewidth=2, label=labels[idx])
        ax.fill(angles, data, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title("\u805a\u7c7b\u7ef4\u5ea6\u7efc\u5408\u5236\u56fe")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))


def cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    values = table.to_numpy()
    n = values.sum()
    if n == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan"), float("nan")
    row_sums = values.sum(axis=1, keepdims=True)
    col_sums = values.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    chi2 = np.nansum((values - expected) ** 2 / np.where(expected == 0, np.nan, expected))
    k = min(table.shape[0] - 1, table.shape[1] - 1)
    v = float(np.sqrt(chi2 / (n * k))) if k > 0 else float("nan")
    return float(chi2), v


def main() -> None:
    ensure_out_dir()
    set_cn_style()

    if not DATA_PATH.exists():
        raise FileNotFoundError("\u672a\u627e\u5230 all_final_data_with_attributes.csv")
    if not CELL_AGG_PATH.exists():
        raise FileNotFoundError("\u672a\u627e\u5230 report_assets/section3/section3_cell_agg.csv")

    cell_agg = pd.read_csv(CELL_AGG_PATH)
    cell_agg["CELL_ID"] = cell_agg["CELL_ID"].astype(int)
    cell_agg_sorted = cell_agg.sort_values("flow_sum", ascending=False)
    if TOP_CELL_COUNT:
        top_cells = cell_agg_sorted.head(TOP_CELL_COUNT)["CELL_ID"].tolist()
    else:
        top_cells = cell_agg_sorted["CELL_ID"].tolist()
    target_cell = int(top_cells[0])
    top_set = set(top_cells)
    cell_index = {cell_id: idx for idx, cell_id in enumerate(top_cells)}

    flow_hour_sum = np.zeros((len(top_cells), 24), dtype=np.float64)
    flow_hour_cnt = np.zeros((len(top_cells), 24), dtype=np.int64)
    user_hour_sum = np.zeros((len(top_cells), 24), dtype=np.float64)
    user_hour_cnt = np.zeros((len(top_cells), 24), dtype=np.int64)

    target_records: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=["CELL_ID", "DATETIME_KEY", "FLOW_SUM", "USER_COUNT"],
        dtype={
            "CELL_ID": "int32",
            "DATETIME_KEY": "string",
            "FLOW_SUM": "float32",
            "USER_COUNT": "float32",
        },
        chunksize=CHUNK_SIZE,
        na_values=["", "NA", "NaN"],
    ):
        chunk.loc[chunk["FLOW_SUM"] < 0, "FLOW_SUM"] = np.nan
        chunk.loc[chunk["USER_COUNT"] < 0, "USER_COUNT"] = np.nan
        dt_values = pd.to_datetime(chunk["DATETIME_KEY"], errors="coerce")
        chunk["hour"] = dt_values.dt.hour

        target_mask = chunk["CELL_ID"] == target_cell
        if target_mask.any():
            target_records.append(
                chunk.loc[target_mask, ["DATETIME_KEY", "FLOW_SUM", "USER_COUNT"]]
            )

        mask = chunk["CELL_ID"].isin(top_set)
        if not mask.any():
            continue

        sub = chunk.loc[mask, ["CELL_ID", "hour", "FLOW_SUM", "USER_COUNT"]].dropna(
            subset=["hour"]
        )
        if sub.empty:
            continue

        grp = sub.groupby(["CELL_ID", "hour"]).agg(
            flow_sum=("FLOW_SUM", "sum"),
            flow_count=("FLOW_SUM", "count"),
            user_sum=("USER_COUNT", "sum"),
            user_count=("USER_COUNT", "count"),
        )

        for (cell_id, hour), row in grp.iterrows():
            if cell_id not in cell_index:
                continue
            if hour < 0 or hour > 23:
                continue
            idx = cell_index[cell_id]
            flow_hour_sum[idx, hour] += row["flow_sum"]
            flow_hour_cnt[idx, hour] += row["flow_count"]
            user_hour_sum[idx, hour] += row["user_sum"]
            user_hour_cnt[idx, hour] += row["user_count"]

    flow_hour_mean = safe_mean(flow_hour_sum, flow_hour_cnt)
    user_hour_mean = safe_mean(user_hour_sum, user_hour_cnt)

    flow_profile_sum = np.nansum(flow_hour_mean, axis=1, keepdims=True)
    user_profile_sum = np.nansum(user_hour_mean, axis=1, keepdims=True)
    flow_profile_share = flow_hour_mean / flow_profile_sum
    user_profile_share = user_hour_mean / user_profile_sum
    valid_mask = np.isfinite(flow_profile_share).all(axis=1) & np.isfinite(
        user_profile_share
    ).all(axis=1)
    flow_profile_share = flow_profile_share[valid_mask]
    user_profile_share = user_profile_share[valid_mask]
    valid_cells = np.array(top_cells)[valid_mask]

    profile_df = pd.DataFrame(
        np.hstack([flow_profile_share, user_profile_share]),
        columns=[f"flow_h{h}" for h in range(24)] + [f"user_h{h}" for h in range(24)],
    )
    cell_meta = cell_agg[cell_agg["CELL_ID"].isin(valid_cells)].copy()
    cell_meta = cell_meta.set_index("CELL_ID").loc[valid_cells].reset_index()

    extra_cols = [
        "flow_mean",
        "user_mean",
        "flow_per_user",
        "activity_mean",
    ]
    extra_df = cell_meta[extra_cols].copy()
    extra_df = (extra_df - extra_df.mean()) / extra_df.std()
    profile_df = pd.concat([profile_df.reset_index(drop=True), extra_df], axis=1)

    ks = list(range(2, 9))
    inertias = []
    silhouettes = []
    ch_scores = []
    db_scores = []
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(profile_df)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(profile_df, labels))
        ch_scores.append(calinski_harabasz_score(profile_df, labels))
        db_scores.append(davies_bouldin_score(profile_df, labels))

    best_k = ks[int(np.argmax(silhouettes))]
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_kmeans.fit_predict(profile_df)
    cell_meta["cluster"] = labels
    flow_hour_valid = flow_hour_mean[valid_mask]
    peak_hours = np.nanargmax(flow_hour_valid, axis=1)
    cell_meta["peak_hour"] = peak_hours

    day_hours = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    night_hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23])
    day_flow = np.nansum(flow_hour_valid[:, day_hours], axis=1)
    night_flow = np.nansum(flow_hour_valid[:, night_hours], axis=1)
    day_night_ratio = day_flow / np.where(night_flow == 0, np.nan, night_flow)
    cell_meta["day_night_ratio"] = day_night_ratio

    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertias, marker="o")
    plt.title("\u805a\u7c7b\u80fd\u91cf\u66f2\u7ebf")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    save_fig("fig01_elbow_kmeans.png")

    plt.figure(figsize=(6, 4))
    plt.plot(ks, silhouettes, marker="o", color="#E45756")
    plt.title("\u8f6e\u5ed3\u7cfb\u6570\u66f2\u7ebf")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    save_fig("fig02_silhouette_scores.png")

    plt.figure(figsize=(6, 4))
    plt.plot(ks, ch_scores, marker="o", color="#4C78A8")
    plt.title("Calinski-Harabasz \u66f2\u7ebf")
    plt.xlabel("K")
    plt.ylabel("CH Score")
    save_fig("fig02b_ch_scores.png")

    plt.figure(figsize=(6, 4))
    plt.plot(ks, db_scores, marker="o", color="#72B7B2")
    plt.title("Davies-Bouldin \u66f2\u7ebf")
    plt.xlabel("K")
    plt.ylabel("DB Score")
    save_fig("fig02c_db_scores.png")

    cluster_profiles = (
        pd.DataFrame(flow_profile_share, columns=[f"h{h}" for h in range(24)])
        .assign(cluster=labels)
        .groupby("cluster")
        .mean()
        * 100
    )

    plt.figure(figsize=(10, 4))
    for cluster_id, row in cluster_profiles.iterrows():
        plt.plot(range(24), row.values, marker="o", label=f"Cluster {cluster_id}")
    plt.title("\u805a\u7c7b\u5e73\u5747\u65e5\u5185\u66f2\u7ebf\uff08\u5360\u6bd4\uff09")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u6d41\u91cf\u5360\u6bd4\uff08%\uff09")
    plt.legend()
    save_fig("fig03_cluster_profiles.png")

    plt.figure(figsize=(8, 4))
    cluster_counts = cell_meta["cluster"].value_counts().sort_index()
    sns.barplot(x=cluster_counts.index.astype(str), y=cluster_counts.values, color="#4C78A8")
    plt.title("\u805a\u7c7b\u5c0f\u533a\u6570\u91cf")
    plt.xlabel("Cluster")
    plt.ylabel("\u5c0f\u533a\u6570\u91cf")
    save_fig("fig04_cluster_sizes.png")

    scene_ct = pd.crosstab(cell_meta["cluster"], cell_meta["SCENE"])
    scene_ct_full = scene_ct.copy()
    top_scenes = scene_ct.sum(axis=0).sort_values(ascending=False).head(10).index
    scene_ct = scene_ct[top_scenes]
    plt.figure(figsize=(10, 4))
    sns.heatmap(scene_ct, cmap="Blues")
    plt.title("\u805a\u7c7b\u4e0e\u573a\u666f\u5173\u7cfb\u70ed\u529b\u56fe")
    plt.xlabel("\u573a\u666f\uff08SCENE\uff09")
    plt.ylabel("Cluster")
    save_fig("fig05_cluster_scene_heatmap.png")

    scene_share = scene_ct.div(scene_ct.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(10, 4))
    sns.heatmap(scene_share, cmap="YlGnBu", fmt=".1f")
    plt.title("\u805a\u7c7b\u00d7\u573a\u666f\u5360\u6bd4\u70ed\u529b\u56fe\uff08%\uff09")
    plt.xlabel("\u573a\u666f\uff08SCENE\uff09")
    plt.ylabel("Cluster")
    save_fig("fig30_cluster_scene_share_heatmap.png")

    cluster_ids = scene_ct.index.tolist()
    x = np.arange(len(top_scenes))
    width = 0.35
    plt.figure(figsize=(10, 4))
    for idx, cluster_id in enumerate(cluster_ids):
        values = scene_ct.loc[cluster_id].values
        plt.bar(x + (idx - 0.5) * width, values, width, label=f"C{cluster_id}")
    plt.xticks(x, top_scenes.astype(int).astype(str), rotation=30)
    plt.title("\u4e0d\u540c\u805a\u7c7b\u7684\u573a\u666f\u6570\u91cf\u5bf9\u6bd4")
    plt.xlabel("\u573a\u666f\uff08SCENE\uff09")
    plt.ylabel("\u5c0f\u533a\u6570\u91cf")
    plt.legend()
    save_fig("fig31_cluster_scene_grouped_bar.png")

    total = scene_ct_full.to_numpy().sum()
    row_sums = scene_ct_full.sum(axis=1).to_numpy()[:, None]
    col_sums = scene_ct_full.sum(axis=0).to_numpy()[None, :]
    expected = row_sums @ col_sums / max(total, 1)
    residual = (scene_ct_full.to_numpy() - expected) / np.sqrt(
        np.where(expected == 0, np.nan, expected)
    )
    residual_df = pd.DataFrame(residual, index=scene_ct_full.index, columns=scene_ct_full.columns)
    residual_df = residual_df[top_scenes]
    plt.figure(figsize=(10, 4))
    sns.heatmap(residual_df, cmap="coolwarm", center=0)
    plt.title("\u805a\u7c7b\u00d7\u573a\u666f\u6807\u51c6\u5316\u6b8b\u5dee")
    plt.xlabel("\u573a\u666f\uff08SCENE\uff09")
    plt.ylabel("Cluster")
    save_fig("fig32_cluster_scene_residual.png")

    type_ct = pd.crosstab(cell_meta["cluster"], cell_meta["TYPE"])
    type_ct_full = type_ct.copy()
    plt.figure(figsize=(8, 4))
    type_ct.plot(kind="bar", stacked=True, colormap="Set2", ax=plt.gca())
    plt.title("\u805a\u7c7b\u4e0e TYPE \u5206\u5e03")
    plt.xlabel("Cluster")
    plt.ylabel("\u5c0f\u533a\u6570\u91cf")
    plt.legend(title="TYPE", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig("fig06_cluster_type_bar.png")

    type_share = type_ct.div(type_ct.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(8, 4))
    type_share.plot(kind="bar", stacked=True, colormap="Set2", ax=plt.gca())
    plt.title("\u805a\u7c7b\u00d7TYPE \u5360\u6bd4\u5206\u5e03\uff08%\uff09")
    plt.xlabel("Cluster")
    plt.ylabel("\u5360\u6bd4\uff08%\uff09")
    plt.legend(title="TYPE", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig("fig33_cluster_type_share_bar.png")

    chi2_scene, cramers_scene = cramers_v(scene_ct_full)
    chi2_type, cramers_type = cramers_v(type_ct_full)

    scene_share_top = scene_ct.div(scene_ct.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(10, 4))
    scene_share_top.plot(kind="bar", stacked=True, colormap="tab20", ax=plt.gca())
    plt.title("\u805a\u7c7b\u00d7\u573a\u666f\u5360\u6bd4\u7ec4\u5408\uff08Top\u573a\u666f\uff09")
    plt.xlabel("Cluster")
    plt.ylabel("\u5360\u6bd4\uff08%\uff09")
    plt.legend(title="SCENE", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig("fig34_cluster_scene_share_bar.png")

    type_total = type_ct_full.to_numpy().sum()
    type_row = type_ct_full.sum(axis=1).to_numpy()[:, None]
    type_col = type_ct_full.sum(axis=0).to_numpy()[None, :]
    type_expected = type_row @ type_col / max(type_total, 1)
    type_residual = (type_ct_full.to_numpy() - type_expected) / np.sqrt(
        np.where(type_expected == 0, np.nan, type_expected)
    )
    type_residual_df = pd.DataFrame(
        type_residual, index=type_ct_full.index, columns=type_ct_full.columns
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(type_residual_df, cmap="coolwarm", center=0)
    plt.title("\u805a\u7c7b\u00d7TYPE \u6807\u51c6\u5316\u6b8b\u5dee")
    plt.xlabel("TYPE")
    plt.ylabel("Cluster")
    save_fig("fig35_cluster_type_residual.png")

    if scene_share_top.shape[0] == 2:
        diff = scene_share_top.iloc[1] - scene_share_top.iloc[0]
        colors = ["#E45756" if v < 0 else "#54A24B" for v in diff.values]
        plt.figure(figsize=(10, 4))
        plt.bar(scene_share_top.columns.astype(int).astype(str), diff.values, color=colors)
        plt.axhline(0, color="#333333", linewidth=1)
        plt.title("\u805a\u7c7b\u95f4\u573a\u666f\u5360\u6bd4\u5dee\u5f02\uff08C1-C0\uff09")
        plt.xlabel("\u573a\u666f\uff08SCENE\uff09")
        plt.ylabel("\u5360\u6bd4\u5dee\u503c\uff08%\uff09")
        save_fig("fig36_cluster_scene_share_diff.png")

    metric_items = [
        ("flow_mean", "\u5e73\u5747\u6d41\u91cf"),
        ("user_mean", "\u5e73\u5747\u7528\u6237\u6570"),
        ("flow_per_user", "\u4eba\u5747\u6d41\u91cf"),
        ("flow_cv", "\u6d41\u91cfCV"),
        ("peak_ratio", "\u5cf0\u5747\u6bd4"),
        ("activity_mean", "\u6d3b\u8dc3\u5ea6"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (col, title) in zip(axes.flat, metric_items):
        sns.boxplot(data=cell_meta, x="cluster", y=col, ax=ax, showfliers=False)
        ax.set_title(title)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("")
    save_fig("fig37_cluster_metric_box.png")

    flow_user_df = cell_meta[["flow_mean", "user_mean", "cluster"]].dropna()
    flow_user_sample = flow_user_df.sample(min(6000, len(flow_user_df)), random_state=42)
    plt.figure(figsize=(6, 5))
    for cluster_id, subset in flow_user_sample.groupby("cluster"):
        plt.scatter(
            np.log1p(subset["flow_mean"]),
            np.log1p(subset["user_mean"]),
            s=8,
            alpha=0.4,
            label=f"Cluster {cluster_id}",
        )
    plt.title("\u805a\u7c7b\u6d41\u91cf-\u7528\u6237\u6570\u5173\u7cfb\uff08log1p\uff09")
    plt.xlabel("log(1+\u5e73\u5747\u6d41\u91cf)")
    plt.ylabel("log(1+\u5e73\u5747\u7528\u6237\u6570)")
    plt.legend()
    save_fig("fig38_cluster_flow_user_scatter.png")

    cv_peak_df = cell_meta[["flow_cv", "peak_ratio", "cluster"]].dropna()
    cv_peak_sample = cv_peak_df.sample(min(6000, len(cv_peak_df)), random_state=42)
    plt.figure(figsize=(6, 5))
    for cluster_id, subset in cv_peak_sample.groupby("cluster"):
        plt.scatter(
            subset["flow_cv"],
            subset["peak_ratio"],
            s=8,
            alpha=0.4,
            label=f"Cluster {cluster_id}",
        )
    plt.title("\u805a\u7c7b\u6d41\u91cf\u6ce2\u52a8-\u5cf0\u5747\u6bd4\u5173\u7cfb")
    plt.xlabel("\u6d41\u91cfCV")
    plt.ylabel("\u5cf0\u5747\u6bd4")
    plt.legend()
    save_fig("fig39_cluster_flowcv_peakratio_scatter.png")

    peak_ct = pd.crosstab(cell_meta["cluster"], cell_meta["peak_hour"])
    peak_share = peak_ct.div(peak_ct.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(10, 4))
    sns.heatmap(peak_share, cmap="YlGnBu")
    plt.title("\u805a\u7c7b\u00d7\u5cf0\u503c\u65f6\u6bb5\u5360\u6bd4\uff08%\uff09")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("Cluster")
    save_fig("fig40_cluster_peak_hour_heatmap.png")

    ratio_df = cell_meta[["cluster", "day_night_ratio"]].replace([np.inf, -np.inf], np.nan)
    ratio_df = ratio_df.dropna()
    ratio_df = ratio_df[ratio_df["day_night_ratio"] > 0]
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=ratio_df, x="cluster", y="day_night_ratio", showfliers=False)
    plt.yscale("log")
    plt.title("\u805a\u7c7b\u65e5/\u591c\u6d41\u91cf\u6bd4\u5206\u5e03")
    plt.xlabel("Cluster")
    plt.ylabel("\u65e5/\u591c\u6d41\u91cf\u6bd4\uff08log\uff09")
    save_fig("fig41_cluster_day_night_ratio.png")

    plt.figure(figsize=(6, 6))
    scatter_df = cell_meta.sample(min(3000, len(cell_meta)), random_state=42)
    unique_clusters = np.sort(scatter_df["cluster"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    for color, cluster_id in zip(colors, unique_clusters):
        subset = scatter_df[scatter_df["cluster"] == cluster_id]
        plt.scatter(
            subset["LONGITUDE"],
            subset["LATITUDE"],
            s=8,
            alpha=0.6,
            color=color,
            label=f"Cluster {cluster_id}",
        )
    plt.legend(title="\u805a\u7c7b\u7c7b\u522b", loc="best")
    plt.title("\u805a\u7c7b\u5730\u7406\u7a7a\u95f4\u5206\u5e03")
    plt.xlabel("\u7ecf\u5ea6")
    plt.ylabel("\u7eac\u5ea6")
    save_fig("fig07_cluster_geo_scatter.png")

    radar_cols = [
        "flow_mean",
        "user_mean",
        "flow_per_user",
        "flow_cv",
        "peak_ratio",
        "activity_mean",
    ]
    radar_data = cell_meta[radar_cols].copy()
    radar_data = (radar_data - radar_data.mean()) / radar_data.std()
    radar_df = radar_data.assign(cluster=cell_meta["cluster"]).groupby("cluster").mean()
    radar_values = radar_df.values
    radar_labels = [f"C{i}" for i in radar_df.index]
    radar_names = [
        "\u6d41\u91cf\u5747\u503c",
        "\u7528\u6237\u5747\u503c",
        "\u4eba\u5747\u6d41\u91cf",
        "\u6d41\u91cfCV",
        "\u5cf0\u5747\u6bd4",
        "\u6d3b\u8dc3\u5ea6",
    ]
    build_radar(radar_names, radar_values, radar_labels)
    save_fig("fig08_cluster_radar.png")

    plt.figure(figsize=(10, 4))
    sns.heatmap(cluster_profiles, cmap="YlOrRd")
    plt.title("\u805a\u7c7b\u5e73\u5747\u6d41\u91cf\u65f6\u95f4\u70ed\u529b\u56fe")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("Cluster")
    save_fig("fig09_cluster_hour_heatmap.png")

    corr_cols = [
        "flow_mean",
        "user_mean",
        "flow_per_user",
        "flow_cv",
        "peak_ratio",
        "activity_mean",
        "silent_ratio",
    ]
    corr_df = cell_agg[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, cmap="coolwarm", center=0, annot=False)
    plt.title("\u6307\u6807\u76f8\u5173\u6027\u70ed\u529b\u56fe")
    save_fig("fig10_correlation_heatmap.png")

    pred_feature_num = [
        "user_mean",
        "user_max",
        "user_std",
        "user_cv",
        "activity_mean",
        "par_mean",
        "silent_ratio",
        "flow_cv",
        "peak_ratio",
        "flow_per_user",
        "LATITUDE",
        "LONGITUDE",
    ]
    pred_feature_cat = ["SCENE", "TYPE"]
    pred_target = "flow_mean"
    pred_df = cell_agg[pred_feature_num + pred_feature_cat + [pred_target]].dropna()
    pred_metrics = None
    if pred_df.shape[0] >= 500:
        X = pred_df[pred_feature_num + pred_feature_cat]
        y = pred_df[pred_target].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), pred_feature_num),
                ("cat", OneHotEncoder(handle_unknown="ignore"), pred_feature_cat),
            ]
        )
        pred_model = GradientBoostingRegressor(
            random_state=42,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.9,
            min_samples_leaf=8,
        )
        pred_pipe = make_pipeline(pre, pred_model)
        pred_pipe.fit(X_train, np.log1p(y_train))
        y_pred = np.expm1(pred_pipe.predict(X_test))

        pred_mae = float(np.mean(np.abs(y_test - y_pred)))
        pred_rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        pred_mape = float(np.mean(np.abs(y_test - y_pred) / np.maximum(y_test, 1e-6)))
        pred_r2 = safe_r2(y_test, y_pred)
        pred_metrics = {
            "target": pred_target,
            "features_num": pred_feature_num,
            "features_cat": pred_feature_cat,
            "mae": pred_mae,
            "rmse": pred_rmse,
            "mape": pred_mape,
            "mape_pct": float(pred_mape * 100),
            "r2": pred_r2,
            "test_size": int(y_test.shape[0]),
        }

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, s=10, alpha=0.4, color="#4C78A8")
        min_val = float(np.nanmin([y_test.min(), y_pred.min()]))
        max_val = float(np.nanmax([y_test.max(), y_pred.max()]))
        ax.plot([min_val, max_val], [min_val, max_val], color="#333333", linewidth=1)
        ax.set_title("\u5c0f\u533a\u5e73\u5747\u6d41\u91cf\u9884\u6d4b\u6563\u70b9")
        ax.set_xlabel("\u771f\u5b9e\u503c\uff08MB\uff09")
        ax.set_ylabel("\u9884\u6d4b\u503c\uff08MB\uff09")
        add_metric_box(
            ax,
            f"MAE={pred_mae:.2f}\nRMSE={pred_rmse:.2f}\nMAPE={pred_mape*100:.1f}%\nR2={pred_r2:.3f}",
        )
        save_fig("fig24_flow_mean_pred_scatter.png")

        feature_names = pred_pipe.named_steps["columntransformer"].get_feature_names_out()
        importances = pred_pipe.named_steps["gradientboostingregressor"].feature_importances_
        grouped = {}
        for name, value in zip(feature_names, importances):
            if name.startswith("num__"):
                base = name.replace("num__", "")
            else:
                base = name.replace("cat__", "").split("_")[0]
            grouped[base] = grouped.get(base, 0.0) + float(value)
        coef_series = pd.Series(grouped).sort_values(ascending=True)
        plt.figure(figsize=(6, 4))
        plt.barh(coef_series.index, coef_series.values, color="#72B7B2")
        plt.title("\u7279\u5f81\u91cd\u8981\u6027\uff08\u7ec4\u5408\u7edf\u8ba1\uff09")
        plt.xlabel("\u91cd\u8981\u6027")
        plt.ylabel("\u6307\u6807")
        save_fig("fig25_flow_mean_pred_features.png")

    plt.figure(figsize=(8, 4))
    sns.countplot(x=peak_hours, color="#72B7B2")
    plt.title("\u5c0f\u533a\u5cf0\u503c\u65f6\u6bb5\u5206\u5e03")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u5c0f\u533a\u6570\u91cf")
    save_fig("fig11_peak_hour_distribution.png")

    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(profile_df)
    plt.figure(figsize=(6, 5))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for color, label in zip(colors, unique_labels):
        mask = labels == label
        plt.scatter(
            pca_points[mask, 0],
            pca_points[mask, 1],
            s=8,
            alpha=0.6,
            color=color,
            label=f"Cluster {label}",
        )
    plt.legend(title="\u805a\u7c7b\u7c7b\u522b", loc="best")
    plt.title("\u805a\u7c7b\u964d\u7ef4\u53ef\u89c6\u5316\uff08PCA\uff09")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    save_fig("fig12_cluster_pca.png")

    target_df = pd.concat(target_records, ignore_index=True)
    target_df["DATETIME_KEY"] = pd.to_datetime(target_df["DATETIME_KEY"], errors="coerce")
    target_df = target_df.dropna(subset=["DATETIME_KEY"]).set_index("DATETIME_KEY")
    hourly = target_df.resample("h")[["FLOW_SUM", "USER_COUNT"]].sum()
    hourly = hourly.asfreq("h", fill_value=0)

    test_hours = TEST_DAYS * 24
    if len(hourly) <= test_hours:
        raise ValueError("\u65f6\u95f4\u5e8f\u5217\u8fc7\u77ed\uff0c\u65e0\u6cd5\u62c6\u5206\u6d4b\u8bd5\u96c6")

    train = hourly.iloc[:-test_hours]
    test = hourly.iloc[-test_hours:]

    train_idx = train.index
    train_key = train_idx.weekday * 24 + train_idx.hour
    train_group = train.groupby(train_key).mean()
    global_flow = train["FLOW_SUM"].mean()
    global_user = train["USER_COUNT"].mean()

    test_idx = test.index
    test_key = test_idx.weekday * 24 + test_idx.hour
    pred_flow = train_group["FLOW_SUM"].reindex(test_key).fillna(global_flow).values
    pred_user = train_group["USER_COUNT"].reindex(test_key).fillna(global_user).values

    actual_flow = test["FLOW_SUM"].values
    actual_user = test["USER_COUNT"].values
    flow_mae = np.mean(np.abs(actual_flow - pred_flow))
    user_mae = np.mean(np.abs(actual_user - pred_user))
    flow_mape = np.mean(np.abs(actual_flow - pred_flow) / np.maximum(actual_flow, 1e-6))
    user_mape = np.mean(np.abs(actual_user - pred_user) / np.maximum(actual_user, 1e-6))
    flow_rmse = np.sqrt(np.mean((actual_flow - pred_flow) ** 2))
    user_rmse = np.sqrt(np.mean((actual_user - pred_user) ** 2))
    flow_r2 = safe_r2(actual_flow, pred_flow)
    user_r2 = safe_r2(actual_user, pred_user)
    flow_mape_pct = flow_mape * 100
    user_mape_pct = user_mape * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test.index, actual_flow, label="\u771f\u5b9e")
    ax.plot(test.index, pred_flow, label="\u9884\u6d4b")
    ax.set_title("\u6d41\u91cf\u9884\u6d4b\u5bf9\u6bd4\uff08\u7c7b\u4f3c\u5b63\u8282\u6027\u57fa\u7ebf\uff09")
    ax.set_xlabel("\u65f6\u95f4")
    ax.set_ylabel("\u6d41\u91cf\uff08MB\uff09")
    ax.legend()
    add_metric_box(
        ax,
        f"MAE={flow_mae:.2f}\nRMSE={flow_rmse:.2f}\nMAPE={flow_mape_pct:.1f}%\nR2={flow_r2:.3f}",
    )
    save_fig("fig13_actual_vs_pred_flow.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test.index, actual_user, label="\u771f\u5b9e")
    ax.plot(test.index, pred_user, label="\u9884\u6d4b")
    ax.set_title("\u7528\u6237\u6570\u9884\u6d4b\u5bf9\u6bd4\uff08\u7c7b\u4f3c\u5b63\u8282\u6027\u57fa\u7ebf\uff09")
    ax.set_xlabel("\u65f6\u95f4")
    ax.set_ylabel("\u7528\u6237\u6570")
    ax.legend()
    add_metric_box(
        ax,
        f"MAE={user_mae:.2f}\nRMSE={user_rmse:.2f}\nMAPE={user_mape_pct:.1f}%\nR2={user_r2:.3f}",
    )
    save_fig("fig14_actual_vs_pred_user.png")

    residual_flow = actual_flow - pred_flow
    plt.figure(figsize=(8, 4))
    sns.histplot(residual_flow, bins=60, color="#E45756")
    plt.title("\u6d41\u91cf\u6b8b\u5dee\u5206\u5e03")
    plt.xlabel("\u6b8b\u5dee")
    plt.ylabel("\u6837\u672c\u6570\u91cf")
    save_fig("fig15_residual_hist_flow.png")

    plt.figure(figsize=(10, 4))
    plt.plot(test.index, residual_flow)
    plt.axhline(0, color="#333333", linewidth=1)
    plt.title("\u6d41\u91cf\u6b8b\u5dee\u8d70\u52bf")
    plt.xlabel("\u65f6\u95f4")
    plt.ylabel("\u6b8b\u5dee")
    save_fig("fig16_residual_ts_flow.png")

    plt.figure(figsize=(6, 5))
    plt.scatter(actual_flow, pred_flow, s=10, alpha=0.5)
    plt.plot([actual_flow.min(), actual_flow.max()], [actual_flow.min(), actual_flow.max()], color="#333333")
    plt.title("\u6d41\u91cf\u771f\u5b9e\u4e0e\u9884\u6d4b\u6563\u70b9")
    plt.xlabel("\u771f\u5b9e\u503c")
    plt.ylabel("\u9884\u6d4b\u503c")
    save_fig("fig17_actual_vs_pred_scatter_flow.png")

    residual_user = actual_user - pred_user
    plt.figure(figsize=(8, 4))
    sns.histplot(residual_user, bins=60, color="#4C78A8")
    plt.title("\u7528\u6237\u6570\u6b8b\u5dee\u5206\u5e03")
    plt.xlabel("\u6b8b\u5dee")
    plt.ylabel("\u6837\u672c\u6570\u91cf")
    save_fig("fig18_residual_hist_user.png")

    plt.figure(figsize=(6, 5))
    plt.scatter(actual_user, pred_user, s=10, alpha=0.5)
    plt.plot([actual_user.min(), actual_user.max()], [actual_user.min(), actual_user.max()], color="#333333")
    plt.title("\u7528\u6237\u6570\u771f\u5b9e\u4e0e\u9884\u6d4b\u6563\u70b9")
    plt.xlabel("\u771f\u5b9e\u503c")
    plt.ylabel("\u9884\u6d4b\u503c")
    save_fig("fig19_actual_vs_pred_scatter_user.png")

    high_load_threshold = cell_agg["flow_sum"].quantile(0.99)
    silent_threshold = 0.5
    high_load_cells = cell_agg[cell_agg["flow_sum"] >= high_load_threshold]
    silent_cells = cell_agg[cell_agg["silent_ratio"] >= silent_threshold]

    def plot_scene_type(df: pd.DataFrame, label: str, file_prefix: str) -> None:
        scene_counts = df["SCENE"].value_counts().head(10)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=scene_counts.index.astype(int).astype(str), y=scene_counts.values, color="#4C78A8")
        plt.title(f"{label}\u573a\u666f\u5206\u5e03")
        plt.xlabel("\u573a\u666f\uff08SCENE\uff09")
        plt.ylabel("\u5c0f\u533a\u6570\u91cf")
        save_fig(f"fig20_{file_prefix}_scene_bar.png")

        type_counts = df["TYPE"].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=type_counts.index.astype(int).astype(str), y=type_counts.values, color="#F58518")
        plt.title(f"{label}TYPE \u5206\u5e03")
        plt.xlabel("TYPE")
        plt.ylabel("\u5c0f\u533a\u6570\u91cf")
        save_fig(f"fig21_{file_prefix}_type_bar.png")

    if not high_load_cells.empty:
        plot_scene_type(high_load_cells, "高负荷", "highload")
        top_highload = high_load_cells.sort_values("flow_sum", ascending=False).head(20)
        plt.figure(figsize=(10, 4))
        labels = top_highload["CELL_ID"].astype(str)
        sns.barplot(x=labels, y=top_highload["flow_sum"], color="#E45756")
        plt.title("TOP20 高负荷小区总流量")
        plt.xlabel("小区ID")
        plt.ylabel("总流量（MB）")
        plt.xticks(rotation=30, ha="right")
        save_fig("fig22_highload_top20.png")
        top_highload.to_csv(OUT_DIR / "highload_top20.csv", index=False)

    if not silent_cells.empty:
        plot_scene_type(silent_cells, "静默", "silent")
        top_silent = silent_cells.sort_values("silent_ratio", ascending=False).head(20)
        plt.figure(figsize=(10, 4))
        labels = top_silent["CELL_ID"].astype(str)
        sns.barplot(x=labels, y=top_silent["silent_ratio"], color="#72B7B2")
        plt.title("TOP20 静默小区比例")
        plt.xlabel("小区ID")
        plt.ylabel("静默比例")
        plt.xticks(rotation=30, ha="right")
        save_fig("fig23_silent_top20.png")
        top_silent.to_csv(OUT_DIR / "silent_top20.csv", index=False)

    stats_out = {
        "clustering": {
            "k": int(best_k),
            "top_cells_used": int(len(valid_cells)),
            "cluster_sizes": cluster_counts.to_dict(),
            "cluster_peak_hour": {
                int(cid): int(cluster_profiles.loc[cid].values.argmax())
                for cid in cluster_profiles.index
            },
            "ch_scores": dict(zip(ks, ch_scores)),
            "db_scores": dict(zip(ks, db_scores)),
            "scene_chi2": chi2_scene,
            "scene_cramers_v": cramers_scene,
            "type_chi2": chi2_type,
            "type_cramers_v": cramers_type,
        },
        "forecast": {
            "target_cell": int(target_cell),
            "test_days": int(TEST_DAYS),
            "flow_mae": float(flow_mae),
            "flow_mape": float(flow_mape),
            "flow_mape_pct": float(flow_mape_pct),
            "flow_rmse": float(flow_rmse),
            "flow_r2": float(flow_r2),
            "user_mae": float(user_mae),
            "user_mape": float(user_mape),
            "user_mape_pct": float(user_mape_pct),
            "user_rmse": float(user_rmse),
            "user_r2": float(user_r2),
        },
        "highload": {
            "threshold": float(high_load_threshold),
            "count": int(high_load_cells.shape[0]),
        },
        "silent": {
            "threshold": float(silent_threshold),
            "count": int(silent_cells.shape[0]),
        },
    }
    if pred_metrics is not None:
        stats_out["feature_prediction"] = pred_metrics

    with open(OUT_DIR / "section5_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
