# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("all_final_data_with_attributes.csv")
CELL_AGG_PATH = Path("report_assets/section3/section3_cell_agg.csv")
OUT_DIR = Path("report_assets/exception")

CHUNK_SIZE = 1_000_000

HOLIDAYS_2021 = {
    date(2021, 2, 11),
    date(2021, 2, 12),
    date(2021, 2, 13),
    date(2021, 2, 14),
    date(2021, 2, 15),
    date(2021, 2, 16),
    date(2021, 2, 17),
    date(2021, 4, 3),
    date(2021, 4, 4),
    date(2021, 4, 5),
    date(2021, 5, 1),
    date(2021, 5, 2),
    date(2021, 5, 3),
    date(2021, 5, 4),
    date(2021, 5, 5),
    date(2021, 6, 12),
    date(2021, 6, 13),
    date(2021, 6, 14),
    date(2021, 9, 19),
    date(2021, 9, 20),
    date(2021, 9, 21),
    date(2021, 10, 1),
    date(2021, 10, 2),
    date(2021, 10, 3),
    date(2021, 10, 4),
    date(2021, 10, 5),
    date(2021, 10, 6),
    date(2021, 10, 7),
}


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


def compute_spike_and_daily() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    last_flow: dict[int, float] = {}
    max_diff: dict[int, float] = {}
    max_time: dict[int, str] = {}
    max_user: dict[int, float] = {}
    max_flow: dict[int, float] = {}

    daily_flow = defaultdict(float)
    daily_user = defaultdict(float)
    flow_wh_sum = np.zeros((7, 24), dtype=np.float64)
    flow_wh_cnt = np.zeros((7, 24), dtype=np.int64)
    user_wh_sum = np.zeros((7, 24), dtype=np.float64)
    user_wh_cnt = np.zeros((7, 24), dtype=np.int64)

    usecols = ["CELL_ID", "DATETIME_KEY", "FLOW_SUM", "USER_COUNT"]
    dtypes = {
        "CELL_ID": "int32",
        "DATETIME_KEY": "string",
        "FLOW_SUM": "float32",
        "USER_COUNT": "float32",
    }

    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=usecols,
        dtype=dtypes,
        chunksize=CHUNK_SIZE,
        na_values=["", "NA", "NaN"],
    ):
        chunk.loc[chunk["FLOW_SUM"] < 0, "FLOW_SUM"] = np.nan
        chunk.loc[chunk["USER_COUNT"] < 0, "USER_COUNT"] = np.nan
        chunk = chunk.dropna(subset=["FLOW_SUM", "USER_COUNT"])
        if chunk.empty:
            continue

        dt_values = pd.to_datetime(chunk["DATETIME_KEY"], errors="coerce")
        date_str = chunk["DATETIME_KEY"].str.slice(0, 10)
        daily_group = chunk.groupby(date_str)[["FLOW_SUM", "USER_COUNT"]].sum()
        for dt_key, row in daily_group.iterrows():
            daily_flow[dt_key] += float(row["FLOW_SUM"])
            daily_user[dt_key] += float(row["USER_COUNT"])

        valid_time = dt_values.notna()
        if valid_time.any():
            sub = chunk.loc[valid_time, ["FLOW_SUM", "USER_COUNT"]].copy()
            sub["weekday"] = dt_values.loc[valid_time].dt.weekday.to_numpy()
            sub["hour"] = dt_values.loc[valid_time].dt.hour.to_numpy()
            wh_group = sub.groupby(["weekday", "hour"]).agg(
                flow_sum=("FLOW_SUM", "sum"),
                flow_count=("FLOW_SUM", "count"),
                user_sum=("USER_COUNT", "sum"),
                user_count=("USER_COUNT", "count"),
            )
            for (weekday, hour), row in wh_group.iterrows():
                flow_wh_sum[int(weekday), int(hour)] += float(row["flow_sum"])
                flow_wh_cnt[int(weekday), int(hour)] += int(row["flow_count"])
                user_wh_sum[int(weekday), int(hour)] += float(row["user_sum"])
                user_wh_cnt[int(weekday), int(hour)] += int(row["user_count"])

        diff = chunk.groupby("CELL_ID")["FLOW_SUM"].diff().astype("float64")
        first_rows = chunk.groupby("CELL_ID", sort=False).head(1)
        prev = first_rows["CELL_ID"].map(last_flow)
        diff.loc[first_rows.index] = first_rows["FLOW_SUM"] - prev.to_numpy()

        pos_mask = diff > 0
        if pos_mask.any():
            sub = chunk.loc[
                pos_mask, ["CELL_ID", "DATETIME_KEY", "FLOW_SUM", "USER_COUNT"]
            ].copy()
            sub["diff"] = diff[pos_mask].to_numpy()
            idx = sub.groupby("CELL_ID")["diff"].idxmax()
            best = sub.loc[idx]
            for row in best.itertuples(index=False):
                cell_id = int(row.CELL_ID)
                diff_val = float(row.diff)
                prev_best = max_diff.get(cell_id, float("-inf"))
                if diff_val > prev_best:
                    max_diff[cell_id] = diff_val
                    max_time[cell_id] = str(row.DATETIME_KEY)
                    max_user[cell_id] = float(row.USER_COUNT)
                    max_flow[cell_id] = float(row.FLOW_SUM)

        last_rows = chunk.groupby("CELL_ID", sort=False).tail(1)
        for cell_id, flow_val in zip(last_rows["CELL_ID"], last_rows["FLOW_SUM"]):
            last_flow[int(cell_id)] = float(flow_val)

    spike_df = pd.DataFrame(
        {
            "CELL_ID": list(max_diff.keys()),
            "max_diff": list(max_diff.values()),
            "max_time": [max_time[c] for c in max_diff.keys()],
            "max_flow": [max_flow[c] for c in max_diff.keys()],
            "max_user": [max_user[c] for c in max_diff.keys()],
        }
    )
    daily_df = pd.DataFrame(
        {
            "date": list(daily_flow.keys()),
            "flow_sum": list(daily_flow.values()),
            "user_sum": list(daily_user.values()),
        }
    )
    flow_wh_mean = np.divide(
        flow_wh_sum, np.where(flow_wh_cnt == 0, np.nan, flow_wh_cnt)
    )
    user_wh_mean = np.divide(
        user_wh_sum, np.where(user_wh_cnt == 0, np.nan, user_wh_cnt)
    )
    return spike_df, daily_df, flow_wh_mean, user_wh_mean


def plot_charts(
    cell_agg: pd.DataFrame,
    spike_df: pd.DataFrame,
    spike_cells: pd.DataFrame,
    spike_top: pd.DataFrame,
    spike_threshold: float,
    long_cells: pd.DataFrame,
    long_thr: float,
    long_stable: pd.DataFrame,
    high_user_low_fpu: pd.DataFrame,
    high_flow_low_user: pd.DataFrame,
    high_user_high_flow: pd.DataFrame,
    daily_df: pd.DataFrame,
    flow_wh_mean: np.ndarray,
    user_wh_mean: np.ndarray,
    user_p99: float,
    flow_p99: float,
    user_p10: float,
    flow_per_user_p10: float,
    std_thr: float,
    p95: float,
    p99: float,
    iso_scores: np.ndarray,
    dist: np.ndarray,
    dist_thr: float,
) -> None:
    weekday_names = weekday_labels(list(range(7)))

    if not spike_df.empty:
        plt.figure(figsize=(7, 4))
        sns.histplot(np.log1p(spike_df["max_diff"]), bins=40, color="#4C78A8")
        plt.axvline(np.log1p(spike_threshold), color="#E45756", linestyle="--")
        plt.title("\u6d41\u91cf\u7a81\u589e\u5dee\u503c\u5206\u5e03\uff08log1p\uff09")
        plt.xlabel("log(1 + max_diff)")
        plt.ylabel("\u9891\u6570")
        save_fig("fig01_spike_diff_hist.png")

    if not spike_cells.empty:
        weekday_counts = spike_cells["weekday"].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=weekday_names, y=weekday_counts.values, color="#72B7B2")
        plt.title("\u7a81\u53d1\u9ad8\u8d1f\u8377\u5468\u5185\u5206\u5e03")
        plt.xlabel("\u661f\u671f")
        plt.ylabel("\u5c0f\u533a\u6570\u91cf")
        save_fig("fig02_spike_weekday_bar.png")

        hour_counts = spike_cells["hour"].value_counts().sort_index()
        plt.figure(figsize=(7, 4))
        sns.barplot(x=hour_counts.index, y=hour_counts.values, color="#54A24B")
        plt.title("\u7a81\u53d1\u9ad8\u8d1f\u8377\u5c0f\u65f6\u5206\u5e03")
        plt.xlabel("\u5c0f\u65f6")
        plt.ylabel("\u5c0f\u533a\u6570\u91cf")
        save_fig("fig03_spike_hour_bar.png")

        scene_counts = spike_cells["SCENE"].value_counts().head(10)
        plt.figure(figsize=(7, 4))
        sns.barplot(x=scene_counts.index.astype(str), y=scene_counts.values, color="#F58518")
        plt.title("\u7a81\u53d1\u9ad8\u8d1f\u8377\u573a\u666f Top10")
        plt.xlabel("SCENE")
        plt.ylabel("\u5c0f\u533a\u6570\u91cf")
        save_fig("fig04_spike_scene_bar.png")

        type_counts = spike_cells["TYPE"].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=type_counts.index.astype(str), y=type_counts.values, color="#B279A2")
        plt.title("\u7a81\u53d1\u9ad8\u8d1f\u8377\u7c7b\u578b\u5206\u5e03")
        plt.xlabel("TYPE")
        plt.ylabel("\u5c0f\u533a\u6570\u91cf")
        save_fig("fig05_spike_type_bar.png")

        plt.figure(figsize=(6, 4))
        plt.scatter(
            spike_cells["max_user"],
            spike_cells["max_diff"],
            s=10,
            alpha=0.5,
            color="#4C78A8",
        )
        plt.title("\u7a81\u53d1\u6d41\u91cf\u589e\u5e45 vs \u7528\u6237\u6570")
        plt.xlabel("\u7528\u6237\u6570")
        plt.ylabel("\u6d41\u91cf\u589e\u5e45")
        save_fig("fig06_spike_diff_user_scatter.png")

        if not spike_top.empty:
            plt.figure(figsize=(8, 4))
            top_sorted = spike_top.sort_values("max_diff", ascending=True)
            plt.barh(top_sorted["CELL_ID"].astype(str), top_sorted["max_diff"], color="#E45756")
            plt.title("\u7a81\u53d1\u6d41\u91cf\u589e\u5e45 Top20")
            plt.xlabel("max_diff")
            plt.ylabel("CELL_ID")
            save_fig("fig07_spike_top20_bar.png")

    plt.figure(figsize=(7, 4))
    sns.histplot(cell_agg["flow_mean"], bins=50, color="#4C78A8")
    plt.axvline(long_thr, color="#E45756", linestyle="--")
    plt.title("\u5c0f\u533a\u5e73\u5747\u6d41\u91cf\u5206\u5e03")
    plt.xlabel("flow_mean")
    plt.ylabel("\u9891\u6570")
    save_fig("fig08_flow_mean_hist.png")

    if not long_cells.empty:
        scene_counts = long_cells["SCENE"].value_counts().head(10)
        plt.figure(figsize=(7, 4))
        sns.barplot(x=scene_counts.index.astype(str), y=scene_counts.values, color="#72B7B2")
        plt.title("\u957f\u671f\u9ad8\u8d1f\u8377\u573a\u666f Top10")
        plt.xlabel("SCENE")
        plt.ylabel("\u5c0f\u533a\u6570\u91cf")
        save_fig("fig09_highload_scene_bar.png")

        type_counts = long_cells["TYPE"].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=type_counts.index.astype(str), y=type_counts.values, color="#54A24B")
        plt.title("\u957f\u671f\u9ad8\u8d1f\u8377\u7c7b\u578b\u5206\u5e03")
        plt.xlabel("TYPE")
        plt.ylabel("\u5c0f\u533a\u6570\u91cf")
        save_fig("fig10_highload_type_bar.png")

        plt.figure(figsize=(6, 4))
        plt.scatter(
            cell_agg["user_mean"],
            cell_agg["flow_mean"],
            s=6,
            alpha=0.2,
            color="#999999",
            label="\u5176\u4ed6\u5c0f\u533a",
        )
        plt.scatter(
            long_cells["user_mean"],
            long_cells["flow_mean"],
            s=10,
            alpha=0.5,
            color="#E45756",
            label="\u9ad8\u8d1f\u8377",
        )
        plt.title("\u7528\u6237\u4e0e\u6d41\u91cf\uff08\u9ad8\u8d1f\u8377\u9ad8\u4eae\uff09")
        plt.xlabel("user_mean")
        plt.ylabel("flow_mean")
        plt.legend(loc="best")
        save_fig("fig11_flow_user_scatter_highload.png")

        plt.figure(figsize=(6, 4))
        data = pd.DataFrame(
            {
                "flow_cv": pd.concat([long_cells["flow_cv"], cell_agg["flow_cv"]]),
                "group": ["\u9ad8\u8d1f\u8377"] * len(long_cells)
                + ["\u5176\u4ed6"] * len(cell_agg),
            }
        )
        sns.boxplot(data=data, x="group", y="flow_cv")
        plt.title("\u6d41\u91cf\u6ce2\u52a8\u7cfb\u6570\u5bf9\u6bd4")
        plt.xlabel("")
        plt.ylabel("flow_cv")
        save_fig("fig12_highload_flow_cv_box.png")

        plt.figure(figsize=(6, 4))
        sns.histplot(long_cells["flow_cv"], bins=40, color="#E45756")
        plt.title("\u9ad8\u8d1f\u8377\u6d41\u91cf\u6ce2\u52a8\u5206\u5e03")
        plt.xlabel("flow_cv")
        plt.ylabel("\u9891\u6570")
        save_fig("fig13_highload_flow_cv_hist.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(
        cell_agg["user_mean"],
        cell_agg["flow_mean"],
        s=6,
        alpha=0.2,
        color="#999999",
        label="\u5176\u4ed6\u5c0f\u533a",
    )
    if not high_user_high_flow.empty:
        plt.scatter(
            high_user_high_flow["user_mean"],
            high_user_high_flow["flow_mean"],
            s=12,
            alpha=0.6,
            color="#E45756",
            label="\u9ad8\u7528\u6237\u9ad8\u6d41\u91cf",
        )
    plt.axvline(user_p99, color="#4C78A8", linestyle="--")
    plt.axhline(flow_p99, color="#4C78A8", linestyle="--")
    plt.title("\u7528\u6237\u4e0e\u6d41\u91cf\u5f02\u5e38\u533a\u57df")
    plt.xlabel("user_mean")
    plt.ylabel("flow_mean")
    plt.legend(loc="best")
    save_fig("fig14_user_flow_scatter_p99.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(
        cell_agg["user_mean"],
        cell_agg["flow_per_user"],
        s=6,
        alpha=0.2,
        color="#999999",
        label="\u5176\u4ed6\u5c0f\u533a",
    )
    if not high_user_low_fpu.empty:
        plt.scatter(
            high_user_low_fpu["user_mean"],
            high_user_low_fpu["flow_per_user"],
            s=12,
            alpha=0.6,
            color="#54A24B",
            label="\u9ad8\u7528\u6237\u4f4e\u4eba\u5747\u6d41\u91cf",
        )
    plt.axvline(user_p99, color="#4C78A8", linestyle="--")
    plt.axhline(flow_per_user_p10, color="#4C78A8", linestyle="--")
    plt.title("\u9ad8\u7528\u6237\u4f4e\u4eba\u5747\u6d41\u91cf\u533a\u57df")
    plt.xlabel("user_mean")
    plt.ylabel("flow_per_user")
    plt.legend(loc="best")
    save_fig("fig15_user_fpu_scatter.png")

    if not high_user_low_fpu.empty:
        top_low_fpu = high_user_low_fpu.sort_values("flow_per_user").head(20)
        plt.figure(figsize=(8, 4))
        plt.barh(top_low_fpu["CELL_ID"].astype(str), top_low_fpu["flow_per_user"], color="#54A24B")
        plt.title("\u9ad8\u7528\u6237\u4f4e\u4eba\u5747\u6d41\u91cf Top20")
        plt.xlabel("flow_per_user")
        plt.ylabel("CELL_ID")
        save_fig("fig16_high_user_low_fpu_top20_bar.png")

        scene_counts = high_user_low_fpu["SCENE"].value_counts().head(10)
        plt.figure(figsize=(7, 4))
        sns.barplot(x=scene_counts.index.astype(str), y=scene_counts.values, color="#F58518")
        plt.title("\u9ad8\u7528\u6237\u4f4e\u4eba\u5747\u6d41\u91cf\u573a\u666f Top10")
        plt.xlabel("SCENE")
        plt.ylabel("\u5c0f\u533a\u6570\u91cf")
        save_fig("fig18_high_user_low_fpu_scene_bar.png")

        type_counts = high_user_low_fpu["TYPE"].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=type_counts.index.astype(str), y=type_counts.values, color="#B279A2")
        plt.title("\u9ad8\u7528\u6237\u4f4e\u4eba\u5747\u6d41\u91cf\u7c7b\u578b\u5206\u5e03")
        plt.xlabel("TYPE")
        plt.ylabel("\u5c0f\u533a\u6570\u91cf")
        save_fig("fig19_high_user_low_fpu_type_bar.png")

    if not high_user_high_flow.empty:
        top_high = high_user_high_flow.sort_values("flow_mean", ascending=False).head(20)
        plt.figure(figsize=(8, 4))
        plt.barh(top_high["CELL_ID"].astype(str), top_high["flow_mean"], color="#E45756")
        plt.title("\u9ad8\u7528\u6237\u9ad8\u6d41\u91cf Top20")
        plt.xlabel("flow_mean")
        plt.ylabel("CELL_ID")
        save_fig("fig17_high_user_high_flow_top20_bar.png")

    daily_df = daily_df.sort_values("date")
    plt.figure(figsize=(9, 4))
    plt.plot(daily_df["date"], daily_df["flow_sum"], color="#4C78A8")
    plt.title("\u65e5\u603b\u6d41\u91cf\u8d8b\u52bf")
    plt.xlabel("\u65e5\u671f")
    plt.ylabel("FLOW_SUM")
    save_fig("fig20_daily_flow_line.png")

    plt.figure(figsize=(9, 4))
    plt.plot(daily_df["date"], daily_df["user_sum"], color="#F58518")
    plt.title("\u65e5\u603b\u7528\u6237\u8d8b\u52bf")
    plt.xlabel("\u65e5\u671f")
    plt.ylabel("USER_COUNT")
    save_fig("fig21_daily_user_line.png")

    weekday_flow = daily_df.groupby(daily_df["date"].dt.weekday)["flow_sum"].mean()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=weekday_labels(list(weekday_flow.index)), y=weekday_flow.values, color="#72B7B2")
    plt.title("\u661f\u671f\u7ef4\u5ea6\u65e5\u6d41\u91cf\u5e73\u5747")
    plt.xlabel("\u661f\u671f")
    plt.ylabel("FLOW_SUM")
    save_fig("fig22_weekday_flow_bar.png")

    holiday_df = daily_df.copy()
    holiday_df["is_holiday"] = holiday_df["date"].dt.date.isin(HOLIDAYS_2021)
    group = holiday_df.groupby("is_holiday")[["flow_sum", "user_sum"]].mean()
    plt.figure(figsize=(6, 4))
    plt.bar(["\u975e\u8282\u5047\u65e5", "\u8282\u5047\u65e5"], group["flow_sum"], color="#4C78A8")
    plt.title("\u8282\u5047\u65e5\u4e0e\u975e\u8282\u5047\u65e5\u6d41\u91cf\u5bf9\u6bd4")
    plt.xlabel("\u7c7b\u578b")
    plt.ylabel("FLOW_SUM")
    save_fig("fig23_holiday_flow_bar.png")

    flow_wh_df = pd.DataFrame(flow_wh_mean, index=weekday_names, columns=list(range(24)))
    plt.figure(figsize=(10, 4))
    sns.heatmap(flow_wh_df, cmap="YlOrRd")
    plt.title("\u661f\u671f\u00d7\u5c0f\u65f6\u6d41\u91cf\u70ed\u529b\u56fe")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u661f\u671f")
    save_fig("fig24_weekday_hour_heatmap_flow.png")

    user_wh_df = pd.DataFrame(user_wh_mean, index=weekday_names, columns=list(range(24)))
    plt.figure(figsize=(10, 4))
    sns.heatmap(user_wh_df, cmap="YlGnBu")
    plt.title("\u661f\u671f\u00d7\u5c0f\u65f6\u7528\u6237\u70ed\u529b\u56fe")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u661f\u671f")
    save_fig("fig25_weekday_hour_heatmap_user.png")

    plt.figure(figsize=(7, 4))
    sns.histplot(cell_agg["flow_mean"], bins=50, color="#4C78A8")
    plt.axvline(p95, color="#F58518", linestyle="--", label="P95")
    plt.axvline(p99, color="#E45756", linestyle="--", label="P99")
    plt.title("\u6d41\u91cf\u5747\u503c\u5206\u4f4d\u6570\u9608\u503c")
    plt.xlabel("flow_mean")
    plt.ylabel("\u9891\u6570")
    plt.legend(loc="best")
    save_fig("fig26_flow_mean_hist_thresholds.png")

    plt.figure(figsize=(7, 4))
    sns.histplot(iso_scores, bins=40, color="#72B7B2")
    plt.title("IsolationForest \u5206\u6570\u5206\u5e03")
    plt.xlabel("\u5206\u6570")
    plt.ylabel("\u9891\u6570")
    save_fig("fig27_isoforest_score_hist.png")

    plt.figure(figsize=(7, 4))
    sns.histplot(dist, bins=40, color="#B279A2")
    plt.axvline(dist_thr, color="#E45756", linestyle="--")
    plt.title("KMeans \u8ddd\u79bb\u5206\u5e03")
    plt.xlabel("\u8ddd\u79bb")
    plt.ylabel("\u9891\u6570")
    save_fig("fig28_kmeans_dist_hist.png")

    plt.figure(figsize=(6, 4))
    counts = [
        int((cell_agg["flow_mean"] > std_thr).sum()),
        int((cell_agg["flow_mean"] > p95).sum()),
        int((cell_agg["flow_mean"] > p99).sum()),
        int((iso_scores < np.quantile(iso_scores, 0.01)).sum()),
        int((dist >= dist_thr).sum()),
    ]
    labels = ["\u6807\u51c6\u5dee", "P95", "P99", "IForest", "KMeans"]
    sns.barplot(x=labels, y=counts, color="#4C78A8")
    plt.title("\u4e0d\u540c\u65b9\u6cd5\u7684\u5f02\u5e38\u6570\u91cf")
    plt.xlabel("\u65b9\u6cd5")
    plt.ylabel("\u5c0f\u533a\u6570\u91cf")
    save_fig("fig29_anomaly_counts_bar.png")


def build_stats() -> dict:
    ensure_out_dir()
    set_cn_style()
    cell_agg = pd.read_csv(CELL_AGG_PATH)

    spike_df, daily_df, flow_wh_mean, user_wh_mean = compute_spike_and_daily()
    spike_df = spike_df.dropna(subset=["max_diff"])
    spike_threshold = float(spike_df["max_diff"].mean() + 2 * spike_df["max_diff"].std())
    spike_df["is_spike"] = spike_df["max_diff"] > spike_threshold
    spike_cells = spike_df[spike_df["is_spike"]].copy()

    cell_meta = cell_agg[["CELL_ID", "SCENE", "TYPE"]].copy()
    spike_cells = spike_cells.merge(cell_meta, on="CELL_ID", how="left")
    spike_cells["max_time"] = pd.to_datetime(spike_cells["max_time"], errors="coerce")
    spike_cells["weekday"] = spike_cells["max_time"].dt.weekday
    spike_cells["hour"] = spike_cells["max_time"].dt.hour

    spike_top = spike_cells.sort_values("max_diff", ascending=False).head(20)
    spike_top.to_csv(OUT_DIR / "sudden_spike_top20.csv", index=False)

    long_thr = float(cell_agg["flow_mean"].quantile(0.9))
    long_cells = cell_agg[cell_agg["flow_mean"] >= long_thr].copy()
    long_cells = long_cells.sort_values("flow_mean", ascending=False)
    long_cells.head(20).to_csv(OUT_DIR / "long_highload_top20.csv", index=False)

    flow_cv_med = float(cell_agg["flow_cv"].median())
    long_stable = long_cells[long_cells["flow_cv"] <= flow_cv_med]

    user_p99 = float(cell_agg["user_mean"].quantile(0.99))
    flow_p99 = float(cell_agg["flow_mean"].quantile(0.99))
    user_p10 = float(cell_agg["user_mean"].quantile(0.1))
    flow_per_user_p10 = float(cell_agg["flow_per_user"].quantile(0.1))

    high_user_low_fpu = cell_agg[
        (cell_agg["user_mean"] >= user_p99)
        & (cell_agg["flow_per_user"] <= flow_per_user_p10)
    ].copy()
    high_user_low_fpu = high_user_low_fpu.sort_values("user_mean", ascending=False)
    high_user_low_fpu.head(20).to_csv(
        OUT_DIR / "abnormal_high_user_low_fpu_top20.csv", index=False
    )

    high_flow_low_user = cell_agg[
        (cell_agg["flow_mean"] >= flow_p99) & (cell_agg["user_mean"] <= user_p10)
    ].copy()
    high_flow_low_user = high_flow_low_user.sort_values("flow_mean", ascending=False)
    high_flow_low_user.head(20).to_csv(
        OUT_DIR / "abnormal_high_flow_low_user_top20.csv", index=False
    )

    high_user_high_flow = cell_agg[
        (cell_agg["user_mean"] >= user_p99) & (cell_agg["flow_mean"] >= flow_p99)
    ].copy()
    high_user_high_flow = high_user_high_flow.sort_values("flow_mean", ascending=False)
    high_user_high_flow.head(20).to_csv(
        OUT_DIR / "abnormal_high_user_high_flow_top20.csv", index=False
    )

    daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
    daily_df = daily_df.dropna(subset=["date"]).sort_values("date")
    daily_df["weekday"] = daily_df["date"].dt.weekday
    daily_df["is_weekend"] = daily_df["weekday"] >= 5
    daily_df["is_holiday"] = daily_df["date"].dt.date.isin(HOLIDAYS_2021)
    daily_df["flow_change"] = daily_df["flow_sum"].diff().abs()

    weekday_flow = daily_df.loc[~daily_df["is_weekend"], "flow_sum"].mean()
    weekend_flow = daily_df.loc[daily_df["is_weekend"], "flow_sum"].mean()
    weekday_user = daily_df.loc[~daily_df["is_weekend"], "user_sum"].mean()
    weekend_user = daily_df.loc[daily_df["is_weekend"], "user_sum"].mean()

    holiday_flow = daily_df.loc[daily_df["is_holiday"], "flow_sum"].mean()
    normal_flow = daily_df.loc[~daily_df["is_holiday"], "flow_sum"].mean()
    holiday_user = daily_df.loc[daily_df["is_holiday"], "user_sum"].mean()
    normal_user = daily_df.loc[~daily_df["is_holiday"], "user_sum"].mean()

    flow_corr = float(daily_df["flow_sum"].corr(daily_df["user_sum"]))
    change_p90 = float(daily_df["flow_change"].quantile(0.9))

    max_day = daily_df.loc[daily_df["flow_sum"].idxmax()]
    min_day = daily_df.loc[daily_df["flow_sum"].idxmin()]

    std_thr = float(cell_agg["flow_mean"].mean() + 2 * cell_agg["flow_mean"].std())
    std_anom = cell_agg[cell_agg["flow_mean"] > std_thr]
    p95 = float(cell_agg["flow_mean"].quantile(0.95))
    p99 = float(cell_agg["flow_mean"].quantile(0.99))

    feature_cols = [
        "flow_mean",
        "user_mean",
        "flow_cv",
        "peak_ratio",
        "silent_ratio",
        "flow_per_user",
    ]
    feature_df = cell_agg[["CELL_ID"] + feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    feature_df = feature_df.dropna().set_index("CELL_ID")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df)

    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    iso_labels = iso.fit_predict(features_scaled)
    iso_scores = iso.decision_function(features_scaled)
    iso_anom = feature_df.loc[iso_labels == -1].copy()
    iso_anom["score"] = iso_scores[iso_labels == -1]
    iso_top = iso_anom.sort_values("score").head(20).reset_index()
    iso_top.to_csv(OUT_DIR / "anomaly_isoforest_top20.csv", index=False)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(features_scaled)
    centers = kmeans.cluster_centers_
    dist = np.linalg.norm(features_scaled - centers[km_labels], axis=1)
    dist_thr = float(np.quantile(dist, 0.99))
    km_anom = feature_df.loc[dist >= dist_thr].copy()
    km_anom["dist"] = dist[dist >= dist_thr]
    km_top = km_anom.sort_values("dist", ascending=False).head(20).reset_index()
    km_top.to_csv(OUT_DIR / "anomaly_kmeans_top20.csv", index=False)

    def top_counts(series: pd.Series, top_n: int = 3) -> list[dict]:
        total = int(series.shape[0])
        counts = series.value_counts(dropna=False).head(top_n)
        return [
            {
                "value": int(idx) if pd.notna(idx) else None,
                "count": int(val),
                "share": float(val / total) if total else 0.0,
            }
            for idx, val in counts.items()
        ]

    stats = {
        "sudden_spike": {
            "threshold": spike_threshold,
            "total_cells": int(spike_df.shape[0]),
            "spike_cells": int(spike_cells.shape[0]),
            "weekday_mode": int(spike_cells["weekday"].mode().iloc[0])
            if not spike_cells.empty
            else None,
            "hour_mode": int(spike_cells["hour"].mode().iloc[0])
            if not spike_cells.empty
            else None,
            "scene_top": top_counts(spike_cells["SCENE"]) if not spike_cells.empty else [],
            "type_top": top_counts(spike_cells["TYPE"]) if not spike_cells.empty else [],
        },
        "long_highload": {
            "threshold_p90": long_thr,
            "high_cells": int(long_cells.shape[0]),
            "stable_high_cells": int(long_stable.shape[0]),
            "total_cells": int(cell_agg.shape[0]),
            "scene_top": top_counts(long_cells["SCENE"]) if not long_cells.empty else [],
            "type_top": top_counts(long_cells["TYPE"]) if not long_cells.empty else [],
        },
        "abnormal_user": {
            "user_p99": user_p99,
            "flow_p99": flow_p99,
            "user_p10": user_p10,
            "flow_per_user_p10": flow_per_user_p10,
            "high_user_low_fpu": int(high_user_low_fpu.shape[0]),
            "high_flow_low_user": int(high_flow_low_user.shape[0]),
            "high_user_high_flow": int(high_user_high_flow.shape[0]),
        },
        "time_series": {
            "weekday_flow": float(weekday_flow),
            "weekend_flow": float(weekend_flow),
            "weekday_user": float(weekday_user),
            "weekend_user": float(weekend_user),
            "holiday_flow": float(holiday_flow),
            "normal_flow": float(normal_flow),
            "holiday_user": float(holiday_user),
            "normal_user": float(normal_user),
            "flow_user_corr": flow_corr,
            "change_p90": change_p90,
            "max_day": {
                "date": max_day["date"].strftime("%Y-%m-%d"),
                "flow_sum": float(max_day["flow_sum"]),
            },
            "min_day": {
                "date": min_day["date"].strftime("%Y-%m-%d"),
                "flow_sum": float(min_day["flow_sum"]),
            },
        },
        "anomaly_methods": {
            "std_threshold": std_thr,
            "std_anom_cells": int(std_anom.shape[0]),
            "p95_threshold": p95,
            "p99_threshold": p99,
            "p95_anom_cells": int((cell_agg["flow_mean"] > p95).sum()),
            "p99_anom_cells": int((cell_agg["flow_mean"] > p99).sum()),
            "isoforest_anom_cells": int((iso_labels == -1).sum()),
            "kmeans_anom_cells": int((dist >= dist_thr).sum()),
            "kmeans_dist_threshold": dist_thr,
        },
    }

    plot_charts(
        cell_agg=cell_agg,
        spike_df=spike_df,
        spike_cells=spike_cells,
        spike_top=spike_top,
        spike_threshold=spike_threshold,
        long_cells=long_cells,
        long_thr=long_thr,
        long_stable=long_stable,
        high_user_low_fpu=high_user_low_fpu,
        high_flow_low_user=high_flow_low_user,
        high_user_high_flow=high_user_high_flow,
        daily_df=daily_df,
        flow_wh_mean=flow_wh_mean,
        user_wh_mean=user_wh_mean,
        user_p99=user_p99,
        flow_p99=flow_p99,
        user_p10=user_p10,
        flow_per_user_p10=flow_per_user_p10,
        std_thr=std_thr,
        p95=p95,
        p99=p99,
        iso_scores=iso_scores,
        dist=dist,
        dist_thr=dist_thr,
    )

    with open(OUT_DIR / "exception_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("all_final_data_with_attributes.csv not found")
    if not CELL_AGG_PATH.exists():
        raise FileNotFoundError("section3_cell_agg.csv not found")
    build_stats()


if __name__ == "__main__":
    main()
