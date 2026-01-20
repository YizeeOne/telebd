# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = Path("all_final_data_with_attributes.csv")
OUT_DIR = Path("report_assets/section3")
CHUNK_SIZE = 1_000_000
HIST_BINS = 60
SCATTER_BINS = 80
FLOW_LOG_MAX = 14.0
USER_LOG_MAX = 10.0
FPU_LOG_MAX = 9.0
PAR_LOG_MAX = 5.5
ACTIVITY_MAX = 0.6

COL_DATE = "\u65e5\u671f"
COL_HOUR = "\u5c0f\u65f6"
COL_WEEKDAY = "\u661f\u671f"
COL_HOLIDAY = "\u662f\u5426\u8282\u5047\u65e5"
HOLIDAYS_2021 = {
    date.fromisoformat(day)
    for day in [
        "2021-02-11",
        "2021-02-12",
        "2021-02-13",
        "2021-02-14",
        "2021-02-15",
        "2021-02-16",
        "2021-02-17",
        "2021-04-03",
        "2021-04-04",
        "2021-04-05",
        "2021-05-01",
        "2021-05-02",
        "2021-05-03",
        "2021-05-04",
        "2021-05-05",
        "2021-06-12",
        "2021-06-13",
        "2021-06-14",
        "2021-09-19",
        "2021-09-20",
        "2021-09-21",
        "2021-10-01",
        "2021-10-02",
        "2021-10-03",
        "2021-10-04",
        "2021-10-05",
        "2021-10-06",
        "2021-10-07",
    ]
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


def init_stats() -> dict:
    return {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": np.inf, "max": -np.inf}


def update_stats(series: pd.Series, stats: dict) -> None:
    values = series.dropna()
    if values.empty:
        return
    stats["count"] += int(values.size)
    stats["sum"] += float(values.sum())
    stats["sumsq"] += float((values ** 2).sum())
    stats["min"] = float(min(stats["min"], values.min()))
    stats["max"] = float(max(stats["max"], values.max()))


def summarize_global(stats: dict) -> dict:
    if stats["count"] == 0:
        return {"count": 0, "mean": np.nan, "var": np.nan, "std": np.nan}
    mean = stats["sum"] / stats["count"]
    var = max(stats["sumsq"] / stats["count"] - mean**2, 0.0)
    return {
        "count": stats["count"],
        "mean": mean,
        "var": var,
        "std": float(np.sqrt(var)),
        "min": stats["min"],
        "max": stats["max"],
    }


def combine_sum(base: pd.DataFrame | None, add: pd.DataFrame) -> pd.DataFrame:
    if base is None:
        return add
    base = base.reindex(base.index.union(add.index)).fillna(0)
    add = add.reindex(base.index).fillna(0)
    return base.add(add, fill_value=0)


def combine_cell_agg(
    base: pd.DataFrame | None,
    add: pd.DataFrame,
    sum_cols: list[str],
    max_cols: list[str],
) -> pd.DataFrame:
    if base is None:
        return add
    base = base.reindex(base.index.union(add.index))
    for col in sum_cols:
        base[col] = base[col].fillna(0) + add[col].reindex(base.index).fillna(0)
    for col in max_cols:
        base[col] = np.maximum(
            base[col].fillna(0), add[col].reindex(base.index).fillna(0)
        )
    return base


def hist_quantile(edges: np.ndarray, counts: np.ndarray, q: float) -> float:
    total = counts.sum()
    if total == 0:
        return np.nan
    target = q * total
    cumulative = np.cumsum(counts)
    idx = int(np.searchsorted(cumulative, target))
    if idx <= 0:
        return float(edges[0])
    if idx >= len(counts):
        return float(edges[-1])
    prev = cumulative[idx - 1]
    count = counts[idx]
    if count == 0:
        return float(edges[idx])
    frac = (target - prev) / count
    return float(edges[idx] + frac * (edges[idx + 1] - edges[idx]))


def smooth_counts(counts: np.ndarray, passes: int = 2) -> np.ndarray:
    kernel = np.array(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float64
    )
    kernel /= kernel.sum()
    smoothed = counts.astype(np.float64)
    for _ in range(passes):
        padded = np.pad(smoothed, 1, mode="edge")
        smoothed = (
            kernel[0, 0] * padded[:-2, :-2]
            + kernel[0, 1] * padded[:-2, 1:-1]
            + kernel[0, 2] * padded[:-2, 2:]
            + kernel[1, 0] * padded[1:-1, :-2]
            + kernel[1, 1] * padded[1:-1, 1:-1]
            + kernel[1, 2] * padded[1:-1, 2:]
            + kernel[2, 0] * padded[2:, :-2]
            + kernel[2, 1] * padded[2:, 1:-1]
            + kernel[2, 2] * padded[2:, 2:]
        )
    return smoothed


def grid_shape(count: int, max_cols: int = 5) -> tuple[int, int]:
    cols = max(1, min(max_cols, count))
    rows = int(np.ceil(count / cols))
    return rows, cols


def annotate_empty(ax: plt.Axes, text: str = "\u65e0\u6570\u636e") -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.set_axis_off()


def bar_subplot(
    ax: plt.Axes,
    labels: pd.Series,
    values: pd.Series,
    title: str,
    ylabel: str,
    color: str = "#4C78A8",
) -> None:
    ax.bar(labels, values, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)


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


def main() -> None:
    ensure_out_dir()
    set_cn_style()

    if not DATA_PATH.exists():
        raise FileNotFoundError("\u672a\u627e\u5230 all_final_data_with_attributes.csv")

    usecols = [
        "CELL_ID",
        "DATETIME_KEY",
        "FLOW_SUM",
        "USER_COUNT",
        "LATITUDE",
        "LONGITUDE",
        "TYPE",
        "SCENE",
        "flow_per_user",
        "PAR",
        "ActivityScore",
    ]
    dtype = {
        "CELL_ID": "int32",
        "DATETIME_KEY": "string",
        "FLOW_SUM": "float32",
        "USER_COUNT": "float32",
        "LATITUDE": "float32",
        "LONGITUDE": "float32",
        "TYPE": "float32",
        "SCENE": "float32",
        "flow_per_user": "float32",
        "PAR": "float32",
        "ActivityScore": "float32",
    }

    flow_stats = init_stats()
    user_stats = init_stats()
    fpu_stats = init_stats()
    par_stats = init_stats()
    activity_stats = init_stats()

    daily_agg: pd.DataFrame | None = None
    hourly_agg: pd.DataFrame | None = None
    weekday_agg: pd.DataFrame | None = None
    holiday_hour_agg: pd.DataFrame | None = None
    heatmap_agg: pd.DataFrame | None = None
    scene_agg: pd.DataFrame | None = None
    type_agg: pd.DataFrame | None = None
    cell_agg: pd.DataFrame | None = None
    cell_attrs: pd.DataFrame | None = None

    flow_bins = np.linspace(0, FLOW_LOG_MAX, HIST_BINS + 1)
    user_bins = np.linspace(0, USER_LOG_MAX, HIST_BINS + 1)
    fpu_bins = np.linspace(0, FPU_LOG_MAX, HIST_BINS + 1)
    par_bins = np.linspace(0, PAR_LOG_MAX, HIST_BINS + 1)
    activity_bins = np.linspace(0, ACTIVITY_MAX, HIST_BINS + 1)
    scatter_x_bins = np.linspace(0, USER_LOG_MAX, SCATTER_BINS + 1)
    scatter_y_bins = np.linspace(0, FLOW_LOG_MAX, SCATTER_BINS + 1)

    flow_hist = np.zeros(HIST_BINS, dtype=np.int64)
    user_hist = np.zeros(HIST_BINS, dtype=np.int64)
    fpu_hist = np.zeros(HIST_BINS, dtype=np.int64)
    par_hist = np.zeros(HIST_BINS, dtype=np.int64)
    activity_hist = np.zeros(HIST_BINS, dtype=np.int64)
    scatter_counts = np.zeros((SCATTER_BINS, SCATTER_BINS), dtype=np.int64)

    for idx, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            usecols=usecols,
            dtype=dtype,
            chunksize=CHUNK_SIZE,
            na_values=["", "NA", "NaN"],
        ),
        start=1,
    ):
        chunk.loc[chunk["FLOW_SUM"] < 0, "FLOW_SUM"] = np.nan
        chunk.loc[chunk["USER_COUNT"] < 0, "USER_COUNT"] = np.nan

        safe_user = chunk["USER_COUNT"].where(chunk["USER_COUNT"] > 0)
        chunk["flow_per_user"] = chunk["flow_per_user"].fillna(
            chunk["FLOW_SUM"] / safe_user
        )
        chunk.loc[chunk["USER_COUNT"] <= 0, "flow_per_user"] = np.nan

        dt_values = pd.to_datetime(chunk["DATETIME_KEY"], errors="coerce")
        chunk[COL_DATE] = dt_values.dt.date
        chunk[COL_HOUR] = dt_values.dt.hour
        chunk[COL_WEEKDAY] = dt_values.dt.weekday
        chunk[COL_HOLIDAY] = dt_values.dt.date.isin(HOLIDAYS_2021).astype(int)

        update_stats(chunk["FLOW_SUM"], flow_stats)
        update_stats(chunk["USER_COUNT"], user_stats)
        update_stats(chunk["flow_per_user"], fpu_stats)
        update_stats(chunk["PAR"], par_stats)
        update_stats(chunk["ActivityScore"], activity_stats)

        flow_log = np.log1p(chunk["FLOW_SUM"].to_numpy())
        user_log = np.log1p(chunk["USER_COUNT"].to_numpy())
        fpu_log = np.log1p(chunk["flow_per_user"].to_numpy())
        par_log = np.log1p(chunk["PAR"].to_numpy())
        activity_vals = chunk["ActivityScore"].to_numpy()

        flow_hist += np.histogram(flow_log[np.isfinite(flow_log)], bins=flow_bins)[0]
        user_hist += np.histogram(user_log[np.isfinite(user_log)], bins=user_bins)[0]
        fpu_hist += np.histogram(fpu_log[np.isfinite(fpu_log)], bins=fpu_bins)[0]
        par_hist += np.histogram(par_log[np.isfinite(par_log)], bins=par_bins)[0]
        activity_hist += np.histogram(
            activity_vals[np.isfinite(activity_vals)], bins=activity_bins
        )[0]

        valid_mask = np.isfinite(flow_log) & np.isfinite(user_log)
        if valid_mask.any():
            hist2d, _, _ = np.histogram2d(
                user_log[valid_mask],
                flow_log[valid_mask],
                bins=[scatter_x_bins, scatter_y_bins],
            )
            scatter_counts += hist2d.astype(np.int64)

        daily = (
            chunk.groupby(COL_DATE, as_index=False)[["FLOW_SUM", "USER_COUNT"]]
            .sum(min_count=1)
            .rename(columns={"FLOW_SUM": "flow_sum", "USER_COUNT": "user_sum"})
        )
        daily_agg = combine_sum(daily_agg, daily.set_index(COL_DATE))

        hourly = chunk.groupby(COL_HOUR).agg(
            flow_sum=("FLOW_SUM", "sum"),
            flow_count=("FLOW_SUM", "count"),
            user_sum=("USER_COUNT", "sum"),
            user_count=("USER_COUNT", "count"),
        )
        hourly_agg = combine_sum(hourly_agg, hourly)

        weekday = chunk.groupby(COL_WEEKDAY).agg(
            flow_sum=("FLOW_SUM", "sum"),
            flow_count=("FLOW_SUM", "count"),
            user_sum=("USER_COUNT", "sum"),
            user_count=("USER_COUNT", "count"),
        )
        weekday_agg = combine_sum(weekday_agg, weekday)

        holiday_hour = chunk.groupby([COL_HOLIDAY, COL_HOUR]).agg(
            flow_sum=("FLOW_SUM", "sum"),
            flow_count=("FLOW_SUM", "count"),
            user_sum=("USER_COUNT", "sum"),
            user_count=("USER_COUNT", "count"),
        )
        holiday_hour_agg = combine_sum(holiday_hour_agg, holiday_hour)

        heatmap = chunk.groupby([COL_WEEKDAY, COL_HOUR]).agg(
            flow_sum=("FLOW_SUM", "sum"),
            flow_count=("FLOW_SUM", "count"),
            user_sum=("USER_COUNT", "sum"),
            user_count=("USER_COUNT", "count"),
        )
        heatmap_agg = combine_sum(heatmap_agg, heatmap)

        scene = chunk.groupby("SCENE").agg(
            flow_sum=("FLOW_SUM", "sum"),
            flow_count=("FLOW_SUM", "count"),
            user_sum=("USER_COUNT", "sum"),
            user_count=("USER_COUNT", "count"),
            activity_sum=("ActivityScore", "sum"),
            activity_count=("ActivityScore", "count"),
            par_sum=("PAR", "sum"),
            par_count=("PAR", "count"),
        )
        scene_agg = combine_sum(scene_agg, scene)

        type_df = chunk.groupby("TYPE").agg(
            flow_sum=("FLOW_SUM", "sum"),
            flow_count=("FLOW_SUM", "count"),
            user_sum=("USER_COUNT", "sum"),
            user_count=("USER_COUNT", "count"),
            activity_sum=("ActivityScore", "sum"),
            activity_count=("ActivityScore", "count"),
            par_sum=("PAR", "sum"),
            par_count=("PAR", "count"),
        )
        type_agg = combine_sum(type_agg, type_df)

        chunk["flow_sq"] = chunk["FLOW_SUM"] ** 2
        chunk["user_sq"] = chunk["USER_COUNT"] ** 2
        chunk["silent"] = (chunk["FLOW_SUM"].fillna(0) == 0) & (chunk["USER_COUNT"] > 0)

        cell_stats = chunk.groupby("CELL_ID").agg(
            flow_sum=("FLOW_SUM", "sum"),
            user_sum=("USER_COUNT", "sum"),
            flow_sumsq=("flow_sq", "sum"),
            user_sumsq=("user_sq", "sum"),
            flow_count=("FLOW_SUM", "count"),
            user_count=("USER_COUNT", "count"),
            flow_max=("FLOW_SUM", "max"),
            user_max=("USER_COUNT", "max"),
            silent_count=("silent", "sum"),
            record_count=("CELL_ID", "size"),
            activity_sum=("ActivityScore", "sum"),
            activity_count=("ActivityScore", "count"),
            par_sum=("PAR", "sum"),
            par_count=("PAR", "count"),
        )

        cell_agg = combine_cell_agg(
            cell_agg,
            cell_stats,
            sum_cols=[
                "flow_sum",
                "user_sum",
                "flow_sumsq",
                "user_sumsq",
                "flow_count",
                "user_count",
                "silent_count",
                "record_count",
                "activity_sum",
                "activity_count",
                "par_sum",
                "par_count",
            ],
            max_cols=["flow_max", "user_max"],
        )

        attrs = chunk[["CELL_ID", "LATITUDE", "LONGITUDE", "TYPE", "SCENE"]].drop_duplicates(
            subset=["CELL_ID"]
        )
        if cell_attrs is None:
            cell_attrs = attrs
        else:
            cell_attrs = pd.concat([cell_attrs, attrs], ignore_index=True).drop_duplicates(
                subset=["CELL_ID"], keep="first"
            )

    daily_agg = daily_agg.reset_index().rename(columns={COL_DATE: "date"})
    daily_agg["date"] = pd.to_datetime(daily_agg["date"], errors="coerce")
    daily_agg = daily_agg.sort_values("date")

    cell_agg = cell_agg.reset_index()
    if cell_attrs is not None:
        cell_agg = cell_agg.merge(cell_attrs, on="CELL_ID", how="left")

    cell_agg["flow_mean"] = cell_agg["flow_sum"] / cell_agg["flow_count"].replace(0, np.nan)
    cell_agg["user_mean"] = cell_agg["user_sum"] / cell_agg["user_count"].replace(0, np.nan)
    cell_agg["flow_var"] = (
        cell_agg["flow_sumsq"] / cell_agg["flow_count"].replace(0, np.nan)
        - cell_agg["flow_mean"] ** 2
    )
    cell_agg["user_var"] = (
        cell_agg["user_sumsq"] / cell_agg["user_count"].replace(0, np.nan)
        - cell_agg["user_mean"] ** 2
    )
    cell_agg["flow_std"] = np.sqrt(np.maximum(cell_agg["flow_var"], 0))
    cell_agg["user_std"] = np.sqrt(np.maximum(cell_agg["user_var"], 0))
    cell_agg["flow_cv"] = cell_agg["flow_std"] / cell_agg["flow_mean"]
    cell_agg["user_cv"] = cell_agg["user_std"] / cell_agg["user_mean"]
    cell_agg["flow_per_user"] = cell_agg["flow_sum"] / cell_agg["user_sum"].replace(0, np.nan)
    cell_agg["peak_ratio"] = cell_agg["flow_max"] / cell_agg["flow_mean"]
    cell_agg["silent_ratio"] = cell_agg["silent_count"] / cell_agg["record_count"].replace(0, np.nan)
    cell_agg["activity_mean"] = cell_agg["activity_sum"] / cell_agg["activity_count"].replace(0, np.nan)
    cell_agg["par_mean"] = cell_agg["par_sum"] / cell_agg["par_count"].replace(0, np.nan)

    flow_global = summarize_global(flow_stats)
    user_global = summarize_global(user_stats)
    fpu_global = summarize_global(fpu_stats)
    par_global = summarize_global(par_stats)
    activity_global = summarize_global(activity_stats)

    high_load_threshold = cell_agg["flow_sum"].quantile(0.99)
    silent_threshold = 0.3
    high_load_cells = cell_agg[cell_agg["flow_sum"] >= high_load_threshold]
    silent_cells = cell_agg[cell_agg["silent_ratio"] >= silent_threshold]

    def save_fig(name: str) -> None:
        plt.tight_layout()
        plt.savefig(OUT_DIR / name, dpi=150)
        plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    flow_centers = (flow_bins[:-1] + flow_bins[1:]) / 2
    user_centers = (user_bins[:-1] + user_bins[1:]) / 2
    axes[0].bar(flow_centers, flow_hist, width=np.diff(flow_bins), color="#4C78A8")
    axes[0].set_title("\u4e1a\u52a1\u6d41\u91cf\u5206\u5e03\uff08log1p\uff09")
    axes[0].set_xlabel("log(1+\u4e1a\u52a1\u6d41\u91cf)")
    axes[0].set_ylabel("\u6837\u672c\u6570\u91cf")
    axes[1].bar(user_centers, user_hist, width=np.diff(user_bins), color="#F58518")
    axes[1].set_title("\u7528\u6237\u6570\u5206\u5e03\uff08log1p\uff09")
    axes[1].set_xlabel("log(1+\u7528\u6237\u6570)")
    axes[1].set_ylabel("\u6837\u672c\u6570\u91cf")
    save_fig("fig01_flow_user_hist.png")

    flow_box = {
        "med": hist_quantile(flow_bins, flow_hist, 0.5),
        "q1": hist_quantile(flow_bins, flow_hist, 0.25),
        "q3": hist_quantile(flow_bins, flow_hist, 0.75),
        "whislo": hist_quantile(flow_bins, flow_hist, 0.05),
        "whishi": hist_quantile(flow_bins, flow_hist, 0.95),
        "fliers": [],
    }
    user_box = {
        "med": hist_quantile(user_bins, user_hist, 0.5),
        "q1": hist_quantile(user_bins, user_hist, 0.25),
        "q3": hist_quantile(user_bins, user_hist, 0.75),
        "whislo": hist_quantile(user_bins, user_hist, 0.05),
        "whishi": hist_quantile(user_bins, user_hist, 0.95),
        "fliers": [],
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bxp([flow_box], showfliers=False)
    axes[0].set_title("\u4e1a\u52a1\u6d41\u91cf\u5206\u4f4d\u7bb1\u56fe\uff08log1p\uff09")
    axes[0].set_ylabel("log(1+\u4e1a\u52a1\u6d41\u91cf)")
    axes[0].set_xticklabels(["FLOW_SUM"])
    axes[1].bxp([user_box], showfliers=False)
    axes[1].set_title("\u7528\u6237\u6570\u5206\u4f4d\u7bb1\u56fe\uff08log1p\uff09")
    axes[1].set_ylabel("log(1+\u7528\u6237\u6570)")
    axes[1].set_xticklabels(["USER_COUNT"])
    save_fig("fig02_flow_user_box.png")

    plt.figure(figsize=(6, 5))
    density_counts = smooth_counts(scatter_counts.T, passes=2)
    density = np.log1p(density_counts)
    plt.imshow(
        density,
        origin="lower",
        aspect="auto",
        extent=[
            scatter_x_bins[0],
            scatter_x_bins[-1],
            scatter_y_bins[0],
            scatter_y_bins[-1],
        ],
        cmap="magma",
    )
    plt.colorbar(label="log(1+\u8ba1\u6570)")
    plt.title("\u4e1a\u52a1\u6d41\u91cf-\u7528\u6237\u6570\u5bc6\u5ea6\uff08log1p\uff09")
    plt.xlabel("log(1+\u7528\u6237\u6570)")
    plt.ylabel("log(1+\u4e1a\u52a1\u6d41\u91cf)")
    save_fig("fig03_flow_user_scatter.png")

    plt.figure(figsize=(8, 4))
    fpu_centers = (fpu_bins[:-1] + fpu_bins[1:]) / 2
    plt.bar(fpu_centers, fpu_hist, width=np.diff(fpu_bins), color="#72B7B2")
    plt.title("\u4eba\u5747\u6d41\u91cf\u5206\u5e03\uff08log1p\uff09")
    plt.xlabel("log(1+\u4eba\u5747\u6d41\u91cf)")
    plt.ylabel("\u6837\u672c\u6570\u91cf")
    save_fig("fig24_flow_per_user_hist.png")

    plt.figure(figsize=(8, 4))
    par_centers = (par_bins[:-1] + par_bins[1:]) / 2
    plt.bar(par_centers, par_hist, width=np.diff(par_bins), color="#E45756")
    plt.title("PAR \u5206\u5e03\uff08log1p\uff09")
    plt.xlabel("log(1+PAR)")
    plt.ylabel("\u6837\u672c\u6570\u91cf")
    save_fig("fig22_par_hist.png")

    plt.figure(figsize=(8, 4))
    activity_centers = (activity_bins[:-1] + activity_bins[1:]) / 2
    plt.bar(activity_centers, activity_hist, width=np.diff(activity_bins), color="#54A24B")
    plt.title("\u6d3b\u8dc3\u5ea6\u8bc4\u5206\u5206\u5e03")
    plt.xlabel("ActivityScore")
    plt.ylabel("\u6837\u672c\u6570\u91cf")
    save_fig("fig21_activityscore_hist.png")

    plt.figure(figsize=(10, 4))
    plt.plot(daily_agg["date"], daily_agg["flow_sum"])
    plt.xticks(rotation=45)
    plt.title("\u5168\u7f51\u65e5\u603b\u6d41\u91cf\u8d70\u52bf")
    plt.xlabel("\u65e5\u671f")
    plt.ylabel("\u4e1a\u52a1\u6d41\u91cf\uff08MB\uff09")
    save_fig("fig04_daily_total_flow.png")

    plt.figure(figsize=(10, 4))
    plt.plot(daily_agg["date"], daily_agg["user_sum"])
    plt.xticks(rotation=45)
    plt.title("\u5168\u7f51\u65e5\u603b\u7528\u6237\u6570\u8d70\u52bf")
    plt.xlabel("\u65e5\u671f")
    plt.ylabel("\u7528\u6237\u6570")
    save_fig("fig05_daily_total_user.png")

    cell_type_counts = cell_agg["TYPE"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=cell_type_counts.index.astype(int), y=cell_type_counts.values, color="#4C78A8")
    plt.title("\u5c0f\u533a\u7c7b\u578b\u6570\u91cf\u5206\u5e03")
    plt.xlabel("TYPE")
    plt.ylabel("\u5c0f\u533a\u6570\u91cf")
    save_fig("fig32_cell_type_count.png")

    cell_scene_counts = cell_agg["SCENE"].value_counts()
    top_scene_counts = cell_scene_counts.head(12)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=top_scene_counts.index.astype(int), y=top_scene_counts.values, color="#72B7B2")
    plt.title("\u5c0f\u533a\u573a\u666f\u6570\u91cf Top12")
    plt.xlabel("SCENE")
    plt.ylabel("\u5c0f\u533a\u6570\u91cf")
    save_fig("fig33_cell_scene_count.png")

    scene_type = cell_agg.pivot_table(
        index="SCENE",
        columns="TYPE",
        values="CELL_ID",
        aggfunc="count",
        fill_value=0,
    )
    top_scene_index = cell_scene_counts.head(12).index
    scene_type = scene_type.loc[scene_type.index.intersection(top_scene_index)]
    if not scene_type.empty:
        plt.figure(figsize=(8, 5))
        sns.heatmap(scene_type, cmap="YlOrRd")
        plt.title("\u573a\u666f\u00d7\u7c7b\u578b\u7ec4\u5408\u5206\u5e03\uff08Top12 \u573a\u666f\uff09")
        plt.xlabel("TYPE")
        plt.ylabel("SCENE")
        save_fig("fig34_scene_type_heatmap.png")

    combo_counts = (
        cell_agg.groupby(["SCENE", "TYPE"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    if not combo_counts.empty:
        labels = [f"S{int(idx[0])}-T{int(idx[1])}" for idx in combo_counts.index]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, combo_counts.values, color="#F58518")
        plt.title("\u573a\u666f\u00d7\u7c7b\u578b\u7ec4\u5408 Top10")
        plt.xlabel("\u7ec4\u5408")
        plt.ylabel("\u5c0f\u533a\u6570\u91cf")
        plt.xticks(rotation=30)
        save_fig("fig35_scene_type_top10.png")

    hourly_agg = hourly_agg.reindex(range(24)).fillna(0)
    hourly_agg["flow_mean"] = hourly_agg["flow_sum"] / hourly_agg["flow_count"].replace(0, np.nan)
    hourly_agg["user_mean"] = hourly_agg["user_sum"] / hourly_agg["user_count"].replace(0, np.nan)

    plt.figure(figsize=(10, 4))
    plt.plot(hourly_agg.index, hourly_agg["flow_mean"], marker="o")
    plt.title("\u5c0f\u65f6\u7ef4\u5ea6\u5e73\u5747\u6d41\u91cf")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u5e73\u5747\u6d41\u91cf\uff08MB\uff09")
    save_fig("fig06_hourly_mean_flow.png")

    plt.figure(figsize=(10, 4))
    plt.plot(hourly_agg.index, hourly_agg["user_mean"], marker="o", color="#F58518")
    plt.title("\u5c0f\u65f6\u7ef4\u5ea6\u5e73\u5747\u7528\u6237\u6570")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u5e73\u5747\u7528\u6237\u6570")
    save_fig("fig07_hourly_mean_user.png")

    weekday_agg = weekday_agg.reindex(range(7)).fillna(0)
    weekday_agg["flow_mean"] = weekday_agg["flow_sum"] / weekday_agg["flow_count"].replace(0, np.nan)
    weekday_agg["user_mean"] = weekday_agg["user_sum"] / weekday_agg["user_count"].replace(0, np.nan)
    weekday_names = weekday_labels(list(weekday_agg.index))

    flow_weekday_mean = weekday_agg["flow_mean"].mean()
    user_weekday_mean = weekday_agg["user_mean"].mean()
    weekday_agg["flow_pct"] = (weekday_agg["flow_mean"] / flow_weekday_mean - 1) * 100
    weekday_agg["user_pct"] = (weekday_agg["user_mean"] / user_weekday_mean - 1) * 100

    plt.figure(figsize=(8, 4))
    sns.barplot(x=weekday_names, y=weekday_agg["flow_pct"], color="#4C78A8")
    plt.axhline(0, color="#333333", linewidth=1)
    plt.title("\u661f\u671f\u7ef4\u5ea6\u5e73\u5747\u6d41\u91cf\u76f8\u5bf9\u5747\u503c")
    plt.xlabel("\u661f\u671f")
    plt.ylabel("\u76f8\u5bf9\u5747\u503c\uff08%\uff09")
    save_fig("fig08_weekday_mean_flow.png")

    plt.figure(figsize=(8, 4))
    sns.barplot(x=weekday_names, y=weekday_agg["user_pct"], color="#F58518")
    plt.axhline(0, color="#333333", linewidth=1)
    plt.title("\u661f\u671f\u7ef4\u5ea6\u5e73\u5747\u7528\u6237\u6570\u76f8\u5bf9\u5747\u503c")
    plt.xlabel("\u661f\u671f")
    plt.ylabel("\u76f8\u5bf9\u5747\u503c\uff08%\uff09")
    save_fig("fig09_weekday_mean_user.png")

    holiday_hour_agg = holiday_hour_agg.reset_index()
    holiday_hour_agg["flow_mean"] = holiday_hour_agg["flow_sum"] / holiday_hour_agg["flow_count"].replace(0, np.nan)
    holiday_hour_agg["user_mean"] = holiday_hour_agg["user_sum"] / holiday_hour_agg["user_count"].replace(0, np.nan)

    plt.figure(figsize=(10, 4))
    for flag, label in [(0, "\u5de5\u4f5c\u65e5"), (1, "\u8282\u5047\u65e5")]:
        subset = holiday_hour_agg[holiday_hour_agg[COL_HOLIDAY] == flag]
        subset = subset.set_index(COL_HOUR).reindex(range(24)).fillna(0)
        plt.plot(subset.index, subset["flow_mean"], marker="o", label=label)
    plt.title("\u8282\u5047\u65e5\u4e0e\u5de5\u4f5c\u65e5\u7684\u5c0f\u65f6\u6d41\u91cf\u5bf9\u6bd4")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u5e73\u5747\u6d41\u91cf\uff08MB\uff09")
    plt.legend()
    save_fig("fig25_hourly_holiday_flow.png")

    plt.figure(figsize=(10, 4))
    for flag, label in [(0, "\u5de5\u4f5c\u65e5"), (1, "\u8282\u5047\u65e5")]:
        subset = holiday_hour_agg[holiday_hour_agg[COL_HOLIDAY] == flag]
        subset = subset.set_index(COL_HOUR).reindex(range(24)).fillna(0)
        plt.plot(subset.index, subset["user_mean"], marker="o", label=label)
    plt.title("\u8282\u5047\u65e5\u4e0e\u5de5\u4f5c\u65e5\u7684\u5c0f\u65f6\u7528\u6237\u6570\u5bf9\u6bd4")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u5e73\u5747\u7528\u6237\u6570")
    plt.legend()
    save_fig("fig26_hourly_holiday_user.png")

    heatmap_agg = heatmap_agg.reset_index()
    heatmap_agg["flow_mean"] = heatmap_agg["flow_sum"] / heatmap_agg["flow_count"].replace(0, np.nan)
    heatmap_agg["user_mean"] = heatmap_agg["user_sum"] / heatmap_agg["user_count"].replace(0, np.nan)

    heat_flow = heatmap_agg.pivot(index=COL_WEEKDAY, columns=COL_HOUR, values="flow_mean").reindex(index=range(7), columns=range(24))
    heat_user = heatmap_agg.pivot(index=COL_WEEKDAY, columns=COL_HOUR, values="user_mean").reindex(index=range(7), columns=range(24))

    plt.figure(figsize=(12, 4))
    sns.heatmap(heat_flow, cmap="YlOrRd")
    plt.title("\u5c0f\u65f6-\u661f\u671f\u7ef4\u5ea6\u6d41\u91cf\u70ed\u529b\u56fe")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u661f\u671f")
    plt.yticks(ticks=np.arange(0.5, 7.5, 1.0), labels=weekday_labels(list(range(7))))
    save_fig("fig15_hour_weekday_heatmap_flow.png")

    plt.figure(figsize=(12, 4))
    sns.heatmap(heat_user, cmap="YlGnBu")
    plt.title("\u5c0f\u65f6-\u661f\u671f\u7ef4\u5ea6\u7528\u6237\u6570\u70ed\u529b\u56fe")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u661f\u671f")
    plt.yticks(ticks=np.arange(0.5, 7.5, 1.0), labels=weekday_labels(list(range(7))))
    save_fig("fig16_hour_weekday_heatmap_user.png")

    scene_agg["flow_mean"] = scene_agg["flow_sum"] / scene_agg["flow_count"].replace(0, np.nan)
    scene_agg["user_mean"] = scene_agg["user_sum"] / scene_agg["user_count"].replace(0, np.nan)
    scene_top = scene_agg.sort_values("flow_sum", ascending=False).head(10)

    plt.figure(figsize=(10, 4))
    sns.barplot(x=scene_top.index.astype(int).astype(str), y=scene_top["flow_sum"], color="#4C78A8")
    plt.title("\u573a\u666f\u603b\u6d41\u91cf TOP10")
    plt.xlabel("\u573a\u666f\uff08SCENE\uff09")
    plt.ylabel("\u603b\u6d41\u91cf\uff08MB\uff09")
    save_fig("fig12_scene_flow_share.png")

    scene_counts = cell_agg["SCENE"].value_counts().head(10).index
    scene_df = cell_agg[cell_agg["SCENE"].isin(scene_counts)]

    plt.figure(figsize=(10, 4))
    sns.boxplot(x="SCENE", y="flow_mean", data=scene_df, color="#72B7B2")
    plt.title("\u4e0d\u540c\u573a\u666f\u7684\u5c0f\u533a\u5e73\u5747\u6d41\u91cf\u5206\u5e03")
    plt.xlabel("\u573a\u666f\uff08SCENE\uff09")
    plt.ylabel("\u5e73\u5747\u6d41\u91cf\uff08MB\uff09")
    save_fig("fig10_scene_flow_box.png")

    plt.figure(figsize=(10, 4))
    sns.boxplot(x="SCENE", y="user_mean", data=scene_df, color="#54A24B")
    plt.title("\u4e0d\u540c\u573a\u666f\u7684\u5c0f\u533a\u5e73\u5747\u7528\u6237\u6570\u5206\u5e03")
    plt.xlabel("\u573a\u666f\uff08SCENE\uff09")
    plt.ylabel("\u5e73\u5747\u7528\u6237\u6570")
    save_fig("fig11_scene_user_box.png")

    type_agg["flow_mean"] = type_agg["flow_sum"] / type_agg["flow_count"].replace(0, np.nan)
    type_agg["user_mean"] = type_agg["user_sum"] / type_agg["user_count"].replace(0, np.nan)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=type_agg.index.astype(int).astype(str), y=type_agg["flow_mean"], color="#9D755D")
    plt.title("\u4e0d\u540c TYPE \u7684\u5e73\u5747\u6d41\u91cf")
    plt.xlabel("TYPE")
    plt.ylabel("\u5e73\u5747\u6d41\u91cf\uff08MB\uff09")
    save_fig("fig13_type_flow_bar.png")

    plt.figure(figsize=(8, 4))
    sns.barplot(x=type_agg.index.astype(int).astype(str), y=type_agg["user_mean"], color="#F28E2B")
    plt.title("\u4e0d\u540c TYPE \u7684\u5e73\u5747\u7528\u6237\u6570")
    plt.xlabel("TYPE")
    plt.ylabel("\u5e73\u5747\u7528\u6237\u6570")
    save_fig("fig14_type_user_bar.png")

    top10_flow = cell_agg.sort_values("flow_sum", ascending=False).head(10)
    top10_flow_labels = (
        top10_flow["CELL_ID"].astype(str)
        + " | \u573a\u666f"
        + top10_flow["SCENE"].fillna(-1).astype(int).astype(str)
    )
    plt.figure(figsize=(10, 4))
    sns.barplot(x=top10_flow_labels, y=top10_flow["flow_sum"], color="#4C78A8")
    plt.title("TOP10 \u5c0f\u533a\u603b\u6d41\u91cf")
    plt.xlabel("\u5c0f\u533aID | \u573a\u666f")
    plt.ylabel("\u4e1a\u52a1\u6d41\u91cf\uff08MB\uff09")
    plt.xticks(rotation=30, ha="right")
    save_fig("fig17_top10_flow.png")

    top10_fpu = cell_agg.sort_values("flow_per_user", ascending=False).head(10)
    top10_fpu_labels = (
        top10_fpu["CELL_ID"].astype(str)
        + " | \u573a\u666f"
        + top10_fpu["SCENE"].fillna(-1).astype(int).astype(str)
    )
    plt.figure(figsize=(10, 4))
    sns.barplot(x=top10_fpu_labels, y=top10_fpu["flow_per_user"], color="#E45756")
    plt.title("TOP10 \u4eba\u5747\u6d41\u91cf")
    plt.xlabel("\u5c0f\u533aID | \u573a\u666f")
    plt.ylabel("\u4eba\u5747\u6d41\u91cf\uff08MB/\u4eba\uff09")
    plt.xticks(rotation=30, ha="right")
    save_fig("fig18_top10_flow_per_user.png")

    top_metrics = [
        ("flow_sum", "TOP10 \u5c0f\u533a\u603b\u6d41\u91cf", "\u603b\u6d41\u91cf(MB)", "#4C78A8"),
        ("user_sum", "TOP10 \u5c0f\u533a\u603b\u7528\u6237\u6570", "\u603b\u7528\u6237\u6570", "#F58518"),
        ("flow_mean", "TOP10 \u5c0f\u533a\u5e73\u5747\u6d41\u91cf", "\u5e73\u5747\u6d41\u91cf(MB)", "#72B7B2"),
        ("user_mean", "TOP10 \u5c0f\u533a\u5e73\u5747\u7528\u6237\u6570", "\u5e73\u5747\u7528\u6237\u6570", "#54A24B"),
        ("flow_per_user", "TOP10 \u4eba\u5747\u6d41\u91cf", "MB/\u4eba", "#E45756"),
        ("peak_ratio", "TOP10 \u5cf0\u5747\u6bd4", "\u5cf0\u5747\u6bd4", "#B279A2"),
        ("flow_cv", "TOP10 \u6d41\u91cfCV", "CV", "#9D755D"),
        ("user_cv", "TOP10 \u7528\u6237CV", "CV", "#4C78A8"),
        ("activity_mean", "TOP10 \u6d3b\u8dc3\u5ea6", "\u6d3b\u8dc3\u5ea6", "#F28E2B"),
        ("par_mean", "TOP10 PAR \u5747\u503c", "PAR", "#76B7B2"),
    ]
    rows, cols = grid_shape(len(top_metrics), max_cols=5)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).reshape(rows, cols)
    for ax, (col, title, ylabel, color) in zip(axes.flat, top_metrics):
        metric_df = cell_agg[["CELL_ID", col]].dropna()
        metric_top = metric_df.nlargest(10, col)
        if metric_top.empty:
            annotate_empty(ax)
            continue
        labels = metric_top["CELL_ID"].astype(str)
        bar_subplot(ax, labels, metric_top[col], title, ylabel, color)
    for ax in axes.flat[len(top_metrics) :]:
        ax.axis("off")
    save_fig("fig28_top_metrics_grid.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(cell_agg["silent_ratio"].dropna(), bins=40, color="#B279A2")
    plt.title("\u5c0f\u533a\u9759\u9ed8\u6bd4\u4f8b\u5206\u5e03")
    plt.xlabel("\u9759\u9ed8\u6bd4\u4f8b\uff08\u6709\u7528\u6237\u65e0\u6d41\u91cf\uff09")
    plt.ylabel("\u5c0f\u533a\u6570\u91cf")
    save_fig("fig19_silent_ratio_hist.png")

    if not silent_cells.empty:
        silent_scene = silent_cells["SCENE"].value_counts().sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=silent_scene.index.astype(int).astype(str), y=silent_scene.values, color="#FF9DA6")
        plt.title("\u9759\u9ed8\u5c0f\u533a\u6570\u91cf\uff08\u6309\u573a\u666f\uff09")
        plt.xlabel("\u573a\u666f\uff08SCENE\uff09")
        plt.ylabel("\u9759\u9ed8\u5c0f\u533a\u6570\u91cf")
        save_fig("fig20_silent_scene.png")

    if not high_load_cells.empty:
        high_scene = high_load_cells["SCENE"].value_counts().sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=high_scene.index.astype(int).astype(str), y=high_scene.values, color="#F28E2B")
        plt.title("\u9ad8\u8d1f\u8377\u5c0f\u533a\u6570\u91cf\uff08\u6309\u573a\u666f\uff09")
        plt.xlabel("\u573a\u666f\uff08SCENE\uff09")
        plt.ylabel("\u9ad8\u8d1f\u8377\u5c0f\u533a\u6570\u91cf")
        save_fig("fig21_highload_scene.png")

    dist_metrics = [
        ("silent_ratio", "\u9759\u9ed8\u6bd4\u4f8b\u5206\u5e03", "\u9759\u9ed8\u6bd4\u4f8b", "#B279A2", False),
        ("peak_ratio", "\u5cf0\u5747\u6bd4\u5206\u5e03\uff0899% \u88c1\u526a\uff09", "\u5cf0\u5747\u6bd4", "#4C78A8", True),
        ("flow_cv", "\u6d41\u91cfCV\u5206\u5e03\uff0899% \u88c1\u526a\uff09", "\u6d41\u91cfCV", "#9D755D", True),
        ("user_cv", "\u7528\u6237CV\u5206\u5e03\uff0899% \u88c1\u526a\uff09", "\u7528\u6237CV", "#F58518", True),
        ("activity_mean", "\u6d3b\u8dc3\u5ea6\u5206\u5e03", "\u6d3b\u8dc3\u5ea6", "#54A24B", False),
        ("flow_per_user", "\u4eba\u5747\u6d41\u91cf\u5206\u5e03\uff0899% \u88c1\u526a\uff09", "MB/\u4eba", "#72B7B2", True),
    ]
    rows, cols = grid_shape(len(dist_metrics), max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).reshape(rows, cols)
    for ax, (col, title, xlabel, color, clip_tail) in zip(axes.flat, dist_metrics):
        series = cell_agg[col].dropna()
        if series.empty:
            annotate_empty(ax)
            continue
        if clip_tail:
            series = series.clip(upper=series.quantile(0.99))
        ax.hist(series, bins=40, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
    for ax in axes.flat[len(dist_metrics) :]:
        ax.axis("off")
    save_fig("fig29_anomaly_dist_grid.png")

    highload_panels = 6
    rows, cols = grid_shape(highload_panels, max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).reshape(rows, cols).flat

    ax = axes[0]
    if high_load_cells.empty:
        annotate_empty(ax)
    else:
        high_scene = high_load_cells["SCENE"].value_counts().sort_values(ascending=False).head(10)
        bar_subplot(ax, high_scene.index.astype(int).astype(str), high_scene.values, "\u9ad8\u8d1f\u8377\u573a\u666f\u5206\u5e03", "\u6570\u91cf")

    ax = axes[1]
    if high_load_cells.empty:
        annotate_empty(ax)
    else:
        high_type = high_load_cells["TYPE"].value_counts().sort_values(ascending=False)
        bar_subplot(ax, high_type.index.astype(int).astype(str), high_type.values, "\u9ad8\u8d1f\u8377 TYPE \u5206\u5e03", "\u6570\u91cf", "#F58518")

    ax = axes[2]
    if high_load_cells.empty:
        annotate_empty(ax)
    else:
        top_highload = high_load_cells.sort_values("flow_sum", ascending=False).head(10)
        bar_subplot(
            ax,
            top_highload["CELL_ID"].astype(str),
            top_highload["flow_sum"],
            "\u9ad8\u8d1f\u8377 TOP10 \u603b\u6d41\u91cf",
            "\u603b\u6d41\u91cf(MB)",
            "#E45756",
        )

    ax = axes[3]
    if high_load_cells.empty:
        annotate_empty(ax)
    else:
        top_highload = high_load_cells.sort_values("peak_ratio", ascending=False).head(10)
        bar_subplot(
            ax,
            top_highload["CELL_ID"].astype(str),
            top_highload["peak_ratio"],
            "\u9ad8\u8d1f\u8377 TOP10 \u5cf0\u5747\u6bd4",
            "\u5cf0\u5747\u6bd4",
            "#9D755D",
        )

    ax = axes[4]
    if high_load_cells.empty:
        annotate_empty(ax)
    else:
        top_highload = high_load_cells.sort_values("flow_per_user", ascending=False).head(10)
        bar_subplot(
            ax,
            top_highload["CELL_ID"].astype(str),
            top_highload["flow_per_user"],
            "\u9ad8\u8d1f\u8377 TOP10 \u4eba\u5747\u6d41\u91cf",
            "MB/\u4eba",
            "#72B7B2",
        )

    ax = axes[5]
    highload_fpu = high_load_cells["flow_per_user"].dropna()
    other_fpu = cell_agg.loc[cell_agg["flow_sum"] < high_load_threshold, "flow_per_user"].dropna()
    if highload_fpu.empty or other_fpu.empty:
        annotate_empty(ax)
    else:
        ax.boxplot(
            [highload_fpu, other_fpu],
            tick_labels=["\u9ad8\u8d1f\u8377", "\u975e\u9ad8\u8d1f\u8377"],
            showfliers=False,
        )
        ax.set_title("\u4eba\u5747\u6d41\u91cf\u5bf9\u6bd4", fontsize=10)
        ax.set_ylabel("MB/\u4eba")

    save_fig("fig30_highload_profile_grid.png")

    silent_panels = 6
    rows, cols = grid_shape(silent_panels, max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).reshape(rows, cols).flat

    ax = axes[0]
    if silent_cells.empty:
        annotate_empty(ax)
    else:
        silent_scene = silent_cells["SCENE"].value_counts().sort_values(ascending=False).head(10)
        bar_subplot(ax, silent_scene.index.astype(int).astype(str), silent_scene.values, "\u9759\u9ed8\u573a\u666f\u5206\u5e03", "\u6570\u91cf", "#FF9DA6")

    ax = axes[1]
    if silent_cells.empty:
        annotate_empty(ax)
    else:
        silent_type = silent_cells["TYPE"].value_counts().sort_values(ascending=False)
        bar_subplot(ax, silent_type.index.astype(int).astype(str), silent_type.values, "\u9759\u9ed8 TYPE \u5206\u5e03", "\u6570\u91cf", "#72B7B2")

    ax = axes[2]
    if silent_cells.empty:
        annotate_empty(ax)
    else:
        top_silent = silent_cells.sort_values("silent_ratio", ascending=False).head(10)
        bar_subplot(
            ax,
            top_silent["CELL_ID"].astype(str),
            top_silent["silent_ratio"],
            "\u9759\u9ed8 TOP10 \u6bd4\u4f8b",
            "\u9759\u9ed8\u6bd4\u4f8b",
            "#B279A2",
        )

    ax = axes[3]
    if silent_cells.empty:
        annotate_empty(ax)
    else:
        top_silent = silent_cells.sort_values("activity_mean", ascending=False).head(10)
        bar_subplot(
            ax,
            top_silent["CELL_ID"].astype(str),
            top_silent["activity_mean"],
            "\u9759\u9ed8 TOP10 \u6d3b\u8dc3\u5ea6",
            "\u6d3b\u8dc3\u5ea6",
            "#54A24B",
        )

    ax = axes[4]
    silent_activity = silent_cells["activity_mean"].dropna()
    other_activity = cell_agg.loc[cell_agg["silent_ratio"] < silent_threshold, "activity_mean"].dropna()
    if silent_activity.empty or other_activity.empty:
        annotate_empty(ax)
    else:
        ax.boxplot(
            [silent_activity, other_activity],
            tick_labels=["\u9759\u9ed8", "\u975e\u9759\u9ed8"],
            showfliers=False,
        )
        ax.set_title("\u6d3b\u8dc3\u5ea6\u5bf9\u6bd4", fontsize=10)
        ax.set_ylabel("\u6d3b\u8dc3\u5ea6")

    ax = axes[5]
    silent_fpu = silent_cells["flow_per_user"].dropna()
    other_fpu = cell_agg.loc[cell_agg["silent_ratio"] < silent_threshold, "flow_per_user"].dropna()
    if silent_fpu.empty or other_fpu.empty:
        annotate_empty(ax)
    else:
        ax.boxplot(
            [silent_fpu, other_fpu],
            tick_labels=["\u9759\u9ed8", "\u975e\u9759\u9ed8"],
            showfliers=False,
        )
        ax.set_title("\u4eba\u5747\u6d41\u91cf\u5bf9\u6bd4", fontsize=10)
        ax.set_ylabel("MB/\u4eba")

    save_fig("fig31_silent_profile_grid.png")

    geo_df = cell_agg.dropna(subset=["LONGITUDE", "LATITUDE", "flow_per_user"])
    if not geo_df.empty:
        plt.figure(figsize=(6, 6))
        sizes = np.log1p(geo_df["flow_sum"].clip(lower=0)) * 4
        plt.scatter(
            geo_df["LONGITUDE"],
            geo_df["LATITUDE"],
            s=sizes,
            c=geo_df["flow_per_user"],
            cmap="viridis",
            alpha=0.4,
        )
        plt.colorbar(label="\u4eba\u5747\u6d41\u91cf\uff08MB/\u4eba\uff09")
        plt.title("\u7ecf\u7eac\u5ea6\u6ce1\u6ce1\u56fe\uff08\u5927\u5c0f=\u6d41\u91cf\uff0c\u989c\u8272=\u4eba\u5747\u6d41\u91cf\uff09")
        plt.xlabel("\u7ecf\u5ea6")
        plt.ylabel("\u7eac\u5ea6")
        save_fig("fig27_geo_bubble_flow.png")

    stats_out = {
        "global": {
            "flow": flow_global,
            "user": user_global,
            "flow_per_user": fpu_global,
            "par": par_global,
            "activity": activity_global,
        },
        "cell_distribution": {
            "type_counts": cell_type_counts.to_dict(),
            "scene_counts_top12": top_scene_counts.to_dict(),
            "scene_type_top10": {
                f"S{int(k[0])}-T{int(k[1])}": int(v) for k, v in combo_counts.items()
            },
        },
        "cells": {
            "total_cells": int(cell_agg.shape[0]),
            "silent_threshold": silent_threshold,
            "silent_cells": int(silent_cells.shape[0]),
            "high_load_threshold": float(high_load_threshold),
            "high_load_cells": int(high_load_cells.shape[0]),
        },
        "derived_metrics": {
            "flow_mean": "\u5c0f\u533a\u5e73\u5747\u4e1a\u52a1\u6d41\u91cf = flow_sum / flow_count",
            "user_mean": "\u5c0f\u533a\u5e73\u5747\u7528\u6237\u6570 = user_sum / user_count",
            "flow_per_user": "\u4eba\u5747\u6d41\u91cf = flow_sum / user_sum",
            "flow_cv": "\u6d41\u91cf\u53d8\u5f02\u7cfb\u6570 = flow_std / flow_mean",
            "silent_ratio": "\u9759\u9ed8\u6bd4\u4f8b = silent_count / record_count",
            "peak_ratio": "\u5cf0\u5747\u6bd4 = flow_max / flow_mean",
            "activity_mean": "\u6d3b\u8dc3\u5ea6\u5747\u503c = activity_sum / activity_count",
            "par_mean": "PAR \u5747\u503c = par_sum / par_count",
        },
        "top10_flow": top10_flow[["CELL_ID", "flow_sum", "SCENE", "TYPE"]].to_dict(
            orient="records"
        ),
        "top10_flow_per_user": top10_fpu[
            ["CELL_ID", "flow_per_user", "SCENE", "TYPE"]
        ].to_dict(orient="records"),
    }

    with open(OUT_DIR / "section3_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)

    cell_agg.to_csv(OUT_DIR / "section3_cell_agg.csv", index=False)


if __name__ == "__main__":
    main()
