# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = Path("processed")
ATTR_PATH = Path("attributes.csv")
OUT_DIR = Path("report_assets/section3")


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


def combine_cell_agg(base: pd.DataFrame | None, add: pd.DataFrame) -> pd.DataFrame:
    if base is None:
        return add
    base = base.reindex(base.index.union(add.index))
    sum_cols = [
        "flow_sum",
        "user_sum",
        "flow_sumsq",
        "user_sumsq",
        "flow_count",
        "user_count",
        "silent_count",
        "record_count",
    ]
    max_cols = ["flow_max", "user_max"]
    for col in sum_cols:
        base[col] = base[col].fillna(0) + add[col].reindex(base.index).fillna(0)
    for col in max_cols:
        base[col] = np.maximum(
            base[col].fillna(0), add[col].reindex(base.index).fillna(0)
        )
    return base


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


def main() -> None:
    ensure_out_dir()
    set_cn_style()

    files = sorted(DATA_DIR.glob("cell_data_b*.csv"))
    if not files:
        raise FileNotFoundError("未找到 processed/cell_data_b*.csv")

    attrs = pd.read_csv(ATTR_PATH)
    flow_stats = init_stats()
    user_stats = init_stats()
    cell_agg: pd.DataFrame | None = None
    daily_agg: pd.DataFrame | None = None
    samples = []

    for idx, path in enumerate(files, start=1):
        df = pd.read_csv(
            path,
            usecols=["CELL_ID", "DATETIME_KEY", "FLOW_SUM", "USER_COUNT"],
            dtype={
                "CELL_ID": "int32",
                "DATETIME_KEY": "string",
                "FLOW_SUM": "float32",
                "USER_COUNT": "float32",
            },
            na_values=["", "NA", "NaN"],
        )

        df.loc[df["FLOW_SUM"] < 0, "FLOW_SUM"] = np.nan
        df.loc[df["USER_COUNT"] < 0, "USER_COUNT"] = np.nan

        update_stats(df["FLOW_SUM"], flow_stats)
        update_stats(df["USER_COUNT"], user_stats)

        if not df.empty:
            sample_n = min(5000, len(df))
            samples.append(
                df[["FLOW_SUM", "USER_COUNT"]].sample(
                    n=sample_n, random_state=42 + idx
                )
            )

        df["DATE"] = df["DATETIME_KEY"].str.slice(0, 10)
        daily = (
            df.groupby("DATE", as_index=False)[["FLOW_SUM", "USER_COUNT"]]
            .sum(min_count=1)
            .rename(columns={"FLOW_SUM": "flow_sum", "USER_COUNT": "user_sum"})
        )
        if daily_agg is None:
            daily_agg = daily
        else:
            daily_agg = (
                daily_agg.set_index("DATE")
                .add(daily.set_index("DATE"), fill_value=0)
                .reset_index()
            )

        df["flow_sq"] = df["FLOW_SUM"] ** 2
        df["user_sq"] = df["USER_COUNT"] ** 2
        df["silent"] = (df["FLOW_SUM"].fillna(0) == 0) & (df["USER_COUNT"] > 0)

        grp = df.groupby("CELL_ID").agg(
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
        )
        cell_agg = combine_cell_agg(cell_agg, grp)

    sample_df = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    daily_agg = daily_agg.sort_values("DATE")

    cell_agg = cell_agg.reset_index().merge(attrs, on="CELL_ID", how="left")
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
    cell_agg["flow_per_user"] = cell_agg["flow_sum"] / cell_agg["user_sum"].replace(
        0, np.nan
    )
    cell_agg["peak_ratio"] = cell_agg["flow_max"] / cell_agg["flow_mean"]
    cell_agg["silent_ratio"] = cell_agg["silent_count"] / cell_agg["record_count"].replace(
        0, np.nan
    )

    flow_global = summarize_global(flow_stats)
    user_global = summarize_global(user_stats)

    high_load_threshold = cell_agg["flow_sum"].quantile(0.99)
    silent_threshold = 0.5
    high_load_cells = cell_agg[cell_agg["flow_sum"] >= high_load_threshold]
    silent_cells = cell_agg[cell_agg["silent_ratio"] >= silent_threshold]

    def save_fig(name: str) -> None:
        plt.tight_layout()
        plt.savefig(OUT_DIR / name, dpi=150)
        plt.close()

    if not sample_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(np.log1p(sample_df["FLOW_SUM"].dropna()), bins=60, ax=axes[0])
        axes[0].set_title("业务流量分布（log1p）")
        axes[0].set_xlabel("log(1+业务流量)")
        axes[0].set_ylabel("样本数量")
        sns.histplot(np.log1p(sample_df["USER_COUNT"].dropna()), bins=60, ax=axes[1])
        axes[1].set_title("用户数分布（log1p）")
        axes[1].set_xlabel("log(1+用户数)")
        axes[1].set_ylabel("样本数量")
        save_fig("fig01_flow_user_hist.png")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.boxplot(
            y=np.log1p(sample_df["FLOW_SUM"].dropna()),
            ax=axes[0],
            color="#4C78A8",
        )
        axes[0].set_title("业务流量箱线图（log1p）")
        axes[0].set_ylabel("log(1+业务流量)")
        sns.boxplot(
            y=np.log1p(sample_df["USER_COUNT"].dropna()),
            ax=axes[1],
            color="#F58518",
        )
        axes[1].set_title("用户数箱线图（log1p）")
        axes[1].set_ylabel("log(1+用户数)")
        save_fig("fig02_flow_user_box.png")

        plt.figure(figsize=(6, 5))
        plt.scatter(
            sample_df["USER_COUNT"],
            sample_df["FLOW_SUM"],
            s=6,
            alpha=0.3,
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.title("业务流量-用户数散点（log-log）")
        plt.xlabel("用户数")
        plt.ylabel("业务流量（MB）")
        save_fig("fig03_flow_user_scatter.png")

    plt.figure(figsize=(10, 4))
    plt.plot(daily_agg["DATE"], daily_agg["flow_sum"])
    plt.xticks(rotation=45)
    plt.title("全网日总流量走势")
    plt.xlabel("日期")
    plt.ylabel("业务流量（MB）")
    save_fig("fig04_daily_total_flow.png")

    plt.figure(figsize=(10, 4))
    plt.plot(daily_agg["DATE"], daily_agg["user_sum"])
    plt.xticks(rotation=45)
    plt.title("全网日总用户数走势")
    plt.xlabel("日期")
    plt.ylabel("用户数")
    save_fig("fig05_daily_total_user.png")

    top10_flow = cell_agg.sort_values("flow_sum", ascending=False).head(10)
    top10_flow_labels = (
        top10_flow["CELL_ID"].astype(str)
        + " | 场景"
        + top10_flow["SCENE"].fillna(-1).astype(int).astype(str)
    )
    plt.figure(figsize=(10, 4))
    sns.barplot(x=top10_flow_labels, y=top10_flow["flow_sum"], color="#4C78A8")
    plt.title("TOP10 小区总流量")
    plt.xlabel("小区ID | 场景")
    plt.ylabel("业务流量（MB）")
    plt.xticks(rotation=30, ha="right")
    save_fig("fig06_top10_flow.png")

    top10_fpu = cell_agg.sort_values("flow_per_user", ascending=False).head(10)
    top10_fpu_labels = (
        top10_fpu["CELL_ID"].astype(str)
        + " | 场景"
        + top10_fpu["SCENE"].fillna(-1).astype(int).astype(str)
    )
    plt.figure(figsize=(10, 4))
    sns.barplot(x=top10_fpu_labels, y=top10_fpu["flow_per_user"], color="#E45756")
    plt.title("TOP10 人均流量")
    plt.xlabel("小区ID | 场景")
    plt.ylabel("人均流量（MB/人）")
    plt.xticks(rotation=30, ha="right")
    save_fig("fig07_top10_flow_per_user.png")

    scene_counts = cell_agg["SCENE"].value_counts().head(10).index
    scene_df = cell_agg[cell_agg["SCENE"].isin(scene_counts)]

    plt.figure(figsize=(10, 4))
    sns.boxplot(x="SCENE", y="flow_mean", data=scene_df, color="#72B7B2")
    plt.title("不同场景的小区平均流量分布")
    plt.xlabel("场景（SCENE）")
    plt.ylabel("平均流量（MB）")
    save_fig("fig08_scene_flow_box.png")

    plt.figure(figsize=(10, 4))
    sns.boxplot(x="SCENE", y="user_mean", data=scene_df, color="#54A24B")
    plt.title("不同场景的小区平均用户数分布")
    plt.xlabel("场景（SCENE）")
    plt.ylabel("平均用户数")
    save_fig("fig09_scene_user_box.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(cell_agg["silent_ratio"].dropna(), bins=40, color="#B279A2")
    plt.title("小区静默比例分布")
    plt.xlabel("静默比例（有用户无流量）")
    plt.ylabel("小区数量")
    save_fig("fig10_silent_ratio_hist.png")

    if not silent_cells.empty:
        silent_scene = (
            silent_cells["SCENE"].value_counts().sort_values(ascending=False)
        )
        plt.figure(figsize=(8, 4))
        sns.barplot(
            x=silent_scene.index.astype(str), y=silent_scene.values, color="#FF9DA6"
        )
        plt.title("静默小区数量（按场景）")
        plt.xlabel("场景（SCENE）")
        plt.ylabel("静默小区数量")
        save_fig("fig11_silent_scene.png")

    if not high_load_cells.empty:
        high_scene = (
            high_load_cells["SCENE"].value_counts().sort_values(ascending=False)
        )
        plt.figure(figsize=(8, 4))
        sns.barplot(
            x=high_scene.index.astype(str), y=high_scene.values, color="#F28E2B"
        )
        plt.title("高负荷小区数量（按场景）")
        plt.xlabel("场景（SCENE）")
        plt.ylabel("高负荷小区数量")
        save_fig("fig12_highload_scene.png")

    stats_out = {
        "global": {
            "flow": flow_global,
            "user": user_global,
        },
        "cells": {
            "total_cells": int(cell_agg.shape[0]),
            "silent_threshold": silent_threshold,
            "silent_cells": int(silent_cells.shape[0]),
            "high_load_threshold": float(high_load_threshold),
            "high_load_cells": int(high_load_cells.shape[0]),
        },
        "derived_metrics": {
            "flow_mean": "小区平均业务流量 = flow_sum / flow_count",
            "user_mean": "小区平均用户数 = user_sum / user_count",
            "flow_per_user": "人均流量 = flow_sum / user_sum",
            "flow_cv": "流量变异系数 = flow_std / flow_mean",
            "silent_ratio": "静默比例 = silent_count / record_count",
            "peak_ratio": "峰均比 = flow_max / flow_mean",
        },
        "top10_flow": top10_flow[["CELL_ID", "flow_sum", "SCENE", "TYPE"]]
        .to_dict(orient="records"),
        "top10_flow_per_user": top10_fpu[
            ["CELL_ID", "flow_per_user", "SCENE", "TYPE"]
        ].to_dict(orient="records"),
    }

    with open(OUT_DIR / "section3_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)

    cell_agg.to_csv(OUT_DIR / "section3_cell_agg.csv", index=False)


if __name__ == "__main__":
    main()
