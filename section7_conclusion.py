# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = Path("all_final_data_with_attributes.csv")
OUT_DIR = Path("report_assets/section7")
CHUNK_SIZE = 1_000_000
SCENES = [2, 6]

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


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=150)
    plt.close()


def main() -> None:
    ensure_out_dir()
    set_cn_style()

    if not DATA_PATH.exists():
        raise FileNotFoundError("\u672a\u627e\u5230 all_final_data_with_attributes.csv")

    daily_flow: dict[date, float] = {}
    daily_user: dict[date, float] = {}
    hour_scene_sum = {scene: np.zeros(24, dtype=np.float64) for scene in SCENES}
    hour_scene_cnt = {scene: np.zeros(24, dtype=np.int64) for scene in SCENES}

    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=["DATETIME_KEY", "FLOW_SUM", "USER_COUNT", "SCENE"],
        dtype={
            "DATETIME_KEY": "string",
            "FLOW_SUM": "float32",
            "USER_COUNT": "float32",
            "SCENE": "float32",
        },
        chunksize=CHUNK_SIZE,
        na_values=["", "NA", "NaN"],
    ):
        chunk.loc[chunk["FLOW_SUM"] < 0, "FLOW_SUM"] = np.nan
        chunk.loc[chunk["USER_COUNT"] < 0, "USER_COUNT"] = np.nan
        dt_values = pd.to_datetime(chunk["DATETIME_KEY"], errors="coerce")
        chunk["date"] = dt_values.dt.date
        chunk["hour"] = dt_values.dt.hour

        daily = (
            chunk.groupby("date", as_index=True)[["FLOW_SUM", "USER_COUNT"]]
            .sum(min_count=1)
            .dropna(how="all")
        )
        for idx, row in daily.iterrows():
            if pd.isna(idx):
                continue
            daily_flow[idx] = daily_flow.get(idx, 0.0) + float(row["FLOW_SUM"])
            daily_user[idx] = daily_user.get(idx, 0.0) + float(row["USER_COUNT"])

        scene_chunk = chunk[chunk["SCENE"].isin(SCENES)].dropna(subset=["hour"])
        if not scene_chunk.empty:
            grp = scene_chunk.groupby(["SCENE", "hour"])["FLOW_SUM"].agg(["sum", "count"])
            for (scene, hour), row in grp.iterrows():
                hour_scene_sum[int(scene)][int(hour)] += float(row["sum"])
                hour_scene_cnt[int(scene)][int(hour)] += int(row["count"])

    daily_df = pd.DataFrame(
        {
            "date": list(daily_flow.keys()),
            "flow_sum": list(daily_flow.values()),
            "user_sum": [daily_user[k] for k in daily_flow.keys()],
        }
    )
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date")
    if daily_df.empty:
        raise ValueError("\u65e5\u7ea7\u7edf\u8ba1\u4e3a\u7a7a")

    full_dates = pd.date_range(daily_df["date"].min(), daily_df["date"].max(), freq="D")
    daily_df = daily_df.set_index("date").reindex(full_dates).fillna(0).rename_axis("date").reset_index()
    daily_df["flow_per_user"] = daily_df["flow_sum"] / daily_df["user_sum"].replace(0, np.nan)

    hours = np.arange(24)
    plt.figure(figsize=(8, 4))
    for scene in SCENES:
        mean_vals = hour_scene_sum[scene] / np.maximum(hour_scene_cnt[scene], 1)
        plt.plot(hours, mean_vals, marker="o", label=f"\u573a\u666f {scene}")
    plt.title("\u5178\u578b 24 \u5c0f\u65f6\u6d41\u91cf\u66f2\u7ebf\uff08\u573a\u666f\u5bf9\u6bd4\uff09")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u5e73\u5747\u6d41\u91cf\uff08MB\uff09")
    plt.legend()
    save_fig("fig01_diurnal_scene_2_6.png")

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(daily_df["date"], daily_df["flow_sum"], color="#4C78A8", label="\u6d41\u91cf")
    ax1.set_xlabel("\u65e5\u671f")
    ax1.set_ylabel("\u4e1a\u52a1\u6d41\u91cf\uff08MB\uff09", color="#4C78A8")
    ax2 = ax1.twinx()
    ax2.plot(daily_df["date"], daily_df["user_sum"], color="#F58518", label="\u7528\u6237\u6570")
    ax2.set_ylabel("\u7528\u6237\u6570", color="#F58518")
    plt.title("\u53cc\u8f74\u8d8b\u52bf\u56fe\uff08\u6d41\u91cf vs \u7528\u6237\u6570\uff09")
    save_fig("fig02_dual_axis_daily.png")

    daily_df["week_idx"] = ((daily_df["date"] - daily_df["date"].min()).dt.days // 7).astype(int)
    daily_df["weekday"] = daily_df["date"].dt.weekday
    heat = daily_df.pivot(index="week_idx", columns="weekday", values="flow_sum")
    plt.figure(figsize=(10, 4))
    sns.heatmap(heat, cmap="YlOrRd")
    plt.title("\u65e5\u5386\u70ed\u529b\u56fe\uff08\u65e5\u603b\u6d41\u91cf\uff09")
    plt.xlabel("\u661f\u671f")
    plt.ylabel("\u5468\u5e8f\u53f7")
    plt.xticks(ticks=np.arange(0.5, 7.5, 1.0), labels=["\u4e00", "\u4e8c", "\u4e09", "\u56db", "\u4e94", "\u516d", "\u65e5"])
    save_fig("fig03_calendar_heatmap_flow.png")

    plt.figure(figsize=(10, 4))
    plt.plot(daily_df["date"], daily_df["flow_per_user"], color="#72B7B2")
    plt.title("\u65e5\u7ea7\u4eba\u5747\u6d41\u91cf\u8d8b\u52bf")
    plt.xlabel("\u65e5\u671f")
    plt.ylabel("\u4eba\u5747\u6d41\u91cf\uff08MB/\u4eba\uff09")
    save_fig("fig04_daily_flow_per_user.png")

    plt.figure(figsize=(6, 5))
    plt.scatter(daily_df["user_sum"], daily_df["flow_sum"], s=18, alpha=0.6)
    plt.title("\u65e5\u7ea7\u6d41\u91cf-\u7528\u6237\u6570\u6563\u70b9")
    plt.xlabel("\u7528\u6237\u6570")
    plt.ylabel("\u6d41\u91cf\uff08MB\uff09")
    save_fig("fig05_daily_flow_user_scatter.png")

    daily_df["holiday"] = daily_df["date"].dt.date.isin(HOLIDAYS_2021)
    holiday_stats = (
        daily_df.groupby("holiday")[["flow_sum", "user_sum"]]
        .mean()
        .rename(index={False: "\u975e\u8282\u5047\u65e5", True: "\u8282\u5047\u65e5"})
    )
    holiday_melt = holiday_stats.reset_index().melt(
        id_vars="holiday",
        value_vars=["flow_sum", "user_sum"],
        var_name="metric",
        value_name="value",
    )
    holiday_melt["metric"] = holiday_melt["metric"].map(
        {"flow_sum": "\u6d41\u91cf", "user_sum": "\u7528\u6237\u6570"}
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x="metric",
        y="value",
        hue="holiday",
        data=holiday_melt,
        palette=["#4C78A8", "#E45756"],
    )
    plt.title("\u8282\u5047\u65e5\u4e0e\u975e\u8282\u5047\u65e5\u5e73\u5747\u6c34\u5e73\u5bf9\u6bd4")
    plt.xlabel("\u6307\u6807")
    plt.ylabel("\u5e73\u5747\u503c")
    save_fig("fig06_holiday_vs_nonholiday.png")

    daily_df["flow_change_pct"] = daily_df["flow_sum"].pct_change().fillna(0) * 100
    plt.figure(figsize=(10, 4))
    plt.plot(daily_df["date"], daily_df["flow_change_pct"], color="#E45756")
    plt.axhline(0, color="#333333", linewidth=1)
    plt.title("\u65e5\u6d41\u91cf\u53d8\u5316\u7387")
    plt.xlabel("\u65e5\u671f")
    plt.ylabel("\u53d8\u5316\u7387\uff08%\uff09")
    save_fig("fig07_daily_flow_change.png")


if __name__ == "__main__":
    main()
