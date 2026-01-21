# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import date
from pathlib import Path

import json
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
CHUXI = date(2021, 2, 11)


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

def series_to_float_dict(series: pd.Series) -> dict[int, float]:
    return {int(k): float(v) for k, v in series.items()}

def sanitize_json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(v) for v in value]
    return value


def main() -> None:
    ensure_out_dir()
    set_cn_style()

    if not DATA_PATH.exists():
        raise FileNotFoundError("\u672a\u627e\u5230 all_final_data_with_attributes.csv")

    daily_flow: dict[date, float] = {}
    daily_user: dict[date, float] = {}
    hour_scene_sum = {scene: np.zeros(24, dtype=np.float64) for scene in SCENES}
    hour_scene_cnt = {scene: np.zeros(24, dtype=np.int64) for scene in SCENES}
    weekday_flow_sum = np.zeros((5, 24), dtype=np.float64)
    weekday_flow_cnt = np.zeros((5, 24), dtype=np.int64)
    weekday_user_sum = np.zeros((5, 24), dtype=np.float64)
    weekday_user_cnt = np.zeros((5, 24), dtype=np.int64)
    scene_weekday_sum: dict[int, np.ndarray] = {}
    scene_weekday_cnt: dict[int, np.ndarray] = {}

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
        chunk["weekday"] = dt_values.dt.weekday

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

        workday_mask = (chunk["weekday"] < 5) & (~chunk["date"].isin(HOLIDAYS_2021))
        workday_chunk = chunk.loc[workday_mask].dropna(subset=["hour"])
        if not workday_chunk.empty:
            grp = workday_chunk.groupby(["weekday", "hour"]).agg(
                flow_sum=("FLOW_SUM", "sum"),
                flow_count=("FLOW_SUM", "count"),
                user_sum=("USER_COUNT", "sum"),
                user_count=("USER_COUNT", "count"),
            )
            for (weekday, hour), row in grp.iterrows():
                w_idx = int(weekday)
                h_idx = int(hour)
                if 0 <= w_idx < 5 and 0 <= h_idx < 24:
                    weekday_flow_sum[w_idx, h_idx] += float(row["flow_sum"])
                    weekday_flow_cnt[w_idx, h_idx] += int(row["flow_count"])
                    weekday_user_sum[w_idx, h_idx] += float(row["user_sum"])
                    weekday_user_cnt[w_idx, h_idx] += int(row["user_count"])

        scene_daily = (
            chunk.dropna(subset=["SCENE", "date", "weekday"])
            .groupby(["SCENE", "date", "weekday"])["FLOW_SUM"]
            .sum(min_count=1)
            .dropna()
        )
        if not scene_daily.empty:
            for (scene, _, weekday), flow_val in scene_daily.items():
                scene_key = int(scene)
                if scene_key not in scene_weekday_sum:
                    scene_weekday_sum[scene_key] = np.zeros(7, dtype=np.float64)
                    scene_weekday_cnt[scene_key] = np.zeros(7, dtype=np.int64)
                scene_weekday_sum[scene_key][int(weekday)] += float(flow_val)
                scene_weekday_cnt[scene_key][int(weekday)] += 1

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
    daily_df["weekday"] = daily_df["date"].dt.weekday
    daily_df["holiday"] = daily_df["date"].dt.date.isin(HOLIDAYS_2021)

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

    thursday_all = daily_df[daily_df["weekday"] == 3]
    thursday_wo_chuxi = thursday_all[thursday_all["date"].dt.date != CHUXI]
    mon_wed = daily_df[(daily_df["weekday"] >= 0) & (daily_df["weekday"] <= 2)]
    th_flow_all = float(thursday_all["flow_sum"].mean())
    th_flow_wo = float(thursday_wo_chuxi["flow_sum"].mean())
    th_user_all = float(thursday_all["user_sum"].mean())
    th_user_wo = float(thursday_wo_chuxi["user_sum"].mean())
    mon_wed_flow_mean_val = float(mon_wed["flow_sum"].mean())
    mon_wed_user_mean_val = float(mon_wed["user_sum"].mean())

    plt.figure(figsize=(7, 4))
    plt.bar(
        ["\u5468\u56db\u5168\u90e8", "\u5264\u9664\u9664\u5915", "\u5468\u4e00-\u5468\u4e09\u5747\u503c"],
        [th_flow_all, th_flow_wo, mon_wed_flow_mean_val],
        color=["#4C78A8", "#72B7B2", "#E45756"],
    )
    plt.title("\u5468\u56db\u65e5\u5747\u6d41\u91cf\uff08\u662f\\u5426\u5264\u9664\\u9664\u5915\uff09")
    plt.xlabel("\u5bf9\u6bd4\u7ec4")
    plt.ylabel("\u5e73\u5747\u6d41\u91cf\uff08MB\uff09")
    save_fig("fig13_thursday_exclusion_flow.png")

    plt.figure(figsize=(7, 4))
    plt.bar(
        ["\u5468\u56db\u5168\u90e8", "\u5264\u9664\u9664\u5915", "\u5468\u4e00-\u5468\u4e09\u5747\u503c"],
        [th_user_all, th_user_wo, mon_wed_user_mean_val],
        color=["#4C78A8", "#72B7B2", "#E45756"],
    )
    plt.title("\u5468\u56db\u65e5\u5747\u7528\u6237\u6570\uff08\u662f\\u5426\u5264\u9664\\u9664\u5915\uff09")
    plt.xlabel("\u5bf9\u6bd4\u7ec4")
    plt.ylabel("\u5e73\u5747\u7528\u6237\u6570")
    save_fig("fig14_thursday_exclusion_user.png")

    daily_excl_chuxi = daily_df[daily_df["date"].dt.date != CHUXI]
    weekday_flow_excl = (
        daily_excl_chuxi.groupby("weekday")["flow_sum"].mean().reindex(range(7))
    )
    weekday_user_excl = (
        daily_excl_chuxi.groupby("weekday")["user_sum"].mean().reindex(range(7))
    )
    weekday_flow_all = daily_df.groupby("weekday")["flow_sum"].mean().reindex(range(7))
    weekday_user_all = daily_df.groupby("weekday")["user_sum"].mean().reindex(range(7))
    weekday_labels_full = [
        "\u5468\u4e00",
        "\u5468\u4e8c",
        "\u5468\u4e09",
        "\u5468\u56db",
        "\u5468\u4e94",
        "\u5468\u516d",
        "\u5468\u65e5",
    ]
    flow_colors = ["#4C78A8"] * 7
    user_colors = ["#4C78A8"] * 7
    flow_colors[3] = "#E45756"
    user_colors[3] = "#E45756"
    flow_excl_base = float(weekday_flow_excl.mean())
    user_excl_base = float(weekday_user_excl.mean())
    flow_all_base = float(weekday_flow_all.mean())
    user_all_base = float(weekday_user_all.mean())
    flow_excl_ratio = weekday_flow_excl / flow_excl_base if flow_excl_base else weekday_flow_excl
    user_excl_ratio = weekday_user_excl / user_excl_base if user_excl_base else weekday_user_excl
    flow_all_ratio = weekday_flow_all / flow_all_base if flow_all_base else weekday_flow_all
    user_all_ratio = weekday_user_all / user_all_base if user_all_base else weekday_user_all
    flow_excl_rel = (flow_excl_ratio - 1) * 100
    user_excl_rel = (user_excl_ratio - 1) * 100
    flow_all_rel = (flow_all_ratio - 1) * 100
    user_all_rel = (user_all_ratio - 1) * 100

    plt.figure(figsize=(8, 4))
    plt.bar(weekday_labels_full, flow_excl_rel.values, color=flow_colors)
    plt.axhline(0.0, color="#333333", linewidth=1)
    plt.title("\u5264\u9664\u9664\u5915\u540e\u5468\u5185\u65e5\u5747\u6d41\u91cf\uff08\u76f8\u5bf9\u5747\u503c\u504f\u79fb\uff0c\u542b\u5468\u672b\uff09")
    plt.xlabel("\u661f\u671f")
    plt.ylabel("\u76f8\u5bf9\u5747\u503c\u504f\u79fb\uff08%\uff09")
    save_fig("fig17_weekday_mean_flow_excl_chuxi.png")

    plt.figure(figsize=(8, 4))
    plt.bar(weekday_labels_full, user_excl_rel.values, color=user_colors)
    plt.axhline(0.0, color="#333333", linewidth=1)
    plt.title("\u5264\u9664\u9664\u5915\u540e\u5468\u5185\u65e5\u5747\u7528\u6237\u6570\uff08\u76f8\u5bf9\u5747\u503c\u504f\u79fb\uff0c\u542b\u5468\u672b\uff09")
    plt.xlabel("\u661f\u671f")
    plt.ylabel("\u76f8\u5bf9\u5747\u503c\u504f\u79fb\uff08%\uff09")
    save_fig("fig18_weekday_mean_user_excl_chuxi.png")

    plt.figure(figsize=(8, 4))
    plt.bar(weekday_labels_full, flow_all_rel.values, color=flow_colors)
    plt.axhline(0.0, color="#333333", linewidth=1)
    plt.title("\u672a\u5254\u9664\u9664\u5915\u7684\u5468\u5185\u65e5\u5747\u6d41\u91cf\uff08\u76f8\u5bf9\u5747\u503c\u504f\u79fb\uff0c\u542b\u5468\u672b\uff09")
    plt.xlabel("\u661f\u671f")
    plt.ylabel("\u76f8\u5bf9\u5747\u503c\u504f\u79fb\uff08%\uff09")
    save_fig("fig19_weekday_mean_flow_with_chuxi.png")

    plt.figure(figsize=(8, 4))
    plt.bar(weekday_labels_full, user_all_rel.values, color=user_colors)
    plt.axhline(0.0, color="#333333", linewidth=1)
    plt.title("\u672a\u5254\u9664\u9664\u5915\u7684\u5468\u5185\u65e5\u5747\u7528\u6237\u6570\uff08\u76f8\u5bf9\u5747\u503c\u504f\u79fb\uff0c\u542b\u5468\u672b\uff09")
    plt.xlabel("\u661f\u671f")
    plt.ylabel("\u76f8\u5bf9\u5747\u503c\u504f\u79fb\uff08%\uff09")
    save_fig("fig20_weekday_mean_user_with_chuxi.png")

    scene_records = []
    for scene_key, sums in scene_weekday_sum.items():
        counts = scene_weekday_cnt[scene_key]
        mean_vals = sums / np.where(counts == 0, np.nan, counts)
        weekly_mean = float(np.nanmean(mean_vals)) if np.isfinite(mean_vals).any() else float("nan")
        thursday_mean = float(mean_vals[3]) if np.isfinite(mean_vals[3]) else float("nan")
        if not np.isfinite(weekly_mean) or weekly_mean == 0:
            continue
        retention = thursday_mean / weekly_mean
        scene_records.append(
            {
                "scene": scene_key,
                "weekly_mean": weekly_mean,
                "thursday_mean": thursday_mean,
                "retention": retention,
            }
        )
    scene_df = pd.DataFrame(scene_records)
    scene_df = scene_df.sort_values("weekly_mean", ascending=False)
    top_scene_df = scene_df.head(12).sort_values("retention")
    colors = []
    for scene_key, retention in zip(top_scene_df["scene"], top_scene_df["retention"]):
        if scene_key == 2:
            colors.append("#F28E2B")
        else:
            colors.append("#E45756" if retention < 1 else "#54A24B")
    plt.figure(figsize=(9, 4))
    plt.bar(top_scene_df["scene"].astype(str), top_scene_df["retention"], color=colors)
    plt.axhline(1.0, color="#333333", linewidth=1)
    plt.title("\u573a\u666f\u5468\u56db\u6d41\u91cf\u4fdd\u7559\u7387\uff08\u5468\u56db/\u5468\u5747\uff09")
    plt.xlabel("\u573a\u666f\uff08SCENE\uff09")
    plt.ylabel("\u4fdd\u7559\u7387")
    save_fig("fig15_scene_thursday_retention.png")

    weekday_flow_per_user = (
        daily_df.groupby("weekday")["flow_per_user"].mean().reindex(range(7))
    )
    plt.figure(figsize=(8, 4))
    plt.plot(range(7), weekday_flow_per_user.values, marker="o", color="#4C78A8")
    plt.xticks(
        ticks=range(7),
        labels=["\u5468\u4e00", "\u5468\u4e8c", "\u5468\u4e09", "\u5468\u56db", "\u5468\u4e94", "\u5468\u516d", "\u5468\u65e5"],
    )
    plt.title("\u5468\u4e00\u81f3\u5468\u65e5\u4eba\u5747\u6d41\u91cf\u8d8b\u52bf")
    plt.xlabel("\u661f\u671f")
    plt.ylabel("\u4eba\u5747\u6d41\u91cf\uff08MB/\u4eba\uff09")
    save_fig("fig16_weekday_flow_per_user.png")

    workday_df = daily_df[(daily_df["weekday"] < 5) & (~daily_df["holiday"])]
    weekday_flow_mean = workday_df.groupby("weekday")["flow_sum"].mean()
    weekday_user_mean = workday_df.groupby("weekday")["user_sum"].mean()
    weekday_labels = ["\u5468\u4e00", "\u5468\u4e8c", "\u5468\u4e09", "\u5468\u56db", "\u5468\u4e94"]

    plt.figure(figsize=(8, 4))
    colors = ["#4C78A8", "#4C78A8", "#4C78A8", "#E45756", "#4C78A8"]
    plt.bar(weekday_labels, weekday_flow_mean.values, color=colors)
    plt.title("\u5de5\u4f5c\u65e5\u65e5\u5747\u6d41\u91cf\uff08\u975e\u8282\u5047\u65e5\uff09")
    plt.xlabel("\u661f\u671f")
    plt.ylabel("\u5e73\u5747\u6d41\u91cf\uff08MB\uff09")
    save_fig("fig08_weekday_flow_bar.png")

    plt.figure(figsize=(8, 4))
    plt.bar(weekday_labels, weekday_user_mean.values, color=colors)
    plt.title("\u5de5\u4f5c\u65e5\u65e5\u5747\u7528\u6237\u6570\uff08\u975e\u8282\u5047\u65e5\uff09")
    plt.xlabel("\u661f\u671f")
    plt.ylabel("\u5e73\u5747\u7528\u6237\u6570")
    save_fig("fig09_weekday_user_bar.png")

    flow_profile = weekday_flow_sum / np.maximum(weekday_flow_cnt, 1)
    user_profile = weekday_user_sum / np.maximum(weekday_user_cnt, 1)
    mon_wed_flow = flow_profile[0:3].mean(axis=0)
    mon_wed_user = user_profile[0:3].mean(axis=0)
    thu_flow = flow_profile[3]
    fri_flow = flow_profile[4]
    thu_user = user_profile[3]
    fri_user = user_profile[4]

    hours = np.arange(24)
    plt.figure(figsize=(9, 4))
    plt.plot(hours, mon_wed_flow, label="\u5468\u4e00-\u5468\u4e09", color="#4C78A8")
    plt.plot(hours, thu_flow, label="\u5468\u56db", color="#E45756")
    plt.plot(hours, fri_flow, label="\u5468\u4e94", color="#72B7B2")
    plt.title("\u5de5\u4f5c\u65e5\u65f6\u6bb5\u6d41\u91cf\u5bf9\u6bd4\uff08\u975e\u8282\u5047\u65e5\uff09")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u5e73\u5747\u6d41\u91cf\uff08MB\uff09")
    plt.legend()
    save_fig("fig10_weekday_hour_flow.png")

    plt.figure(figsize=(9, 4))
    plt.plot(hours, mon_wed_user, label="\u5468\u4e00-\u5468\u4e09", color="#4C78A8")
    plt.plot(hours, thu_user, label="\u5468\u56db", color="#E45756")
    plt.plot(hours, fri_user, label="\u5468\u4e94", color="#72B7B2")
    plt.title("\u5de5\u4f5c\u65e5\u65f6\u6bb5\u7528\u6237\u6570\u5bf9\u6bd4\uff08\u975e\u8282\u5047\u65e5\uff09")
    plt.xlabel("\u5c0f\u65f6")
    plt.ylabel("\u5e73\u5747\u7528\u6237\u6570")
    plt.legend()
    save_fig("fig11_weekday_hour_user.png")

    midday_hours = np.arange(10, 18)
    evening_hours = np.arange(18, 23)

    def mean_slice(profile: np.ndarray, hours_slice: np.ndarray) -> float:
        vals = profile[hours_slice]
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    def peak_duration(profile: np.ndarray) -> int:
        sub = profile[evening_hours]
        peak = np.nanmax(sub)
        if not np.isfinite(peak) or peak <= 0:
            return 0
        return int((sub >= peak * 0.9).sum())

    duration_vals = [peak_duration(mon_wed_flow), peak_duration(thu_flow), peak_duration(fri_flow)]
    plt.figure(figsize=(6, 4))
    plt.bar(["\u5468\u4e00-\u5468\u4e09", "\u5468\u56db", "\u5468\u4e94"], duration_vals, color=["#4C78A8", "#E45756", "#72B7B2"])
    plt.title("\u665a\u9ad8\u5cf0\u5f3a\u5ea6\u6301\u7eed\u65f6\u957f\uff08>=90% \u5cf0\u503c\uff09")
    plt.xlabel("\u5de5\u4f5c\u65e5")
    plt.ylabel("\u65f6\u957f\uff08\u5c0f\u65f6\uff09")
    save_fig("fig12_weekday_peak_duration.png")

    stats_out = {
        "weekday_flow_mean": series_to_float_dict(weekday_flow_mean),
        "weekday_user_mean": series_to_float_dict(weekday_user_mean),
        "thu_flow_rank": int((weekday_flow_mean.rank(ascending=True).loc[3])),
        "thu_user_rank": int((weekday_user_mean.rank(ascending=True).loc[3])),
        "midday_hours": [10, 17],
        "evening_hours": [18, 22],
        "midday_flow_mean": {
            "mon_wed": mean_slice(mon_wed_flow, midday_hours),
            "thu": mean_slice(thu_flow, midday_hours),
            "fri": mean_slice(fri_flow, midday_hours),
        },
        "midday_user_mean": {
            "mon_wed": mean_slice(mon_wed_user, midday_hours),
            "thu": mean_slice(thu_user, midday_hours),
            "fri": mean_slice(fri_user, midday_hours),
        },
        "evening_flow_mean": {
            "mon_wed": mean_slice(mon_wed_flow, evening_hours),
            "thu": mean_slice(thu_flow, evening_hours),
            "fri": mean_slice(fri_flow, evening_hours),
        },
        "evening_user_mean": {
            "mon_wed": mean_slice(mon_wed_user, evening_hours),
            "thu": mean_slice(thu_user, evening_hours),
            "fri": mean_slice(fri_user, evening_hours),
        },
        "peak_duration_hours": {
            "mon_wed": duration_vals[0],
            "thu": duration_vals[1],
            "fri": duration_vals[2],
        },
        "thursday_exclusion": {
            "thursday_flow_all": th_flow_all,
            "thursday_flow_wo_chuxi": th_flow_wo,
            "thursday_user_all": th_user_all,
            "thursday_user_wo_chuxi": th_user_wo,
            "mon_wed_flow_mean": mon_wed_flow_mean_val,
            "mon_wed_user_mean": mon_wed_user_mean_val,
        },
        "scene_thursday_retention": scene_df.set_index("scene")[
            ["weekly_mean", "thursday_mean", "retention"]
        ].to_dict(orient="index"),
        "weekday_flow_per_user": series_to_float_dict(weekday_flow_per_user),
        "weekday_flow_mean_excl_chuxi": series_to_float_dict(weekday_flow_excl),
        "weekday_user_mean_excl_chuxi": series_to_float_dict(weekday_user_excl),
        "weekday_flow_mean_all": series_to_float_dict(weekday_flow_all),
        "weekday_user_mean_all": series_to_float_dict(weekday_user_all),
    }
    stats_out = sanitize_json_value(stats_out)
    with open(OUT_DIR / "section7_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
