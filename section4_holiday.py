# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = Path("all_final_data_with_attributes_holiday.csv")
CELL_AGG_PATH = Path("report_assets/section3/section3_cell_agg.csv")
OUT_DIR = Path("report_assets/section4")
CHUNK_SIZE = 1_000_000

SPRING_START = date(2021, 2, 11)
SPRING_END = date(2021, 2, 17)
CHUXI = date(2021, 2, 11)
CHUYI = date(2021, 2, 12)
NORMAL_SUN = date(2021, 3, 7)


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
    attr_cols: list[str],
) -> pd.DataFrame:
    if base is None:
        return add
    base = base.reindex(base.index.union(add.index))
    for col in sum_cols:
        base[col] = base[col].fillna(0) + add[col].reindex(base.index).fillna(0)
    for col in attr_cols:
        base[col] = base[col].fillna(add[col].reindex(base.index))
    return base


def pick_target_cell() -> int:
    if CELL_AGG_PATH.exists():
        cell_agg = pd.read_csv(CELL_AGG_PATH)
        top = cell_agg.sort_values("flow_sum", ascending=False).head(1)
        if not top.empty:
            return int(top.iloc[0]["CELL_ID"])
    return 0


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
    ax.set_title("\u76ee\u6807\u5c0f\u533a\u8282\u5047\u65e5\u6307\u6807\u5bf9\u6bd4")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))


def main() -> None:
    ensure_out_dir()
    set_cn_style()

    if not DATA_PATH.exists():
        raise FileNotFoundError("all_final_data_with_attributes_holiday.csv not found")

    target_cell = pick_target_cell()
    target_dates = {CHUXI, CHUYI, NORMAL_SUN}

    scene_stats = {"spring": None, "march_weekday": None}
    daily_agg: pd.DataFrame | None = None
    march20_cells: pd.DataFrame | None = None
    chuxi20_cells: pd.DataFrame | None = None

    target_flow = {d: [] for d in target_dates}
    target_user = {d: [] for d in target_dates}

    usecols = [
        "CELL_ID",
        "DATETIME_KEY",
        "FLOW_SUM",
        "USER_COUNT",
        "LATITUDE",
        "LONGITUDE",
        "TYPE",
        "SCENE",
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
    }

    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=usecols,
        dtype=dtype,
        chunksize=CHUNK_SIZE,
        na_values=["", "NA", "NaN"],
    ):
        chunk.loc[chunk["FLOW_SUM"] < 0, "FLOW_SUM"] = np.nan
        chunk.loc[chunk["USER_COUNT"] < 0, "USER_COUNT"] = np.nan

        dt_values = pd.to_datetime(chunk["DATETIME_KEY"], errors="coerce")
        valid = dt_values.notna()
        if not valid.any():
            continue

        chunk = chunk.loc[valid].copy()
        dt_values = dt_values.loc[valid]
        chunk["date"] = dt_values.dt.date
        chunk["hour"] = dt_values.dt.hour
        chunk["weekday"] = dt_values.dt.weekday

        daily = chunk.groupby("date")[["FLOW_SUM", "USER_COUNT"]].sum(min_count=1)
        daily_agg = combine_sum(daily_agg, daily)

        spring_mask = (chunk["date"] >= SPRING_START) & (chunk["date"] <= SPRING_END)
        if spring_mask.any():
            spring_df = chunk.loc[spring_mask]
            spring_scene = spring_df.groupby("SCENE").agg(
                flow_sum=("FLOW_SUM", "sum"),
                user_sum=("USER_COUNT", "sum"),
                count=("FLOW_SUM", "count"),
            )
            scene_stats["spring"] = combine_sum(scene_stats["spring"], spring_scene)

        march_mask = (
            (dt_values.dt.year == 2021)
            & (dt_values.dt.month == 3)
            & (chunk["weekday"] < 5)
        )
        if march_mask.any():
            march_df = chunk.loc[march_mask]
            march_scene = march_df.groupby("SCENE").agg(
                flow_sum=("FLOW_SUM", "sum"),
                user_sum=("USER_COUNT", "sum"),
                count=("FLOW_SUM", "count"),
            )
            scene_stats["march_weekday"] = combine_sum(
                scene_stats["march_weekday"], march_scene
            )

        march20_mask = march_mask & (chunk["hour"] == 20)
        if march20_mask.any():
            m20 = chunk.loc[
                march20_mask, ["CELL_ID", "FLOW_SUM", "LATITUDE", "LONGITUDE", "SCENE"]
            ]
            m20_grp = m20.groupby("CELL_ID").agg(
                flow_sum=("FLOW_SUM", "sum"),
                count=("FLOW_SUM", "count"),
                LATITUDE=("LATITUDE", "first"),
                LONGITUDE=("LONGITUDE", "first"),
                SCENE=("SCENE", "first"),
            )
            march20_cells = combine_cell_agg(
                march20_cells, m20_grp, sum_cols=["flow_sum", "count"], attr_cols=["LATITUDE", "LONGITUDE", "SCENE"]
            )

        chuxi20_mask = (chunk["date"] == CHUXI) & (chunk["hour"] == 20)
        if chuxi20_mask.any():
            c20 = chunk.loc[
                chuxi20_mask, ["CELL_ID", "FLOW_SUM", "LATITUDE", "LONGITUDE", "SCENE"]
            ]
            c20_grp = c20.groupby("CELL_ID").agg(
                flow_sum=("FLOW_SUM", "sum"),
                count=("FLOW_SUM", "count"),
                LATITUDE=("LATITUDE", "first"),
                LONGITUDE=("LONGITUDE", "first"),
                SCENE=("SCENE", "first"),
            )
            chuxi20_cells = combine_cell_agg(
                chuxi20_cells, c20_grp, sum_cols=["flow_sum", "count"], attr_cols=["LATITUDE", "LONGITUDE", "SCENE"]
            )

        if target_cell is not None:
            target_mask = (chunk["CELL_ID"] == target_cell) & (
                chunk["date"].isin(target_dates)
            )
            if target_mask.any():
                sub = chunk.loc[target_mask, ["date", "FLOW_SUM", "USER_COUNT"]]
                for d, grp in sub.groupby("date"):
                    target_flow[d].extend(grp["FLOW_SUM"].dropna().tolist())
                    target_user[d].extend(grp["USER_COUNT"].dropna().tolist())

    spring_df = scene_stats["spring"].copy()
    march_df = scene_stats["march_weekday"].copy()
    spring_df["flow_mean"] = spring_df["flow_sum"] / spring_df["count"].replace(0, np.nan)
    spring_df["user_mean"] = spring_df["user_sum"] / spring_df["count"].replace(0, np.nan)
    march_df["flow_mean"] = march_df["flow_sum"] / march_df["count"].replace(0, np.nan)
    march_df["user_mean"] = march_df["user_sum"] / march_df["count"].replace(0, np.nan)

    scene_compare = pd.DataFrame(
        {
            "spring_flow": spring_df["flow_mean"],
            "march_flow": march_df["flow_mean"],
            "spring_user": spring_df["user_mean"],
            "march_user": march_df["user_mean"],
        }
    ).fillna(0)
    scene_compare["flow_change_pct"] = (
        scene_compare["spring_flow"] / scene_compare["march_flow"].replace(0, np.nan) - 1
    ) * 100
    scene_compare["user_change_pct"] = (
        scene_compare["spring_user"] / scene_compare["march_user"].replace(0, np.nan) - 1
    ) * 100
    scene_compare = scene_compare.replace([np.inf, -np.inf], np.nan).fillna(0)

    top_scenes = (
        scene_compare["march_flow"].sort_values(ascending=False).head(10).index
    )
    top_scene_df = scene_compare.loc[top_scenes].copy()

    x = np.arange(len(top_scene_df))
    width = 0.35
    plt.figure(figsize=(9, 4))
    plt.bar(x - width / 2, top_scene_df["spring_flow"], width, label="\u6625\u8282")
    plt.bar(x + width / 2, top_scene_df["march_flow"], width, label="3\u6708\u5de5\u4f5c\u65e5")
    plt.xticks(x, [int(s) for s in top_scene_df.index], rotation=30)
    plt.title("\u4e0d\u540c\u573a\u666f\u5e73\u5747\u6d41\u91cf\uff08\u6625\u8282 vs 3\u6708\u5de5\u4f5c\u65e5\uff09")
    plt.xlabel("SCENE")
    plt.ylabel("\u5e73\u5747\u6d41\u91cf\uff08MB\uff09")
    plt.legend()
    save_fig("fig01_scene_flow_grouped.png")

    plt.figure(figsize=(9, 4))
    plt.bar(x - width / 2, top_scene_df["spring_user"], width, label="\u6625\u8282")
    plt.bar(x + width / 2, top_scene_df["march_user"], width, label="3\u6708\u5de5\u4f5c\u65e5")
    plt.xticks(x, [int(s) for s in top_scene_df.index], rotation=30)
    plt.title("\u4e0d\u540c\u573a\u666f\u5e73\u5747\u7528\u6237\u6570\uff08\u6625\u8282 vs 3\u6708\u5de5\u4f5c\u65e5\uff09")
    plt.xlabel("SCENE")
    plt.ylabel("\u5e73\u5747\u7528\u6237\u6570")
    plt.legend()
    save_fig("fig02_scene_user_grouped.png")

    flow_change = scene_compare["flow_change_pct"].copy()
    flow_change = flow_change.reindex(flow_change.abs().sort_values(ascending=False).head(10).index)
    plt.figure(figsize=(9, 4))
    colors = ["#E45756" if v < 0 else "#54A24B" for v in flow_change.values]
    plt.bar([int(s) for s in flow_change.index], flow_change.values, color=colors)
    plt.axhline(0, color="#333333", linewidth=1)
    plt.title("\u573a\u666f\u6d41\u91cf\u53d8\u5316\uff08\u6625\u8282 vs 3\u6708\u5de5\u4f5c\u65e5\uff09")
    plt.xlabel("SCENE")
    plt.ylabel("\u53d8\u5316\u6bd4\u4f8b\uff08%\uff09")
    save_fig("fig03_scene_flow_change.png")

    user_change = scene_compare["user_change_pct"].copy()
    user_change = user_change.reindex(user_change.abs().sort_values(ascending=False).head(10).index)
    plt.figure(figsize=(9, 4))
    colors = ["#E45756" if v < 0 else "#54A24B" for v in user_change.values]
    plt.bar([int(s) for s in user_change.index], user_change.values, color=colors)
    plt.axhline(0, color="#333333", linewidth=1)
    plt.title("\u573a\u666f\u7528\u6237\u6570\u53d8\u5316\uff08\u6625\u8282 vs 3\u6708\u5de5\u4f5c\u65e5\uff09")
    plt.xlabel("SCENE")
    plt.ylabel("\u53d8\u5316\u6bd4\u4f8b\uff08%\uff09")
    save_fig("fig04_scene_user_change.png")

    daily_agg = daily_agg.reset_index().rename(columns={"index": "date"})
    daily_agg["date"] = pd.to_datetime(daily_agg["date"], errors="coerce")
    daily_agg = daily_agg.dropna(subset=["date"]).sort_values("date")
    daily_agg["weekday"] = daily_agg["date"].dt.weekday

    spring_daily = daily_agg[
        (daily_agg["date"] >= pd.Timestamp("2021-02-11"))
        & (daily_agg["date"] <= pd.Timestamp("2021-02-17"))
    ]
    march_weekday_daily = daily_agg[
        (daily_agg["date"] >= pd.Timestamp("2021-03-01"))
        & (daily_agg["date"] <= pd.Timestamp("2021-03-31"))
        & (daily_agg["weekday"] < 5)
    ]
    spring_flow_mean = float(spring_daily["FLOW_SUM"].mean())
    march_flow_mean = float(march_weekday_daily["FLOW_SUM"].mean())
    spring_user_mean = float(spring_daily["USER_COUNT"].mean())
    march_user_mean = float(march_weekday_daily["USER_COUNT"].mean())
    flow_change_pct = (
        (spring_flow_mean / march_flow_mean - 1) * 100 if march_flow_mean else 0.0
    )
    user_change_pct = (
        (spring_user_mean / march_user_mean - 1) * 100 if march_user_mean else 0.0
    )
    daily_subset = daily_agg[
        (daily_agg["date"] >= pd.Timestamp("2021-02-01"))
        & (daily_agg["date"] <= pd.Timestamp("2021-03-31"))
    ]

    plt.figure(figsize=(9, 4))
    plt.plot(daily_subset["date"], daily_subset["FLOW_SUM"], color="#4C78A8")
    plt.axvspan(pd.Timestamp("2021-02-11"), pd.Timestamp("2021-02-17"), color="#F58518", alpha=0.2)
    plt.title("\u6625\u8282\u524d\u540e\u65e5\u6d41\u91cf\u8d8b\u52bf")
    plt.xlabel("\u65e5\u671f")
    plt.ylabel("FLOW_SUM")
    save_fig("fig05_daily_flow_feb_mar.png")

    plt.figure(figsize=(9, 4))
    plt.plot(daily_subset["date"], daily_subset["USER_COUNT"], color="#E45756")
    plt.axvspan(pd.Timestamp("2021-02-11"), pd.Timestamp("2021-02-17"), color="#F58518", alpha=0.2)
    plt.title("\u6625\u8282\u524d\u540e\u65e5\u7528\u6237\u6570\u8d8b\u52bf")
    plt.xlabel("\u65e5\u671f")
    plt.ylabel("USER_COUNT")
    save_fig("fig06_daily_user_feb_mar.png")

    radar_metrics = []
    radar_labels = []
    for d, label in [(CHUXI, "\u9664\u5915"), (CHUYI, "\u521d\u4e00"), (NORMAL_SUN, "\u666e\u901a\u5468\u65e5")]:
        flows = np.array(target_flow.get(d, []), dtype=np.float64)
        users = np.array(target_user.get(d, []), dtype=np.float64)
        mean_flow = float(np.nanmean(flows)) if flows.size else 0.0
        peak_user = float(np.nanmax(users)) if users.size else 0.0
        flow_per_user = (
            float(np.nansum(flows) / np.nansum(users)) if np.nansum(users) > 0 else 0.0
        )
        flow_cv = float(np.nanstd(flows) / mean_flow) if mean_flow > 0 else 0.0
        radar_metrics.append([mean_flow, peak_user, flow_per_user, flow_cv])
        radar_labels.append(label)

    radar_values = np.array(radar_metrics, dtype=np.float64)
    radar_values = radar_values / np.maximum(radar_values.max(axis=0, keepdims=True), 1e-6)
    build_radar(["\u5e73\u5747\u6d41\u91cf", "\u5cf0\u503c\u7528\u6237", "\u4eba\u5747\u6d41\u91cf", "\u6d41\u91cf\u6ce2\u52a8"], radar_values, radar_labels)
    save_fig("fig07_target_cell_radar.png")

    if chuxi20_cells is not None and not chuxi20_cells.empty:
        chuxi20_cells["flow_mean"] = chuxi20_cells["flow_sum"] / chuxi20_cells["count"].replace(0, np.nan)
        plt.figure(figsize=(6, 5))
        plt.hist2d(
            chuxi20_cells["LONGITUDE"],
            chuxi20_cells["LATITUDE"],
            weights=chuxi20_cells["flow_mean"],
            bins=60,
            cmap="magma",
        )
        plt.colorbar(label="\u6d41\u91cf\u5bc6\u5ea6")
        plt.title("\u9664\u5915\u591c 20:00 \u5730\u7406\u70ed\u529b\u5206\u5e03")
        plt.xlabel("\u7ecf\u5ea6")
        plt.ylabel("\u7eac\u5ea6")
        save_fig("fig08_geo_heatmap_chuxi_20.png")

    if march20_cells is not None and not march20_cells.empty:
        march20_cells["flow_mean"] = march20_cells["flow_sum"] / march20_cells["count"].replace(0, np.nan)
        plt.figure(figsize=(6, 5))
        plt.hist2d(
            march20_cells["LONGITUDE"],
            march20_cells["LATITUDE"],
            weights=march20_cells["flow_mean"],
            bins=60,
            cmap="magma",
        )
        plt.colorbar(label="\u6d41\u91cf\u5bc6\u5ea6")
        plt.title("\u5de5\u4f5c\u65e5 20:00 \u5730\u7406\u70ed\u529b\u5206\u5e03\uff083\u6708\uff09")
        plt.xlabel("\u7ecf\u5ea6")
        plt.ylabel("\u7eac\u5ea6")
        save_fig("fig09_geo_heatmap_weekday_20.png")

    if chuxi20_cells is not None and not chuxi20_cells.empty:
        sizes = np.log1p(chuxi20_cells["flow_sum"].clip(lower=0)) * 6
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(
            chuxi20_cells["LONGITUDE"],
            chuxi20_cells["LATITUDE"],
            s=sizes,
            c=chuxi20_cells["SCENE"],
            cmap="tab20",
            alpha=0.6,
        )
        plt.colorbar(sc, label="SCENE")
        plt.title("\u9664\u5915\u591c 20:00 \u6d41\u91cf\u6c14\u6ce1\u56fe")
        plt.xlabel("\u7ecf\u5ea6")
        plt.ylabel("\u7eac\u5ea6")
        save_fig("fig10_bubble_chuxi_20.png")

    if march20_cells is not None and not march20_cells.empty:
        sizes = np.log1p(march20_cells["flow_mean"].clip(lower=0)) * 6
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(
            march20_cells["LONGITUDE"],
            march20_cells["LATITUDE"],
            s=sizes,
            c=march20_cells["SCENE"],
            cmap="tab20",
            alpha=0.6,
        )
        plt.colorbar(sc, label="SCENE")
        plt.title("\u5de5\u4f5c\u65e5 20:00 \u6d41\u91cf\u6c14\u6ce1\u56fe\uff083\u6708\uff09")
        plt.xlabel("\u7ecf\u5ea6")
        plt.ylabel("\u7eac\u5ea6")
        save_fig("fig11_bubble_weekday_20.png")

    top_increase = scene_compare["flow_change_pct"].idxmax()
    top_decrease = scene_compare["flow_change_pct"].idxmin()

    stats_out = {
        "target_cell": target_cell,
        "overall_compare": {
            "spring_flow_mean": spring_flow_mean,
            "march_weekday_flow_mean": march_flow_mean,
            "flow_change_pct": flow_change_pct,
            "spring_user_mean": spring_user_mean,
            "march_weekday_user_mean": march_user_mean,
            "user_change_pct": user_change_pct,
        },
        "periods": {
            "spring": [SPRING_START.isoformat(), SPRING_END.isoformat()],
            "march_weekday": "2021-03 weekdays",
            "target_dates": {
                "chuxi": CHUXI.isoformat(),
                "chuyi": CHUYI.isoformat(),
                "normal_sun": NORMAL_SUN.isoformat(),
            },
        },
        "scene_compare": {
            "top_increase_scene": int(top_increase),
            "top_decrease_scene": int(top_decrease),
            "scene_stats": {
                str(k): {
                    "spring_flow": float(v["spring_flow"]),
                    "march_flow": float(v["march_flow"]),
                    "flow_change_pct": float(v["flow_change_pct"]),
                    "spring_user": float(v["spring_user"]),
                    "march_user": float(v["march_user"]),
                    "user_change_pct": float(v["user_change_pct"]),
                }
                for k, v in scene_compare.iterrows()
            },
        },
        "radar_metrics": {
            "metrics": ["mean_flow", "peak_user", "flow_per_user", "flow_cv"],
            "dates": [CHUXI.isoformat(), CHUYI.isoformat(), NORMAL_SUN.isoformat()],
            "values": radar_metrics,
        },
    }

    with open(OUT_DIR / "section4_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
