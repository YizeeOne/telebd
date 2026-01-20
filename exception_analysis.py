# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
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


def compute_spike_and_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    last_flow: dict[int, float] = {}
    max_diff: dict[int, float] = {}
    max_time: dict[int, str] = {}
    max_user: dict[int, float] = {}
    max_flow: dict[int, float] = {}

    daily_flow = defaultdict(float)
    daily_user = defaultdict(float)

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

        date_str = chunk["DATETIME_KEY"].str.slice(0, 10)
        daily_group = chunk.groupby(date_str)[["FLOW_SUM", "USER_COUNT"]].sum()
        for dt_key, row in daily_group.iterrows():
            daily_flow[dt_key] += float(row["FLOW_SUM"])
            daily_user[dt_key] += float(row["USER_COUNT"])

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
    return spike_df, daily_df


def build_stats() -> dict:
    ensure_out_dir()
    cell_agg = pd.read_csv(CELL_AGG_PATH)

    spike_df, daily_df = compute_spike_and_daily()
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
