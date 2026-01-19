# -*- coding: utf-8 -*-
"""
保姆级实现：重点交通场景移动网络质量数据集（Cat1 / NB-IoT / Redcap）
- 清洗、质量检查、KPI衍生、加权汇总、运营商排名、导出Excel报表
运行示例：
    python run_pipeline.py --input "重点交通场景移动网络质量数据集.xlsx" --outdir outputs
"""

from __future__ import annotations
import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# 0) 配置区：你想改权重/阈值就在这
# -----------------------------

@dataclass(frozen=True)
class ScoreWeights:
    """综合分权重：可按业务偏好调"""
    coverage: float = 0.4
    experience: float = 0.4
    stability: float = 0.2


WEIGHTS = ScoreWeights()

# 覆盖阈值（说明文档：RSRP/SINR 或 SS-RSRP/SS-SINR）
COVERAGE_THRESHOLDS = {
    "cov1": {"rsrp": -105.0, "sinr": -3.0},
    "cov2": {"rsrp": -110.0, "sinr": -3.0},
}

# 速率达标阈值（说明文档）
RATE_THRESHOLDS = {
    "dl_mbps": 2.0,    # 下行 >= 2Mbps
    "ul_mbps": 0.5,    # 上行 >= 0.5Mbps
}

# 你的Excel公共字段（真实存在）
DIM_COLS = ["设备厂家", "地市", "一级场景", "二级场景", "运营商", "测试总里程", "测试总时长", "平均车速"]

# 运营商编码映射
OP_MAP = {1: "移动", 2: "电信", 3: "联通"}

# 口径：采样点 vs 25m栅格
SCOPE_MAP = {"1": "sample_point", "2": "grid_25m"}


# -----------------------------
# 1) 工具函数：列名规范化与别名修正
# -----------------------------

def normalize_columns(cols: List[str]) -> List[str]:
    """去空格、统一大小写、统一全角字符等"""
    out = []
    for c in cols:
        if c is None:
            out.append(c)
            continue
        s = str(c).strip()
        # 常见的全角空格/不可见字符
        s = re.sub(r"\s+", "", s)
        # 统一 x2/X2 这种
        s = s.replace("X2_", "x2_").replace("X1_", "x1_")
        out.append(s)
    return out


def apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    修复你这份Excel里出现的轻微列名不一致/笔误。
    你文件中已观察到：
    - Cat1: z2_应用层上行前10%峰值率 (缺“速”) -> 统一为 z2_应用层上行前10%峰值速率
    - Redcap: z1_下行速率达标速率 (明显笔误) -> 统一为 z1_上行速率达标率
    """
    aliases = {
        "z2_应用层上行前10%峰值率": "z2_应用层上行前10%峰值速率",
        "z1_下行速率达标速率": "z1_上行速率达标率",
        "z2_应用层上行速率达标率": "z2_上行速率达标率",  # 有些表会多“应用层”
    }
    rename = {c: aliases[c] for c in df.columns if c in aliases}
    if rename:
        df = df.rename(columns=rename)
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """尽可能把数值列转成 numeric"""
    # 维度列以外的都尝试转数值（失败则NaN）
    for c in df.columns:
        if c in ["设备厂家", "地市", "一级场景", "二级场景"]:
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")
    # 对明显应该是数值的列，强制转 numeric
    maybe_num = [c for c in df.columns if any(k in c for k in ["覆盖率", "吞吐率", "速率", "次数", "里程", "时长", "车速", "RSRP", "SINR", "驻留"])]
    for c in maybe_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -----------------------------
# 2) 读取Excel（三个sheet）
# -----------------------------

def read_dataset(xlsx_path: str) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(xlsx_path)
    sheets = xls.sheet_names

    required = {"Cat1", "NB-IoT", "Redcap"}
    missing = required - set(sheets)
    if missing:
        raise ValueError(f"Excel缺少sheet: {missing}, 实际为: {sheets}")

    data = {}
    for name in ["Cat1", "NB-IoT", "Redcap"]:
        df = pd.read_excel(xlsx_path, sheet_name=name)
        df.columns = normalize_columns(list(df.columns))
        df = apply_column_aliases(df)
        df = coerce_numeric(df)
        df["dataset"] = name
        # 运营商编码映射
        if "运营商" in df.columns:
            df["运营商名称"] = df["运营商"].map(OP_MAP).fillna(df["运营商"].astype(str))
        data[name] = df
    return data


# -----------------------------
# 3) 数据质量检查（输出告警表）
# -----------------------------

def _is_ratio_col(c: str) -> bool:
    return any(k in c for k in ["覆盖率", "达标率", "掉线率", "驻留比"])


def _is_count_col(c: str) -> bool:
    return "次数" in c


def _is_nonneg_col(c: str) -> bool:
    return any(k in c for k in ["吞吐率", "速率", "次数", "里程", "时长", "车速", "驻留时长"])


def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    输出长表：每行一个问题
    columns: [issue_type, column, value, row_index, ...维度字段]
    """
    issues = []

    # 1) 比例类是否在[0,1]
    ratio_cols = [c for c in df.columns if _is_ratio_col(c)]
    for c in ratio_cols:
        s = df[c]
        bad = s.notna() & ((s < 0) | (s > 1))
        for idx in df.index[bad]:
            issues.append(_issue_row(df, idx, "ratio_out_of_range", c, df.at[idx, c]))

    # 2) 非负类是否出现负值
    nonneg_cols = [c for c in df.columns if _is_nonneg_col(c)]
    for c in nonneg_cols:
        s = df[c]
        bad = s.notna() & (s < 0)
        for idx in df.index[bad]:
            issues.append(_issue_row(df, idx, "negative_value", c, df.at[idx, c]))

    # 3) FTP 掉线率对账：掉线次数/尝试次数 vs 字段掉线率
    ftp_specs = [
        ("y1_FTP下载尝试次数", "y1_FTP下载掉线次数", "y1_FTP下载掉线率"),
        ("y2_FTP下载尝试次数", "y2_FTP下载掉线次数", "y2_FTP下载掉线率"),
        ("z1_FTP上传尝试次数", "z1_FTP上传掉线次数", "z1_FTP上传掉线率"),
        ("z2_FTP上传尝试次数", "z2_FTP上传掉线次数", "z2_FTP上传掉线率"),
    ]
    for a, d, r in ftp_specs:
        if all(col in df.columns for col in [a, d, r]):
            denom = df[a]
            num = df[d]
            rate = df[r]
            calc = np.where(denom.fillna(0) > 0, num / denom, np.nan)
            # 容差：1e-3（你也可以改大一点）
            diff = np.abs(calc - rate)
            bad = (~np.isnan(calc)) & rate.notna() & (diff > 1e-3)
            for idx in df.index[bad]:
                issues.append(_issue_row(df, idx, "ftp_rate_mismatch", r, df.at[idx, r],
                                         extra={"calc_rate": float(calc[idx]) if not np.isnan(calc[idx]) else np.nan,
                                                "diff": float(diff[idx])}))
    return pd.DataFrame(issues)


def _issue_row(df: pd.DataFrame, idx: int, issue_type: str, col: str, val, extra: dict | None = None) -> dict:
    base = {
        "dataset": df.at[idx, "dataset"] if "dataset" in df.columns else None,
        "row_index": int(idx),
        "issue_type": issue_type,
        "column": col,
        "value": val,
    }
    # 加上关键维度便于定位
    for c in ["设备厂家", "地市", "一级场景", "二级场景", "运营商", "运营商名称"]:
        if c in df.columns:
            base[c] = df.at[idx, c]
    if extra:
        base.update(extra)
    return base


# -----------------------------
# 4) KPI 衍生：把三个数据集统一成可分析结构
# -----------------------------

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 计算FTP掉线率（优先用次数回算，作为calc列）
    - 计算 x1 vs x2 的差值（delta列）
    - 计算综合分 score_sample / score_grid（各口径）
    """
    out = df.copy()

    # ---- 4.1 FTP掉线率（回算）与成功率 ----
    def safe_rate(num, den):
        return np.where(den.fillna(0) > 0, num / den, np.nan)

    pairs = [
        ("y1", "FTP下载"),
        ("y2", "FTP下载"),
        ("z1", "FTP上传"),
        ("z2", "FTP上传"),
    ]
    for prefix, name in pairs:
        a = f"{prefix}_{name}尝试次数"
        s = f"{prefix}_{name}成功次数"
        d = f"{prefix}_{name}掉线次数"
        r = f"{prefix}_{name}掉线率"
        if all(c in out.columns for c in [a, d]):
            out[f"{prefix}_{name}掉线率_calc"] = safe_rate(out[d], out[a])
        if all(c in out.columns for c in [a, s]):
            out[f"{prefix}_{name}成功率_calc"] = safe_rate(out[s], out[a])

    # ---- 4.2 x1 vs x2 差值（覆盖/信号/吞吐）----
    # Cat1/NB：RSRP/SINR；Redcap：SS-RSRP/SS-SINR
    if "x1_平均RSRP" in out.columns and "x2_平均RSRP" in out.columns:
        out["delta_avg_rsrp"] = out["x1_平均RSRP"] - out["x2_平均RSRP"]
    if "x1_平均SINR" in out.columns and "x2_平均SINR" in out.columns:
        out["delta_avg_sinr"] = out["x1_平均SINR"] - out["x2_平均SINR"]
    if "x1_平均SS-RSRP" in out.columns and "x2_平均SS-RSRP" in out.columns:
        out["delta_avg_ss_rsrp"] = out["x1_平均SS-RSRP"] - out["x2_平均SS-RSRP"]
    if "x1_平均SS-SINR" in out.columns and "x2_平均SS-SINR" in out.columns:
        out["delta_avg_ss_sinr"] = out["x1_平均SS-SINR"] - out["x2_平均SS-SINR"]

    # ---- 4.3 综合分（每条记录、每种口径各算一个）----
    out["score_sample_point"] = compute_score(out, scope="1")
    out["score_grid_25m"] = compute_score(out, scope="2")

    return out


def compute_score(df: pd.DataFrame, scope: str) -> np.ndarray:
    """
    综合分 = coverage * 覆盖率2 + experience * 下行达标率 + stability * (1 - 下行FTP掉线率_calc)
    - NB-IoT 没有业务体验/稳定性字段时，会自动退化只看覆盖
    - Redcap 也会走同样逻辑（覆盖率字段仍是 x{scope}_覆盖率2）
    """
    scope_prefix = f"x{scope}_"
    y_prefix = f"y{scope}_"

    # 覆盖：优先用覆盖率2
    cov2_col = scope_prefix + "覆盖率2"
    cov = df[cov2_col].to_numpy() if cov2_col in df.columns else np.full(len(df), np.nan)

    # 体验：下行速率达标率
    exp_col = y_prefix + "下行速率达标率"
    exp = df[exp_col].to_numpy() if exp_col in df.columns else np.full(len(df), np.nan)

    # 稳定性：下行FTP掉线率（用回算的calc优先）
    drop_calc = f"y{scope}_FTP下载掉线率_calc"
    drop_raw = y_prefix + "FTP下载掉线率"
    if drop_calc in df.columns:
        drop = df[drop_calc].to_numpy()
    elif drop_raw in df.columns:
        drop = df[drop_raw].to_numpy()
    else:
        drop = np.full(len(df), np.nan)

    # 对缺失字段进行退化：如果 exp/drop 全缺失，就只看覆盖
    has_exp = ~np.isnan(exp)
    has_drop = ~np.isnan(drop)

    score = np.full(len(df), np.nan)

    # 完整版
    full = (~np.isnan(cov)) & has_exp & has_drop
    score[full] = (
        WEIGHTS.coverage * cov[full] +
        WEIGHTS.experience * exp[full] +
        WEIGHTS.stability * (1.0 - drop[full])
    )

    # 退化：只有覆盖
    only_cov = (~np.isnan(cov)) & (~has_exp) & (~has_drop)
    score[only_cov] = cov[only_cov]

    # 半退化：cov + exp
    cov_exp = (~np.isnan(cov)) & has_exp & (~has_drop)
    score[cov_exp] = 0.6 * cov[cov_exp] + 0.4 * exp[cov_exp]

    # 半退化：cov + stability
    cov_stab = (~np.isnan(cov)) & (~has_exp) & has_drop
    score[cov_stab] = 0.7 * cov[cov_stab] + 0.3 * (1.0 - drop[cov_stab])

    return score


# -----------------------------
# 5) 汇总与排名（加权均值）
# -----------------------------

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    v = values[mask].astype(float)
    w = weights[mask].astype(float)
    return float(np.average(v, weights=w))


def pick_weight(df: pd.DataFrame) -> pd.Series:
    """
    权重优先用 测试总时长，其次 测试总里程。
    （CQT很多情况下里程可能是0；DT时长也可能更稳）
    """
    time_w = df["测试总时长"] if "测试总时长" in df.columns else pd.Series(np.nan, index=df.index)
    mile_w = df["测试总里程"] if "测试总里程" in df.columns else pd.Series(np.nan, index=df.index)
    w = time_w.copy()
    w = w.where((w.notna()) & (w > 0), mile_w)
    return w


def summarize_kpis(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    输出：按 城市/场景/运营商/口径 汇总的KPI（均值 + 加权均值 + 样本量）
    """
    base_cols = ["dataset", "设备厂家", "地市", "一级场景", "二级场景", "运营商", "运营商名称"]
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan

    weight = pick_weight(df)

    # 每个dataset按已有字段挑KPI
    kpi_candidates = [
        # 覆盖（RSRP/SINR 或 SS-RSRP/SS-SINR）
        "x1_平均RSRP", "x1_平均SINR", "x1_覆盖率1", "x1_覆盖率2",
        "x2_平均RSRP", "x2_平均SINR", "x2_覆盖率1", "x2_覆盖率2",
        "x1_平均SS-RSRP", "x1_平均SS-SINR", "x1_覆盖率3", "x1_覆盖率4",
        "x2_平均SS-RSRP", "x2_平均SS-SINR", "x2_覆盖率3", "x2_覆盖率4",

        # 下行/上行体验与稳定性
        "y1_应用层下行平均吞吐率", "y1_应用层下行前10%峰值速率", "y1_下行速率达标率",
        "y1_FTP下载掉线率", "y1_FTP下载掉线率_calc",
        "z1_应用层上行平均吞吐率", "z1_应用层上行前10%峰值速率", "z1_上行速率达标率",
        "z1_FTP上传掉线率", "z1_FTP上传掉线率_calc",

        "y2_应用层下行平均吞吐率", "y2_应用层下行前10%峰值速率", "y2_下行速率达标率",
        "y2_FTP下载掉线率", "y2_FTP下载掉线率_calc",
        "z2_应用层上行平均吞吐率", "z2_应用层上行前10%峰值速率", "z2_上行速率达标率",
        "z2_FTP上传掉线率", "z2_FTP上传掉线率_calc",

        # Redcap可用性
        "RedCap驻留时长", "RedCap时长驻留比",

        # 口径差异
        "delta_avg_rsrp", "delta_avg_sinr", "delta_avg_ss_rsrp", "delta_avg_ss_sinr",

        # 综合分
        "score_sample_point", "score_grid_25m"
    ]
    kpis = [c for c in kpi_candidates if c in df.columns]

    # 做成两份：sample_point 与 grid_25m
    scopes = [("sample_point", "1"), ("grid_25m", "2")]

    rows = []
    group_cols = ["dataset", "地市", "一级场景", "二级场景", "运营商", "运营商名称"]
    for scope_name, s in scopes:
        # scope内适配：只保留 relevant 列（例如 x1/y1/z1 或 x2/y2/z2）
        # 这里简单做：不过滤，让汇总自然处理缺失（NB没有y/z就会NaN）
        sub = df.copy()
        sub["_scope"] = scope_name
        sub["_weight"] = weight

        gb = sub.groupby(group_cols + ["_scope"], dropna=False)

        # 均值
        mean_df = gb[kpis].mean(numeric_only=True).reset_index()
        mean_df = mean_df.rename(columns={c: f"{c}__mean" for c in kpis})

        # 加权均值
        w_rows = []
        for key, g in gb:
            rec = dict(zip(group_cols + ["_scope"], key))
            w = g["_weight"]
            for c in kpis:
                rec[f"{c}__wmean"] = weighted_mean(g[c], w)
            rec["n_records"] = int(len(g))
            rec["sum_time_h"] = float(g["测试总时长"].fillna(0).sum()) if "测试总时长" in g.columns else np.nan
            rec["sum_mileage_km"] = float(g["测试总里程"].fillna(0).sum()) if "测试总里程" in g.columns else np.nan
            w_rows.append(rec)
        wmean_df = pd.DataFrame(w_rows)

        merged = mean_df.merge(wmean_df, on=group_cols + ["_scope"], how="left")
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True)
    out["dataset"] = dataset_name
    return out


def rank_operators(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    每个 dataset + 地市 + 场景 + 口径 下，对运营商按综合分（加权）排名。
    """
    df = summary_df.copy()

    # 选择该口径的综合分列（加权）
    # sample: score_sample_point__wmean；grid: score_grid_25m__wmean
    def pick_score_col(scope):
        return "score_sample_point__wmean" if scope == "sample_point" else "score_grid_25m__wmean"

    df["score_wmean"] = df.apply(lambda r: r.get(pick_score_col(r["_scope"]), np.nan), axis=1)

    keys = ["dataset", "地市", "一级场景", "二级场景", "_scope"]
    df["rank"] = df.groupby(keys)["score_wmean"].rank(ascending=False, method="min")
    df = df.sort_values(keys + ["rank", "运营商"])
    return df


# -----------------------------
# 6) 导出报表
# -----------------------------

def export_excel(path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            # Excel sheet名最长31字符
            sheet_name = name[:31]
            df.to_excel(writer, index=False, sheet_name=sheet_name)


# -----------------------------
# 7) 主流程
# -----------------------------

def main(input_path: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    data = read_dataset(input_path)

    # 7.1 衍生指标 + 质量检查
    quality_all = []
    cleaned = {}
    for name, df in data.items():
        df2 = add_derived_metrics(df)
        cleaned[name] = df2

        q = quality_checks(df2)
        if not q.empty:
            quality_all.append(q)

        # 保存parquet中间数据
        df2.to_parquet(os.path.join(outdir, f"cleaned_{name}.parquet"), index=False)

    # 7.2 导出质量报告
    quality_df = pd.concat(quality_all, ignore_index=True) if quality_all else pd.DataFrame(
        columns=["dataset", "row_index", "issue_type", "column", "value"]
    )
    export_excel(os.path.join(outdir, "data_quality_report.xlsx"), {"quality_issues": quality_df})

    # 7.3 KPI汇总
    summary_list = []
    for name, df in cleaned.items():
        s = summarize_kpis(df, dataset_name=name)
        summary_list.append(s)
    summary_df = pd.concat(summary_list, ignore_index=True)
    export_excel(os.path.join(outdir, "kpi_summary.xlsx"), {"kpi_summary": summary_df})

    # 7.4 排名
    ranking_df = rank_operators(summary_df)
    export_excel(os.path.join(outdir, "rankings.xlsx"), {"rankings": ranking_df})

    print(f"Done. Outputs saved to: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入Excel路径")
    parser.add_argument("--outdir", default="outputs", help="输出目录")
    args = parser.parse_args()

    main(args.input, args.outdir)
