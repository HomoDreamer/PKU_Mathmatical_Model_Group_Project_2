#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本功能：
    - 读取 newdiffangle_logs_seed*_logs.xlsx 文件。
    - 从 Run 列提取 cos[ A, ... ] 的 A。
    - 按 (cos_A, Remaining) 分组，计算平均 SimTime(s)。
    - 绘制平均 SimTime(s) vs Remaining 散点图，不同 cos_A 用不同颜色。
"""

import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# 用户配置
DATA_DIR     = r"D:\luoyuzhang\pku\数模\newdiffangle"
PATTERN      = os.path.join(DATA_DIR, "newdiffangle_logs_seed*.xlsx")
OUTPUT_DIR   = os.path.join(DATA_DIR, "plots_simtime_by_cos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COL_RUN      = "Run"
COL_SIMTIME  = "SimTime(s)"
COL_REMAIN   = "Remaining"

def extract_cos_A(run_str):
    m = re.search(r"cos\[\s*([^,]+),", str(run_str))
    return float(m.group(1)) if m else None

def load_and_prepare(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"未找到匹配文件：{pattern}")
    dfs = []
    for fp in files:
        df = pd.read_excel(fp, engine="openpyxl")
        for col in (COL_RUN, COL_SIMTIME, COL_REMAIN):
            if col not in df.columns:
                raise KeyError(f"{os.path.basename(fp)} 缺少列 {col}")
        df['cos_A'] = df[COL_RUN].apply(extract_cos_A)
        df = df[df['cos_A'].notnull()].copy()
        dfs.append(df[[ 'cos_A', COL_SIMTIME, COL_REMAIN ]])
    return pd.concat(dfs, ignore_index=True)

def compute_avg_simtime(df):
    return (
        df
        .groupby(['cos_A', COL_REMAIN], as_index=False)[COL_SIMTIME]
        .mean()
        .rename(columns={COL_SIMTIME: "avg_simtime"})
    )

def plot_scatter(df_avg):
    cmap = plt.get_cmap("tab10")
    cos_values = sorted(df_avg['cos_A'].unique())
    plt.figure(figsize=(8,6))
    for idx, a in enumerate(cos_values):
        sub = df_avg[df_avg['cos_A'] == a]
        plt.scatter(
            sub["avg_simtime"],
            sub[COL_REMAIN],
            label=f"cos A={a}",
            s=50,
            alpha=0.7,
            color=cmap(idx % 10)
        )
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    plt.xlabel("平均 SimTime(s)")
    plt.ylabel("Remaining")
    plt.title("按 mincosA 分组的 平均 SimTime(s) vs Remaining")
    plt.legend(title="mincosA", bbox_to_anchor=(1.05,1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "avg_simtime_vs_remaining_by_cosA.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[INFO] 图已保存至 {out_png}")

def main():
    print("[INFO] 加载并准备数据…")
    df = load_and_prepare(PATTERN)
    print(f"[INFO] 有效记录: {len(df)}, 共 {df['cos_A'].nunique()} 个不同 cos_A")
    print("[INFO] 计算平均 SimTime(s)…")
    df_avg = compute_avg_simtime(df)
    print("[INFO] 绘制散点图…")
    plot_scatter(df_avg)
    print("[INFO] 完成。")

if __name__ == "__main__":
    main()
