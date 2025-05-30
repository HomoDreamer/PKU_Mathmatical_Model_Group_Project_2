#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本功能：
    1. 批量读取指定目录下所有 Excel 文件（例如 *.xlsx）。
    2. Excel 中：
        - 第一列 Run，格式为 "seed<S>_merge<M>"，其中 S 为 seed 值，M 为 merge 参数（3.0、3.5、4.0、4.5、5.0）。
        - 其他列包括 SimTime(s) 和 Remaining。
    3. 解析 Run 列，提取 seed 和 merge 值，过滤掉不符合格式的行。
    4. 按（merge, Remaining）分组，计算平均 SimTime(s)。
    5. 绘制散点图：横轴为平均 SimTime(s)，纵轴为 Remaining，不同 merge 值用不同颜色区分。
    6. 保存图像到 ./plots/avg_simtime_vs_remaining_by_merge.png。
"""

import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# ——— 用户配置区 ———
DATA_DIR   = r"D:\luoyuzhang\pku\数模\diffmerge"      # 修改为你的数据目录
PATTERN    = os.path.join(DATA_DIR, "*.xlsx")
OUTPUT_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 列名
COL_RUN     = "Run"
COL_SIMTIME = "SimTime(s)"
COL_REMAIN  = "Remaining"

def parse_run(run_str):
    """
    从 Run 字符串中提取 seed 和 merge 值。
    例如 "seed5673_merge3.0" -> (5673, 3.0)
    解析失败返回 None。
    """
    m = re.match(r"seed(\d+)_merge([\d\.]+)", str(run_str))
    if m:
        seed = int(m.group(1))
        merge = float(m.group(2))
        return seed, merge
    return None

def load_and_concat(pattern):
    """
    读取所有匹配的 Excel 文件，解析 Run 列，合并成一个 DataFrame。
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"未找到匹配的文件：{pattern}")

    df_list = []
    for fp in files:
        df = pd.read_excel(fp, engine="openpyxl")
        # 检查必需列
        for col in (COL_RUN, COL_SIMTIME, COL_REMAIN):
            if col not in df.columns:
                raise KeyError(f"文件 {os.path.basename(fp)} 缺少列 {col}")

        # 解析 Run 列
        parsed = df[COL_RUN].apply(parse_run)
        df = df[parsed.notnull()].copy()
        df[['seed', 'merge']] = pd.DataFrame(parsed[parsed.notnull()].tolist(), index=df.index)

        df_list.append(df[[COL_RUN, 'seed', 'merge', COL_SIMTIME, COL_REMAIN]])

    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

def compute_avg_simtime(df):
    """
    按 merge 和 Remaining 分组，计算平均 SimTime(s)。
    """
    return (
        df
        .groupby(['merge', COL_REMAIN], as_index=False)[COL_SIMTIME]
        .mean()
        .rename(columns={COL_SIMTIME: 'avg_simtime'})
    )

def plot_scatter(df_avg):
    """
    绘制散点图：x 轴 avg_simtime，y 轴 Remaining，不同 merge 用不同颜色。
    """
    cmap = plt.get_cmap('tab10')
    merges = sorted(df_avg['merge'].unique())
    plt.figure(figsize=(8, 6))
    for idx, m in enumerate(merges):
        sub = df_avg[df_avg['merge'] == m]
        plt.scatter(
            sub['avg_simtime'],
            sub[COL_REMAIN],
            label=f"merge {m}",
            s=50,
            alpha=0.7,
            color=cmap(idx % 10)
        )

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

    plt.xlabel('平均 SimTime(s)')
    plt.ylabel('Remaining')
    plt.title('不同 merge 值下 平均 SimTime(s) vs Remaining（散点图）')
    plt.legend(title='merge 参数')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, 'avg_simtime_vs_remaining_by_merge.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"图像已保存到：{out_path}")

def main():
    print("开始加载并合并数据…")
    df_all = load_and_concat(PATTERN)
    print(f"共加载 {len(df_all)} 条记录，包含 merge 值：{sorted(df_all['merge'].unique())}")

    print("计算平均 SimTime(s)…")
    df_avg = compute_avg_simtime(df_all)

    print("绘制散点图…")
    plot_scatter(df_avg)
    print("完成。")

if __name__ == "__main__":
    main()
