#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本功能：
    1. 批量读取指定目录下所有 diffdis_logs_seed_*_logs.xlsx 文件。
    2. 文件名格式：diffdis_logs_seed_S_logs.xlsx，S 为 seed 值。
    3. Excel 中：
        - 第一列 Run，格式为区间字符串，例如 "100–200R"，或值 "FINAL"。
        - 过滤掉 Run 列中值为 FINAL 或无法解析区间的行。
    4. 按 (Run 区间, Remaining) 分组，计算平均 Step。
    5. 绘制散点图：横轴为平均 Step，纵轴为 Remaining，不同 Run 区间用不同颜色区分。
    6. 保存图像到 ./plots/avg_step_vs_remaining_by_run.png。
"""

import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# ——— 用户配置区 ———
DATA_DIR     = r"D:\luoyuzhang\pku\数模\newdiffdis"  # 修改为你的数据目录
PATTERN      = os.path.join(DATA_DIR, "nnewdiffdis_logs_seed_*_logs.xlsx")
OUTPUT_DIR   = r"D:\luoyuzhang\pku\数模\newdiffdis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COL_RUN      = "Run"
COL_STEP     = "Step"
COL_REMAIN   = "Remaining"

def parse_run_interval(run_str):
    """用正则提取 Run 字符串中的两个数字（区间下限和上限），否则返回 None。"""
    nums = re.findall(r"(\d+)", str(run_str))
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    return None

def load_and_concat(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"未找到匹配文件：{pattern}")
    df_list = []
    for fp in files:
        df = pd.read_excel(fp, engine="openpyxl")
        for col in (COL_RUN, COL_STEP, COL_REMAIN):
            if col not in df.columns:
                raise KeyError(f"文件 {os.path.basename(fp)} 缺少列 {col}")
        # 解析区间，并过滤掉解析失败的行
        df['interval'] = df[COL_RUN].apply(parse_run_interval)
        df = df[df['interval'].notnull()].copy()
        # 从文件名提取 seed
        m = re.search(r"seed_(\d+)_logs", os.path.basename(fp))
        df['file_seed'] = m.group(1) if m else ""
        df_list.append(df[[COL_RUN, 'file_seed', COL_STEP, COL_REMAIN, 'interval']])
    return pd.concat(df_list, ignore_index=True)

def compute_avg_step(df):
    return (
        df
        .groupby([COL_RUN, COL_REMAIN], as_index=False)[COL_STEP]
        .mean()
        .rename(columns={COL_STEP: 'avg_step'})
    )

def plot_scatter(df_avg, intervals):
    cmap = plt.get_cmap('tab10')
    # 根据区间的下限排序
    sorted_runs = sorted(intervals.items(), key=lambda kv: kv[1][0])

    plt.figure(figsize=(8,6))
    for idx, (run, (low, _)) in enumerate(sorted_runs):
        sub = df_avg[df_avg[COL_RUN] == run]
        plt.scatter(
            sub['avg_step'],
            sub[COL_REMAIN],
            label=run,
            s=50,
            alpha=0.7,
            color=cmap(idx % 10)
        )
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    plt.xlabel('平均 Step')
    plt.ylabel('Remaining')
    plt.title('不同 R 区间下 平均 Step vs Remaining（散点图）')
    plt.legend(title='R 区间')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'avg_step_vs_remaining_by_run.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"图像已保存到：{out_path}")

def main():
    print("开始加载并合并数据…")
    df_all = load_and_concat(PATTERN)
    print(f"共加载 {len(df_all)} 条记录，有效 Run 区间：{df_all[COL_RUN].unique()}")

    print("计算平均 Step…")
    df_avg = compute_avg_step(df_all)

    # 收集区间信息，用于排序
    intervals = {
        run: df_all[df_all[COL_RUN] == run]['interval'].iloc[0]
        for run in df_all[COL_RUN].unique()
    }

    print("绘制散点图…")
    plot_scatter(df_avg, intervals)
    print("完成。")

if __name__ == "__main__":
    main()
