#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本功能：
    1. 批量读取指定目录下所有 diffmass_logs_seed*.xls* 文件。
    2. 每个文件中第一列 Run 的格式假设为 "<seed>_<M>"，例如 "seedA_500"、"seedB_1000" 等。
       第二列是 step，第三列是 Remaining。
    3. 拆分 Run 得到 seed（字符串）和 M（整数）。
    4. 合并所有文件后，对于每个唯一的 seed，绘制一张散点图：x=step，y=Remaining；
       同一张图内不同 M 值用不同颜色区分。
    5. 将输出图片保存到 ./plots/ 目录，文件名格式为 scatter_seed_<seed>.png。

使用方法：
    1. 将本脚本（例如命名为 plot_step_vs_remaining.py）放到你的工作目录，或修改 DATA_DIR 指向实际存放 Excel 文件的目录。
    2. 安装依赖：pip install pandas matplotlib openpyxl
    3. 运行：python plot_step_vs_remaining.py
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ——— 配置区 ———
DATA_DIR     = r"D:\luoyuzhang\pku\数模\newdiffmass"  # 修改为你的数据目录
FILE_PATTERN = os.path.join(DATA_DIR, "nnewdiffmass_logs_seed*.xls*")

# 列名：确保与你的 Excel 表格列名一致
COL_RUN      = "Run"        # 第一列，格式假设为 "<seed>_<M>"
COL_STEP     = "Step"       # 第二列
COL_REMAIN   = "Remaining"  # 第三列

# 输出图像保存目录
OUTPUT_DIR   = r"D:\luoyuzhang\pku\数模\newdiffmass"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# —————————

def parse_run_column(run_str):
    """
    将 Run 字符串拆分为 seed（字符串）和 M（整数）两部分。
    例如：“seedA_500” → ("seedA", 500)
    """
    try:
        seed_part, m_part = run_str.split("_M")
        return seed_part, int(m_part)
    except Exception as e:
        raise ValueError(f"无法解析 Run 列的值 '{run_str}'，应为 '<seed>_<M>' 形式（例如 seedA_500）") from e

def load_and_combine(data_dir, pattern):
    """
    批量读取所有符合模式的 Excel 文件，返回一个包含 seed、M、step、Remaining 四列的 DataFrame。
    """
    all_files = glob.glob(pattern)
    if not all_files:
        raise FileNotFoundError(f"在目录 {data_dir} 下未找到任何符合模式 '{pattern}' 的文件。")
    
    df_list = []
    for filepath in all_files:
        df = pd.read_excel(filepath, engine="openpyxl")
        
        # 检查必要列是否存在
        for col in (COL_RUN, COL_STEP, COL_REMAIN):
            if col not in df.columns:
                raise KeyError(f"文件 '{os.path.basename(filepath)}' 中缺少列 '{col}'，请确认列名一致。")
        
        # 从 Run 列拆分 seed 和 M
        parsed = df[COL_RUN].astype(str).apply(parse_run_column)
        df["seed"] = parsed.apply(lambda t: t[0])
        df["M"]    = parsed.apply(lambda t: t[1])
        
        # 只保留我们关心的四列
        df_subset = df[["seed", "M", COL_STEP, COL_REMAIN]].copy()
        df_list.append(df_subset)
    
    # 合并所有文件
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

def plot_step_vs_remaining_by_seed(df_all):
    """
    对于 df_all 中每个唯一的 seed，绘制一张散点图（step vs Remaining），
    同一张图中不同 M 值用不同颜色区分，并保存到 OUTPUT_DIR。
    """
    unique_seeds = sorted(df_all["seed"].unique())
    unique_Ms    = sorted(df_all["M"].unique())  # 修改为 M

    # 使用 tab20 调色板保证足够多的颜色
    cmap = plt.get_cmap("tab20")
    color_map = { m: cmap(idx % 20) for idx, m in enumerate(unique_Ms) }

    for seed_value in unique_seeds:
        df_seed = df_all[df_all["seed"] == seed_value]
        if df_seed.empty:
            continue

        plt.figure(figsize=(6, 5))
        for M_value in unique_Ms:
            df_sub = df_seed[df_seed["M"] == M_value]
            if df_sub.empty:
                continue
            plt.scatter(
                df_sub[COL_STEP],
                df_sub[COL_REMAIN],
                label=f"M={M_value}",
                alpha=0.7,
                color=color_map[M_value],
                s=30,
                edgecolors="none"
            )

        plt.xlabel(COL_STEP)
        plt.ylabel(COL_REMAIN)
        plt.title(f"step vs Remaining (seed = {seed_value})")
        plt.legend(title="M 值", fontsize="small", frameon=True, ncol=2)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, f"scatter_seed_{seed_value}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"已保存：{out_path}")

def main():
    print("1. 开始加载并解析所有 Excel 文件……")
    df_all = load_and_combine(DATA_DIR, FILE_PATTERN)
    print(f"   共读取到 {len(df_all)} 行数据；包含的 Seeds: {sorted(df_all['seed'].unique())}；M 值: {sorted(df_all['M'].unique())}")
    
    print("\n2. 根据不同的 seed 绘制散点图……")
    plot_step_vs_remaining_by_seed(df_all)
    
    print(f"\n全部绘图完成，图片保存在：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
