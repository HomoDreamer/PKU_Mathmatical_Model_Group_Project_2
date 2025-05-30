# PKU_Mathmatical_Model_Group_Project_2

# 太空碎石聚合模拟 (Space Debris Aggregation Simulation)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

## 项目简介

本项目实现了一个基于经典牛顿引力模型的三维太空碎石多体动力学模拟系统，支持粒子间引力计算与聚合合并机制。  
系统采用暴力法完成多体引力计算，支持四阶 Runge-Kutta 时间积分。  
项目旨在研究不同初始条件（如中心天体质量、粒子初始半径分布、速度方向、合并阈值）对碎石聚合行为及稳态结构的影响。

我们也尝试过基于 Barnes–Hut 八叉树加速算法以降低计算复杂度，但目前因实现难度仍未稳定，相关内容保留为未来优化方向。

---

## 主要功能

- 三维多体引力仿真，支持 $N=100$ 粒子规模；
- 基于距离阈值的碎石合并机制，保证动量与质量守恒；
- 参数化初始条件配置，支持多组参数敏感性实验；
- 可视化轨迹与聚合结果展示（支持静态图和动画帧导出）；
- 支持双时间尺度模拟分析（绝对时间与轨道周期归一化步长）；
- 详尽代码实现，方便二次开发和扩展。

---

## 运行环境

- Python 3.8 及以上
- 依赖库：
  - numpy
  - matplotlib
  - tqdm（可选，用于显示进度条）

安装依赖示例：

```bash
pip install numpy matplotlib tqdm
