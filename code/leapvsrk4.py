import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import datetime
import pandas as pd
import os

# == 参数 ==
G = 1.0       # 引力常数
M = 10000.0   # 中心质量
N = 100       # 初始碎石数量
R = 0.5       # 单位长度缩放因子
DT = 0.05     # 时间步长
STEPS = 3000000
MERGE_THRESHOLD = 3.0
SKIP = 5
TRAIL_LEN = 20  # 拖尾长度（帧数）

# 日志参数
LOG_FILE = 'merge_log_vi_leapfrog.xlsx'
total_steps = 0

# ====== 记录合并信息 ======
def create_log_file():
    df = pd.DataFrame(columns=["step", "merge_events", "remaining_rocks", "simulation_time", "real_time"])
    df.to_excel(LOG_FILE, index=False)


def log_merge_info(step, merge_events, remaining_rocks, simulation_time, real_time):
    data = pd.DataFrame([{
        "step": step,
        "merge_events": merge_events,
        "remaining_rocks": remaining_rocks,
        "simulation_time": simulation_time,
        "real_time": real_time.strftime('%Y-%m-%d %H:%M:%S')
    }])
    if os.path.exists(LOG_FILE):
        with pd.ExcelWriter(LOG_FILE, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            sheet = writer.sheets['Sheet1']
            data.to_excel(writer, index=False, header=False, startrow=sheet.max_row)
    else:
        create_log_file()
        log_merge_info(step, merge_events, remaining_rocks, simulation_time, real_time)

# ====== 岩石类 ======
class Rock:
    def __init__(self, pos, vel, mass):
        self.pos = np.array(pos, float)
        self.vel = np.array(vel, float)
        self.mass = mass
        self.history = deque(maxlen=TRAIL_LEN)

# ====== 初始化 ======
def init_rocks(n):
    rocks = []
    for _ in range(n):
        r = np.random.uniform(100*R, 150*R)
        phi = np.random.uniform(0, 2*np.pi)
        costheta = np.random.uniform(0.8, 1)
        theta = np.arccos(costheta)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        u = np.array([x, y, z]) / r
        v_ref = np.array([1, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1, 0])
        tangent = np.cross(u, v_ref)
        tangent /= np.linalg.norm(tangent)
        vel = np.sqrt(G*M/r) * tangent
        rocks.append(Rock([x, y, z], vel, np.random.uniform(0.5, 1.5)))
    return rocks

# ====== 计算加速度 ======
def compute_acc(rocks):
    pos = np.array([r.pos for r in rocks])
    acc = np.zeros_like(pos)
    # 中心质量引力
    d0 = np.linalg.norm(pos, axis=1)
    acc += -G * M * pos / d0[:, None]**3
    # 碎石间相互作用
    masses = np.array([r.mass for r in rocks])
    for i in range(len(rocks)):
        diff = pos - pos[i]
        dist3 = np.linalg.norm(diff, axis=1)**3
        dist3[i] = np.inf
        acc[i] += G * np.sum((masses[:, None] * diff) / dist3[:, None], axis=0)
    return acc

# ====== Leapfrog 单步 ======
def step_leapfrog(rocks):
    pos0 = np.array([r.pos for r in rocks])
    vel0 = np.array([r.vel for r in rocks])
    m0 = np.array([r.mass for r in rocks])

    # 计算加速度函数
    def a(p, v):
        return compute_acc([Rock(p[i], v[i], m0[i]) for i in range(len(rocks))])

    # 初始加速度
    acc0 = a(pos0, vel0)
    # 半步速度更新
    vel_half = vel0 + 0.5 * DT * acc0
    # 全步位置更新
    pos_new = pos0 + DT * vel_half
    # 新加速度
    acc_new = a(pos_new, vel_half)
    # 全步速度更新
    vel_new = vel_half + 0.5 * DT * acc_new

    return [Rock(pos_new[i], vel_new[i], m0[i]) for i in range(len(rocks))]

# ====== 合并逻辑 ======
def merge(rocks):
    global total_steps
    new, used = [], set()
    merge_events = 0
    for i, r1 in enumerate(rocks):
        if i in used: continue
        group = [i]
        for j, r2 in enumerate(rocks[i+1:], start=i+1):
            if j in used: continue
            if np.linalg.norm(r1.pos - r2.pos) < MERGE_THRESHOLD * R:
                group.append(j)
                used.add(j)
        if len(group) == 1:
            new.append(r1)
        else:
            merge_events += 1
            ms = np.array([rocks[k].mass for k in group])
            ps = np.array([rocks[k].pos for k in group])
            vs = np.array([rocks[k].vel for k in group])
            Mtot = ms.sum()
            cm_pos = (ms[:,None]*ps).sum(axis=0)/Mtot
            cm_vel = (ms[:,None]*vs).sum(axis=0)/Mtot
            new.append(Rock(cm_pos, cm_vel, Mtot))
    if merge_events > 0:
        sim_time = total_steps * DT
        real_now = datetime.datetime.now()
        log_merge_info(total_steps, merge_events, len(new), sim_time, real_now)
        print(f"[Merge] step={total_steps}, events={merge_events}, left={len(new)}, sim_time={sim_time:.2f}s, real={real_now}")
    return new

# ====== 主程序 ======
if __name__ == "__main__":
    rocks = init_rocks(N)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = None
    origin = ax.scatter([0], [0], [0], c='red', s=100, label='Origin')
    ax.legend()
    ax.set_xlim(-200,200); ax.set_ylim(-200,200); ax.set_zlim(-200,200)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=45)

    norm = plt.Normalize(vmin=0, vmax=200)

    while True:
        for _ in range(SKIP):
            total_steps += 1
            rocks = step_leapfrog(rocks)

        rocks = merge(rocks)

        if len(rocks) < 21:
            pos_arr = np.array([r.pos for r in rocks])
            xs, ys, zs = pos_arr[:,0], pos_arr[:,1], pos_arr[:,2]
            ds = np.linalg.norm(pos_arr, axis=1)

            if total_steps % 10000 == 0:
                dist_str = ", ".join(f"{d:.2f}" for d in ds)
                print(f"[Step {total_steps}] 距离中心: [{dist_str}]")

            for r in rocks:
                r.history.append(r.pos.copy())

            if scatter is None:
                scatter = ax.scatter(xs, ys, zs, c=ds, cmap='viridis', s=50, norm=norm)
                plt.colorbar(scatter, ax=ax, label='Distance')
            else:
                scatter._offsets3d = (xs, ys, zs)
                scatter.set_array(ds)

            for r in rocks:
                trail = np.array(r.history)
                if len(trail) > 1:
                    ax.plot(trail[:,0], trail[:,1], trail[:,2], linewidth=1)

            ax.set_title(f"Step {total_steps} — Bodies: {len(rocks)}")
            plt.draw()
            plt.pause(0.001)

        if len(rocks) <= 1:
            print("所有碎石已合并为一颗，模拟结束。")
            break

    plt.ioff()
    plt.show()
