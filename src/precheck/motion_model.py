"""
MGA · MotionModel：纯数学运动模型（零神经网络）
==============================================
预判帧 System1 的核心：只跟踪障碍物的水平位置，用**最小二乘**拟合它的水平速度，
再预测它到达小恐龙所在 x 的剩余时间 ETA。全程无 NN、无大模型，单帧微秒级。

接口与「手写 pipeline」对齐：
    m = MotionModel(dino_x=50.0)
    m.update(cx, timestamp)          # 每帧喂障碍中心 x + 时间戳
    pred = m.predict(dt)             # -> {"velocity": -320.0, "dino_x": 50.0}
    t_arrive = m.arrival_time(cx)    # -> 到达 dino_x 的剩余秒数(inf 表示不会到)

多个障碍物时由 TrackerHub 做跨帧关联（按水平位移连贯性分配稳定 id）。
"""

import numpy as np


class MotionModel:
    """单个障碍物的水平运动模型：最小二乘估速度 + ETA。"""

    def __init__(self, dino_x: float = 50.0, initial_vx: float = -320.0,
                 window: int = 8):
        self.dino_x = float(dino_x)
        self.vx = float(initial_vx)          # 初速度(向左为负)
        self.window = int(window)            # 最小二乘所用最近样本数
        self._samples: list = []             # [(t, x)]
        self.last_cx = None
        self.last_t = None

    def update(self, cx, timestamp):
        """喂一帧实测的障碍物中心 x 与时间戳，刷新速度估计。"""
        self._samples.append((float(timestamp), float(cx)))
        if len(self._samples) > self.window:
            self._samples.pop(0)
        if len(self._samples) >= 2:
            ts = np.array([s[0] for s in self._samples], dtype=float)
            xs = np.array([s[1] for s in self._samples], dtype=float)
            span = ts[-1] - ts[0]
            if span > 1e-4:
                # 最小二乘斜率 = 水平速度(px/s)，向左为负
                self.vx = float(np.polyfit(ts - ts[0], xs, 1)[0])
        self.last_cx = float(cx)
        self.last_t = float(timestamp)

    def predict(self, dt: float = 0.0) -> dict:
        """返回当前速度估计。dt 保留作未来外推，这里直接返回即时估计。"""
        return {"velocity": float(self.vx), "dino_x": float(self.dino_x)}

    def arrival_time(self, cx=None) -> float:
        """障碍物中心到达 dino_x 的剩余时间(秒)。不向左(vx>=0)则 inf。

        注：小恐龙游戏里障碍从右向左移(vx<0)。ETA = (cx - dino_x) / (-vx)：
        分子(障碍在右为正) / 分母(速度绝对值) = 正实数。
        """
        if cx is None:
            cx = self.last_cx
        if cx is None or self.vx >= 0:
            return float("inf")
        return (cx - self.dino_x) / (-self.vx)

    def reset(self, initial_vx: float = -320.0):
        self._samples.clear()
        self.vx = float(initial_vx)
        self.last_cx = None
        self.last_t = None


class TrackerHub:
    """多障碍物跨帧关联：给每个障碍分配稳定 id，复用对应 MotionModel。

    关联策略：新帧的障碍优先匹配「水平位移连贯(同帧位移<60px)且未老化」的已有 tracker，
    否则新建 id。连续 max_age 帧未见则淘汰。
    """

    def __init__(self, dino_x: float = 50.0, max_age: int = 5):
        self.dino_x = float(dino_x)
        self.max_age = int(max_age)
        self.trackers: dict = {}          # id -> MotionModel
        self.age: dict = {}               # id -> 多少帧没出现
        self._next = 0

    def update(self, obstacles, t) -> list:
        """obstacles: list of (cx, is_bird)。返回 [(id, MotionModel, is_bird, cx), ...]。"""
        live = []
        for cx, is_bird in obstacles:
            best_id = None
            best_d = 1e9
            for tid, mm in self.trackers.items():
                if self.age.get(tid, 0) > self.max_age or mm.last_cx is None:
                    continue
                d = abs(mm.last_cx - cx)
                if d < best_d and d < 60:       # 同一障碍水平位移不应过大
                    best_d, best_id = d, tid
            if best_id is None:
                best_id = self._next
                self._next += 1
                self.trackers[best_id] = MotionModel(dino_x=self.dino_x)
            mm = self.trackers[best_id]
            mm.update(cx, t)
            self.age[best_id] = 0
            live.append((best_id, mm, is_bird, cx))
        # 老化淘汰
        seen = {x[0] for x in live}
        for tid in list(self.trackers.keys()):
            if tid not in seen:
                self.age[tid] = self.age.get(tid, 0) + 1
                if self.age[tid] > self.max_age:
                    del self.trackers[tid]
                    del self.age[tid]
        return live
