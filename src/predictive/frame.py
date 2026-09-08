"""
MGA · System 1 预判帧模块（src/predictive/frame.py）
=====================================================
设计原则（与报告 / 对话一致）：
- System 1 是「可训练的习惯性下意识」：解析物理为基，习得残差为进化。
- 90% 的视频帧从「神经网络推理」降级为「纯数学运算 + 一次比较」。
- 闭环：物理预判 -> 预判帧比较 -> 无偏差静默 / 有偏差关键帧 -> 关键帧
        重新理解并输出补偿动作 -> 状态更新 -> 循环。
- SE(3) 就绪：状态可以是 (x,y) 或 (x,y,z,roll,pitch,yaw)，运算同构。
- 元认知门控：用残差方差估计置信度，检测分布偏移（反经验主义）。
- 知识蒸馏：System 2 的每一条纠正样本回流训练 System 1 的残差模型，
  使「习惯」可习得（learned residual）。

依赖：仅 numpy（零框架污染）。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 在线学习的残差模型（System2 -> System1 蒸馏落点）
# ---------------------------------------------------------------------------
class LearnedResidual:
    """
    轻量在线线性残差模型：correction = W @ feature + b。
    这是 System 1 的「习得习惯」——从 System 2 的纠正样本里长出来。
    用递归最小二乘（RLS）在线更新，无需攒批、无需 torch。
    """

    def __init__(self, dim: int, forget: float = 0.99):
        self.dim = dim
        self.W = np.zeros((dim, dim))
        self.b = np.zeros(dim)
        self.P = np.eye(dim) * 1e3
        self.forget = forget

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.W @ state + self.b

    def train_sample(self, state: np.ndarray, correction: np.ndarray):
        x = np.asarray(state, dtype=float)
        y = np.asarray(correction, dtype=float)
        Px = self.P @ x
        denom = self.forget + x @ Px
        K = Px / denom
        e = y - (self.W @ x + self.b)
        self.W += np.outer(K, e)
        self.b += K * e
        self.P = (self.P - np.outer(Px, Px) / denom) / self.forget


# ---------------------------------------------------------------------------
# 启动校准窗口：0.1s 内反推真实初速度，全程无需 AI 参与
# ---------------------------------------------------------------------------
class CalibrationWindow:
    def __init__(self, window: float = 0.15, dt_probe: float = 0.05):
        self.window = window
        self.t = 0.0
        self.samples = []
        self.done = False

    def feed(self, t: float, pos: np.ndarray) -> Optional[np.ndarray]:
        self.t = t
        self.samples.append((t, np.asarray(pos, dtype=float)))
        if not self.done and self.t >= self.window and len(self.samples) >= 2:
            (t1, p1), (t2, p2) = self.samples[0], self.samples[1]
            dt = t2 - t1
            if dt > 1e-6:
                self.done = True
                return (p2 - p1) / dt
        return None


# ---------------------------------------------------------------------------
# 单对象跟踪器：一步前向预测 + 在线速度估计 + 残差回流
# ---------------------------------------------------------------------------
class Track:
    """对单个被跟踪对象做纯数学的一步前向预测（last_pos + v*dt + 残差）。"""

    def __init__(self, pos, vel, dim: int, threshold: float = 0.1):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.dim = dim
        self.threshold = threshold
        self.last_pos = self.pos.copy()
        self.last_t = 0.0
        self.calib = CalibrationWindow()
        self.calibrated = False
        self.residual = LearnedResidual(dim)

    def feed(self, t: float, m: np.ndarray) -> bool:
        """喂实测；校准阶段反推速度。返回是否已校准。"""
        if not self.calibrated:
            v = self.calib.feed(t, m)
            if v is not None:
                self.vel = v
                self.calibrated = True
                self.last_pos = np.asarray(m, dtype=float).copy()
                self.last_t = t
                return True
            self.last_pos = np.asarray(m, dtype=float).copy()
            return False
        return True

    def reset(self, vel):
        """新跟踪对象出现时重置(保留习得残差模型)。避免跨对象位置跳变污染速度估计。"""
        self.vel = np.asarray(vel, dtype=float)
        self.calibrated = False
        self.calib = CalibrationWindow()
        self.last_pos = self.pos.copy()
        self.last_t = 0.0

    def forecast(self, t: float) -> np.ndarray:
        """预测 t 时刻位置：一步前向 + 习得残差。"""
        dt = t - self.last_t
        return self.last_pos + self.vel * dt + self.residual.predict(self.last_pos)

    def update(self, t: float, m: np.ndarray, sys2: Optional[np.ndarray] = None):
        m = np.asarray(m, dtype=float)
        dt = t - self.last_t
        if dt > 1e-6:                       # 在线速度估计（指数平滑）
            nv = (m - self.last_pos) / dt
            self.vel = 0.7 * self.vel + 0.3 * nv
        if sys2 is not None:               # System2 纠正回流 -> 训练残差
            target = np.asarray(sys2, dtype=float) - (self.last_pos + self.vel * dt)
            self.residual.train_sample(self.last_pos, target)
        self.last_pos = m.copy()
        self.last_t = t


# ---------------------------------------------------------------------------
# 置信度 / 元认知门控（反经验主义）
# ---------------------------------------------------------------------------
class ConfidenceMixin:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self._err_ema = 1e-3
        self._history = []

    def _update_confidence(self, err: float) -> float:
        self._err_ema = 0.9 * self._err_ema + 0.1 * (err ** 2)
        self._history.append(err)
        self._history = self._history[-50:]
        return float(np.clip(np.exp(-self._err_ema / (self.threshold ** 2)), 0.0, 1.0))

    def novelty_detected(self) -> bool:
        """分布偏移检测：近期残差显著大于历史 -> 不盲信习惯。"""
        if len(self._history) < 10:
            return False
        hist = np.array(self._history[:-5])
        recent = np.array(self._history[-5:])
        if hist.mean() < 1e-9:
            return bool(recent.mean() > self.threshold)
        return bool(recent.mean() > 2.0 * hist.mean() + self.threshold)


@dataclass
class FrameResult:
    kind: str
    trigger: bool
    error: float
    confidence: float
    info: str = ""


# ---------------------------------------------------------------------------
# 三类预判帧
# ---------------------------------------------------------------------------
class SelfMotionFrame(ConfidenceMixin):
    """① 自身运动预判：预判自己 t 秒后位置，与实际比较。"""

    def __init__(self, pos, vel, threshold: float = 0.1, dim: int = 2):
        super().__init__(threshold)
        self.track = Track(pos, vel, dim, threshold)

    def step(self, t: float, measurement: np.ndarray,
             sys2_correction: Optional[np.ndarray] = None) -> FrameResult:
        m = np.asarray(measurement, dtype=float)
        if not self.track.feed(t, m):
            return FrameResult("self_motion", False, 0.0, 1.0, "calibrating")
        pred = self.track.forecast(t)
        err = float(np.linalg.norm(m - pred))
        conf = self._update_confidence(err)
        trigger = (err > self.threshold) or self.novelty_detected()
        if trigger and sys2_correction is not None:
            self.track.update(t, m, sys2=sys2_correction)
        else:
            self.track.update(t, m)
        return FrameResult("self_motion", trigger, err, conf,
                           f"pred={np.round(pred, 2)} meas={np.round(m, 2)}")


class ExternalObjectFrame(ConfidenceMixin):
    """② 外部物体预判：预判外部物体何时到达我方位置附近。"""

    def __init__(self, pos, vel, self_pos, threshold=0.1, dim=2, arrive_th=0.5):
        super().__init__(threshold)
        self.track = Track(pos, vel, dim, threshold)
        self.self_pos = np.asarray(self_pos, dtype=float)
        self.arrive_th = arrive_th

    def step(self, t: float, measurement: np.ndarray) -> FrameResult:
        m = np.asarray(measurement, dtype=float)
        if not self.track.feed(t, m):
            return FrameResult("external_object", False, 0.0, 1.0, "calibrating")
        pred = self.track.forecast(t)
        err = float(np.linalg.norm(m - pred))
        conf = self._update_confidence(err)
        dist = float(np.linalg.norm(m - self.self_pos))
        trigger = (dist < self.arrive_th) or self.novelty_detected()
        self.track.update(t, m)
        return FrameResult("external_object", trigger, dist, conf,
                           f"dist_to_self={dist:.3f}")


class JointFrame(ConfidenceMixin):
    """③ 联合预判：预判外部物体位置 vs 我方预测位置的碰撞风险。"""

    def __init__(self, self_pos, self_vel, obj_pos, obj_vel,
                 threshold=0.1, dim=2, collide_th=0.3):
        super().__init__(threshold)
        self.self_track = Track(self_pos, self_vel, dim, threshold)
        self.obj_track = Track(obj_pos, obj_vel, dim, threshold)
        self.collide_th = collide_th

    def step(self, t: float, self_meas: np.ndarray, obj_meas: np.ndarray) -> FrameResult:
        sm = np.asarray(self_meas, dtype=float)
        om = np.asarray(obj_meas, dtype=float)
        if not self.self_track.feed(t, sm):
            self.obj_track.feed(t, om)
            return FrameResult("joint", False, 0.0, 1.0, "calibrating")
        self.obj_track.feed(t, om)
        self_pred = self.self_track.forecast(t)
        obj_pred = self.obj_track.forecast(t)
        err = float(np.linalg.norm(sm - self_pred))
        conf = self._update_confidence(err)
        gap = float(np.linalg.norm(self_pred - obj_pred))
        trigger = (gap < self.collide_th) or self.novelty_detected()
        self.self_track.update(t, sm)
        self.obj_track.update(t, om)
        return FrameResult("joint", trigger, gap, conf, f"collision_gap={gap:.3f}")


# ---------------------------------------------------------------------------
# self-check：自身运动 + 偏差触发 + System2 回流 的玩具闭环
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    st_pos = np.array([0.0, 0.0])
    st_vel = np.array([1.0, 0.2])
    f = SelfMotionFrame(st_pos, st_vel, threshold=0.15, dim=2)
    print("== SelfMotionFrame 闭环演示（t=0.7~0.9 世界突变，触发关键帧）==")
    for i in range(12):
        t = i * 0.1
        true = st_pos + st_vel * t
        if 0.7 <= t < 0.9:
            true = true + np.array([0.5, 0.3])   # 物体被推了一下
        r = f.step(t, true)
        tag = "KEYFRAME(S2)" if r.trigger else "silent(S1)"
        print(f"t={t:.1f} {tag:14s} err={r.error:6.3f} conf={r.confidence:.2f} {r.info}")
        if r.trigger:
            f.step(t, true, sys2_correction=true)   # System2 纠正回流
            print("        -> 纠正已回流训练 System1 残差模型")
    print("校准速度:", np.round(f.track.vel, 3), " 校准完成:", f.track.calibrated)
