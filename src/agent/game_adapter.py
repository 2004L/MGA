"""
game_adapter.py —— 游戏适配器：把"具体游戏"接到通用智能体上
============================================================
智能体核心（orchestrator / system1 / system2）**不认识任何具体游戏**，只认这个接口：

    locate()        定位游戏区域（computer-use：找到窗口/画布在哪）
    grab()          截一帧
    perceive(scene) 像素 → 结构化 Scene（三级降级感知）
    act(action)     执行动作
    tick(dt)        推进一步（sim 用；真机为空操作）

Phase 1：DinoAdapter   —— chrome://dino（简单：完全可观测 / 离散动作）
Phase 2：GTA5Adapter   —— 占位，待接入（复杂：部分可观测 / 连续控制 / 长任务）

复用：感知直接用 demo_dino_real 已真机验证的 detect（YOLOE 主通道 + CV 兜底）。
"""

from __future__ import annotations

import random
import time
from typing import Optional, Tuple

import numpy as np

from .types import Scene


class GameAdapter:
    """游戏适配器抽象基类。新增游戏 = 实现这几个方法，智能体核心一行不用改。"""

    name = "base"
    ACTIONS = ("jump", "squat")

    def locate(self) -> Optional[Tuple[int, int, int, int]]:
        raise NotImplementedError

    def grab(self) -> np.ndarray:
        raise NotImplementedError

    def perceive(self, scene: np.ndarray) -> Scene:
        raise NotImplementedError

    def act(self, action: str) -> None:
        raise NotImplementedError

    def tick(self, dt: float) -> None:
        """推进世界一步（真机不需要；sim 用于推进障碍运动）。"""
        return None


class DinoAdapter(GameAdapter):
    """Phase 1：chrome://dino 小恐龙。"""

    name = "chrome-dino"
    ACTIONS = ("jump", "squat")

    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None,
                 use_dl: bool = False, sim: bool = False,
                 executor=None, verbose: bool = True):
        import demo_dino_real as D          # 复用已真机验证的感知实现
        self.D = D
        self.region = region
        self.use_dl = use_dl
        self.sim = sim
        self.executor = executor
        self.verbose = verbose
        self._sct = None
        # ---- sim 状态 ----
        self.W = region[2] if region else 1000
        self.H = region[3] if region else 180
        self._dino_x = 44.0
        self._obs_x = float(self.W - 30)
        self._vx = -300.0
        self._obs_kind = "ground"
        self._active = 0
        self._next_spawn = 0.6
        # ---- sim 碰撞/跳跃物理（让离线环境也能真正"撞死"→ 驱动 System2 纠错闭环）----
        self._air = 0.0          # 恐龙离地高度(px)
        self._vy = 0.0           # 垂直速度(px/s)
        self._squat = 0.0        # 下蹲剩余时间(s)
        self._dead_flag = False  # 本局是否已撞死
        self._JUMP_V = 600.0     # 起跳初速(px/s)
        self._G = 1800.0         # 重力(px/s^2)
        self._DINO_W = 28
        self._SAFE_AIR = 42.0    # 越过地面障碍所需最小离地高度
        self._t = 0.0            # sim 已运行时间(s)，用于难度爬升

    # ---------------- 定位游戏区域 ----------------
    def locate(self) -> Optional[Tuple[int, int, int, int]]:
        if self.sim:
            self.region = self.region or (0, 0, self.W, self.H)
            return self.region
        # 关键：先聚焦游戏窗口再截图定位。否则全屏截到的是别的窗口，
        # CV 当然找不到游戏区（真机踩过：Chrome 被盖住 → 定位必失败）。
        if self.executor is not None:
            try:
                self.executor.ensure_focus()
                time.sleep(0.4)
            except Exception as e:
                # 不再静默：注释明写「不聚焦就截到别的窗口 → 定位必失败」。
                # 这里失败等于后续定位注定失败，必须留痕。
                print(f"  [定位] ⚠️ 聚焦游戏窗口失败（可能截到别的窗口）："
                      f"{type(e).__name__}: {e}")
        try:
            full = self._grab_full()
            if full is None:
                return None
            try:
                reg = self.D._find_region_from_full(full)
            except TypeError:
                reg = self.D.scan_for_game(full)
            if reg:
                self.region = tuple(reg)
                try:
                    self.D.save_region(self.region)
                except Exception as e:
                    # 不再静默：保存失败 = 下次冷启动无法回落，定位又要重来
                    print(f"  [定位] ⚠️ 游戏区保存失败：{type(e).__name__}: {e}")
                if self.verbose:
                    print(f"  [定位] 自动识别游戏区={self.region}")
                return self.region
        except Exception as e:
            print(f"  [定位] 自动识别失败：{type(e).__name__}: {e}")

        # 兜底：用上次保存的区域（窗口位置没大变时依然有效）
        try:
            saved = self.D.load_region()
            if saved:
                self.region = tuple(saved)
                print(f"  [定位] 回落到上次保存的区域={self.region}")
                return self.region
        except Exception as e:
            print(f"  [定位] ⚠️ 读取上次保存区域失败：{type(e).__name__}: {e}")
        return self.region

    def _grab_full(self) -> Optional[np.ndarray]:
        try:
            if self._sct is None:
                self._sct = self.D._get_sct()
            mons = self._sct.monitors
            mon = mons[1] if len(mons) > 1 else mons[0]
            return self.D._mss_to_rgb(self._sct.grab(mon))
        except Exception as e:
            print(f"  [定位] 全屏截图失败：{type(e).__name__}: {e}")
            return None

    # ---------------- 截图 ----------------
    def grab(self) -> np.ndarray:
        if self.sim:
            obs_x = self._obs_x if (self._active and 0 <= self._obs_x < self.W) else None
            return np.array(self.D.synthetic_shot(
                self.W, self.H, self._dino_x, obs_x,
                self._obs_kind == "bird",
                dino_y=int(self._air), squat=(self._squat > 0)).convert("RGB"))
        if self.region is None:
            raise RuntimeError("未定位游戏区域，请先 locate()")
        if self._sct is None:
            self._sct = self.D._get_sct()
        m = {"top": self.region[1], "left": self.region[0],
             "width": self.region[2], "height": self.region[3]}
        return self.D._mss_to_rgb(self._sct.grab(m))

    # ---------------- 感知（三级降级） ----------------
    def perceive(self, scene: np.ndarray) -> Scene:
        dino_x, obstacles = self.D.detect(scene, self.region, use_dl=self.use_dl)
        return Scene(frame=scene, dino_x=dino_x, obstacles=obstacles,
                     meta={"region": self.region})

    # ---------------- 执行 ----------------
    def act(self, action: str) -> None:
        if self.sim:
            if action == "jump" and self._air <= 0.0:
                self._vy = self._JUMP_V          # 仅在地面时起跳
            elif action == "squat":
                self._squat = 0.35               # 下蹲保持 0.35s
            return
        if self.executor is not None:
            self.executor.act(action)

    # ---------------- sim 推进 ----------------
    def tick(self, dt: float) -> None:
        if not self.sim:
            return
        self._t += dt
        base = 300.0 + min(420.0, self._t * 16.0)   # 难度随时长爬升(越玩越快)
        # 跳跃物理（恐龙垂直运动）
        if self._air > 0.0 or self._vy > 0.0:
            self._vy -= self._G * dt
            self._air += self._vy * dt
            if self._air <= 0.0:
                self._air = 0.0
                self._vy = 0.0
        if self._squat > 0.0:
            self._squat = max(0.0, self._squat - dt)
        # 障碍运动
        if self._active:
            self._obs_x += self._vx * dt
            if self._obs_x < -40:
                self._active = 0
        else:
            self._next_spawn -= dt
            if self._next_spawn <= 0:
                self._active = 1
                self._obs_x = float(self.W - 30)
                self._vx = -base * random.uniform(1.0, 1.6)
                self._obs_kind = "bird" if random.random() < 0.25 else "ground"
                self._next_spawn = random.uniform(0.12, 0.45)   # 短间隔→偶发双障碍
        # 碰撞判定（内部真值，独立于 detect 漏检；产生"死亡"以驱动 System2 纠错）
        if self._active and not self._dead_flag:
            ox0, ox1 = ((self._obs_x, self._obs_x + 18) if self._obs_kind == "ground"
                        else (self._obs_x - 6, self._obs_x + 36))
            dx0, dx1 = self._dino_x, self._dino_x + self._DINO_W
            if ox1 > dx0 and ox0 < dx1:                 # 水平重叠
                if self._obs_kind == "bird":
                    if self._squat <= 0.0:               # 没下蹲 → 撞鸟
                        self._dead_flag = True
                else:
                    if self._air < self._SAFE_AIR:        # 没跳够高 → 撞地刺
                        self._dead_flag = True

    # ---- sim 重开 / 死亡查询（供 orchestrator 在离线环境驱动纠错闭环）----
    def sim_reset(self) -> None:
        """sim 死亡后清场重开（替代真机空格重开）。"""
        self._dead_flag = False
        self._active = 0
        self._obs_x = float(self.W - 30)
        self._air = 0.0
        self._vy = 0.0
        self._squat = 0.0
        self._next_spawn = 0.6

    @property
    def sim_dead(self) -> bool:
        return bool(self._dead_flag)


class GTA5Adapter(GameAdapter):
    """Phase 2 占位：GTA5（复杂游戏）。

    与 dino 的差距（决定了这里要补什么）：
      · 部分可观测：3D 场景、遮挡、HUD 干扰 → 感知要升级到 3D 场景理解
        （YOLOE 开放词汇检车辆/行人/目标物 + 深度/小地图融合）
      · 连续动作：转向/油门/刹车/射击/视角 → 动作空间从 2 个离散变成连续向量，
        System1 小模型要输出连续控制量（残差模型直接预测控制补偿）
      · 稀疏奖励 + 长任务：填一个任务要几百步 → 必须有任务状态机 + 目标进度记忆
      · System2 职责更重：任务分解、导航规划、战术选择，并把战术经验写入记忆

    接入时实现：locate/grab/perceive/act 四个方法即可，orchestrator 与 system1/2 不用改。
    """

    name = "gta5"
    ACTIONS = ("steer", "throttle", "brake", "shoot", "look")

    def __init__(self, *a, **kw):
        raise NotImplementedError(
            "GTA5Adapter 是 Phase 2 扩展位，尚未接入。"
            "接入要点见本类 docstring：3D 感知 / 连续动作 / 任务状态机 / 长程记忆。"
        )
