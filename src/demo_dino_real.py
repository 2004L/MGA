"""
MGA · 真实 Chrome 小恐龙 Pilot（端到端真实演示）
==============================================
目标：真实桌面上的 Chrome 小恐龙被系统自动控制；System1 纯数学预判(无 NN 介入)
      的实测延迟可录屏展示。这是能拿手机/OBS 录给别人看的第一个端到端演示。

链路（对应你列的 6 步）：
  ① 截图        : mss 直接读帧缓冲(比 pyautogui 快，拉高 fps) → 手动框选/自动量游戏区
  ② 检测障碍物  : detect() = 深度学习教育主通道 + CV 兜底。
                 深度学习：YOLOE 开放词汇检测器按名检测 dinosaur/cactus/bird（见
                 perception/dino_detect.py），不受单色配色影响、抗误检，是"用深度学习
                 替代纯色 CV"的落地；权重缺失/未装 ultralytics → 自动降级 CV。
                 CV 兜底：颜色初筛(#535353) + 形状佐证(多证据融合)标出 cactus/bird。
                 恐龙须过形状几何约束(贴地/近方形/剪影占位比)，障碍须过分叉/双翼结构佐证，
                 否则不采纳 —— 单靠颜色会把任务栏/窗口边误认。
                 注：ScreenParser(55 类 UI) 词汇里没有游戏元素，它在本项目负责 GUI 桌面
                 访问（定位 Chrome 窗口），已在 exec/gui_agent.locate_game_window 接好。
  ③ 预判到达时间 : MotionModel 纯数学(最小二乘估速度→ETA)，每个障碍独立 trackers[id]
  ④ 决策+执行   : System2 阈值(arrival < reaction)→ SendInput 原生按键(跳/蹲)
                  （不用 pyautogui：管理员 Chrome 会被 UIPI 吞键，是之前撞车真根因）
  ⑤ 跑通循环    : while 循环持续跑，推测游戏结束自动停（Ctrl+C 也可随时停）
  ⑥ 录屏        : 弹「MGA · Dino Pilot (AI 视角)」标注窗口，手机相机/OBS 录即可

说明：
- System1/System2 分层是核心：90%+ 帧 System1 静默(纯物理)，仅临近才唤醒 System2(决策)。
- 纯数学 ETA 由 src/precheck/motion_model.MotionModel 提供（零神经网络）。

模式：
  --sim             合成截图自测(沙箱可跑，验证 CV+MotionModel+闭环+标注窗口)
  --region X,Y,W,H  真实游戏区屏幕绝对坐标(手动覆盖；不传则自动量/复用存盘)
  --pick-region     弹全屏图拖拽框选游戏区一次并存盘(.dino_region.json)，之后自动复用
  --prefer sendinput|pyautogui   输入后端(默认 sendinput 原生，失败降级 pyautogui)
  --show / --no-show            是否弹标注窗口(默认真机开、sim 关，可手动开)
  --frames N        最大帧数(默认持续跑，撞死自动停)
  --reaction N      触发阈值(秒，默认 0.22；真机验证 0.22 远优于 0.45，越小越晚跳)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from PIL import Image
try:
    import pyautogui
    _HAS_GUI = True
except Exception:
    _HAS_GUI = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    import mss
    _HAS_MSS = True
except Exception:
    _HAS_MSS = False

# mss 单例：直接读帧缓冲，比 pyautogui.screenshot 快很多(拉高 fps，避免漏判)
_SCT = None
def _get_sct():
    global _SCT
    if _SCT is None:
        _SCT = mss.MSS()
    return _SCT

def _mss_to_rgb(shot) -> np.ndarray:
    """mss.ScreenShot.rgb 是 BGR(A) 原始字节 → 转 RGB numpy(对 3/4 通道都稳健)。"""
    w, h = shot.width, shot.height
    buf = shot.rgb
    ch = len(buf) // (w * h)                     # 3(BGR) 或 4(BGRA)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, ch)
    bgr = arr[:, :, :3]                           # 去 alpha(若有)
    return bgr[:, :, ::-1].copy()                # BGR→RGB

# 游戏区存盘：手动/自动量到后存下，下次直接复用(省得每次量坐标)
_REGION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            ".dino_region.json")
def save_region(region: Tuple[int, int, int, int]):
    try:
        with open(_REGION_FILE, "w") as f:
            json.dump({"region": list(region)}, f)
    except Exception:
        pass
def load_region() -> Optional[Tuple[int, int, int, int]]:
    try:
        with open(_REGION_FILE) as f:
            r = json.load(f).get("region")
        if r and len(r) == 4:
            return tuple(int(x) for x in r)
    except Exception:
        pass
    return None


# ============================ 检测（步骤②） ============================
# 小恐龙游戏全元素单色 #535353 on #f7f7f7，靠空间位置区分(无需 cv2，纯 numpy+PIL)。
BG = (247, 247, 247)
FG = (83, 83, 83)        # #535353 所有元素(恐龙/仙人掌/鸟/地平线)单色

# 标注窗口用的配色（BGR：cv2 默认）
C_DINO = (0, 200, 0)       # 绿：恐龙
C_GROUND = (0, 0, 220)     # 红：地面障碍(需跳) = cactus
C_BIRD = (0, 215, 255)     # 黄：高空鸟(需蹲) = bird
C_TXT = (255, 255, 255)     # 白：HUD 文字


@dataclass
class Obstacle:
    x_rel: float          # 相对游戏区左缘的 x(像素，障碍左缘)
    y_bottom: float       # 障碍底部相对 y
    is_bird: bool         # True=高空飞鸟(需蹲/bird)，False=地面(需跳/cactus)


# -------- 形状佐证层（computer use 感知：颜色初筛 + 形状佐证，多证据融合） --------
# 单靠颜色(#535353)不可靠：桌面任务栏/窗口边缘也是深灰横线，会被误认成"地平线"。
# 因此检测必须经过几何形状佐证：恐龙有固定的剪影轮廓，游戏地面有周期点纹，
# 仙人掌/鸟有特定分叉/双翼结构。颜色只做初筛，形状做"作证"，二者齐全才认。

def _dark_mask(scene: np.ndarray) -> np.ndarray:
    """颜色初筛：接近游戏元素色 #535353 的像素 → bool mask。"""
    d = np.abs(scene.astype(int) - np.array(FG)).sum(2)
    return d < 60


def _ground_texture_present(dark: np.ndarray, ground_y: int) -> bool:
    """形状佐证①：地平线下方 3~15px 内是否存在周期性深色点纹（chrome://dino 地面特征）。
    任务栏/窗口边缘下方是纯色或窗口内容，没有这种均匀间隔的小点 —— 最强区分证据。"""
    y0 = int(ground_y) + 3
    y1 = min(dark.shape[0], int(ground_y) + 15)
    if y1 <= y0:
        return False
    band = dark[y0:y1, :]
    colproj = band.sum(0)                       # 每列在 band 内的深灰像素数
    hot = np.where(colproj > 0)[0]
    if len(hot) < 10:
        return False
    # 把连续深灰列段算作"一个地面点"，统计点的数量（点纹=多个分散小点，非一整条长线）
    gaps = np.diff(hot)
    n_dots = 1
    run = 1
    for g in gaps:
        if g <= 3:
            run += 1
        else:
            if run >= 1:
                n_dots += 1
            run = 1
    short_runs = sum(1 for part in np.split(hot, np.where(gaps > 3)[0] + 1) if len(part) <= 6)
    return n_dots >= 5 and short_runs >= 4


def _kill_full_width_rows(mask: np.ndarray, thr: float = 0.6) -> np.ndarray:
    """去掉"横跨整宽"的行(地平线/窗口边框)：它们会把恐龙和右侧障碍连成一片，
    使列连续性失效。全屏 dino 实测地平线行占满 1920px，不去掉就无法分离恐龙与障碍。"""
    m = mask.copy()
    m[m.mean(1) > thr] = False
    return m


def _left_blob_extent(mask: np.ndarray, gap_tol: int = 10):
    """最左侧连续暗列组 → (x0, x1)；gap_tol = 允许跨越的空列数。

    用来求恐龙的**真实横向范围**。旧的固定 90px 左窗在真机全屏下会把恐龙截断：
    全屏恐龙连尾巴实测约 270px 宽(1920x1080)，90px 窗口只截到 x=0~89，
    剩下的身体 x=94~269 被判成"紧贴恐龙的仙人掌" → AI 对着恐龙自己无脑连跳。
    """
    col = mask.any(0)
    idx = np.where(col)[0]
    if len(idx) == 0:
        return None
    x0 = int(idx[0])
    x1 = prev = x0
    for c in idx[1:]:
        c = int(c)
        if c - prev <= gap_tol:
            x1 = c
        else:
            break                      # 出现大间隙 → 恐龙结束，后面是障碍/其他元素
        prev = c
    return (x0, x1)


def _column_groups(mask: np.ndarray, gap_tol: int = 12):
    """整幅 mask 按列切成若干连通块 → [(x0, x1), ...]；gap_tol = 允许跨越的空列数。

    用途：一次返回**所有**候选障碍，而不是只取最近的一个。
    真机实测：右上角的分数文字(HI 00371)是静止的，若只取最近障碍，
    它会一直"遮挡"从右侧逼近的真实仙人掌 —— 改为全部返回后，
    静止物由 MotionModel 按速度(vx≈0 → ETA 无穷)自然过滤掉，永不开火。
    """
    col = mask.any(0)
    idx = np.where(col)[0]
    if len(idx) == 0:
        return []
    groups = []
    x0 = prev = int(idx[0])
    for c in idx[1:]:
        c = int(c)
        if c - prev <= gap_tol:
            prev = c
        else:
            groups.append((x0, prev))
            x0 = prev = c
    groups.append((x0, prev))
    return groups


def _dino_shape_ok(dark: np.ndarray, ground_y: int):
    """形状佐证②：取最左侧暗色块(恐龙)，验证其几何。返回 (ok, dino_x, dino_right)。

    真机修正(2026-09-06)：不再用固定 90px 左窗，改为
    「去地平线行 → 按列连续性求恐龙真实范围」，否则全屏恐龙会被自身截断误判成障碍。
    几何约束：高度 / 宽高比 / 贴地 / 占位比（宽高比放宽：全屏带尾≈2.2，普通窗口≈0.65）。
    """
    dark = dark.copy()
    dark[int(ground_y):, :] = False             # 排除地平线及其下方
    dark = _kill_full_width_rows(dark)          # 再去掉横跨整宽的行(地平线本身)
    ext = _left_blob_extent(dark)
    if ext is None:
        return (False, None, 90)
    x0, x1 = ext
    sub = dark[:, x0:x1 + 1]
    ys, xs = np.where(sub)
    if len(xs) == 0:
        return (False, None, 90)
    y_min, y_max = int(ys.min()), int(ys.max())
    w = x1 - x0 + 1
    h = y_max - y_min + 1
    if not (18 <= h <= 260):
        # 上限 260：全屏/高分屏 chrome://dino 画布自适应放大，实测恐龙高约 120px
        return (False, None, 90)
    if not (0.35 <= w / h <= 3.0):
        # 全屏带尾巴 w/h≈2.25；普通窗口≈0.65。放宽但仍能挡掉细长/扁平的桌面线条
        return (False, None, 90)
    if not (ground_y - 35 <= y_max <= ground_y + 4):
        # 地平线行已被剔除，恐龙底部会略高于 ground_y，故下限放宽到 -35
        return (False, None, 90)
    fill = len(xs) / (w * h)
    if fill < 0.25:                            # 太稀疏(边框/细线)不像恐龙
        return (False, None, 90)
    dino_x = float(xs.mean() + x0)
    dino_right = int(x1) + 8                   # 留 8px 余量，避免尾巴边缘被当成障碍
    return (True, dino_x, dino_right)


def _obstacle_shape(omask: np.ndarray, ground_y: int):
    """形状佐证③：候选障碍(已排除恐龙列)的形状过滤。
    仙人掌：瘦高(主干) 或 含分叉手臂(多条横向带)；鸟：扁 + 高空 + 双横线(翅膀)。
    返回 (is_cactus, is_bird)。"""
    ys, xs = np.where(omask)
    if len(xs) == 0:
        return (False, False)
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    if h < 15:
        return (False, False)                  # 矮纹理(地面碎点)排除
    # 仙人掌：瘦高主干 或 含分叉手臂(一行内有 >=8px 横向游程，且 >=2 个不同高度出现)
    row_runs = 0
    for yy in range(ys.min(), ys.max() + 1):
        row = omask[yy]
        if not row.any():
            continue
        rr = np.diff(np.where(row)[0])
        if len(rr) and rr.max() >= 8:
            row_runs += 1
    if row_runs >= 2 and h >= 20:
        return (True, False)
    if (w / h) < 0.55 and h >= 25:
        return (True, False)
    # 鸟：扁 + 位于地平线上方 30~130px + 上下各有一道横向带(双翼)
    if h < 20 and (ys.min() < ground_y - 30) and (ground_y - ys.min()) <= 130:
        up = omask[ys.min():ys.min() + max(1, h // 2), :].sum(1).max()
        dn = omask[ys.min() + max(1, h // 2):ys.max() + 1, :].sum(1).max()
        if up > 0 and dn > 0:
            return (False, True)
    return (False, False)


# 深度学习障碍检测（YOLOE 开放词汇）懒加载单例 —— 见 perception/dino_detect.py
_DEEP_DETECTOR = None
_DL_WEIGHTS_SPEC = ""      # 由 --dl-weights 命令行参数写入，优先于自动查找
def _get_deep_detector(weights: str = ""):
    """返回 DeepDinoDetector 单例；创建失败(缺依赖/缺权重)返回 None（不崩）。"""
    global _DEEP_DETECTOR
    if _DEEP_DETECTOR is None:
        try:
            from perception.dino_detect import DeepDinoDetector
            spec = weights or _DL_WEIGHTS_SPEC
            _DEEP_DETECTOR = (DeepDinoDetector(weights=spec)
                              if spec else DeepDinoDetector())
        except Exception as e:
            _DEEP_DETECTOR = None
            print(f"  [DL] 深度学习障碍检测不可用：{type(e).__name__}: {e}")
    return _DEEP_DETECTOR


def detect(scene: np.ndarray, region: Tuple[int, int, int, int] = None,
           use_dl: bool = True) -> Tuple[Optional[float], List[Obstacle]]:
    """障碍检测分发器（步骤②）。

    优先用 YOLOE 开放词汇深度学习检测恐龙/仙人掌/鸟（不受单色配色影响、抗误检）；
    深度学习不可用 / 没看到恐龙 / 未装权重 → 自动降级到颜色+形状 CV 兜底。
    返回 (dino_x_rel, obstacles)。"""
    if use_dl:
        det = _get_deep_detector()
        if det is not None and det.load():
            dino_x, raw = det.detect(scene)
            if dino_x is not None:
                obstacles = [Obstacle(x_rel=x, y_bottom=yb, is_bird=ib)
                             for (x, yb, ib) in raw]
                return dino_x, obstacles
            # 深度学习没看到恐龙（可能没截到游戏区）→ 不信任，降级 CV 再确认一次
    return _detect_cv(scene, region)


def _detect_cv(scene: np.ndarray, region: Tuple[int, int, int, int] = None
               ) -> Tuple[Optional[float], List[Obstacle]]:
    """颜色初筛 + 形状佐证（CV 兜底通道）：返回 (dino_x_rel, obstacles)。
    scene: (H,W,3) RGB uint8。仅颜色命中不足够 —— 恐龙/障碍都须经过几何形状佐证才采纳。"""
    H, W = scene.shape[:2]
    ground_y = H - 25                       # 地面线(地平线)相对 y
    mask = _dark_mask(scene)
    mask[ground_y:, :] = False              # 排除地平线及其下方(全宽线会污染检测)
    # 恐龙：左 90px 内、形状佐证通过(贴地站立 + 近方形 + 占位比合理)
    dino_x = None
    dino_right = 90
    ok, dino_x, dino_right = _dino_shape_ok(mask, ground_y)
    if not ok:
        dino_x = None
    # 障碍：恐龙右侧出现的 mask，逐个形状佐证(仙人掌分叉 / 鸟双翼)
    obstacles: List[Obstacle] = []
    if dino_x is not None:
        right = mask.copy()
        right[:, :dino_right] = False          # 按恐龙真实右缘排除(尾巴不再漏成障碍)
        # 去掉横跨整宽的地平线：否则分数文字会被地平线竖向"接"到地面，
        # 每一列都变成百来像素高的"障碍"(全屏实测：分数 x=912 被判成 y_bottom=154 的高障碍)。
        right = _kill_full_width_rows(right)
        # 按列切块 → 每个连通块单独判定并返回(不再只取最近的一个)
        for (bx0, bx1) in _column_groups(right, gap_tol=12):
            band = right[:, bx0:bx1 + 1]
            ys, xs = np.where(band)
            if len(ys) == 0:
                continue
            h = int(ys.max() - ys.min() + 1)
            wc = bx1 - bx0 + 1
            if h < 15 or wc < 4:
                continue                        # 矮纹理/地面碎点(真机 16px 的地面装饰)排除
            is_c, is_b = _obstacle_shape(band, ground_y)
            if not (is_c or is_b):
                continue
            obstacles.append(Obstacle(x_rel=float(bx0), y_bottom=float(ys.max()),
                                      is_bird=is_b))
    return dino_x, obstacles


def _obstacle_box(o: Obstacle):
    """返回 (x1,y1,x2,y2, cx) 用于绘制与 MotionModel 喂中心 x。"""
    if o.is_bird:
        w, h, top = 36, 18, int(o.y_bottom) - 18
    else:
        w, h, top = 18, 42, int(o.y_bottom) - 42
    x1 = int(o.x_rel)
    return (x1, top, x1 + w, int(o.y_bottom)), x1 + w // 2


def synthetic_shot(W: int, H: int, dino_x: float, obs_x: Optional[float], is_bird: bool,
                   dino_y: int = 0, squat: bool = False) -> Image.Image:
    """合成小恐龙截图(沙箱自测用,复刻单色配色)。obs_x=None 或越界时只画恐龙不画障碍。
    dino_y=恐龙离地高度(px,>0 表示起跳); squat=下蹲(画矮一点)。"""
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    img[H - 25:H - 23, :] = FG                              # 地平线
    dino_h = 16 if squat else 32
    dy = H - 25 - dino_h - int(dino_y)
    img[dy:dy + dino_h, int(dino_x):int(dino_x) + 28] = FG       # 恐龙(随跳跃抬高/下蹲变矮)
    if obs_x is not None and 0 <= obs_x < W:
        if is_bird:
            by = H - 25 - 52
            img[by:by + 18, int(obs_x):int(obs_x) + 30] = FG
            img[by + 4:by + 14, int(obs_x) - 6:int(obs_x) + 36] = FG
        else:
            img[H - 25 - 42:H - 25, int(obs_x):int(obs_x) + 18] = FG
    return Image.fromarray(img)


# ============================ 自动量游戏区（步骤①兜底） ============================
def find_canvas_bbox(full: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """端侧自动量游戏区：在全屏图里找 chrome://dino 的浅灰 canvas(#f7f7f7)矩形块。"""
    d = np.abs(full.astype(int) - np.array([247, 247, 247])).sum(2)
    mask = d < 20
    row = mask.mean(1)
    col = mask.mean(0)
    rmax = row.max(); cmax = col.max()
    if rmax < 0.1 or cmax < 0.1:
        return None
    def _longest(a, thr):
        best_s = best_e = 0; cur_s = 0; cur = 0
        for i, v in enumerate(a):
            if v > thr:
                if cur == 0:
                    cur_s = i
                cur += 1
                if cur > best_e - best_s:
                    best_s, best_e = cur_s, cur_s + cur - 1
            else:
                cur = 0
        return best_s, best_e
    y0, y1 = _longest(row, rmax * 0.15)
    x0, x1 = _longest(col, cmax * 0.15)
    x = max(0, int(x0) - 4)
    y = max(0, int(y0) - 4)
    w = int(x1 - x0 + 1) + 8
    h = int(y1 - y0 + 1) + 8
    return (x, y, w, h)


def screen_stats(full: np.ndarray) -> str:
    """全屏颜色诊断：主色分布 + 目标色命中数。"""
    H, W = full.shape[:2]
    lines = [f"  屏幕截图尺寸={W}x{H}"]
    flat = full.reshape(-1, 3).astype(int)
    q = flat // 16
    keys, counts = np.unique(q, axis=0, return_counts=True)
    for i in np.argsort(-counts)[:5]:
        c = (keys[i] * 16).tolist()
        lines.append(f"  主色 RGB~{c}  占比={100 * counts[i] / len(flat):.2f}%")
    for name, t in [("#f7f7f7 浅灰(游戏背景)", (247, 247, 247)),
                    ("#535353 深灰(游戏元素)", (83, 83, 83)),
                    ("#ffffff 纯白(页面背景)", (255, 255, 255))]:
        d = np.abs(full.astype(int) - np.array(t)).sum(2)
        lines.append(f"  接近 {name}: 阈值20→{int((d < 20).sum())}px  阈值60→{int((d < 60).sum())}px")
    mean = float(flat.mean())
    lines.append(f"  全屏平均亮度={mean:.1f}")
    if mean < 100:
        lines.append("  提示：亮度<100 可能是 Chrome 深色模式/游戏页反色")
    elif mean > 240:
        lines.append("  提示：亮度>240 说明背景接近纯白，新版 chrome://dino 常为纯白，建议 --pick-region 手动框选")
    return "\n".join(lines)


def find_ground_bbox(full: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """备用定位：背景不是浅灰时(新版 Chrome 页面为纯白)，改用地平线深灰长线定位。"""
    d = np.abs(full.astype(int) - np.array(FG)).sum(2)
    mask = d < 60
    row = mask.sum(1)
    hy = int(np.argmax(row))
    if row[hy] < 300:
        return None
    xs = np.where(mask[hy])[0]
    x0, x1 = int(xs.min()), int(xs.max())
    h = 150
    y0 = max(0, hy - (h - 25))
    x = max(0, x0 - 4)
    w = min(full.shape[1] - x, x1 - x0 + 9)
    return (x, y0, w, h)


def _validate_region_contains_game(full: np.ndarray, region: Tuple[int, int, int, int],
                                   strict: bool = True) -> bool:
    """防误检（多证据融合）：候选区必须同时具备 ①全宽/局部地平线 ②恐龙形状佐证
    ③地面周期点纹(或恐龙剪影占位比在剪影区间) 才认作真 chrome://dino。
    strict=True 要求地平线几乎全宽覆盖（旧 canvas/ground 用法）；
    strict=False 用于 scan_for_game 窗口化场景，地平线只需覆盖 crop 的 >=50%。"""
    x, y, w, h = region
    x = max(0, x); y = max(0, y)
    crop = full[y:y + h, x:x + w]
    if crop.size == 0 or crop.shape[0] < 60:
        return False
    Hc, Wc = crop.shape[:2]
    ground_y = Hc - 25
    dark = _dark_mask(crop)
    # ① 地平线：用**宽松**暗色阈值（真机实测新版 dino 地平线是极淡浅灰，
    #    严格 #535353 阈值检不出 → 曾导致窗口开着却永远定位不到游戏）；
    #    恐龙形状(②)仍用严格阈值把关，不放松。
    dark_loose = (crop.astype(int).sum(2) < 3 * 170)
    horizon_thr = 0.8 if strict else 0.5
    if dark_loose[ground_y, :].mean() < horizon_thr:
        return False
    # ② 恐龙形状佐证：贴地站立 + 近方形 + 占位比合理(非桌面图标/边框)
    ok, _, _ = _dino_shape_ok(dark, ground_y)
    if not ok:
        return False
    # ③ 地面周期点纹(游戏独有，任务栏/窗口边没有) —— 最强区分证据
    if _ground_texture_present(dark, ground_y):
        return True
    # 降级：未见点纹，但恐龙是"剪影"特征(占位比在剪影区间而非实心矩形/细线)也接受，
    # 避免点纹检测过严误拒真游戏。实心矩形(fill~1.0)、细线条(fill<0.3)均排除 → 挡住任务栏图标。
    # 用清理后的 mask 算 fill，避免全宽地平线把恐龙 bbox 拉大、fill 被压到阈值以下误拒真游戏。
    dark_clean = dark.copy()
    dark_clean[ground_y:, :] = False
    left = dark_clean[:, :min(90, Wc)]
    ys, xs = np.where(left)
    if len(xs):
        fill = len(xs) / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        if 0.30 <= fill <= 0.65:
            return True
    return False


def scan_for_game(full: np.ndarray, min_line: int = 200, max_candidates: int = 200,
                  y_lo: float = 0.5
                  ) -> Optional[Tuple[int, int, int, int]]:
    """    全屏扫描：找「局部深灰地平线 + 地平线左侧有恐龙 + 地面点纹」的区域。
    解决窗口化 chrome://dino（地平线未跨全屏、背景纯白）时 auto_region 失效的问题。

    真机实测坑（2026-09-06）：新版 chrome://dino 的地平线是**极淡的浅灰**
    （远浅于恐龙的 #535353），用 _dark_mask 的严格阈值根本检不出地平线 ——
    这就是"窗口明明开着却怎么都定位不到"的真根因之一。
    所以这里用自适应暗色（每通道 <170，显著比白背景暗即可）找地平线候选；
    恐龙/障碍的确认仍由 validator 用严格 #535353 阈值把关，不放松。
    """
    dark = (full.astype(int).sum(2) < 3 * 170)
    H, W = dark.shape
    candidates = []
    # 默认只扫下半屏（全屏场景，游戏通常在下部，省一半时间）；
    # 若已锁定窗口客户区(y_lo=0)，则全高扫描 —— chrome://dino 画布是垂直居中的。
    for y in range(int(H * y_lo), H):
        row = dark[y]
        if row.sum() < min_line:
            continue
        # 用整行暗像素的**跨度**(最左→最右)，而不是"最长连续段"。
        # 真机实测(2026-09-06 全屏 chrome://dino)：地平线会被障碍物/地面纹理/UI 打断成
        # 多段(实测 x224~1139 与 x1216~1912 两段)，而恐龙在 x≈58，比最长段的起点还靠左。
        # 只取最长连续段 → crop 从 x≈209 开始 → 恐龙被排除 → 形状佐证永远失败。
        # 取跨度则 x0≈8，恐龙自然落进 crop；误检仍由下方 validator 把关。
        xs = np.where(row)[0]
        x0, x1 = int(xs.min()), int(xs.max())
        seg_w = x1 - x0 + 1
        if seg_w >= min_line:
            candidates.append((seg_w, x0, x1, y))
    if not candidates:
        return None
    # 优先验证长线段（更可能是游戏地平线）
    candidates.sort(reverse=True)
    for _, x0, x1, y in candidates[:max_candidates]:
        line_w = x1 - x0 + 1
        # 左留少量余量给恐龙（恐龙在 horizon 左端右侧 ~50px，落进 90px 左窗内即可）；
        # 余量过大反而把恐龙推到左窗外被剪裁 → 误拒。右留 40px 给障碍。
        x = max(0, x0 - 15)
        w = min(W - x, line_w + 100)
        # 让地平线落在 crop 底部往上 25px（与 detect 约定一致）。
        # 数学上：horizon 在 crop 内相对行 = y - top；Hc = bot - top，
        # 取 bot = y+25、top = y-155 → 恒有 horizon 行 = Hc-25，与 detect 约定一致。
        # 越界只裁下边(bot 夹到 H)，绝不重算 top —— 否则地平线对齐会被破坏。
        top = y - 155
        bot = y + 25
        if top < 0:
            top = 0
        if bot > H:
            bot = H
        region = (x, top, w, bot - top)
        if _validate_region_contains_game(full, region, strict=False):
            return region
    return None


def _best_horizon_row(dark: np.ndarray, y_lo: int, y_hi: int):
    """在 [y_lo, y_hi] 行范围内找「最长连续深灰段」→ (y, x0, x1, 段长)。

    用来把大模型给的粗略 ground_y 锚点精修成真实地平线行。
    """
    H, W = dark.shape
    lo, hi = max(0, int(y_lo)), min(H - 1, int(y_hi))
    best = (None, 0, 0, 0)
    if hi < lo:
        return best
    for y in range(lo, hi + 1):
        row = dark[y]
        if not row.any():
            continue
        padded = np.concatenate(([False], row, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        if len(starts) == 0:
            continue
        lens = ends - starts + 1
        i = int(np.argmax(lens))
        if lens[i] > best[3]:
            best = (y, int(starts[i]), int(ends[i]), int(lens[i]))
    return best


def _build_region_from_anchors(full: np.ndarray, ground_y: int, dino_x: int
                               ) -> Optional[Tuple[int, int, int, int]]:
    """用大模型的 ground_y / dino_x 锚点直接构造游戏区（不再要求 CV 点头）。

    布局约定与 detect() 一致：地平线落在 crop 底部往上 25px 处，
    恐龙左侧留 60px，右侧取 1000px 视距（够跟踪 1.5s 以上）。
    """
    H, W = full.shape[:2]
    if not (0 <= ground_y < H):
        return None
    h = 150
    x = max(0, int(dino_x) - 60)
    # 地平线落在 crop 底部 25px（与 detect 约定一致）；越界则重新对齐到 Hc-25
    top = int(ground_y) - (h - 25)
    bot = int(ground_y) + 25
    if top < 0:
        top = 0
    if bot > H:
        bot = H
        top = max(0, int(ground_y) - (H - 25))
    w = min(W - x, 1000)
    if w < 200:
        return None
    return (x, top, w, bot - top)


def _refine_vision_hint(full: np.ndarray, info: dict) -> Optional[Tuple[int, int, int, int]]:
    """用 CV 把大模型的粗锚点精修成精确游戏区。

    大模型语义强但像素定位弱（实测 bbox 误差可达数百 px），
    所以用它的 ground_y / dino_x 当**搜索先验**，由 CV 在附近找真实地平线。
    """
    H, W = full.shape[:2]
    dark = _dark_mask(full)
    gy_hint, dx_hint = info.get("ground_y"), info.get("dino_x")
    reg_hint = info.get("region")

    if gy_hint is not None:
        lo, hi = gy_hint - 70, gy_hint + 70
    elif reg_hint:
        lo, hi = reg_hint[1], reg_hint[1] + reg_hint[3]
    else:
        lo, hi = H // 2, H - 1

    y, x0, x1, ln = _best_horizon_row(dark, lo, hi)
    if ln < 150:                       # 没有像样的地平线 → 精修失败
        return None

    # 恐龙 x：模型给的锚点落在地平线段内才采信，否则用线段左端 +45
    if dx_hint is not None and (x0 - 10) <= dx_hint <= (x1 + 10):
        dino_x = int(dx_hint)
    else:
        dino_x = int(x0) + 45

    h = 150
    x = max(0, dino_x - 60)            # 左侧留 60px 给恐龙
    # 地平线落在 crop 底部 25px（与 detect 约定一致）；越界则重新对齐到 Hc-25
    top = y - (h - 25)
    bot = y + 25
    if top < 0:
        top = 0
    if bot > H:
        bot = H
        top = max(0, y - (H - 25))
    w = min(W - x, 1000)               # 视距 1000px 足够，省算力提帧率
    if w < 200:
        return None
    return (x, top, w, bot - top)


def vision_locate_region(full: np.ndarray,
                         model: str = None) -> Optional[Tuple[int, int, int, int]]:
    """System2 兜底：让大模型看屏定位游戏区，再用 CV 精修。失败返回 None。"""
    try:
        try:
            from perception.vision_locate import locate_game_region
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from perception.vision_locate import locate_game_region
    except Exception as e:
        print(f"  [Vision] 模块不可用：{e}")
        return None

    info = locate_game_region(full, model=model)
    if not info:
        return None

    # ① 语义级已确认是游戏且给了地平线锚点 → 直接构造游戏区。
    #    不再让 CV 二次否决：Game Over 画面(有 GAME OVER 文字/重玩按钮)与运行中画面
    #    像素分布不同，CV 形状佐证会误杀它 —— 而它同样是需要操作的游戏状态。
    gy, dx = info.get("ground_y"), info.get("dino_x")
    if gy is not None and dx is not None:
        region = _build_region_from_anchors(full, gy, dx)
        if region:
            print(f"  [Vision] 按锚点构造游戏区={region}"
                  f"（ground_y={gy} dino_x={dx}）")
            return region

    # ② 没给锚点 → CV 在模型粗框附近精修（仍需形状佐证，避免误检桌面）
    region = _refine_vision_hint(full, info)
    if region and _validate_region_contains_game(full, region, strict=False):
        print(f"  [Vision+CV] 精修后游戏区={region}")
        return region
    raw = info.get("region")
    if raw and _validate_region_contains_game(full, raw, strict=False):
        print(f"  [Vision] 精修失败，采用模型粗框={raw}")
        return raw
    print("  [Vision] 大模型指出了区域，但缺少可用锚点且 CV 未确认")
    return None


def _find_region_from_full(full: np.ndarray,
                           use_vision: bool = True) -> Tuple[int, int, int, int]:
    # 1) 新扫描：支持窗口化游戏（地平线未跨全屏）
    bb = scan_for_game(full)
    if bb is not None:
        print(f"  [auto-region] 扫描定位到游戏区={bb}")
        return bb
    # 2) 旧方法：全屏 canvas/ground（全屏游戏时更快）
    bb = find_canvas_bbox(full) or find_ground_bbox(full)
    if bb is not None and _validate_region_contains_game(full, bb):
        return bb
    # 3) System2 兜底：大模型看屏定位（约 2 秒，只跑一次）
    if use_vision:
        print("  [auto-region] CV 没定位到，改用大模型看屏定位...")
        bb = vision_locate_region(full)
        if bb is not None:
            return bb
    # 失败诊断
    print("  [诊断] 全屏颜色统计（定位为什么找不到游戏区）：")
    print(screen_stats(full))
    raise RuntimeError("没找到游戏区。请先打开 chrome://dino，或手动 --pick-region / --region X,Y,W,H")


def _region_from_window(full: np.ndarray,
                        win_region: Tuple[int, int, int, int],
                        use_vision: bool = True
                        ) -> Optional[Tuple[int, int, int, int]]:
    """在**已知的 Chrome 窗口客户区**内用 CV 找游戏地平线。

    多证据融合的关键一环：先用 OS 拿到窗口矩形（语义级，零误差），
    再只在这个窗口里找地平线（像素级）—— 搜索空间从"整个桌面"缩小到"一个窗口"，
    任务栏/其他窗口的横线彻底不可能再被误检成游戏地面。
    """
    x, y, w, h = win_region
    H, W = full.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    crop = full[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    # 窗口内全高扫描（dino 画布垂直居中，不能只扫下半部分）
    bb = scan_for_game(crop, y_lo=0.0)
    if bb is None and use_vision:
        # 大模型看这个窗口的截图（范围已缩小到单个窗口，定位精度远高于全屏）
        bb = vision_locate_region(crop)
    if bb is None:
        return None
    return (x0 + bb[0], y0 + bb[1], bb[2], bb[3])


def _grab_full() -> np.ndarray:
    """截全屏一次（mss 优先，pyautogui 兜底）。"""
    if _HAS_MSS:
        try:
            sct = _get_sct()
            return _mss_to_rgb(sct.grab(sct.monitors[0]))
        except Exception as e:
            print(f"  [auto-region] mss 全屏失败，降级 pyautogui：{e}")
    if not _HAS_GUI:
        raise RuntimeError("自动量 region 需要 mss 或 pyautogui + 真实桌面")
    return np.array(pyautogui.screenshot().convert("RGB"))


def auto_region(use_vision: bool = True,
                use_window: bool = True) -> Tuple[int, int, int, int]:
    """定位游戏区，三级证据链（从最可靠到最兜底）：

      ① OS 窗口级（GUI 桌面访问）：枚举真实窗口找 Chrome → 最小化则还原 → 聚焦
         → 取客户区屏幕矩形（坐标来自操作系统，零误差；顺手解决
         "窗口最小化导致怎么找都找不到"的顽疾）→ 窗口内用 CV 形状佐证找地平线
      ② 全屏 CV 扫描：scan_for_game（支持地平线不跨全屏的窗口化场景）
      ③ 大模型看屏定位：Vision LLM 约 2 秒，只跑一次，成功即存盘复用

    use_window=False 可退回纯像素方案；use_vision=False 可关大模型(离线/省 token)。
    """
    # ① 窗口级
    if use_window:
        try:
            from exec.gui_agent import locate_game_window
        except Exception as e:
            print(f"  [auto-region] 窗口级定位不可用（{type(e).__name__}），走像素方案")
        else:
            win = None
            try:
                win = locate_game_window(verbose=True)
            except Exception as e:
                print(f"  [auto-region] 窗口定位异常：{type(e).__name__}: {e}")
            if win:
                full = _grab_full()
                r = _region_from_window(full, win, use_vision=use_vision)
                if r:
                    print(f"  [auto-region] 窗口内定位到游戏区={r}")
                    return r
                if _validate_region_contains_game(full, win, strict=False):
                    print(f"  [auto-region] 整个客户区即游戏区={win}")
                    return win
                print("  [auto-region] Chrome 窗口内没检测到游戏地平线，转全屏像素扫描...")
                return _find_region_from_full(full, use_vision=use_vision)
    # ②③ 像素级 + 大模型兜底
    return _find_region_from_full(_grab_full(), use_vision=use_vision)


def pick_region() -> Tuple[int, int, int, int]:
    """拖拽框选游戏区一次，存盘(.dino_region.json)，后续直接复用(省时间)。"""
    if not _HAS_CV2:
        raise RuntimeError("pick-region 需要 cv2(请 pip install opencv-python)")
    if not _HAS_MSS:
        raise RuntimeError("pick-region 需要 mss(请 pip install mss)")
    sct = _get_sct()
    full = _mss_to_rgb(sct.grab(sct.monitors[0]))
    H, W = full.shape[:2]
    scale = min(1.0, 1280.0 / W)
    disp = cv2.resize(full[..., ::-1], (int(W * scale), int(H * scale)))
    ix = iy = -1
    drawing = False
    rect = None
    win = "pick region (左键拖拽框选, 回车确认, q 取消)"
    def cb(event, x, y, flags, param):
        nonlocal ix, iy, drawing, rect
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing, ix, iy = True, x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect = (ix, iy, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            rect = (ix, iy, x, y)
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cb)
    print("  [pick-region] 在弹出的全屏图上拖拽框选 Chrome 小恐龙游戏区，回车确认")
    while True:
        img = disp.copy()
        if rect:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 200, 0), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        if k == 13 and rect:
            x0, y0, x1, y1 = rect
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            region = (int(x0 / scale), int(y0 / scale),
                      int((x1 - x0) / scale), int((y1 - y0) / scale))
            save_region(region)
            print(f"  [pick-region] 已保存游戏区={region} → {_REGION_FILE}")
            cv2.destroyAllWindows()
            return region
        if k == ord('q'):
            cv2.destroyAllWindows()
            raise RuntimeError("已取消 pick-region")


# ============================ Pilot（步骤①③④⑤⑥） ============================
from precheck.motion_model import MotionModel, TrackerHub   # 步骤③：纯数学 ETA
from predictive.action import ActionLatencyPredictor         # System1 深化：执行延迟预判
from exec.backends import build_backend                      # 步骤④：SendInput 原生后端
from economy.token import TokenEconomy


class ChromeDinoPilot:
    # 紧密双障碍判定窗口：两障碍 ETA 之差小于此值视为"成对"(秒)
    PAIR_GAP = 0.55

    def __init__(self, region: Tuple[int, int, int, int], sim: bool = False,
                 reaction_time: float = 0.45, dino_x: float = 44.0,
                 prefer: str = "sendinput", show: bool = False,
                 use_dl: bool = True, reaction_pair: float = 0.30,
                 airtime: float = 0.6,
                 adapt: bool = False, adapt_floor: float = 0.6,
                 adapt_vref: float = 320.0, adapt_k: float = 0.5):
        self.region = region
        self.sim = sim
        self.show = show and _HAS_CV2
        self.W, self.H = region[2], region[3]
        self.dino_x = dino_x                  # System1 触发阈值用的恐龙位(真机首帧校准一次)
        self._dino_draw_x = dino_x
        self._dino_refined = False
        self.reaction_time = reaction_time
        self.reaction_pair = reaction_pair    # 紧密双障碍(均地面)触发阈值：晚于单障碍，把死亡带移位
        self.airtime = airtime                # 跳跃滞空估计(秒)：空中抑制窗，避免滞空期误触发/误标记
        self.adapt = adapt                    # 速度自适应反应：高速时把触发阈值随 vx 压缩(反应更早)
        self.adapt_floor = adapt_floor        # 自适应压缩下限系数(0.6=高速时最多压到 60%)
        self.adapt_vref = adapt_vref          # 参考速度(px/s)，此速度及以下不压缩
        self.adapt_k = adapt_k                # 每超过 (vx/vref - 1) 倍的压缩强度
        self.use_dl = use_dl                  # 步骤②：深度学习障碍检测开关(默认开)
        self.hub = TrackerHub(dino_x=dino_x, max_age=5)   # 步骤③：每障碍独立 MotionModel
        self.eco = TokenEconomy(init=1000.0, safe=400.0, cost_sys1=0.1, cost_sys2=100.0)
        self.latencies = []          # MotionModel.arrival_time 纯数学耗时(秒)
        self.frame_times = []
        self.act_lat = ActionLatencyPredictor(window=30, default_ms=80.0)
        self.t = 0.0
        self._fired = set()          # 已对哪些障碍 id 触发过 System2(避免重复扣费/重复跳)
        self._backend_name = "?"
        self._il = "?"
        if not self.sim:
            try:
                self.backend = build_backend(prefer=prefer)
                self._backend_name = self.backend.name
            except Exception as e:
                print(f"  [后端] 创建输入后端失败：{e}，动作将不执行")
                self.backend = None
        else:
            self.backend = None
        # sim 状态
        self._obs_x = float(self.W - 30)
        self._obs_kind = "ground"
        self._vx = -300.0
        self._spawn_gap = (0.45, 1.0)
        self._next_spawn = 0.5
        self._active = 0
        self._gap_streak = 0
        self._ever_fired = False
        self.deaths = 0                     # 本局累计撞死次数(自动重开后累加)
        self.auto_restart = True            # 撞死后自动按空格重开，让闭环真正无限自主
        self._airborne_until = 0.0          # 空中抑制：此时间戳前不重复发跳(游戏无二段跳)
        self._suppressed = 0                # 空中抑制触发次数(调试用)

    def _adapt_react(self, react, vx):
        # 速度自适应反应：高速时障碍 ETA 缩小、系统绝对延迟(按键注入+帧间隔)占比变大，
        # 固定阈值相对偏晚→高速单障碍易死。按 vx 把阈值向早压缩，封底 react*floor。
        if not self.adapt or vx <= 0:
            return react
        ratio = vx / self.adapt_vref
        if ratio <= 1.0:
            return react
        f = 1.0 - self.adapt_k * (ratio - 1.0)
        f = max(self.adapt_floor, f)
        return react * f

    # ---- 步骤① 截图 ----
    def _grab(self) -> np.ndarray:
        if self.sim:
            obs_x = self._obs_x if (self._active and 0 <= self._obs_x < self.W) else None
            return np.array(synthetic_shot(self.W, self.H, self._dino_draw_x, obs_x,
                                           self._obs_kind == "bird").convert("RGB"))
        if _HAS_MSS:
            try:
                m = {"top": self.region[1], "left": self.region[0],
                     "width": self.region[2], "height": self.region[3]}
                return _mss_to_rgb(_get_sct().grab(m))
            except Exception as e:
                print(f"  [截图] mss 失败，降级 pyautogui：{e}")
        if not _HAS_GUI:
            raise RuntimeError("真机模式需要 mss 或 pyautogui + 真实桌面")
        shot = pyautogui.screenshot(region=self.region)
        return np.array(shot.convert("RGB"))

    # ---- 步骤④ 执行 ----
    def _act(self, action: str):
        if self.sim or self.backend is None:
            return
        key = "space" if action == "jump" else "down"
        self.backend.key_down(key)
        time.sleep(0.05)                 # 长按 50ms，极短按下游戏可能采样不到
        self.backend.key_up(key)

    # ---- 步骤②+③ 绘制标注窗口 ----
    def _draw_overlay(self, scene, dino_x, obstacles, near_info, action):
        if not self.show:
            return
        H, W = scene.shape[:2]
        frame = scene[..., ::-1].copy()
        if dino_x is not None:
            x0, x1 = int(dino_x), int(dino_x) + 28
            y0, y1 = H - 25 - 47, H - 25
            cv2.rectangle(frame, (x0, y0), (x1, y1), C_DINO, 2)
            cv2.putText(frame, "DINO", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_DINO, 1)
        for o in obstacles:
            (bx0, by0, bx1, by1), _ = _obstacle_box(o)
            col = C_BIRD if o.is_bird else C_GROUND
            cv2.rectangle(frame, (bx0, by0), (bx1, by1), col, 2)
            cv2.putText(frame, "BIRD" if o.is_bird else "CACTUS",
                        (bx0, by0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        fps = (1000.0 / np.array(self.frame_times[-20:]).mean() / 1000.0) if self.frame_times else 0
        lat = np.array(self.latencies) * 1e6 if self.latencies else np.array([0.0])
        info = near_info or {}
        lines = [
            f"ETA={info.get('eta', 0):.3f}s  v={info.get('vx', 0):.0f}px/s  -> {action.upper() if action else '...'}",
            f"S2阈值={self.reaction_time:.2f}s  backend={self._backend_name}  IL={self._il}",
            f"fps={fps:.1f}  ETA算={lat.mean():.1f}µs  lead={self.act_lat.predict('sense_ms')/1000+self.act_lat.predict('act_ms')/1000:.3f}s",
        ]
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (8, 18 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TXT, 1)
        if action:
            cv2.rectangle(frame, (2, 2), (W - 3, H - 3), C_DINO, 3)
        try:
            cv2.imshow("MGA · Dino Pilot (AI 视角)", frame)
            cv2.waitKey(1)
        except Exception as e:
            print(f"  [标注窗口] 无法显示(无 GUI?)，已关闭：{e}")
            self.show = False

    def _keyboard_selftest(self) -> bool:
        """真机自检：按一次空格看画面是否变化 → 判定按键是否真作用到游戏窗口。
        并打印输入后端 + 自身完整性级别(IL)：若自身 medium IL 而目标 Chrome 是 high IL，
        UIPI 会直接丢弃输入 —— 这是"按键进不去"的真根因。"""
        if self.backend is not None and hasattr(self.backend, "integrity_level"):
            try:
                self._il = self.backend.integrity_level()
            except Exception:
                self._il = "?"
        print(f"  [后端] {self._backend_name}  自身完整性级别(IL)={self._il}")
        if self._il == "medium":
            print("  [提示] 若 Chrome 以管理员身份运行，UIPI 会丢弃本进程的输入；"
                  "请让 Chrome 以普通权限启动，或把本脚本也提权运行。")

        # 关键：Chrome 窗口聚焦 ≠ 页面内容聚焦。焦点常在地址栏，空格会被它吃掉。
        # Computer use 的标准做法：先点一下游戏区，让页面 canvas 拿到焦点，再发键。
        # 死局(GAME OVER)时空格=重开；若首帧差异过小说明键没送达，重试一次(再点回+重发)。
        if not self.sim:
            def _focus_and_probe() -> float:
                cx = self.region[0] + self.region[2] // 2
                cy = self.region[1] + self.region[3] // 2
                try:
                    if hasattr(self.backend, "click"):
                        self.backend.click(cx, cy)
                    elif _HAS_GUI:
                        pyautogui.click(cx, cy)
                except Exception as e:
                    print(f"  [聚焦] 点击游戏区失败：{type(e).__name__}: {e}")
                time.sleep(0.35)
                a = self._grab()
                self._act("jump")
                time.sleep(0.4)
                b = self._grab()
                return float(np.abs(a.astype(int) - b.astype(int)).mean())

            diff = _focus_and_probe()
            # 差异过小：可能窗口被盖住 / 游戏处于 GAME OVER。再试一次(空格在 GAME OVER 后=重开)
            if diff <= 0.5:
                print(f"  [按键自检] 首帧差异={diff:.2f} 过小，重试一次(点回+重发空格/重开)")
                time.sleep(0.3)
                diff = _focus_and_probe()
            ok = diff > 0.5
            print(f"  [按键自检] 画面差异={diff:.2f} → "
                  f"{'按键有效(游戏有响应)' if ok else '按键无效! 请点回 Chrome 窗口 / 检查 Chrome 是否管理员运行'}")
            return ok
        # sim 模式无真实键盘，直接返回 True
        return True

    # ---- 步骤⑤ 主循环 ----
    def run(self, n_frames: int = 400, dt: float = 0.05, seed: int = 3,
            max_gap_frames: int = 240):
        rng = np.random.default_rng(seed)
        s1_t = s2_t = hits = misses = 0
        print(f"== Chrome 小恐龙 Pilot ({'SIM 自测' if self.sim else 'REAL 真机'}) ==")
        print(f"   region={self.region} dt={dt}s 反应阈值={self.reaction_time}s"
              f"  速度自适应={'开' if self.adapt else '关'}(vref={self.adapt_vref:.0f} floor={self.adapt_floor:.2f} k={self.adapt_k:.2f})"
              f"  标注窗口={'开' if self.show else '关'}")
        if not self.sim:
            self._keyboard_selftest()
        t_start = time.perf_counter()
        last_action = ""
        last_info: dict = {}
        for i in range(n_frames):
            f0 = time.perf_counter()
            self.t = (self.t + dt) if self.sim else (f0 - t_start)
            # sim：生成/移动障碍
            if self.sim:
                if self._active == 0 and self.t >= self._next_spawn:
                    self._active = 1
                    self._obs_x = float(self.W - 30)
                    self._obs_kind = "bird" if rng.random() < 0.3 else "ground"
                    self._next_spawn = self.t + rng.uniform(*self._spawn_gap)
                if self._active:
                    self._obs_x += self._vx * dt
                    if self._obs_x < -40:
                        self._active = 0
            # 步骤② 检测
            sense_t0 = time.perf_counter()
            scene = self._grab()
            dino_x, obstacles = detect(scene, self.region, use_dl=self.use_dl)
            self.act_lat.record("sense_ms", (time.perf_counter() - sense_t0) * 1000)
            self.frame_times.append(time.perf_counter() - f0)
            if dino_x is not None and not self._dino_refined:
                self.dino_x = dino_x
                self.hub.dino_x = dino_x           # 真机首帧校准恐龙位
                self._dino_refined = True
            # 步骤③ 喂 MotionModel（每障碍独立 tracker）
            obs_pairs = [(_obstacle_box(o)[1], o.is_bird) for o in obstacles]
            live = self.hub.update(obs_pairs, self.t)
            # 用「是否有在移动的障碍」判间隙：GAME OVER / 暂停屏上常有静止的分数文字等
            # 假障碍(vx≈0)，若用 live 非空判间隙会被这些静止假阳性骗过 → 永远不触发重开。
            has_moving = any(abs(mm.vx) > 30.0 for (_, mm, _, _) in live)
            if not has_moving:
                # 间隙：无障碍(在移动)
                self.eco.act(use_sys2=False, env_reward=0.1)
                s1_t += 1
                last_action = ""
                last_info = {}
                self._gap_streak += 1
                if self._gap_streak > max_gap_frames and self.t > 4.0:
                    # 连续 240 帧无障碍 ⇒ 游戏已结束(GAME OVER)/窗口失焦。
                    # 真实对局里障碍间隔远小于 240 帧，故这里可靠等价于"撞死"。
                    # 去掉 _ever_fired 前置：开局即死局、或首障前就死，也能自动重开。
                    self.deaths += 1
                    if self.auto_restart:
                        # 撞死 → 自动重开(chrome://dino 在 GAME OVER 后空格/上键重开)，
                        # 让 AI 真正"自主"地一直玩下去，不让单局死亡打断闭环。
                        print(f"\n  [游戏结束] 第{self.deaths}次撞死，自动重开继续自主游玩…"
                              "（Ctrl+C 随时停）")
                        try:
                            cx = self.region[0] + self.region[2] // 2
                            cy = self.region[1] + self.region[3] // 2
                            if hasattr(self.backend, "click"):
                                self.backend.click(cx, cy)     # 先点回 canvas 拿焦点
                            elif _HAS_GUI:
                                pyautogui.click(cx, cy)
                        except Exception:
                            pass
                        self._act("jump")
                        time.sleep(0.6)
                        # 重置本局状态进入下一局
                        self._gap_streak = 0
                        self._ever_fired = False
                        self._fired.clear()
                        self.hub.trackers.clear()
                        self.hub.age.clear()
                        self._dino_refined = False   # 下一局首帧重新校准恐龙位
                        self._active = 0
                        continue
                    else:
                        print(f"\n  [推测游戏结束] 已连续 {self._gap_streak} 帧无障碍，自动退出。"
                              "（Ctrl+C 也可随时停）")
                        break
                self._draw_overlay(scene, dino_x, obstacles, last_info, last_action)
                continue
            self._gap_streak = 0
            # 步骤④：从所有"未触发过"的障碍里挑最近(ETA 最小)做决策；trail=次近(用于双障碍判定)
            cand = []
            for tid, mm, is_bird, cx in live:
                t0 = time.perf_counter()
                arr = mm.arrival_time(cx)
                self.latencies.append(time.perf_counter() - t0)
                if tid not in self._fired:
                    cand.append((arr, tid, mm, is_bird, cx))
            if not cand:
                # 全部已触发(都在飞过/已标记)：本帧无需决策
                self._draw_overlay(scene, dino_x, obstacles, last_info, last_action)
                continue
            cand.sort(key=lambda c: c[0])
            best = cand[0]
            trail = cand[1] if len(cand) > 1 else None
            arr, tid, mm, is_bird, cx = best
            # 执行延迟补偿：把"自己这一侧开销"算进提前量(否则帧率低系统性偏晚)
            lead = (self.act_lat.predict("sense_ms") + self.act_lat.predict("act_ms")) / 1000.0
            # 紧密双障碍(lead 与 trail 都是地面障碍且间距近)→ 用更晚的触发阈值，
            # 把"单跳清不掉又来不及补跳"的死亡带移到游戏更少出现的间距。
            pair_ground = (trail is not None and (not is_bird) and (not trail[3])
                           and (trail[0] - arr) < self.PAIR_GAP)
            react = self.reaction_pair if pair_ground else self.reaction_time
            react = self._adapt_react(react, mm.vx)   # 速度自适应：高速时反应更早
            action = "squat" if is_bird else "jump"      # 步骤④ System2 决策
            # 空中抑制：滞空期间不重复发跳(游戏不接受二段跳；旧逻辑会在此把尾随障碍误标记已触发，
            # 导致落地后不再补跳→撞死)。滞空时只静默等待，落地后由 trail 重新触发。
            airborne = self.t < self._airborne_until
            suppress_jump = airborne and action == "jump"
            if suppress_jump:
                self._suppressed += 1
            need_fire = (arr < react + lead) and (tid not in self._fired) and (not suppress_jump)
            # 漏判：障碍已越过恐龙却从未触发
            if cx < self.dino_x - 30 and tid not in self._fired:
                misses += 1
                self._fired.add(tid)
                self.eco.act(use_sys2=False, env_reward=-50.0)
            if need_fire:
                self._fired.add(tid)
                self._ever_fired = True
                act_t0 = time.perf_counter()
                self._act(action)                            # 步骤④ 执行
                self.act_lat.record("act_ms", (time.perf_counter() - act_t0) * 1000)
                if action == "jump":
                    self._airborne_until = self.t + self.airtime
                self.eco.act(use_sys2=True, env_reward=30.0)
                s2_t += 1
                hits += 1
                last_action = action
                last_info = {"eta": arr, "vx": mm.vx, "pair": pair_ground, "react": react}
                if i % 10 == 0 or not self.sim:
                    tag = " PAIR" if pair_ground else ""
                    print(f"  t={self.t:5.2f} KEYFRAME(S2{tag}) ETA={arr:6.3f}s R={react:.2f} "
                          f"{'bird' if is_bird else 'cactus'}→{action} {self.eco.report()}")
            else:
                self.eco.act(use_sys2=False, env_reward=0.1)
                s1_t += 1
                last_action = ""
                last_info = {"eta": arr, "vx": mm.vx}
            self._draw_overlay(scene, dino_x, obstacles, last_info, last_action)
        if self.show:
            cv2.destroyAllWindows()
        self._report(s1_t, s2_t, hits, misses)

    def _report(self, s1, s2, hits, misses):
        total = s1 + s2
        print("\n== 结果 ==")
        print(f"  System1静默={s1}({100*s1/total:.1f}%)  System2关键帧={s2}({100*s2/total:.1f}%)")
        print(f"  触发决策={hits+misses}  成功躲过={hits}  漏判撞车={misses}")
        if self._suppressed:
            print(f"  空中抑制(滞空期拦截的重复跳)={self._suppressed}")
        if self.deaths:
            print(f"  撞死次数={self.deaths}  (已自动重开，闭环未中断)")
        if self.latencies:
            lat = np.array(self.latencies) * 1e6
            print(f"  [MotionModel 纯数学 ETA] avg={lat.mean():.2f}µs  p99={np.percentile(lat,99):.2f}µs  "
                  f"max={lat.max():.2f}µs  (零神经网络介入)")
        if self.frame_times:
            ft = np.array(self.frame_times) * 1000
            print(f"  [帧率] avg={ft.mean():.0f}ms/帧 (≈{1000/ft.mean():.1f}fps)  p99={np.percentile(ft,99):.0f}ms")
        print(f"  [执行延迟预判] {self.act_lat.stats()}")
        print(f"  [TokenEconomy] {self.eco.report()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="合成截图自测(沙箱可跑)")
    ap.add_argument("--region", default="", help="真实游戏区 X,Y,W,H(屏幕绝对像素)")
    ap.add_argument("--pick-region", action="store_true",
                    help="弹全屏图拖拽框选游戏区一次并存盘(.dino_region.json)，之后自动复用")
    ap.add_argument("--frames", type=int, default=20000, help="最大帧数(默认 20000，撞死自动停)")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--reaction", type=float, default=0.22, help="触发阈值(秒)，真机验证 0.22 远优于 0.45(0.45 跳太早→落地即撞)；你伪代码用 0.3 也可")
    ap.add_argument("--reaction-pair", type=float, default=0.30,
                    help="紧密双障碍(均地面)专用触发阈值(秒，默认 0.30，晚于单障碍以移位死亡带)；"
                         "越小越早跳覆盖更宽间距，越大越晚跳")
    ap.add_argument("--airtime", type=float, default=0.6,
                    help="跳跃滞空估计(秒，默认 0.6)，空中抑制窗长；估值偏大则抑制略久(安全)，偏小则可能漏抑制")
    ap.add_argument("--adapt-floor", type=float, default=0.6,
                    help="速度自适应反应压缩下限系数(默认 0.6)：高速时反应阈值最多压到原值的 60%")
    ap.add_argument("--adapt-vref", type=float, default=320.0,
                    help="速度自适应参考速度 px/s(默认 320)：此速度及以下不压缩")
    ap.add_argument("--adapt-k", type=float, default=0.5,
                    help="速度自适应压缩强度(默认 0.5)：每超过参考速度 1 倍压缩 0.5")
    ap.add_argument("--adapt", action="store_true",
                    help="开启速度自适应反应(默认关；真机 10000 帧对照显示对 chrome dino 略增死亡 7.4%% vs 4.95%%，"
                         "因固定阈值几何上已速度无关，提前跳反伤双障碍场景)")
    ap.add_argument("--prefer", default="sendinput", choices=["sendinput", "pyautogui"],
                    help="输入后端：sendinput=Windows 原生(默认)，pyautogui=跨平台兜底")
    ap.add_argument("--show", dest="show", action="store_true", help="弹标注窗口(供 OBS/手机录屏)")
    ap.add_argument("--no-show", dest="show", action="store_const", const=False, help="不弹标注窗口")
    ap.add_argument("--no-vision", action="store_true",
                    help="关闭大模型看屏定位兜底(离线/省 token)，只用端侧 CV")
    ap.add_argument("--no-window", action="store_true",
                    help="关闭 OS 窗口级定位(不还原/聚焦 Chrome)，只用像素扫描")
    ap.add_argument("--no-dl", action="store_true",
                    help="关闭深度学习障碍检测(YOLOE 开放词汇)，只用颜色+形状 CV 兜底")
    ap.add_argument("--no-restart", action="store_true",
                    help="关闭撞死后自动重开(默认自动重开，让 AI 一直自主玩)")
    ap.add_argument("--dl-weights", default="",
                    help="指定 YOLOE 权重路径/文件名(默认在 blobs/weights 找 yoloe*.pt)")
    ap.add_argument("--vision-model", default="",
                    help="视觉模型名(默认 hy-vision-2.0-instruct)")
    args = ap.parse_args()
    default_show = (not args.sim)
    if args.dl_weights:
        global _DL_WEIGHTS_SPEC
        _DL_WEIGHTS_SPEC = args.dl_weights
    show = default_show if args.show is None else args.show
    if args.pick_region:
        try:
            region = pick_region()
        except Exception as e:
            print(f"pick-region 失败：{e}")
            return
    elif args.sim:
        region = (0, 0, 600, 150)
    elif args.region:
        region = tuple(int(x) for x in args.region.split(","))
        if len(region) != 4:
            print("region 格式错误，应为 X,Y,W,H")
            return
    else:
        try:
            region = auto_region(use_vision=(not args.no_vision),
                                 use_window=(not args.no_window))
            save_region(region)
            if region[2] > 1000:
                region = (region[0], region[1], 1000, region[3])
                save_region(region)
                print(f"  [auto-region] 游戏区过宽，截取左 1000px 提帧率 → {region}")
            else:
                print(f"  [auto-region] 端侧自动量到游戏区={region}")
        except Exception as e:
            saved = load_region()
            if saved:
                try:
                    full = _mss_to_rgb(_get_sct().grab(_get_sct().monitors[0]))
                    if _validate_region_contains_game(full, saved):
                        print(f"自动量失败({e})，复用上次保存且校验通过的游戏区 {saved}")
                        region = saved
                    else:
                        print(f"上次保存的游戏区 {saved} 校验未通过(游戏未开/区域变了)")
                        print("请先开 chrome://dino，或 --pick-region 重新框选")
                        return
                except Exception:
                    print(f"自动量失败({e})，复用上次保存的游戏区 {saved}(截屏校验跳过)")
                    region = saved
            else:
                print(f"自动量 region 失败：{e}")
                print("请先 --pick-region 框选一次，或手动 --region X,Y,W,H")
                return
    pilot = ChromeDinoPilot(region, sim=args.sim, reaction_time=args.reaction,
                            prefer=args.prefer, show=show,
                            use_dl=(not args.no_dl),
                            reaction_pair=args.reaction_pair, airtime=args.airtime,
                            adapt=args.adapt, adapt_floor=args.adapt_floor,
                            adapt_vref=args.adapt_vref, adapt_k=args.adapt_k)
    pilot.auto_restart = (not args.no_restart)
    if not args.sim:
        # 主动聚焦游戏窗口（GUI 桌面访问）：不再依赖用户手动点，拿到 hwnd 直接置前
        focused = False
        if not args.no_window:
            try:
                from exec.gui_agent import locate_game_window
                win = locate_game_window(verbose=False, do_focus=True)
                focused = win is not None
            except Exception:
                focused = False
        if not focused:
            print(">>> 请在 3 秒内点击 Chrome 小恐龙窗口使其获得焦点（按键需前台窗口），"
                  "然后保持窗口在前台别切走")
            time.sleep(3)
        else:
            time.sleep(0.5)
    pilot.run(n_frames=args.frames, dt=args.dt)


if __name__ == "__main__":
    main()
