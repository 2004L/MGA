"""
verify_dino_pipeline.py —— 沙箱内自测 auto_region 的 CV 链路 + DL 可用性探测。
不依赖真实桌面/Chrome：用合成全屏图验证「几何佐证」仍能定位游戏区。
用法：python -m verify_dino_pipeline
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_dino_real as D

BG = (247, 247, 247)
HORIZON = (120, 120, 120)   # 极淡浅灰地平线(avg<170 → dark_loose 能检出)
FG = (83, 83, 83)           # #535353 恐龙/障碍(严格阈值)


def build_synthetic_full(W=1280, H=800) -> np.ndarray:
    """合成一张「带 chrome://dino 画布的白底桌面(viewport)截图」。

    关键：地平线严格落在 canvas 底部往上 25px（与 validator 约定一致），
    且 viewport 足够高，使 scan_for_game 构造的 crop 不被夹紧 —— 复刻真实窗口级场景。
    """
    img = np.full((H, W, 3), (255, 255, 255), dtype=np.uint8)
    x, y, cw, ch = 300, 600, 820, 200          # canvas 贴在 viewport 底部
    img[y:y + ch, x:x + cw] = BG               # canvas 浅灰块
    gy = y + ch - 25                           # 地平线 = canvas 底部往上 25px
    img[gy:gy + 2, x:x + cw] = HORIZON
    dy = gy - 44
    img[dy:gy, x + 50:x + 50 + 28] = FG        # 恐龙(贴地、近方形，距 canvas 左 50px)
    for dx in range(x + 5, x + cw - 5, 7):     # 地面周期点纹
        img[gy + 4:gy + 6, dx:dx + 3] = FG
    return img


def build_game_scene(cw=820, ch=150) -> np.ndarray:
    """合成一张「游戏区」截图：地平线严格落在底部往上 25px（与 validator 约定一致）。"""
    img = np.full((ch, cw, 3), BG, dtype=np.uint8)
    gy = ch - 25
    img[gy:gy + 2, :] = HORIZON                        # 横贯全宽地平线(极淡灰)
    dy = gy - 44
    img[dy:gy, 30:30 + 28] = FG                        # 恐龙(贴地、近方形)
    for dx in range(5, cw - 5, 7):                     # 地面周期点纹
        img[gy + 4:gy + 6, dx:dx + 3] = FG
    return img


def main():
    print("== 1) auto_region CV 链路（合成 viewport）==")
    full = build_synthetic_full()
    bb = D.find_canvas_bbox(full)
    print(f"  find_canvas_bbox → {bb}")
    region = D.scan_for_game(full)
    print(f"  scan_for_game(全屏) → {region}")
    ok_scan = D._validate_region_contains_game(full, region, strict=False) if region else False
    # 窗口级路径：locate_game_window 返回的是整个浏览器客户区(viewport)，
    # 本合成的 full 就是 viewport，故 scan_for_game(full) 即等价窗口级主路径。
    # 注意：find_canvas_bbox 的结果**不能**当 scan_for_game 的输入 —— 地平线行不是
    # 画布背景色，会打断 BG 连续段，导致它返回的区域底边就贴在地平线上(约 4px)，
    # 地平线不再位于 Hc-25，喂进去必然校验失败。
    reg2 = region
    ok2 = ok_scan
    print(f"  窗口级(=viewport 客户区): scan_for_game → {reg2}  validate → {ok2}")
    # 直接几何佐证（最严格的核心）
    scene = build_game_scene()
    fh = np.zeros_like(scene); fh[:, :] = scene
    ok_canvas = D._validate_region_contains_game(fh, (0, 0, scene.shape[1], scene.shape[0]), strict=True)
    print(f"  几何佐证(直接游戏区): _validate_region_contains_game → {ok_canvas}")
    print("  [结论] CV 几何佐证定位游戏区:", "OK" if (region and ok_scan and reg2 and ok2 and ok_canvas) else "FAIL")

    print("\n== 2) 深度学习障碍检测可用性探测 ==")
    try:
        from perception.dino_detect import DeepDinoDetector
        det = DeepDinoDetector()
        avail = det.load()
        print(f"  DeepDinoDetector.load() → available={avail}")
        if avail:
            # 在一张合成 dino 帧上真跑一次 DL 检测
            scene = np.full((150, 600, 3), BG, dtype=np.uint8)
            scene[150 - 25:150 - 23, :] = HORIZON
            scene[150 - 25 - 44:150 - 25, 30:30 + 28] = FG
            scene[150 - 25 - 18:150 - 25, 400:430] = FG   # 假鸟
            dx, raw = det.detect(scene)
            print(f"  detect(scene) → dino_x={dx}  obstacles={raw}")
        else:
            print(f"  （权重缺失，已降级 CV；load_error={det.load_error}）")
    except Exception as e:
        print(f"  DL 探测异常：{type(e).__name__}: {e}")

    print("\n== 3) detect() 分发器降级一致性（合成 dino 帧）==")
    scene = np.full((150, 600, 3), BG, dtype=np.uint8)
    scene[150 - 25:150 - 23, :] = HORIZON
    scene[150 - 25 - 44:150 - 25, 30:30 + 28] = FG
    dx_cv, obs_cv = D._detect_cv(scene)
    dx_dl, obs_dl = D.detect(scene, use_dl=False)
    print(f"  CV(兜底): dino_x={dx_cv} obstacles={[(round(o.x_rel), o.is_bird) for o in obs_cv]}")
    print(f"  detect(use_dl=False): dino_x={dx_dl} obstacles={[(round(o.x_rel), o.is_bird) for o in obs_dl]}")
    print("  [结论] 降级一致性:", "OK" if dx_cv == dx_dl else "WARN(不一致)")


if __name__ == "__main__":
    main()
