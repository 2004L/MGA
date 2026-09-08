"""临时：截微信最近会话列表图 + UIA 取零误差名字，供用户肉眼核验挑人。"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path: sys.path.insert(0, _p)

from agent.desktop_adapter import DesktopAdapter
from PIL import Image
import numpy as np

OUT = os.path.join(_ROOT, "wechat_list.png")
OUT_FULL = os.path.join(_ROOT, "wechat_full.png")

adapter = DesktopAdapter(dry_run=False, focus_title="微信")
adapter.enable()

# 先聚焦微信到前台，否则 grab() 截的是全屏最上层窗口（可能是抖音/其他）
print("FOCUS_BEFORE", adapter.cu.foreground_window())
hwnd = adapter.cu.focus_window("微信")
if hwnd is None:
    # 尝试启动微信
    adapter.act("hotkey win r"); time.sleep(0.5)
    adapter.act("type wechat"); time.sleep(0.3); adapter.act("press enter"); time.sleep(3.0)
    hwnd = adapter.cu.focus_window("微信")
if hwnd is None:
    print("NO_WECHAT"); sys.exit(1)
time.sleep(0.5)
print("FOCUS_AFTER", adapter.cu.foreground_window(), "HWND", hwnd)

rect = adapter.cu.client_rect("微信")
if rect is None:
    print("NO_RECT"); sys.exit(1)

frame = adapter.grab()
print("FRAME_SHAPE", frame.shape, "WECHAT_RECT", rect)
h_img, w_img = frame.shape[:2]
# screenshot() 截的是微信窗口本身（原点 0,0），client_rect 是虚拟桌面负坐标会越界，
# 因此裁剪一律用【图像相对坐标】。
x1, y1 = 0, int(0.06 * h_img)
x2, y2 = int(0.32 * w_img), int(0.96 * h_img)
crop = frame[y1:y2, x1:x2]
print("CROP_BOX", (x1, y1, x2, y2), "CROP_SHAPE", crop.shape)
Image.fromarray(frame).save(OUT_FULL)
Image.fromarray(crop).save(OUT)
print("SHOT_SAVED", OUT, OUT_FULL)

try:
    import uiautomation as auto
    wx = auto.WindowControl(ClassName="WeChatMainWndForPC")
    if not wx.Exists(0):
        wx = auto.WindowControl(searchDepth=1, ClassName="WeChatMainWndForPC")
    names = []
    def walk(ctrl, d=0):
        if d > 10 or len(names) > 80: return
        try:
            for c in ctrl.GetChildren():
                n = (c.Name or "").strip()
                if n and 1 <= len(n) <= 20:
                    names.append(n)
                walk(c, d + 1)
        except Exception:
            pass
    walk(wx)
    seen = set(); uniq = []
    for n in names:
        if n not in seen:
            seen.add(n); uniq.append(n)
    print("UIA_NAMES", uniq[:50])
except Exception as e:
    print("UIA_FAIL", type(e).__name__, e)
