# Синтетический снимок для проверки собранной программы: светлый фон, тёмные круглые «частицы»
import sys, json, os
import numpy as np
from PIL import Image
rng = np.random.default_rng(0)
H = W = 700
yy, xx = np.mgrid[0:H, 0:W]
img = np.full((H, W), 200.0)
def disc(cy, cx, r, depth):
    d = np.hypot(yy - cy, xx - cx)
    img[...] -= depth * np.clip((r - d) / 2.0 + 0.5, 0, 1) * np.sqrt(np.clip(1 - (d / r) ** 2, 0, 1))
for i in range(12):                                   # цепочка мелких частиц ~10 нм (при 0,5 нм/пикс)
    disc(200 + 3 * i, 120 + 17 * i, 10, 70)
disc(480, 450, 45, 140)                               # крупная частица ~45 нм
img += rng.normal(0, 4, img.shape)
out = sys.argv[1]
Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(out, "smoke.png"))
print("smoke.png written")
