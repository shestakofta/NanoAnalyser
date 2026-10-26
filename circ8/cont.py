# -*- coding: utf-8 -*-
"""Неразрывность ветвей: участки осевой линии контура, не накрытые кругами, заполняются."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage import measure
from .util import circ_iou

def disks_mask(shape, A, scale=1.0):
    H, W = shape; m = np.zeros(shape, bool)
    for (y, x, R) in A:
        r = R * scale
        a, b = int(max(0, y - r)), int(min(H, y + r + 1)); c, d = int(max(0, x - r)), int(min(W, x + r + 1))
        if a < b and c < d:
            yy, xx = np.ogrid[a:b, c:d]; m[a:b, c:d] |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return m

def fill_gaps(M, A, cands, probs, s, thr_gap=0.08, min_len=0.5, ov_max=0.6, dt_min=0.35, max_iter=3, kR=1.1, rcap=1.5):
    A = list(A)
    if not A:
        return A
    dt = ndi.distance_transform_edt(M)
    skel = skeletonize(M)
    r_typ = float(np.median([a[2] for a in A]))
    skel &= dt >= dt_min * r_typ                       # отбрасываем мелкие отростки скелета у края
    for _ in range(max_iter):
        cov = disks_mask(M.shape, A, 1.0)
        gap = skel & ~cov
        lab = measure.label(gap, connectivity=2)
        added = 0
        for p in measure.regionprops(lab):
            if p.area < min_len * r_typ:
                continue
            pts = p.coords
            best = None
            for i in np.argsort(-probs):
                if probs[i] < thr_gap:
                    break
                y, x, R = cands[i]
                inside = ((pts[:, 0] - y) ** 2 + (pts[:, 1] - x) ** 2 <= R * R).mean()
                if inside < 0.5 or R > rcap * r_typ:
                    continue
                ok = True
                for (y2, x2, R2) in A:
                    if np.hypot(y - y2, x - x2) < R + R2:
                        I = circ_iou((y, x, R), (y2, x2, R2))
                        inter = I * (np.pi * R * R + np.pi * R2 * R2) / (1 + I)
                        if inter / (np.pi * R * R) > ov_max or inter / (np.pi * R2 * R2) > 0.85:
                            ok = False; break
                if ok:
                    best = (y, x, R); break
            if best is None:
                # круг по ширине ветви в середине разрыва
                k = len(pts) // 2; y, x = pts[k]; R = min(kR * dt[y, x], rcap * r_typ)
                if R >= 0.4 * r_typ:
                    best = (float(y), float(x), float(R))
            if best is not None:
                A.append(best); added += 1
        if not added:
            break
    return A
