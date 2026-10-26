# -*- coding: utf-8 -*-
"""Запрет ложных частиц: круг, площадь которого в основном накрыта соседями похожего размера, — лишний.
Удаляем по одному, начиная с наименее вероятного. Соседи сильно другого размера не считаются
(маленькая частица поверх крупной допустима)."""
import numpy as np
_t = np.linspace(-1, 1, 21); _yy, _xx = np.meshgrid(_t, _t); _m = _yy ** 2 + _xx ** 2 <= 1
GY, GX = _yy[_m], _xx[_m]

def union_cov(c, others, ratio):
    y, x, R = c
    py, px = y + R * GY, x + R * GX; cov = np.zeros(len(py), bool)
    for (y2, x2, R2) in others:
        if np.hypot(y - y2, x - x2) < R + R2 and max(R, R2) / min(R, R2) < ratio:
            cov |= (py - y2) ** 2 + (px - x2) ** 2 <= R2 * R2
    return cov.mean()

def prune(A, probs, u0=0.6, ratio=1.6):
    A = list(A); P = list(probs)
    changed = True
    while changed:
        changed = False
        for i in np.argsort(P):
            others = A[:i] + A[i + 1:]
            if union_cov(A[i], others, ratio) >= u0:
                del A[i]; del P[i]; changed = True
                break
    return A


def prune_bridges(A, probs, Dg, tau=0.2, ang_min=90.0, dark_margin=0.0):
    """«Мост» — круг, лежащий между двумя соседями (каждый накрывает >= tau его площади, соседи по разные
    стороны: угол между направлениями на них > ang_min), причём его центр не темнее центров соседей.
    У настоящей частицы центр — самое толстое (тёмное) место; у ложного круга на стыке — промежуток."""
    from .util import circ_iou
    A = list(A); P = list(probs)
    def dval(y, x):
        H, W = Dg.shape
        return Dg[int(min(max(round(y), 0), H - 1)), int(min(max(round(x), 0), W - 1))]
    changed = True
    while changed:
        changed = False
        Aa = np.asarray(A, float)
        for i in np.argsort(P):
            y, x, R = A[i]; nb = []
            js = np.nonzero(np.hypot(Aa[:, 0] - y, Aa[:, 1] - x) < R + Aa[:, 2])[0]
            for j in js:
                if j == i:
                    continue
                y2, x2, R2 = A[j]
                I = circ_iou((y, x, R), (y2, x2, R2)); inter = I * (np.pi * R * R + np.pi * R2 * R2) / (1 + I)
                if inter / (np.pi * R * R) >= tau:
                    nb.append((y2, x2, R2))
            if len(nb) < 2:
                continue
            angs = [np.arctan2(b[0] - y, b[1] - x) for b in nb]
            opp = any(abs((np.degrees(a1 - a2) + 180) % 360 - 180) > ang_min for k, a1 in enumerate(angs) for a2 in angs[k + 1:])
            if not opp:
                continue
            dc = dval(y, x); dn = max(dval(b[0], b[1]) for b in nb)
            if dc <= dn + dark_margin:
                del A[i]; del P[i]; changed = True
                break
    return A


def prune_covering(A, probs, cov_nb=0.5, ratio_max=1.8, min_nb=2):
    """Круг, сильно накрывающий двух (и более) соседей похожего размера, — ложный: две частицы
    «поверх» одной правдоподобны только при большой разнице радиусов (ratio_max)."""
    from .util import circ_iou
    A = list(A); P = list(probs)
    changed = True
    while changed:
        changed = False
        for i in np.argsort(P):
            y, x, R = A[i]; k = 0
            for j, (y2, x2, R2) in enumerate(A):
                if j == i or R2 > R or R / R2 >= ratio_max or np.hypot(y - y2, x - x2) >= R + R2:
                    continue
                I = circ_iou((y, x, R), (y2, x2, R2)); inter = I * (np.pi * R * R + np.pi * R2 * R2) / (1 + I)
                if inter / (np.pi * R2 * R2) >= cov_nb:
                    k += 1
            if k >= min_nb:
                del A[i]; del P[i]; changed = True
                break
    return A
