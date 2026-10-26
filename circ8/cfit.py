# -*- coding: utf-8 -*-
"""Уточнение крупных кругов по контуру агломерата (лучевой метод).

Из центра круга выпускаются лучи; точка выхода луча из (залитой) маски — точка края.
Лучи, уходящие дальше hi*R (перетяжка к соседней частице), и выходы ближе lo*R (вогнутость)
не используются. К оставшимся точкам робастно подгоняется окружность. Так круг частицы,
лежащей на краю агломерата, прижимается к контуру там, где контур её действительно ограничивает.
"""
import numpy as np
from scipy import ndimage as ndi
from . import geo


def ray_exits(Mf, y, x, R, n_dirs=72, lo=0.75, hi=1.4):
    th = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    rr = np.arange(0.3 * R, hi * R + 0.5, 0.5)
    Y = y + np.outer(np.sin(th), rr); X = x + np.outer(np.cos(th), rr)
    H, W = Mf.shape
    inside = (Y >= 0) & (Y < H - 0.5) & (X >= 0) & (X < W - 0.5)
    v = np.zeros(Y.shape, bool)
    v[inside] = Mf[Y[inside].round().astype(int), X[inside].round().astype(int)]
    out = ~v
    first = np.where(out.any(1), out.argmax(1), -1)
    re = np.where(first >= 0, rr[np.maximum(first, 0)], np.inf)
    ok = (first >= 0) & (re >= lo * R) & inside[np.arange(n_dirs), np.maximum(first, 0)]
    return th, re, ok


def refit_contour(Mf, y, x, R, min_frac=0.35, min_span=150, max_shift=0.3, rlo=0.85, rhi=1.3, tol=0.08):
    th, re, ok = ray_exits(Mf, y, x, R)
    if ok.mean() < min_frac:
        return None
    span = len(np.unique((np.degrees(th[ok]) // 20).astype(int))) * 20
    if span < min_span:
        return None
    px = x + re * np.cos(th); py = y + re * np.sin(th)
    s = ok.copy()
    yc, xc, Rc = geo.fit_circle(px[s], py[s])
    for _ in range(3):
        if not np.isfinite(Rc):
            return None
        res = np.abs(np.hypot(px - xc, py - yc) - Rc)
        s2 = ok & (res <= max(2.0, tol * Rc))
        if s2.sum() < min_frac * len(th) * 0.8:
            break
        yc, xc, Rc = geo.fit_circle(px[s2], py[s2])
    if not np.isfinite(Rc) or np.hypot(yc - y, xc - x) > max_shift * R or not (rlo * R <= Rc <= rhi * R):
        return None
    return float(yc), float(xc), float(Rc)


def edge_step(Dg, y, x, R, n_dirs=72, a=0.9, b=1.1):
    """Средний перепад затемнения поперёк окружности (внутри минус снаружи) — сила края на круге."""
    th = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    vi = ndi.map_coordinates(Dg, [y + a * R * np.sin(th), x + a * R * np.cos(th)], order=1, mode="nearest")
    vo = ndi.map_coordinates(Dg, [y + b * R * np.sin(th), x + b * R * np.cos(th)], order=1, mode="nearest")
    return float(np.mean(vi - vo))
