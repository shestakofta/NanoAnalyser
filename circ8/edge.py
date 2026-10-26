# -*- coding: utf-8 -*-
"""Край частицы по контрасту снимка: оценка доли периметра с краем и уточнение круга по краю."""
import numpy as np
from scipy import ndimage as ndi
from . import geo


def radial_edges(Dg, y, x, R, n_dirs=48, lo=0.7, hi=1.3):
    """Для каждого направления: положение (радиус) и сила края — максимум падения затемнения наружу."""
    th = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    rr = np.arange(lo * R, hi * R + 0.5, 0.5)
    if len(rr) < 4:
        rr = np.linspace(lo * R, hi * R, 4)
    Y = y + np.outer(np.sin(th), rr); X = x + np.outer(np.cos(th), rr)
    v = ndi.map_coordinates(Dg, [Y.ravel(), X.ravel()], order=1, mode="nearest").reshape(Y.shape)
    g = -np.gradient(v, rr, axis=1)            # падение затемнения наружу (на пиксель)
    k = g.argmax(1)
    return th, rr[k], g[np.arange(n_dirs), k]


def edge_score(Dg, noise, y, x, R, tau=1.0):
    """Доля периметра, где есть край: падение затемнения на ширине ~R/10 больше tau шумов."""
    th, re, e = radial_edges(Dg, y, x, R, lo=0.85, hi=1.15)
    step = e * max(R / 10.0, 1.0) / max(noise, 1e-6)
    return float((step > tau).mean()), float(np.median(step))


def refine(Dg, y, x, R, iters=2, q=0.4, max_shift=0.25, rlo=0.8, rhi=1.25):
    """Подгонка круга к краям: берём направления с самым сильным краем (верхние 1-q),
    подгоняем окружность к точкам края. Возвращает уточнённые (y, x, R)."""
    y0, x0, R0 = y, x, R
    for _ in range(iters):
        th, re, e = radial_edges(Dg, y0, x0, R0)
        sel = (e >= np.quantile(e, q)) & (e > 0)
        if sel.sum() < 8:
            break
        px = x0 + re[sel] * np.cos(th[sel]); py = y0 + re[sel] * np.sin(th[sel])
        y1, x1, R1 = geo.fit_circle(px, py)
        if not np.isfinite(R1) or np.hypot(y1 - y, x1 - x) > max_shift * R or not (rlo * R <= R1 <= rhi * R):
            break
        y0, x0, R0 = y1, x1, R1
    return y0, x0, R0


def refine_robust(Dg, y, x, R, q=0.4, max_shift=0.25, rlo=0.8, rhi=1.25, tol=0.06, n_dirs=72):
    """Как refine, но с отбрасыванием выбросов: точки края, отстоящие от подогнанной окружности
    больше чем на tol*R (прилипшие соседи, тени), исключаются, окружность подгоняется заново."""
    th, re, e = radial_edges(Dg, y, x, R, n_dirs=n_dirs)
    sel = (e >= np.quantile(e, q)) & (e > 0)
    if sel.sum() < 8:
        return y, x, R
    px = x + re * np.cos(th); py = y + re * np.sin(th)
    yc, xc, Rc = geo.fit_circle(px[sel], py[sel])
    for _ in range(3):
        if not np.isfinite(Rc):
            return y, x, R
        res = np.abs(np.hypot(px - xc, py - yc) - Rc)
        s2 = sel & (res <= max(2.0, tol * Rc))
        if s2.sum() < 8:
            break
        yc, xc, Rc = geo.fit_circle(px[s2], py[s2])
    if not np.isfinite(Rc) or np.hypot(yc - y, xc - x) > max_shift * R or not (rlo * R <= Rc <= rhi * R):
        return y, x, R
    return yc, xc, Rc
