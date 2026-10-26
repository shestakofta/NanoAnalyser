# -*- coding: utf-8 -*-
"""
Прототип v2: кандидаты из дуг контура (кривизна) + из карты расстояний, отбор «хороших» кругов.

1. Контур маски сглаживается, считается кривизна. Вогнутые точки (кривизна < 0) и неглубокие
   минимумы кривизны режут контур на выпуклые дуги. К каждой дуге подгоняется круг: его центр —
   центр кривизны, радиус — радиус кривизны дуги. Так находятся частицы, касающиеся края агломерата.
2. Кандидаты из максимумов карты расстояний (вписанные круги) с уточнением по лучам (как v1) —
   для частиц внутри агломерата.
3. Оценка «хорошести» круга: доля круга внутри контура, доля окружности, совпадающая с краем
   (касание контура), согласие с картой расстояний. Жадный отбор по оценке с ограничением перекрытия.
4. Остаток контура, не покрытый кругами, дозаполняется (как v1).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from skimage.feature import peak_local_max
from . import geo

DEF2 = dict(
    smooth_nm=1.0,       # сглаживание контура вдоль дуги, нм
    neck_ratio=0.5,      # неглубокая шейка: минимум кривизны < neck_ratio * меньшего из соседних максимумов
    min_span_deg=50,     # минимальный угол дуги для подгонки круга
    max_rms=0.12,        # макс. ошибка подгонки (доля R)
    inside_min=0.75,     # доля круга внутри контура
    ov_max=0.5,          # макс. доля круга внутри одного уже принятого круга
    w_support=1.0, w_inside=1.0, w_dt=0.5,
    support_tol=0.08,    # окружность «касается» края, если до края < support_tol*R (+1 px)
)


def contour_arcs(M, s, P, return_pts=False):
    """Выпуклые дуги контура и подогнанные к ним круги: список (y, x, R, span_deg, rms)."""
    out = []
    sig = max(1.5, P["smooth_nm"] / s)
    Mp = np.pad(M, 2)
    for C in measure.find_contours(Mp.astype(float), 0.5):
        C = C - 2
        if len(C) < 12:
            continue
        # равномерная перевыборка с шагом 1 px
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]
        L = d[-1]
        if L < 10:
            continue
        closed = np.allclose(C[0], C[-1])
        t = np.arange(0, L, 1.0)
        y = np.interp(t, d, C[:, 0]); x = np.interp(t, d, C[:, 1])
        mode = "wrap" if closed else "nearest"
        ys = ndi.gaussian_filter1d(y, sig, mode=mode); xs = ndi.gaussian_filter1d(x, sig, mode=mode)
        dy = np.gradient(ys); dx = np.gradient(xs); ddy = np.gradient(dy); ddx = np.gradient(dx)
        k = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
        # ориентация: материал должен быть слева (выпуклость -> k > 0)
        n = len(ys); i = n // 3
        nx, ny = -dy[i], dx[i]; nn = np.hypot(nx, ny) + 1e-9
        py, px = int(round(ys[i] + 2 * ny / nn)), int(round(xs[i] + 2 * nx / nn))
        left_inside = 0 <= py < M.shape[0] and 0 <= px < M.shape[1] and M[py, px]
        if not left_inside:
            k = -k
        # точки разреза: вогнутости и неглубокие минимумы кривизны
        cut = set(np.nonzero(k <= 0)[0].tolist())
        kk = ndi.gaussian_filter1d(k, 2, mode=mode)
        for j in range(1, n - 1):
            if kk[j] < kk[j - 1] and kk[j] <= kk[j + 1] and kk[j] > 0:
                w = int(max(5, 3 * sig))
                lo, hi = kk[max(0, j - w):j].max() if j > 0 else kk[j], kk[j + 1:j + 1 + w].max() if j < n - 1 else kk[j]
                if kk[j] < P["neck_ratio"] * min(lo, hi):
                    cut.add(j)
        lab = np.ones(n, bool); lab[list(cut)] = False
        # связные выпуклые куски (с учётом замкнутости)
        segs = []
        idx = np.nonzero(lab)[0]
        if len(idx) == 0:
            continue
        start = idx[0]; prev = idx[0]
        for j in idx[1:]:
            if j != prev + 1:
                segs.append((start, prev)); start = j
            prev = j
        segs.append((start, prev))
        if closed and len(segs) > 1 and segs[0][0] == 0 and segs[-1][1] == n - 1:
            a = segs.pop(0); b = segs.pop(-1); segs.append((b[0], a[1] + n))
        for a, b in segs:
            ii = np.arange(a, b + 1) % n
            if len(ii) < 6:
                continue
            py_, px_ = y[ii], x[ii]
            yc, xc, R = geo.fit_circle(px_, py_)[0:3]
            if not np.isfinite(R) or R < 1:
                continue
            ang = np.unwrap(np.arctan2(py_ - yc, px_ - xc))
            span = np.degrees(abs(ang[-1] - ang[0]))
            rms = np.sqrt(np.mean((np.hypot(py_ - yc, px_ - xc) - R) ** 2)) / R
            out.append((yc, xc, R, min(span, 360.0), rms, py_, px_) if return_pts else (yc, xc, R, min(span, 360.0), rms))
    return out


def circle_scores(M, dt, edge_dist, c, P):
    y, x, R = c
    H, W = M.shape
    sl, dm = geo.disk_win(M.shape, y, x, R)
    area = dm.sum()
    if area == 0:
        return None
    inside = (dm & M[sl]).sum() / area
    th = np.linspace(0, 2 * np.pi, max(24, int(2 * np.pi * R / 2)), endpoint=False)
    py = np.clip(np.round(y + R * np.sin(th)).astype(int), 0, H - 1); px = np.clip(np.round(x + R * np.cos(th)).astype(int), 0, W - 1)
    support = float((edge_dist[py, px] <= P["support_tol"] * R + 1.0).mean())
    yi, xi = int(round(min(max(y, 0), H - 1))), int(round(min(max(x, 0), W - 1)))
    dtc = dt[yi, xi] / R if M[yi, xi] else 0.0
    return inside, support, min(dtc, 1.2)


def place(M, s, params=None):
    P = dict(geo.DEF); P.update(DEF2); P.update(params or {})
    rmin = P["rmin_nm"] / s; rmax = P["rmax_nm"] / s
    dt = ndi.distance_transform_edt(M)
    edge = M ^ ndi.binary_erosion(M)
    edge_dist = ndi.distance_transform_edt(~edge)
    cands = []
    for (y, x, R, span, rms) in contour_arcs(M, s, P):
        if span >= P["min_span_deg"] and rms <= P["max_rms"] and rmin <= R <= rmax:
            cands.append(("arc", y, x, R))
    pk = peak_local_max(dt, min_distance=max(2, int(round(0.5 * rmin))), threshold_abs=rmin, labels=M.astype(int),
                        exclude_border=False)
    for (y, x) in pk:
        yc, xc, R, fr = geo.measure_particle(M, float(y), float(x), float(dt[y, x]), P, rmax)
        cands.append(("dt", yc, xc, R))
    scored = []
    for kind, y, x, R in cands:
        sc = circle_scores(M, dt, edge_dist, (y, x, R), P)
        if sc is None:
            continue
        inside, support, dtc = sc
        if inside < P["inside_min"]:
            continue
        score = P["w_support"] * support + P["w_inside"] * inside + P["w_dt"] * dtc
        scored.append((score, y, x, R, kind))
    scored.sort(key=lambda t: -t[0])
    acc = []
    U = np.zeros(M.shape, bool)
    for score, y, x, R, kind in scored:
        ok = True
        for (y2, x2, R2) in acc:
            dd = np.hypot(y - y2, x - x2)
            if dd < R + R2:
                from .util import circ_iou
                I = circ_iou((y, x, R), (y2, x2, R2))
                inter = I * (np.pi * R * R + np.pi * R2 * R2) / (1 + I)
                if inter / (np.pi * R * R) > P["ov_max"] or inter / (np.pi * R2 * R2) > 0.85:
                    ok = False; break
        if ok:
            acc.append((y, x, R)); U |= geo.disk(M.shape, y, x, R)
    # остаток
    for _ in range(P["resid_passes"]):
        Rm = M & ~U
        rr = P["rmin_resid_nm"] / s
        dtr = ndi.distance_transform_edt(Rm)
        c2 = peak_local_max(dtr, min_distance=max(2, int(round(rr))), threshold_abs=rr, exclude_border=False)
        if len(c2) == 0:
            break
        n0 = len(acc)
        for (y, x) in c2[np.argsort(-dtr[c2[:, 0], c2[:, 1]])]:
            if U[y, x]:
                continue
            yc, xc, R, fr = geo.measure_particle(M, float(y), float(x), float(dtr[y, x]), P, rmax)
            dm = geo.disk(M.shape, yc, xc, R)
            if (dm & U).sum() / max(dm.sum(), 1) > P["dom_frac"]:
                continue
            acc.append((yc, xc, R)); U |= dm
        if len(acc) == n0:
            break
    return acc
