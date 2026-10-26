# -*- coding: utf-8 -*-
"""
Прототип новой разметки кругов по геометрии контура (без машинного обучения).

Идея: частица = «утолщение» контура. Кандидаты в центры — максимумы карты расстояний до края
(центры вписанных кругов). Для каждого кандидата бросаем лучи до края контура: у отдельной
частицы длина луча ~ R во все стороны, а там, где к ней прилегают соседи, лучи длиннее.
Радиус = нижняя часть распределения длин лучей; центр уточняется подгонкой круга к «свободной»
дуге контура (точкам, где луч упёрся в край, а не ушёл в соседа). Кандидаты, чей круг в основном
лежит внутри уже принятого круга, отбрасываются (лечит «большая частица = семейство мелких»).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

DEF = dict(
    q_radius=0.35,      # квантиль длин лучей -> радиус
    free_tol=1.15,      # луч «свободный», если короче free_tol * R
    min_free=0.25,      # доля свободных лучей для подгонки центра
    max_shift=0.5,      # макс. сдвиг центра при подгонке (доля R)
    dom_frac=0.6,       # кандидат отбрасывается, если доля его круга внутри принятых > dom_frac
    rmin_nm=2.0,        # минимальный радиус частицы, нм
    rmax_nm=80.0,       # максимальный радиус, нм
    n_rays=72,
    resid_passes=2,
    resid_min_frac=0.5, # остаток заполняется, если его площадь > resid_min_frac * pi*rmin_resid^2
    rmin_resid_nm=3.0,
)


def ray_lengths(M, y, x, n, rmax):
    """Длины лучей из (y,x) до первого выхода из маски M (в пикселях)."""
    H, W = M.shape
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    steps = np.arange(0, rmax, 0.5)
    ys = y + np.outer(np.sin(th), steps); xs = x + np.outer(np.cos(th), steps)
    yi = np.clip(np.round(ys).astype(int), 0, H - 1); xi = np.clip(np.round(xs).astype(int), 0, W - 1)
    inside = M[yi, xi] & (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    out = ~inside
    first = np.where(out.any(1), out.argmax(1), len(steps) - 1)
    rho = steps[first]
    edge = (ys[np.arange(n), first] < 0) | (ys[np.arange(n), first] >= H) | (xs[np.arange(n), first] < 0) | \
           (xs[np.arange(n), first] >= W)
    return th, rho, edge


def fit_circle(px, py):
    A = np.c_[2 * px, 2 * py, np.ones_like(px)]
    b = px ** 2 + py ** 2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    x0, y0 = c[0], c[1]
    return y0, x0, np.sqrt(max(c[2] + x0 * x0 + y0 * y0, 0))


def measure_particle(M, y, x, r0, P, rmax_px):
    th, rho, edge = ray_lengths(M, y, x, P["n_rays"], min(rmax_px * 1.6, 6 * r0 + 10))
    R = float(np.quantile(rho, P["q_radius"]))
    R = max(R, r0)
    free = (rho <= P["free_tol"] * R) & ~edge
    yc, xc = y, x
    if free.mean() >= P["min_free"] and free.sum() >= 6:
        px = x + rho[free] * np.cos(th[free]); py = y + rho[free] * np.sin(th[free])
        y1, x1, R1 = fit_circle(px, py)
        if np.hypot(y1 - y, x1 - x) <= P["max_shift"] * R and 0.6 * R <= R1 <= 1.6 * R and \
                0 <= y1 < M.shape[0] - 0.5 and 0 <= x1 < M.shape[1] - 0.5 and M[int(round(y1)), int(round(x1))]:
            yc, xc, R = y1, x1, R1
    return yc, xc, float(min(R, rmax_px)), float(free.mean())


def disk(shape, y, x, r):
    H, W = shape
    a, b = int(max(0, np.floor(y - r))), int(min(H, np.ceil(y + r) + 1))
    c, d = int(max(0, np.floor(x - r))), int(min(W, np.ceil(x + r) + 1))
    m = np.zeros(shape, bool)
    if a < b and c < d:
        yy, xx = np.ogrid[a:b, c:d]
        m[a:b, c:d] = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return m


def disk_win(shape, y, x, r):
    """Круг в окне вокруг него: (срез окна, маска окна). Те же границы, что у disk(), но без маски всего снимка."""
    H, W = shape
    a, b = int(max(0, np.floor(y - r))), int(min(H, np.ceil(y + r) + 1))
    c, d = int(max(0, np.floor(x - r))), int(min(W, np.ceil(x + r) + 1))
    if a >= b or c >= d:
        return (slice(0, 0), slice(0, 0)), np.zeros((0, 0), bool)
    yy, xx = np.ogrid[a:b, c:d]
    return (slice(a, b), slice(c, d)), (yy - y) ** 2 + (xx - x) ** 2 <= r * r


def place(M, s, params=None):
    P = dict(DEF); P.update(params or {})
    rmin = P["rmin_nm"] / s; rmax = P["rmax_nm"] / s
    dt = ndi.distance_transform_edt(M)
    acc = []
    U = np.zeros(M.shape, bool)

    def consider(cands, dtv):
        nonlocal U
        for (y, x) in cands:
            r0 = float(dtv[y, x])
            if r0 < rmin:
                continue
            if U[y, x]:
                # центр внутри уже принятого круга: проверим, не «спутник» ли это
                d0 = disk(M.shape, y, x, r0)
                if (d0 & U).sum() / max(d0.sum(), 1) > P["dom_frac"]:
                    continue
            yc, xc, R, fr = measure_particle(M, float(y), float(x), r0, P, rmax)
            dm = disk(M.shape, yc, xc, R)
            if (dm & U).sum() / max(dm.sum(), 1) > P["dom_frac"]:
                continue
            acc.append((yc, xc, R, fr)); U |= dm

    cand = peak_local_max(dt, min_distance=max(2, int(round(0.5 * rmin))), threshold_abs=rmin, labels=M.astype(int),
                          exclude_border=False)
    order = np.argsort(-dt[cand[:, 0], cand[:, 1]])
    consider([tuple(c) for c in cand[order]], dt)
    for _ in range(P["resid_passes"]):
        R_ = M & ~U
        rr = P["rmin_resid_nm"] / s
        dtr = ndi.distance_transform_edt(R_)
        c2 = peak_local_max(dtr, min_distance=max(2, int(round(rr))), threshold_abs=rr, exclude_border=False)
        if len(c2) == 0:
            break
        c2 = c2[np.argsort(-dtr[c2[:, 0], c2[:, 1]])]
        n0 = len(acc)
        # для остатка радиус меряем по всему контуру, но вписанный — по остатку
        consider([tuple(c) for c in c2], dtr)
        if len(acc) == n0:
            break
    return [(a[0], a[1], a[2]) for a in acc]
