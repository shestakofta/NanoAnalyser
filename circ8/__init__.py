# -*- coding: utf-8 -*-
"""
circ8 — разметка первичных частиц кругами, версия 8 (подобрана по ручной разметке снимков Al).

Шаги (подробности — в отчёте «Разметка первичных частиц кругами», раздел «Состав метода»):
  1. Кандидаты: дуги контура в нескольких масштабах сглаживания и на уровнях затемнения, склейка дуг
     одной окружности, круги «по лучам», вписанные круги, круги вдоль осевой линии, тёмные пятна.
  2. Оценка каждого кандидата формулой из 23 признаков; жадный отбор с ограничением перекрытий.
  3. Модель цепочек (добор кругов вдоль ветвей), заполнение разрывов оси.
  4. Запрет «мостов», круги у края кадра, второй проход по незанятым кускам контура.
  5. Подгонка по краю на снимке; крупных — по контуру (с проверкой края).
  6. Тёмное ядро той же частицы, крупные/мелкие круги без края, почти совпадающие круги.
"""
import numpy as np
from scipy import ndimage as ndi

from . import pool3, chain, cont, prune, pass2, edge, cfit, geo
from .model import MODEL

CH = dict(lam=8.0, mu=2.0, om=0.5)
GAP = dict(thr_gap=0.03, min_len=0.8, dt_min=0.5, kR=1.1, rcap=1.5)
BRIDGE = dict(tau=0.35, ang_min=60)
FRAME = dict(f_small=0.5, f_big=0.75, kbig=1.5)
P2 = dict(p2=0.15, use_inscribed=False, use_mid=True, a_frac=0.25, thick_min=0.35, mid_unc=0.55)
CF = dict(kmin=1.5, scale=0.97, rhi=1.15, edge_k=1.0)
INNER = dict(f=0.75, p=0.3, rmin=0.45)
BIG = dict(kr=1.3, e=1.5)
SMALL = dict(kr=0.6, e=1.5)
CAP = 0.65
RK = 1.02

_mean = np.array(MODEL["mean"]); _scale = np.array(MODEL["scale"])
_coef = np.array(MODEL["coef"]); _b = MODEL["intercept"]


def _inter(c, o):
    d = np.hypot(c[0] - o[0], c[1] - o[1]); r1, r2 = c[2], o[2]
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a = r1 * r1 * np.arccos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    b = r2 * r2 * np.arccos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    return a + b - 0.5 * np.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))


def score(X):
    z = (np.asarray(X, float) - _mean) / _scale
    return 1.0 / (1.0 + np.exp(-(z @ _coef + _b)))


def place(M, D, lvl1, s, valid=None, frame_shape=None, rmax_nm=None):
    """
    Круги первичных частиц внутри маски M (агломераты).
      D — затемнение (фон минус яркость), lvl1 — контраст одиночной первичной частицы, s — нм/пиксель.
      rmax_nm — наибольший радиус кандидатов, нм (None — по умолчанию, 80 нм).
    Возвращает список (y, x, R) в пикселях.
    """
    M = np.asarray(M, bool)
    if M.sum() < 20:
        return []
    H, W = M.shape
    C = pool3.candidates(M, D, lvl1, s, params=None if rmax_nm is None else dict(rmax_nm=float(rmax_nm)))
    if not C:
        return []
    cands = [tuple(map(float, c)) for c, _ in C]; F = [f for _, f in C]
    X = np.array([[f[k] for k in pool3.FEATS] for f in F])
    PR = score(X)
    A = pool3.select(cands, PR, F, **MODEL["kw"])
    if not A:
        return []
    pi = {tuple(map(float, c)): p for c, p in zip(cands, PR)}
    DG = ndi.gaussian_filter(D, max(1.0, 0.7 / s))
    v = np.ones(M.shape, bool) if valid is None else valid
    bgv = DG[~ndi.binary_dilation(M, iterations=5) & v]
    noise = float(1.4826 * np.median(np.abs(bgv - np.median(bgv))) + 1e-6) if bgv.size > 100 else float(np.std(DG) + 1e-6)

    A = chain.chain_fill(M, cands, PR, A, s, **CH)
    A = cont.fill_gaps(M, A, cands, PR, s, **GAP)
    pr = [pi.get(tuple(map(float, a)), 0.02) for a in A]
    A = prune.prune_bridges(A, pr, DG, **BRIDGE)
    # край кадра — не край частицы
    Hh, Ww = frame_shape or (H, W)
    medf = np.median([a[2] for a in A]) if A else 0
    yy, xx = np.mgrid[-1:1:31j, -1:1:31j]; mm = yy ** 2 + xx ** 2 <= 1; yy, xx = yy[mm], xx[mm]

    def _fr(c):
        Y = c[0] + c[2] * yy; X_ = c[1] + c[2] * xx
        return ((Y >= 0) & (Y < Hh) & (X_ >= 0) & (X_ < Ww)).mean()
    A = [a for a in A if not (_fr(a) < FRAME["f_small"] or (a[2] >= FRAME["kbig"] * medf and _fr(a) < FRAME["f_big"]))]
    A = pass2.second_pass(M, A, cands, PR, s, **P2)
    if not A:
        return []
    med = np.median([a[2] for a in A]); out = []; pr2 = []
    for a in A:
        r = edge.refine_robust(DG, *a, q=0.4, tol=0.06) if a[2] >= 2 * med else edge.refine(DG, *a, q=0.5, max_shift=0.25)
        out.append((r[0], r[1], r[2] * RK)); pr2.append(pi.get(tuple(map(float, a)), 0.02))
    # крупные круги — по контуру, если край на новой окружности не слабее
    Mf = ndi.binary_fill_holes(M); med2 = np.median([a[2] for a in out]); o2 = []
    for a in out:
        r = cfit.refit_contour(Mf, *a, rhi=CF["rhi"]) if a[2] >= CF["kmin"] * med2 else None
        if r is not None:
            r = (r[0], r[1], r[2] * CF["scale"])
            if cfit.edge_step(DG, *r) < CF["edge_k"] * cfit.edge_step(DG, *a):
                r = None
        o2.append(r if r is not None else a)
    out = o2
    # тёмное ядро той же частицы
    keep = []
    for i, c in enumerate(out):
        bad = False
        if pr2[i] < INNER["p"]:
            for o in out:
                if o[2] > c[2] and c[2] >= INNER["rmin"] * o[2] and _inter(c, o) / (np.pi * c[2] ** 2) >= INNER["f"]:
                    bad = True; break
        if not bad:
            keep.append(i)
    out = [out[i] for i in keep]; pr2 = [pr2[i] for i in keep]
    # крупные, затем мелкие круги — только при выраженном крае
    for rule, big in ((BIG, True), (SMALL, False)):
        if not out:
            break
        medr = np.median([a[2] for a in out]); keep = []
        for i, a in enumerate(out):
            hit = a[2] >= rule["kr"] * medr if big else a[2] < rule["kr"] * medr
            if hit and cfit.edge_step(DG, *a) / noise < rule["e"]:
                continue
            keep.append(i)
        out = [out[i] for i in keep]; pr2 = [pr2[i] for i in keep]
    if not out:
        return []
    out = chain.mutual_cap(out, pr2, CAP)
    return [(float(y), float(x), float(R)) for (y, x, R) in out if R > 0.5]
