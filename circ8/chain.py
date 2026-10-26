# -*- coding: utf-8 -*-
"""
Модель цепочки: вдоль каждой ветви осевой линии контура выбирается последовательность кругов,
которая лучше всего объясняет ветвь — без разрывов и без избыточных перекрытий.

Ветвь — путь по скелету между развилками/концами. Каждый круг-кандидат, центр которого лежит
у оси ветви, накрывает отрезок пути [начало, конец]. Динамическим программированием выбирается
набор отрезков с минимальной «стоимостью»:
  стоимость круга      = -ln(p)                       (p — вероятность «хорошего» круга по формуле);
  стоимость разрыва    = lam * (длина ненакрытого участка оси) / r_typ;
  стоимость перекрытия = mu  * (перекрытие отрезков сверх доли om от меньшего) / r_typ.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage import measure

_NB = np.ones((3, 3), int)


def branches(M, dt, r_typ, spur=0.5, dt_min=0.25):
    """Ветви скелета как упорядоченные списки пикселей (без коротких отростков у края)."""
    sk = skeletonize(M) & (dt >= dt_min * r_typ)
    nb = ndi.convolve(sk.astype(int), _NB, mode="constant") - 1
    junc = sk & (nb >= 3)
    seg = sk & ~ndi.binary_dilation(junc)
    lab = measure.label(seg, connectivity=2)
    jl = ndi.binary_dilation(junc, iterations=2)
    out = []
    for p in measure.regionprops(lab):
        pts = [tuple(c) for c in p.coords]
        S = set(pts)
        def nbrs(q):
            return [(q[0] + a, q[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a or b) and (q[0] + a, q[1] + b) in S]
        ends = [q for q in pts if len(nbrs(q)) <= 1]
        start = ends[0] if ends else pts[0]
        path = [start]; seen = {start}
        while True:
            nx = [q for q in nbrs(path[-1]) if q not in seen]
            if not nx:
                break
            path.append(nx[0]); seen.add(nx[0])
        if len(path) < 2:
            continue
        a = np.array(path)
        # отросток: один конец свободный (не у развилки) и короткий
        e0 = jl[a[0][0], a[0][1]]; e1 = jl[a[-1][0], a[-1][1]]
        if len(a) < spur * r_typ and not (e0 and e1):
            continue
        out.append(a)
    return out


def dp_branch(path, cands, probs, r_typ, lam=1.0, mu=2.0, om=0.3, p_min=0.05, near=0.6, tip=0.5, bonus=None):
    """Выбор кругов вдоль одной ветви. Возвращает индексы выбранных кандидатов."""
    L = len(path)
    items = []
    py, px = path[:, 0].astype(float), path[:, 1].astype(float)
    Ca = np.asarray(cands, float); Pa = np.asarray(probs)
    # предварительный отбор: центр не дальше R от прямоугольника ветви (иначе ни одна точка ветви не внутри круга)
    sel = np.nonzero((Pa >= p_min) & (Ca[:, 0] >= py.min() - Ca[:, 2]) & (Ca[:, 0] <= py.max() + Ca[:, 2]) &
                     (Ca[:, 1] >= px.min() - Ca[:, 2]) & (Ca[:, 1] <= px.max() + Ca[:, 2]))[0]
    for i in sel:
        y, x, R = Ca[i]
        if probs[i] < p_min:
            continue
        d2 = (py - y) ** 2 + (px - x) ** 2
        if d2.min() > (near * R) ** 2:
            continue
        inside = np.nonzero(d2 <= R * R)[0]
        if len(inside) < 2:
            continue
        items.append((inside.min(), inside.max(), i))
    if not items:
        return []
    items.sort()
    n = len(items)
    cost = np.full(n, np.inf); prev = np.full(n, -1)
    tipl = tip * r_typ
    for k, (a, b, i) in enumerate(items):
        ck = -np.log(max(probs[i], 1e-6)) - (bonus[i] if bonus is not None else 0.0)
        cost[k] = ck + lam * max(0.0, a - tipl) / r_typ
        for j in range(k):
            a2, b2, i2 = items[j]
            if not (a > a2 and b > b2):
                continue
            gap = max(0, a - b2 - 1)
            ov = max(0, b2 - a + 1)
            exc = max(0.0, ov - om * min(b - a + 1, b2 - a2 + 1))
            c = cost[j] + ck + lam * gap / r_typ + mu * exc / r_typ
            if c < cost[k]:
                cost[k] = c; prev[k] = j
    fin = cost + np.array([lam * max(0.0, (L - 1 - b) - tipl) / r_typ for a, b, i in items])
    empty = lam * max(0.0, L - 2 * tipl) / r_typ
    k = int(np.argmin(fin))
    if fin[k] >= empty:
        return []
    sel = []
    while k >= 0:
        sel.append(items[k][2]); k = prev[k]
    return sel[::-1]


def chain_select(M, cands, probs, base, s, **kw):
    """Круги по модели цепочек + круги базового отбора, не конфликтующие с ними."""
    from .util import circ_iou
    if not base:
        return base
    r_typ = float(np.median([b[2] for b in base]))
    Mf = ndi.binary_fill_holes(M)
    dt = ndi.distance_transform_edt(Mf)
    bk = {k: kw.pop(k) for k in ("spur", "dt_min") if k in kw}
    keep = kw.pop("keep", 0.0)
    bset = {tuple(map(float, b)) for b in base}
    bonus = np.array([keep if tuple(map(float, c)) in bset else 0.0 for c in cands])
    chosen = set()
    for path in branches(Mf, dt, r_typ, **bk):
        chosen.update(dp_branch(path, cands, probs, r_typ, bonus=bonus, **kw))
    A = [tuple(cands[i]) for i in sorted(chosen, key=lambda i: -probs[i])]
    # дедупликация внутри цепочек и добавление базовых кругов вне цепочек
    out = []
    for c in A + [tuple(b) for b in base]:
        ok = True
        for o in out:
            if np.hypot(c[0] - o[0], c[1] - o[1]) < c[2] + o[2]:
                I = circ_iou(c, o); inter = I * (np.pi * c[2] ** 2 + np.pi * o[2] ** 2) / (1 + I)
                if inter / (np.pi * c[2] ** 2) > 0.6 or inter / (np.pi * o[2] ** 2) > 0.6:
                    ok = False; break
        if ok:
            out.append(c)
    return out


def mutual_cap(A, probs, tau=0.65):
    """Запрет почти совпадающих кругов: если пересечение > tau площади КАЖДОГО из двух — оставляем более вероятный."""
    from .util import circ_iou
    order = np.argsort(-np.asarray(probs))
    out = []
    for i in order:
        c = A[i]; ok = True
        for o in out:
            if np.hypot(c[0] - o[0], c[1] - o[1]) < c[2] + o[2]:
                I = circ_iou(c, o); inter = I * (np.pi * c[2] ** 2 + np.pi * o[2] ** 2) / (1 + I)
                if inter / (np.pi * c[2] ** 2) > tau and inter / (np.pi * o[2] ** 2) > tau:
                    ok = False; break
        if ok:
            out.append(c)
    return out


def chain_fill(M, cands, probs, base, s, ov_new=0.7, ov_base=0.6, **kw):
    """Модель цепочки как «умное» заполнение разрывов: базовые круги сохраняются (большой бонус),
    DP по каждой ветви добирает круги-кандидаты там, где ось не накрыта."""
    from .util import circ_iou
    if not base:
        return base
    kw = dict(kw); kw["keep"] = 50.0
    bk = {k: kw.pop(k) for k in ("spur", "dt_min") if k in kw}
    keep = kw.pop("keep")
    r_typ = float(np.median([b[2] for b in base]))
    Mf = ndi.binary_fill_holes(M)
    dt = ndi.distance_transform_edt(Mf)
    bset = {tuple(map(float, b)) for b in base}
    bonus = np.array([keep if tuple(map(float, c)) in bset else 0.0 for c in cands])
    chosen = set()
    for path in branches(Mf, dt, r_typ, **bk):
        chosen.update(dp_branch(path, cands, probs, r_typ, bonus=bonus, **kw))
    out = [tuple(b) for b in base]
    for i in sorted(chosen, key=lambda i: -probs[i]):
        c = tuple(cands[i])
        if tuple(map(float, c)) in bset:
            continue
        ok = True
        for o in out:
            if np.hypot(c[0] - o[0], c[1] - o[1]) < c[2] + o[2]:
                I = circ_iou(c, o); inter = I * (np.pi * c[2] ** 2 + np.pi * o[2] ** 2) / (1 + I)
                if inter / (np.pi * c[2] ** 2) > ov_new or inter / (np.pi * o[2] ** 2) > ov_base:
                    ok = False; break
        if ok:
            out.append(c)
    return out
