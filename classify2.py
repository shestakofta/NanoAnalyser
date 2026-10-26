# -*- coding: utf-8 -*-
"""
classify2.py — решение «модифицированная / скопление (модифицированные + первичные) / НФ / агломерат»
для одного крупного тела (версия 4.1).

Порядок проверки:
  0. Пёстрое тело (сильная мелкая неоднородность затемнения внутри и нет осесимметричного профиля)
     — группа мелких частиц → агломерат (заполняется первичными).
  1. Контур близок к кругу или эллипсу (или окружность по участкам, граничащим с плёнкой)
     и внутри нет очень глубокой светлой долины → модифицированная.
  2. Светлые линии: водораздел по затемнению; соседние куски сливаются, если долина между ними
     мелкая (медиана затемнения вдоль общей границы близка к пику слабейшего куска).
     Кусок < отсечки — первичная; кусок с круговой внешней дугой или эллиптический — модифицированная.
  3. Перетяжка контура (EDM) — принимается, только если все куски круглые.
  4. Круги v8 без ограничения размера; круг принят, если его окружность подтверждена краем
     (контур или светлая линия).
  5. Иначе: плотное однородное тело → НФ; бледное или пёстрое → агломерат.
  6. Если модифицированные покрывают < 60 % тела, разбиение отвергается (тело — НФ или агломерат).

Опора: маркерный водораздел с отбором по глубине долины (Beucher & Meyer, 1993; Najman & Schmitt, 1996),
разделение перекрывающихся выпуклых объектов по контуру (Zafari et al., arXiv:1906.01049).
Остальные правила подобраны по визуальной разметке 1160 тел серии W 130 мВт.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, feature, segmentation

P0 = dict(t_round=0.40, t_valley=0.20, md_nm=7.5, ar_max=1.40, eiou_min=0.87, sol_min=0.92,
          g_ar=1.8, g_e=(0.84, 0.88), g_sol=0.90, t_tex=0.07, t_rad=0.40, cov_min=0.60,
          sup_min=0.75, in_min=0.80, ov_acc=0.60, arc_rms=0.08, arc_cov=0.45, arc_iou=0.75,
          k_nf=3.5, neck=0.80, dcut=25.0, ov_max=0.40, eiou_piece=0.85, sol_piece=0.90, big_nf=1.5)


# ------------------------------------------------------------------ признаки формы и затемнения
def shape_feats(m, s):
    """Эквивалентный диаметр, отношение осей, выпуклость и совпадение (IoU) с эллипсом тех же моментов."""
    rp = measure.regionprops(m.astype(np.uint8))[0]
    a = max(rp.major_axis_length, 1); b = max(rp.minor_axis_length, 1)
    cy, cx = rp.centroid; th = rp.orientation
    y0, x0, y1, x1 = rp.bbox
    pad = int(max(a, b)) + 2
    ya, yb, xa, xb = max(0, y0 - pad), min(m.shape[0], y1 + pad), max(0, x0 - pad), min(m.shape[1], x1 + pad)
    Y, X = np.mgrid[ya:yb, xa:xb]
    k = np.sqrt(rp.area / (np.pi * a * b / 4)); A = a / 2 * k; B = b / 2 * k
    u = (Y - cy) * np.cos(th) + (X - cx) * np.sin(th); v = -(Y - cy) * np.sin(th) + (X - cx) * np.cos(th)
    E = (u / A) ** 2 + (v / B) ** 2 <= 1
    mm = m[ya:yb, xa:xb]
    iou = (E & mm).sum() / max((E | mm).sum(), 1)
    return dict(deq=rp.equivalent_diameter_area * s, ar=a / b, sol=rp.solidity, eiou=iou)


def texture(D, m, s):
    """Пёстрость: ст. откл. полосового (0,7–2,5 нм) затемнения внутри тела (без 2 нм у края) / медиана."""
    a = ndi.gaussian_filter(D, 0.7 / s); b = ndi.gaussian_filter(D, 2.5 / s)
    inner = ndi.binary_erosion(m, iterations=max(1, int(round(2 / s))))
    if inner.sum() < 10:
        inner = m
    med = max(float(np.median(a[inner])), 1e-6)
    return float(np.std((a - b)[inner]) / med)


def radial_r2(Ds, m):
    """Доля дисперсии затемнения, объяснённая осесимметричным (эллиптическим) профилем."""
    rp = measure.regionprops(m.astype(np.uint8))[0]
    a = max(rp.major_axis_length, 2) / 2; b = max(rp.minor_axis_length, 2) / 2
    k = np.sqrt(rp.area / (np.pi * a * b)); a *= k; b *= k
    cy, cx = rp.centroid; th = rp.orientation
    ys, xs = np.nonzero(m)
    u = (ys - cy) * np.cos(th) + (xs - cx) * np.sin(th); v = -(ys - cy) * np.sin(th) + (xs - cx) * np.cos(th)
    q = np.sqrt((u / a) ** 2 + (v / b) ** 2); R = np.sqrt(a * b)
    w = q < 1.0; y = Ds[ys, xs][w]
    if y.size < 5:
        return 1.0
    nb = max(4, int(R)); idx = np.minimum((q[w] * nb).astype(int), nb - 1)
    mean = np.bincount(idx, y, nb) / np.maximum(np.bincount(idx, None, nb), 1)
    return float(1 - np.sum((y - mean[idx]) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-9))


def light_lines(D, s, lvl1, sig_nm=(1.0, 1.5, 2.0)):
    """Светлые линии: наибольшее собственное число гессиана затемнения (долина затемнения), в долях lvl1."""
    best = np.zeros(D.shape)
    for sg in [max(0.8, x / s) for x in sig_nm]:
        Hm = feature.hessian_matrix(D, sigma=sg, order="rc", use_gaussian_derivatives=False)
        best = np.maximum(best, sg ** 2 * feature.hessian_matrix_eigvals(Hm)[0])
    return best / lvl1


def support(m, L, cy, cx, r, thrL=0.3, tol_px=2):
    """Доля окружности, подтверждённая краем: контур тела (в пределах tol) или светлая линия внутри."""
    din = ndi.distance_transform_edt(m); dout = ndi.distance_transform_edt(~m)
    Lm = ndi.maximum_filter(L, size=2 * tol_px + 1)
    n = max(24, int(2 * np.pi * r)); ok = []
    for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
        good = False
        for rr in (0.92 * r, r, 1.06 * r):
            y = int(round(cy + rr * np.sin(t))); x = int(round(cx + rr * np.cos(t)))
            if y < 0 or x < 0 or y >= m.shape[0] or x >= m.shape[1]:
                break
            if (m[y, x] and din[y, x] <= tol_px) or (not m[y, x] and dout[y, x] <= tol_px) or (m[y, x] and Lm[y, x] >= thrL):
                good = True; break
        ok.append(good)
    return float(np.mean(ok))


def kasa(y, x):
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]; b = x * x + y * y
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return cy, cx, np.sqrt(max(c + cx * cx + cy * cy, 1e-9))


def arc_round(pm, m, s, P, near=None):
    """Внешняя граница куска (на контуре тела или у плёнки) — дуга окружности?"""
    bd = pm & ~ndi.binary_erosion(pm)
    outer = bd & (ndi.binary_dilation(~m) if near is None else near)
    if outer.sum() < 8:
        return False, None
    y, x = np.nonzero(outer); cy, cx, R = kasa(y.astype(float), x.astype(float))
    rms = np.sqrt(np.mean((np.hypot(y - cy, x - cx) - R) ** 2)) / R
    ang = np.arctan2(y - cy, x - cx); cov = len(np.unique((ang / (2 * np.pi) * 36).astype(int))) / 36
    Y, X = np.mgrid[:m.shape[0], :m.shape[1]]; disc = (np.hypot(Y - cy, X - cx) <= R) & m
    iou = (disc & pm).sum() / max((disc | pm).sum(), 1)
    ok = rms <= P["arc_rms"] and cov >= P["arc_cov"] and iou >= P["arc_iou"] and 2 * R * s >= 0.8 * P["dcut"]
    return ok, dict(rms=rms, cov=cov, iou=iou, d=2 * R * s)


def is_round(f, P):
    return f["ar"] <= P["ar_max"] and f["eiou"] >= P["eiou_min"] and f["sol"] >= P["sol_min"]


# ------------------------------------------------------------------ разбиение
def regions_split(Ds, m, s, ped, rel_t, norm_min, md_nm, Dm, bq=0.5, min_area_nm2=12.0):
    """Водораздел по затемнению; соседние куски сливаются, пока относительная глубина долины < rel_t.
    Глубина = (пик слабейшего − квантиль bq затемнения вдоль общей границы) / max(пик − подложка, norm_min)."""
    pk = feature.peak_local_max(np.where(m, Dm, -1e9), min_distance=max(2, int(round(md_nm / s))),
                                labels=m.astype(int), exclude_border=False)
    if len(pk) == 0:
        return m.astype(np.int32)
    mk = np.zeros(m.shape, np.int32)
    for i, (y, x) in enumerate(pk, 1):
        mk[y, x] = i
    lab = segmentation.watershed(-Ds, mk, mask=m, watershed_line=False)
    while True:
        ids = [i for i in np.unique(lab) if i > 0]
        if len(ids) < 2:
            break
        peak = {i: Ds[lab == i].max() for i in ids}
        area = {i: (lab == i).sum() * s * s for i in ids}
        Pp = []; V = []
        for dy, dx in ((0, 1), (1, 0), (1, 1)):
            a = lab[:lab.shape[0] - dy, :lab.shape[1] - dx]; b = lab[dy:, dx:]
            da = Ds[:Ds.shape[0] - dy, :Ds.shape[1] - dx]; db = Ds[dy:, dx:]
            sel = (a != b) & (a > 0) & (b > 0)
            Pp.append(np.stack([np.minimum(a[sel], b[sel]), np.maximum(a[sel], b[sel])], 1)); V.append(np.minimum(da[sel], db[sel]))
        Pp = np.concatenate(Pp); V = np.concatenate(V); best = {}
        if len(Pp):
            key = Pp[:, 0].astype(np.int64) * 100000 + Pp[:, 1]; o = np.lexsort((V, key)); key = key[o]; V = V[o]
            st = np.r_[0, np.nonzero(key[1:] != key[:-1])[0] + 1]; en = np.r_[st[1:], len(key)]
            for a0, b0 in zip(st, en):
                if b0 - a0 < 3:
                    continue
                k = key[a0]; best[(int(k // 100000), int(k % 100000))] = V[a0:b0][min(b0 - a0 - 1, int(bq * (b0 - a0)))]
        if not best:
            break
        cand = []
        for (i, j), sad in best.items():
            lo = min(peak[i], peak[j]); rel = (lo - sad) / max(lo - ped, norm_min)
            cand.append((rel * (0.5 if min(area[i], area[j]) < min_area_nm2 else 1.0), i, j))
        cand.sort()
        f, i, j = cand[0]
        if f >= rel_t:
            break
        lab[lab == j] = i
    return lab


def neck_split(m, s, dcut, neck_ratio):
    """Разбиение по перетяжке контура (максимумы карты расстояний; шейка < neck_ratio меньшего радиуса)."""
    edm = ndi.distance_transform_edt(m)
    pk = feature.peak_local_max(edm, min_distance=max(2, int(4 / s)), threshold_abs=0.25 * dcut / 2 / s,
                                labels=m.astype(int), exclude_border=False)
    pk = sorted([tuple(q) for q in pk], key=lambda q: -edm[q]); keep = []

    def neck(q, k):
        L = int(np.hypot(q[0] - k[0], q[1] - k[1])) + 2
        yy = np.linspace(q[0], k[0], L).round().astype(int); xx = np.linspace(q[1], k[1], L).round().astype(int)
        return edm[yy, xx].min()
    for q in pk:
        ok = True
        for k in keep:
            if np.hypot(q[0] - k[0], q[1] - k[1]) < 0.5 * max(edm[q], edm[k]) or neck(q, k) > neck_ratio * min(edm[q], edm[k]):
                ok = False; break
        if ok:
            keep.append(q)
    if len(keep) < 2:
        return None
    mk = np.zeros(m.shape, np.int32)
    for i, q in enumerate(keep, 1):
        mk[q] = i
    return segmentation.watershed(-edm, mk, mask=m)


# ------------------------------------------------------------------ решение
class Body:
    """Контекст тела: маска m в окне, затемнение Ds (σ 1 нм), подложка ped, маска вещества, функция кругов v8."""
    def __init__(self, m, Ds, ped, s, lvl1, m_all, place):
        self.m, self.Ds, self.ped, self.s, self.lvl1, self.m_all, self.place = m, Ds, ped, s, lvl1, m_all, place


def mottled(m, Dp, s, P):
    return texture(Dp, m, s) >= P["t_tex"] and radial_r2(Dp, m) < P["t_rad"]


def nf_or_agg(m, Dp, s, lvl1, P, tag=""):
    k = float(np.median(Dp[m]) / lvl1)
    if k >= P["k_nf"] and not mottled(m, Dp, s, P):
        return (m, "nf", "однородное плотное" + tag)
    return (m, "agg", ("бледное" if k < P["k_nf"] else "пёстрое") + tag)


def classify(B, P):
    """Части тела: список (маска, класс, причина), класс: mod / nf / agg / prim."""
    s = B.s; m = B.m; Dp = B.Ds - B.ped
    if shape_feats(m, s)["deq"] >= P["dcut"] and mottled(m, Dp, s, P):
        return [(m, "agg", "пёстрое")]
    out = _body(B, m, P)
    mods = [q for q, c, w in out if c == "mod"]
    if mods and (len(mods) > 1 or len(out) > 1):
        cov = sum(q.sum() for q in mods) / m.sum()
        if cov < P["cov_min"]:
            return [nf_or_agg(m, Dp, s, B.lvl1, P, " (модиф. покрывают {:.0%})".format(cov))]
    return out


def _body(B, m, P, depth=0, whole=None):
    s, lvl1, Ds, ped = B.s, B.lvl1, B.Ds, B.ped
    whole = m if whole is None else whole
    f = shape_feats(m, s)
    if f["deq"] < P["dcut"]:
        return [(m, "prim", "мельче отсечки")]
    D2 = ndi.gaussian_filter(Ds, np.sqrt(3) / s)
    emin = float(np.interp(f["deq"], [1.6 * P["dcut"], 3.2 * P["dcut"]], P["g_e"]))
    rnd = f["ar"] <= P["g_ar"] and f["eiou"] >= emin and f["sol"] >= P["g_sol"]
    if not rnd and depth == 0:
        near = ndi.binary_dilation(~B.m_all, iterations=max(2, int(round(6 / s))))
        okw, iw = arc_round(m, m, s, dict(P, arc_cov=0.40), near=near)
        rnd = okw and iw["iou"] >= 0.88
    if rnd:
        lab = regions_split(Ds, m, s, ped, P["t_round"], 1.0 * lvl1, P["md_nm"], D2)
        ids = [i for i in np.unique(lab) if i > 0]
        if len(ids) > 1:
            good = all(shape_feats(lab == i, s)["deq"] < P["dcut"] or arc_round(lab == i, whole, s, P)[0] for i in ids)
            nbig = sum(shape_feats(lab == i, s)["deq"] >= P["dcut"] for i in ids)
            if not good or nbig == 0:
                ids = [ids[0]]; lab = m.astype(np.int32) * ids[0]
    else:
        lab = regions_split(Ds, m, s, ped, P["t_valley"], 0.5 * lvl1, P["md_nm"], D2)
        ids = [i for i in np.unique(lab) if i > 0]
    if len(ids) == 1:
        if rnd:
            return [(m, "mod", "близко к кругу / эллипсу")]
        if depth < 2:
            ns = neck_split(m, s, P["dcut"], P["neck"])
            if ns is not None:
                sub = [ns == j for j in np.unique(ns) if j > 0]
                if all(shape_feats(q, s)["deq"] < P["dcut"] or is_round(shape_feats(q, s), P) or arc_round(q, m, s, P)[0]
                       for q in sub):
                    return [(q, "prim" if shape_feats(q, s)["deq"] < P["dcut"] else "mod", "перетяжка") for q in sub]
        C = B.place(m)
        ov = None
        if len(C) >= 2:
            ov = 0.0
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    (y1, x1, r1), (y2, x2, r2) = C[i], C[j]
                    ov = max(ov, (r1 + r2 - np.hypot(y1 - y2, x1 - x2)) / (2 * min(r1, r2)))
        L = light_lines(Ds - ped, s, lvl1)
        acc = []
        Y, X = np.ogrid[:m.shape[0], :m.shape[1]]
        for (y, x, r) in sorted(C, key=lambda c: -c[2]):
            disc = (Y - y) ** 2 + (X - x) ** 2 <= r * r
            if disc.sum() == 0 or (disc & m).sum() / disc.sum() < P["in_min"]:
                continue
            if support(m, L, y, x, r) < P["sup_min"]:
                continue
            if any((r + r2 - np.hypot(y - y2, x - x2)) / (2 * min(r, r2)) > P["ov_acc"] for y2, x2, r2 in acc):
                continue
            acc.append((y, x, r))
        circ = acc if (acc and max(2 * r * s for _, _, r in acc) >= P["dcut"]) else (
            C if ov is not None and ov <= P["ov_max"] else None)
        if circ:
            out = []; Yg, Xg = np.mgrid[:m.shape[0], :m.shape[1]]
            dist = np.stack([np.hypot(Yg - y, Xg - x) / r for y, x, r in circ]); own = dist.argmin(0)
            used = np.zeros_like(m)
            why = "круги с краем" if circ is acc else "круги почти не перекрываются"
            for j, (y, x, r) in enumerate(circ):
                q = m & (own == j) & (dist[j] <= 1.0); used |= q
                if q.sum() >= 4:
                    out.append((q, "mod" if 2 * r * s >= P["dcut"] else "prim", why))
            rest = ndi.binary_opening(m & ~used, iterations=1); rl = measure.label(rest)
            for j in range(1, rl.max() + 1):
                q = rl == j
                if q.sum() >= 4:
                    out.append((q, "agg", "остаток"))
            return out
        return [nf_or_agg(m, Ds - ped, s, lvl1, P)]
    out = []
    for i in ids:
        pm = lab == i
        if pm.sum() < 4:
            continue
        fp = shape_feats(pm, s)
        if fp["deq"] < P["dcut"]:
            out.append((pm, "prim", "мельче отсечки")); continue
        ok, _ = arc_round(pm, whole, s, P)
        if ok or (fp["ar"] <= P["ar_max"] and fp["eiou"] >= P["eiou_piece"] and fp["sol"] >= P["sol_piece"]):
            out.append((pm, "mod", "кусок по светлым линиям")); continue
        sub = _body(B, pm, P, depth + 1, whole)
        out += [(q, "agg" if (c == "nf" and shape_feats(q, s)["deq"] < P["big_nf"] * P["dcut"]) else c, w) for q, c, w in sub]
    return out
