# -*- coding: utf-8 -*-
"""
Пул кандидатов v3 — три новых правила:
  1) склейка разорванных дуг одного круга (крупная частица с прилипшими соседями);
  2) край частицы по затемнению (радиальный перепад на окружности) и тёмные пятна внутри контура;
  3) центры на осевой линии ветви + круги вдоль ветви с радиусом по её ширине.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from . import geo, geo2, blobs

FEATS = ["support", "inside", "dtc", "span", "rms", "logR", "d_in", "d_ring", "edge_c", "d_cv", "d_ctr",
         "grad_ring", "grad_cov", "cover", "n_arcs", "dist_skel",
         "is_arc", "is_merged", "is_dt", "is_snap", "is_chain", "is_blob", "is_radial"]

PDEF = dict(smooth_nm=2.0, smooth_scales=(2.0, 5.0, 10.0), levels=(None, 0.45, 0.6, 0.8, 1.0, 1.3), merge_tol=0.08, merge_min_span=20,
            chain_k=(1.0, 1.2), chain_sep=0.9)


def arcs_all(M, Ds, s, P):
    """Дуги контура: на уровнях затемнения и — для самого контура — в нескольких масштабах сглаживания
    (крупное сглаживание убирает мелкие бугорки от прилипших частиц и проявляет край большой частицы)."""
    out = []
    for L in P["levels"]:
        ML = M if L is None else (M & (Ds > L))
        if ML.sum() < 20:
            continue
        scales = P["smooth_scales"] if L is None else (P["smooth_nm"],)
        for sm in scales:
            PP = dict(P); PP["smooth_nm"] = sm
            for a in geo2.contour_arcs(ML, s, PP, return_pts=True):
                out.append(dict(y=a[0], x=a[1], R=a[2], span=a[3], rms=a[4], py=a[5], px=a[6], lev=L, sm=sm))
    return out


def merge_arcs(arcs, P, rmin, rmax, seeds=()):
    """Склейка дуг, лежащих на одной окружности: совместная подгонка, суммарный угловой охват."""
    A = [a for a in arcs if a["span"] >= P["merge_min_span"] and a["rms"] <= 0.2 and len(a["py"]) >= 6]
    if not A:
        return []
    cen = np.array([[a["py"].mean(), a["px"].mean()] for a in A])
    out = []; seen = set()
    starts = [(A[i]["y"], A[i]["x"], A[i]["R"], i) for i in sorted(range(len(A)), key=lambda i: -len(A[i]["py"]))]
    starts += [(y, x, R, None) for (y, x, R) in seeds]
    for (y0, x0, R0, i) in starts:
        if not (rmin <= R0 <= 1.5 * rmax):
            continue
        for it in range(2):
            dc = np.hypot(cen[:, 0] - y0, cen[:, 1] - x0)
            near = np.nonzero((dc > 0.6 * R0) & (dc < 1.4 * R0))[0]
            members = []
            tol = max(1.5, P["merge_tol"] * R0)
            for j in near:
                b = A[j]
                r = np.hypot(b["py"] - y0, b["px"] - x0)
                if np.mean(np.abs(r - R0) <= tol) >= 0.8:
                    members.append(j)
            if i is not None and i not in members:
                members.append(i)
            if not members:
                break
            py = np.concatenate([A[j]["py"] for j in members]); px = np.concatenate([A[j]["px"] for j in members])
            y0, x0, R0 = geo.fit_circle(px, py)
            if not np.isfinite(R0):
                break
        if not members or len(members) < 2 or not np.isfinite(R0) or not (rmin <= R0 <= rmax):
            continue
        key = (round(y0 / 3), round(x0 / 3), round(R0 / 3))
        if key in seen:
            continue
        seen.add(key)
        ang = np.degrees(np.arctan2(py - y0, px - x0)) % 360
        cover = len(np.unique((ang // 10).astype(int))) * 10.0
        rms = np.sqrt(np.mean((np.hypot(py - y0, px - x0) - R0) ** 2)) / R0
        out.append(dict(y=y0, x=x0, R=R0, cover=cover, n=len(members), rms=rms))
    return out


def radial_modes(Mf, y, x, rmin, rmax, n_rays=120, min_sup=0.3):
    """Круги, касающиеся контура на большом угле: моды распределения длин лучей из центра-кандидата.
    Для каждой моды центр и радиус уточняются подгонкой круга к концам лучей этой моды."""
    th, rho, edge = geo.ray_lengths(Mf, y, x, n_rays, rmax * 1.05)
    ok = ~edge
    if ok.sum() < 0.5 * n_rays:
        return []
    Rg = np.exp(np.arange(np.log(max(rmin, 2)), np.log(rmax), 0.04))
    sup = np.array([np.mean(np.abs(rho[ok] - R) <= max(1.5, 0.06 * R)) for R in Rg])
    out = []
    for k in range(len(Rg)):
        if sup[k] < min_sup or sup[k] < sup[max(0, k - 1)] or sup[k] < sup[min(len(Rg) - 1, k + 1)]:
            continue
        R = Rg[k]
        sel = ok & (np.abs(rho - R) <= max(1.5, 0.08 * R))
        if sel.sum() < 8:
            continue
        px = x + rho[sel] * np.cos(th[sel]); py = y + rho[sel] * np.sin(th[sel])
        y1, x1, R1 = geo.fit_circle(px, py)
        if not np.isfinite(R1) or np.hypot(y1 - y, x1 - x) > 0.5 * R or not (0.7 * R <= R1 <= 1.3 * R):
            continue
        ang = np.degrees(np.arctan2(py - y1, px - x1)) % 360
        cover = len(np.unique((ang // 10).astype(int))) * 10.0
        out.append((y1, x1, R1, cover))
    return out


def chain_candidates(M, dt, skel, s, P, rmin):
    """Круги вдоль осевой линии: центр на оси, радиус по ширине ветви (жадно по убыванию толщины)."""
    ys, xs = np.nonzero(skel & (dt >= rmin))
    if len(ys) == 0:
        return []
    order = np.argsort(-dt[ys, xs])
    acc = []
    AY = np.empty(len(order)); AX = np.empty(len(order)); AR = np.empty(len(order)); n = 0
    for k in order:
        y, x = ys[k], xs[k]; r = dt[y, x]
        if n and np.any(np.hypot(y - AY[:n], x - AX[:n]) < P["chain_sep"] * (r + AR[:n])):
            continue
        acc.append((y, x, r)); AY[n], AX[n], AR[n] = y, x, r; n += 1
    out = []
    for (y, x, r) in acc:
        for kf in P["chain_k"]:
            out.append((float(y), float(x), float(r * kf)))
    return out


def candidates(M, D, lvl1, s, params=None):
    P = dict(geo.DEF); P.update(geo2.DEF2); P.update(PDEF); P.update(params or {})
    rmin = P["rmin_nm"] / s; rmax = P["rmax_nm"] / s
    H, W = M.shape
    dt = ndi.distance_transform_edt(M)
    edge = M ^ ndi.binary_erosion(M)
    edge_dist = ndi.distance_transform_edt(~edge)
    skel = skeletonize(M)
    skel_dist = ndi.distance_transform_edt(~skel)
    Dg = ndi.gaussian_filter(D, max(1.0, 0.7 / s))
    Ds = Dg / max(lvl1, 1e-6)
    gy, gx = np.gradient(Dg)
    C = []
    Mf = ndi.binary_fill_holes(M)                 # светлые «дыры» внутри агломерата человек не учитывает
    dtf = ndi.distance_transform_edt(Mf)
    arcs = arcs_all(Mf, Ds, s, P)
    seeds = []
    pkf = peak_local_max(dtf, min_distance=max(2, int(round(rmin))), threshold_abs=2 * rmin, labels=Mf.astype(int),
                         exclude_border=False)
    for (y, x) in pkf:
        for (y1, x1, R1, cov) in radial_modes(Mf, float(y), float(x), rmin, rmax):
            C.append(dict(kind="radial", y=y1, x=x1, R=R1, span=cov, rms=0.0, cover=cov, n=1))
            seeds.append((y1, x1, R1))
    for a in arcs:
        if a["span"] >= 70 and a["rms"] <= 0.2 and rmin <= a["R"] <= rmax:
            C.append(dict(kind="arc", y=a["y"], x=a["x"], R=a["R"], span=a["span"], rms=a["rms"], cover=a["span"], n=1))
    for m in merge_arcs(arcs, P, rmin, rmax, seeds):
        C.append(dict(kind="merged", y=m["y"], x=m["x"], R=m["R"], span=m["cover"], rms=m["rms"], cover=m["cover"], n=m["n"]))
    pk = peak_local_max(dt, min_distance=max(2, int(round(0.5 * rmin))), threshold_abs=rmin, labels=M.astype(int),
                        exclude_border=False)
    for (y, x) in pk:
        yc, xc, R, fr = geo.measure_particle(M, float(y), float(x), float(dt[y, x]), P, rmax)
        C.append(dict(kind="dt", y=yc, x=xc, R=R))
        C.append(dict(kind="dt", y=float(y), x=float(x), R=float(dt[y, x])))
        # притяжение к осевой линии: ближайшая точка оси
        yi, xi = int(round(min(max(yc, 0), H - 1))), int(round(min(max(xc, 0), W - 1)))
        if skel_dist[yi, xi] <= 0.4 * R:
            a, b = int(max(0, yi - R)), int(min(H, yi + R + 1)); c, d = int(max(0, xi - R)), int(min(W, xi + R + 1))
            sy, sx = np.nonzero(skel[a:b, c:d])
            if len(sy):
                j = np.argmin((sy + a - yc) ** 2 + (sx + c - xc) ** 2)
                C.append(dict(kind="snap", y=float(sy[j] + a), x=float(sx[j] + c), R=R))
    for (y, x, R) in chain_candidates(M, dt, skel, s, P, rmin):
        C.append(dict(kind="chain", y=y, x=x, R=R))
    for (y, x, R) in blobs.blobs(D, M, s, thr=0.03, overlap=0.5):
        C.append(dict(kind="blob", y=y, x=x, R=R))

    out = []
    for c in C:
        y, x, R = c["y"], c["x"], c["R"]
        if not (0 <= y < H and 0 <= x < W) or R < 1 or R > rmax:
            continue
        sc = geo2.circle_scores(M, dt, edge_dist, (y, x, R), P)
        if sc is None:
            continue
        inside, support, dtc = sc
        if inside < 0.5:
            continue
        a, b = int(max(0, y - 1.4 * R)), int(min(H, y + 1.4 * R + 1)); cc, d = int(max(0, x - 1.4 * R)), int(min(W, x + 1.4 * R + 1))
        yy, xx = np.ogrid[a:b, cc:d]; rr = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
        Dp = Ds[a:b, cc:d]
        inn = rr <= 0.8 * R; ring_in = (rr > 0.8 * R) & (rr <= R); ring_out = (rr > R) & (rr <= 1.25 * R); ctr = rr <= 0.3 * R
        if inn.sum() < 3 or ring_out.sum() < 3:
            continue
        d_in = float(Dp[inn].mean()); d_ring = float(Dp[ring_in].mean()) if ring_in.any() else d_in
        d_out = float(Dp[ring_out].mean()); d_cv = float(Dp[inn].std() / (abs(d_in) + 1e-3))
        d_ctr = float(Dp[ctr].mean() - d_in) if ctr.any() else 0.0
        # край по затемнению: радиальный перепад на окружности (среднее и доля направлений, где край есть)
        ring = (rr > 0.85 * R) & (rr <= 1.15 * R)
        gr = -(gy[a:b, cc:d] * (yy - y) + gx[a:b, cc:d] * (xx - x)) / np.maximum(rr, 1e-6) * R / max(lvl1, 1e-6)
        if ring.any():
            grad_r = float(gr[ring].mean())
            th = np.arctan2((yy - y) * np.ones_like(xx), (xx - x) * np.ones_like(yy))[ring]
            bins = ((th + np.pi) / (2 * np.pi) * 24).astype(int) % 24
            gm = np.bincount(bins, weights=gr[ring], minlength=24) / np.maximum(np.bincount(bins, minlength=24), 1)
            grad_cov = float((gm > 0.15).mean())
        else:
            grad_r, grad_cov = 0.0, 0.0
        yi, xi = min(int(round(y)), H - 1), min(int(round(x)), W - 1)
        f = dict(support=support, inside=inside, dtc=dtc, span=c.get("span", 0) / 360.0, rms=c.get("rms", 0),
                 logR=np.log(R * s), d_in=d_in, d_ring=d_ring, edge_c=d_ring - d_out, d_cv=d_cv, d_ctr=d_ctr,
                 grad_ring=grad_r, grad_cov=grad_cov, cover=c.get("cover", 0) / 360.0, n_arcs=min(c.get("n", 0), 6),
                 dist_skel=min(skel_dist[yi, xi] / R, 2.0),
                 is_arc=float(c["kind"] == "arc"), is_merged=float(c["kind"] == "merged"), is_dt=float(c["kind"] == "dt"),
                 is_snap=float(c["kind"] == "snap"), is_chain=float(c["kind"] == "chain"), is_blob=float(c["kind"] == "blob"),
                 is_radial=float(c["kind"] == "radial"))
        out.append(((y, x, R), f))
    return out


def select(cands, probs, feats, thr=0.2, ov_max=0.5, ov_big=0.85, inner_grad=0.6, inner_dark=0.15, order_pow=0.0,
           big_k=None, big_cover=0.5, big_support=0.45, replace_rho=None, replace_contain=0.75):
    """Жадный отбор «хороших» кругов. Круг внутри принятого допускается, только если у него
    есть собственный край (перепад затемнения по окружности) и он темнее окружения — частица поверх."""
    from .util import circ_iou
    key = probs * np.array([c[2] for c in cands]) ** order_pow
    order = np.argsort(-key)
    acc = []; accf = []
    good = [cands[i][2] for i in range(len(cands)) if probs[i] >= thr]
    R_typ = float(np.median(good)) if good else 0.0
    for i in order:
        if probs[i] < thr:
            continue
        y, x, R = cands[i]
        # крупный круг должен опираться на контур: большой угловой охват дуг или касание края
        if big_k and R > big_k * R_typ and not (feats[i]["cover"] >= big_cover or feats[i]["support"] >= big_support):
            continue
        ok = True
        for (y2, x2, R2), f2 in zip(acc, accf):
            if np.hypot(y - y2, x - x2) < R + R2:
                I = circ_iou((y, x, R), (y2, x2, R2))
                inter = I * (np.pi * R * R + np.pi * R2 * R2) / (1 + I)
                if inter / (np.pi * R2 * R2) > ov_big:
                    ok = False; break
                if inter / (np.pi * R * R) > ov_max:
                    f = feats[i]
                    interior = R < 0.6 * R2 and f["grad_cov"] >= inner_grad and f["d_in"] - f2["d_in"] >= inner_dark
                    if not interior:
                        ok = False; break
        if ok:
            acc.append((y, x, R)); accf.append(feats[i])
    if replace_rho is not None:
        acc, accf = replace_families(cands, probs, feats, acc, accf, thr, replace_rho, replace_contain, inner_grad, inner_dark)
    return acc


def replace_families(cands, probs, feats, acc, accf, thr, rho, contain, inner_grad, inner_dark):
    """Против «семейств»: если один круг-кандидат накрывает >=2 принятых кругов и объясняет область
    не хуже них (вероятность >= rho * средней вероятности накрытых), заменяем семейство одним кругом.
    Внутри оставляем только круги с собственным краем, которые темнее (частицы поверх)."""
    from .util import circ_iou
    pidx = {tuple(c): i for i, c in enumerate(cands)}
    order = sorted(range(len(cands)), key=lambda i: -cands[i][2])
    accset = {tuple(a) for a in acc}
    for i in order:
        if probs[i] < thr or tuple(cands[i]) in accset:
            continue
        y, x, R = cands[i]
        inside = []
        for k, (y2, x2, R2) in enumerate(acc):
            if R2 >= R or np.hypot(y - y2, x - x2) >= R + R2:
                continue
            I = circ_iou((y, x, R), (y2, x2, R2))
            inter = I * (np.pi * R * R + np.pi * R2 * R2) / (1 + I)
            if inter / (np.pi * R2 * R2) >= contain:
                inside.append(k)
        if len(inside) < 2:
            continue
        pin = np.mean([probs[pidx[tuple(acc[k])]] for k in inside])
        if probs[i] < rho * pin:
            continue
        keep = []
        for k in inside:
            f = accf[k]
            if acc[k][2] < 0.6 * R and f["grad_cov"] >= inner_grad and f["d_in"] - feats[i]["d_in"] >= inner_dark:
                keep.append(k)
        drop = set(inside) - set(keep)
        # новый круг не должен сильно перекрывать оставшиеся (не накрытые) круги
        bad = False
        for k, (y2, x2, R2) in enumerate(acc):
            if k in drop or k in keep or np.hypot(y - y2, x - x2) >= R + R2:
                continue
            I = circ_iou((y, x, R), (y2, x2, R2))
            inter = I * (np.pi * R * R + np.pi * R2 * R2) / (1 + I)
            if inter / (np.pi * R2 * R2) > 0.7:
                bad = True; break
        if bad:
            continue
        acc = [a for k, a in enumerate(acc) if k not in drop] + [(y, x, R)]
        accf = [a for k, a in enumerate(accf) if k not in drop] + [feats[i]]
        accset = {tuple(a) for a in acc}
    return acc, accf
