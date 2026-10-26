# -*- coding: utf-8 -*-
"""
Второй проход: дополнительные круги в незанятых участках контура (выступы, пустоты в цепочках).

После основного отбора ищутся куски контура, не накрытые кругами. Для каждого куска площадью
больше a_frac * (площадь типичной частицы):
  1) лучший кандидат из пула с ослабленным порогом вероятности p2, центр которого лежит в этом куске
     и чей круг в основном (>= unc_min) приходится на незанятую площадь;
  2) если такого нет — круг по дуге контура, ограничивающей кусок (выступ: центр и радиус кривизны
     края), при условии, что он почти целиком внутри контура;
  3) иначе — вписанный в кусок круг (если кусок достаточно толстый).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from . import cont, geo


def _ok_overlap(c, A, ov_new, ov_old):
    from .util import circ_iou
    for o in A:
        if np.hypot(c[0] - o[0], c[1] - o[1]) < c[2] + o[2]:
            I = circ_iou(c, o); inter = I * (np.pi * c[2] ** 2 + np.pi * o[2] ** 2) / (1 + I)
            if inter / (np.pi * c[2] ** 2) > ov_new or inter / (np.pi * o[2] ** 2) > ov_old:
                return False
    return True


def second_pass(M, A, cands, probs, s, p2=0.05, a_frac=0.25, unc_min=0.5, inside_min=0.8, ov_new=0.6, ov_old=0.5,
                rmax_f=1.5, rmin_f=0.3, use_arc=True, use_inscribed=True, rounds=2, thick_min=0.0, use_mid=False, mid_k=1.05, mid_unc=0.4):
    A = list(A)
    if not A:
        return A
    r_typ = float(np.median([a[2] for a in A]))
    Mf = ndi.binary_fill_holes(M)
    edge = Mf ^ ndi.binary_erosion(Mf)
    H, W = M.shape
    C = np.asarray(cands, float); P = np.asarray(probs)
    dtF = ndi.distance_transform_edt(Mf)
    for _ in range(rounds):
        U = cont.disks_mask(M.shape, A, 1.0)
        Rm = Mf & ~U
        Rm = ndi.binary_opening(Rm, iterations=max(1, int(0.12 * r_typ)))     # убираем тонкие щели между кругами
        lab = measure.label(Rm)
        added = 0
        for p in sorted(measure.regionprops(lab), key=lambda p: -p.area):
            if p.area < a_frac * np.pi * r_typ ** 2:
                continue
            comp = lab == p.label
            y0_, x0_, y1_, x1_ = p.bbox
            bb = (slice(max(0, y0_ - 1), min(H, y1_ + 1)), slice(max(0, x0_ - 1), min(W, x1_ + 1)))
            if thick_min and ndi.distance_transform_edt(comp[bb]).max() < thick_min * r_typ:
                continue                                   # тонкая кромка/щель — человек такие не заполняет
            best = None
            # 1) кандидат из пула
            cy, cx = C[:, 0].round().astype(int).clip(0, H - 1), C[:, 1].round().astype(int).clip(0, W - 1)
            idx = np.nonzero(comp[cy, cx] & (P >= p2) & (C[:, 2] <= rmax_f * r_typ) & (C[:, 2] >= rmin_f * r_typ))[0]
            bs = -1
            for i in idx:
                y, x, R = C[i]
                sl, dm = geo.disk_win(M.shape, y, x, R); area = dm.sum()
                if area == 0:
                    continue
                unc = (dm & ~U[sl]).sum() / area; ins = (dm & Mf[sl]).sum() / area
                if unc < unc_min or ins < inside_min:
                    continue
                sc = P[i] * unc
                if sc > bs and _ok_overlap((y, x, R), A, ov_new, ov_old):
                    bs = sc; best = (float(y), float(x), float(R))
            # 1b) круг, вписанный в КОНТУР (касается краёв агломерата), с центром в незанятом куске:
            #     для выступа это частица, «торчащая» из цепочки; перекрытие с уже принятыми кругами допускается
            if best is None and use_mid:
                yy, xx = np.nonzero(comp)
                Rr = dtF[yy, xx] * mid_k
                ok = (Rr >= rmin_f * r_typ) & (Rr <= rmax_f * r_typ)
                yy, xx, Rr = yy[ok], xx[ok], Rr[ok]
                if len(yy):
                    o = np.argsort(-Rr)[:400:4] if len(yy) > 100 else np.argsort(-Rr)
                    bs2 = 0
                    for i in o:
                        y, x, R = float(yy[i]), float(xx[i]), float(Rr[i])
                        sl, dm = geo.disk_win(M.shape, y, x, R); area = max(dm.sum(), 1)
                        unc = (dm & ~U[sl]).sum() / area
                        if unc < mid_unc or (dm & Mf[sl]).sum() / area < inside_min:
                            continue
                        gain = (dm & comp[sl]).sum()
                        if gain > bs2 and _ok_overlap((y, x, R), A, ov_new, ov_old):
                            bs2 = gain; best = (y, x, R)
            # 2) круг по дуге контура выступа
            if best is None and use_arc:
                bb2 = (slice(max(0, y0_ - 3), min(H, y1_ + 3)), slice(max(0, x0_ - 3), min(W, x1_ + 3)))
                ey, ex = np.nonzero(edge[bb2] & ndi.binary_dilation(comp[bb2], iterations=2))
                ey = ey + bb2[0].start; ex = ex + bb2[1].start
                if len(ey) >= 10:
                    y, x, R = geo.fit_circle(ex.astype(float), ey.astype(float))
                    if np.isfinite(R) and rmin_f * r_typ <= R <= rmax_f * r_typ and 0 <= y < H and 0 <= x < W:
                        ang = np.degrees(np.arctan2(ey - y, ex - x)) % 360
                        span = len(np.unique((ang // 15).astype(int))) * 15
                        sl, dm = geo.disk_win(M.shape, y, x, R); area = max(dm.sum(), 1)
                        if span >= 120 and (dm & Mf[sl]).sum() / area >= inside_min and (dm & ~U[sl]).sum() / area >= unc_min \
                                and _ok_overlap((y, x, R), A, ov_new, ov_old):
                            best = (float(y), float(x), float(R))
            # 3) вписанный круг
            if best is None and use_inscribed:
                dtc = ndi.distance_transform_edt(comp[bb])
                y, x = np.unravel_index(np.argmax(dtc), dtc.shape); R = float(dtc[y, x]) * 1.1
                y, x = y + bb[0].start, x + bb[1].start
                if R >= rmin_f * r_typ and _ok_overlap((y, x, R), A, ov_new, ov_old):
                    best = (float(y), float(x), R)
            if best is not None:
                A.append(best); added += 1
                sl, dm = geo.disk_win(M.shape, *best); U[sl] |= dm
        if not added:
            break
    return A
