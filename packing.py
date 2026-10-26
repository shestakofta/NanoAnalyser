# -*- coding: utf-8 -*-
"""
packing.py — расстановка кругов (первичных частиц) внутри контура агломерата
"методом баланса площадей".

Идея простыми словами:
  * у нас есть контур агломерата (маска: True — вещество, False — фон);
  * мы ставим внутрь круги заданного диапазона диаметров;
  * круги могут перекрываться (частицы спекаются и меняют форму);
  * центр каждого круга обязательно лежит внутри контура;
  * в конце все радиусы чуть-чуть дружно подправляются так, чтобы
      площадь контура, не накрытая кругами  ==  площадь кругов, вылезшая за контур.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import measure


def _init_circles(m, s, r_min_nm, r_cap_nm, c_max=0.3):
    """Начальные круги: вписанные окружности в самых "толстых" местах маски."""
    edm = ndi.distance_transform_edt(m)          # расстояние от каждой точки до края
    cand = peak_local_max(edm, min_distance=max(2, int(1.5 / s)), threshold_abs=r_min_nm / s,
                          labels=m.astype(int), exclude_border=False)
    if len(cand) == 0:
        return []
    vals = np.minimum(edm[cand[:, 0], cand[:, 1]], r_cap_nm / s)
    acc = []
    for i in np.argsort(-vals):                  # от самых больших к маленьким
        y, x = cand[i]; r = vals[i]
        if all(np.hypot(y - a, x - b) >= (r + c) * (1 - c_max) for a, b, c in acc):
            acc.append([float(y), float(x), float(r)])
    return acc


class _Packer:
    def __init__(self, m, s, dmin, dmax, thr):
        self.m = m; self.s = s; self.H, self.W = m.shape
        self.RMIN = dmin / 2 / s; self.RMAX = dmax / 2 / s; self.THR = thr
        self.cnt = np.zeros(m.shape, np.int16)  # сколько кругов накрывает каждый пиксель
        self.circ = []

    def _patch(self, y, x, r):
        a, b = int(max(0, np.floor(y - r))), int(min(self.H, np.ceil(y + r) + 1))
        c, d = int(max(0, np.floor(x - r))), int(min(self.W, np.ceil(x + r) + 1))
        yy, xx = np.ogrid[a:b, c:d]
        return (slice(a, b), slice(c, d)), (yy - y) ** 2 + (xx - x) ** 2 <= r * r

    def _add(self, cc, sign=1):
        sl, D = self._patch(*cc[:3]); self.cnt[sl] += sign * D.astype(np.int16)

    def _gain(self, cc):
        """Насколько изменится (непокрытая площадь + вылет за контур), если добавить круг. Меньше — лучше."""
        sl, D = self._patch(*cc[:3]); c = self.cnt[sl]; mm = self.m[sl]; new = D & (c == 0)
        return -(new & mm).sum() + (new & ~mm).sum()

    def _loss_if_removed(self, cc):
        sl, D = self._patch(*cc[:3]); c = self.cnt[sl]; mm = self.m[sl]; lost = D & (c == 1)
        return (lost & mm).sum() - (lost & ~mm).sum()

    def _balance(self):
        U = self.cnt > 0
        return (self.m & ~U).sum(), (U & ~self.m).sum()

    def run(self, init):
        for c in init:
            c = [c[0], c[1], float(np.clip(c[2], self.RMIN, self.RMAX))]
            self.circ.append(c); self._add(c)
        rgrid = np.arange(self.RMIN, self.RMAX + 1e-6, max(0.5 / self.s, 0.5))
        thr_px = self.THR * np.pi * self.RMIN ** 2
        # 1) жадно добавляем круги в непокрытые места
        for _ in range(200):
            R = self.m & (self.cnt == 0)
            e = ndi.distance_transform_edt(R)
            cand = peak_local_max(e, min_distance=3, threshold_abs=1, exclude_border=False, num_peaks=4000)
            res = []
            for (y, x) in cand:
                best = (0, None)
                for r in rgrid:
                    g = self._gain((y, x, r))
                    if g < best[0]: best = (g, r)
                if best[1] is not None and -best[0] >= thr_px:
                    res.append((best[0], y, x, best[1]))
            if not res:
                break
            res.sort(); taken = []
            for g, y, x, r in res:
                if any(np.hypot(y - ty, x - tx) < r + tr + 2 for ty, tx, tr in taken):
                    continue
                c = [float(y), float(x), float(r)]
                if -self._gain(c) >= thr_px:
                    self.circ.append(c); self._add(c); taken.append((y, x, r))
        # 2) убираем лишние круги (те, что больше вылезают за контур, чем закрывают внутри)
        changed = True
        while changed:
            changed = False; rm = []
            for i in sorted(range(len(self.circ)), key=lambda i: self.circ[i][2]):
                if self._loss_if_removed(self.circ[i]) <= 0:
                    self._add(self.circ[i], -1); rm.append(i); changed = True
            for i in sorted(rm, reverse=True):
                self.circ.pop(i)
        # 3) немного двигаем каждый круг, если так становится лучше
        moves = [(2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 1), (0, 0, -1), (1, 1, 0), (-1, -1, 0)]
        for _ in range(3):
            for i in range(len(self.circ)):
                c = self.circ[i]; self._add(c, -1); best = (self._gain(c), c)
                for dy, dx, dr in moves:
                    ny, nx = c[0] + dy, c[1] + dx; nr = float(np.clip(c[2] + dr, self.RMIN, self.RMAX))
                    iy, ix = int(round(ny)), int(round(nx))
                    if not (0 <= iy < self.H and 0 <= ix < self.W and self.m[iy, ix]):
                        continue                      # центр должен остаться внутри контура
                    g = self._gain((ny, nx, nr))
                    if g < best[0]: best = (g, [ny, nx, nr])
                self.circ[i] = best[1]; self._add(self.circ[i])
        # 4) общий сдвиг всех радиусов до равенства площадей (бисекция)
        base = [list(c) for c in self.circ]
        def at(dr):
            self.cnt[:] = 0
            self.circ = [[y, x, float(np.clip(r + dr, self.RMIN * 0.9, self.RMAX * 1.05))] for y, x, r in base]
            for c in self.circ: self._add(c)
            b, g = self._balance(); return b - g
        lo, hi = -1.5 / self.s, 1.5 / self.s
        for _ in range(20):
            mid = (lo + hi) / 2
            if at(mid) > 0: lo = mid
            else: hi = mid
        at((lo + hi) / 2)
        return self.circ


def pack(mask, s, dmin, dmax, thr=0.5):
    """Круги для одной маски. Возвращает список (y, x, радиус_в_пикселях)."""
    if mask.sum() == 0:
        return []
    init = _init_circles(mask, s, dmin / 2, 0.7 * dmax / 2)
    return _Packer(mask, s, dmin, dmax, thr).run(init)


def pack_components(mask, s, dmin, dmax, thr=0.5, labels=None):
    """
    То же, но отдельно для каждого связного кусочка маски (так быстрее).
    Возвращает список кругов (y, x, d_нм, номер_компоненты).
    """
    if labels is None:
        labels = measure.label(mask)
    out = []
    pad = int(np.ceil(dmax / s)) + 3
    H, W = mask.shape
    min_area_px = 0.5 * np.pi * (dmin / 2 / s) ** 2
    for p in measure.regionprops(labels):
        if p.area < min_area_px:
            continue
        y0, x0, y1, x1 = p.bbox
        a, b, c, d = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
        sub = labels[a:b, c:d] == p.label
        for (y, x, r) in pack(sub, s, dmin, dmax, thr):
            out.append((y + a, x + c, 2 * r * s, p.label))
    return out
