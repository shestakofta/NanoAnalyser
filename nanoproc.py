# -*- coding: utf-8 -*-
"""
nanoproc.py — "мозг" программы: вся обработка снимков без окон и кнопок.

Что делает этот модуль (по шагам, как в нашем ручном анализе):
  1. Читает снимок и узнаёт масштаб (нм на пиксель): из служебных данных TIFF
     (Olympus analySIS) или по масштабной линейке + длине, которую ввёл пользователь.
  2. Вычитает неравномерный фон и находит всё вещество на снимке
     (контур по половине контраста одиночной первичной частицы).
  3. Ищет крупные тела (больше отсечки, по умолчанию 25 нм) и классифицирует их (classify2):
     близкие к кругу/эллипсу — модифицированные; скопления делятся по светлым линиям на
     модифицированные и первичные; плотные однородные неправильные — НФ; пёстрые — агломерат.
  4. Всё остальное вещество (включая налипшие на крупные частицы обрывки) — агломераты.
  5. Внутри агломератов расставляет круги первичных частиц (метод v8, пакет circ8:
     дуги контура, край частиц на снимке, модель цепочек; подобран по ручной разметке).
  6. Считает массы: модифицированных (как сфер), тел неправильной формы (НФ — эллипсоид с
     плотной упаковкой шаров) и агломератов — плоско (только видимые частицы) и объёмно
     (с учётом частиц, скрытых в глубине, формула Кёйлю).

Каждую функцию можно вызывать отдельно; окно программы (app.py) просто вызывает их по очереди.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation, feature
from skimage.morphology import skeletonize

import time

from packing import pack, pack_components
import circ8
import classify2

try:
    import tifffile
except ImportError:          # программа будет работать и без tifffile, но масштаб из TIFF не прочитает
    tifffile = None

# ---------------------------------------------------------------------------
# Плотности чистых металлов, г/см³ (справочные значения, CRC Handbook)
# ---------------------------------------------------------------------------
MATERIALS = {
    "W": 19.25, "Au": 19.32, "Ag": 10.49, "Pt": 21.45, "Ti": 4.51, "Al": 2.70,
    "Cu": 8.96, "Ge": 5.32, "Ni": 8.91, "Fe": 7.87, "Cr": 7.19, "Zn": 7.14,
}

# Параметры по умолчанию. Окно программы меняет часть из них.
DEFAULTS = dict(
    contrast=0.5,        # отсечка контраста: контур = эта доля контраста одиночной первичной частицы
    denoise=1.0,         # сила шумоподавления (1 = как в нашем анализе; больше = сильнее сглаживание)
    d_cut=25.0,          # отсечка "первичные / модифицированные", нм
    density=19.25,       # плотность материала, г/см³
    nm_per_px=None,      # масштаб вручную (если в файле нет данных), нм/пиксель
    bar_nm=None,         # или: длина масштабной линейки на снимке, нм (программа найдёт её длину в пикселях)
    include_cut=True,    # учитывать объекты, обрезанные краем кадра (их видимую часть)
    prim_dmin=4.0,       # круги первичных мельче этого диаметра отбрасываются, нм
    prim_dmax=None,      # и не крупнее этого (None = до отсечки d_cut: первичная не больше модифицированной)
    mode="full",         # "full" — всё; "no_mod" — не искать модифицированные; "only_mod" — только модифицированные
    nf_packing=0.74,     # доля заполнения НФ шарами (плотная упаковка шаров одного размера)
    bar_px=None,         # длина линейки в пикселях, заданная вручную (красная линия на снимке)
    edge=0.25,           # край крупной частицы: 25 % от (максимум - подложка); подобрано на модельных сферах
    extend=1.0,          # 1 — крупные частицы забирают узкие полоски вещества вдоль своего края, если форма от этого
                         # не портится (extend_objects): бледный край крупной частицы — часть частицы, а не агломерат
    halo_nm=3.0,         # полоса размытия вокруг модифицированной частицы, не считается агломератом
    film=1.0,            # 1 — убирать из контура тёмные участки подложки (ступенька толщины плёнки), 0 — нет
    k_hyst_lo=2.2,       # пороги поиска вещества относительно шума фона
    k_hyst_hi=4.0,
    ka=1.15, alpha=1.09, # объёмная оценка числа первичных: N = ka * (S / Ap)^alpha (Köylü et al., 1995)
    # Правила отбора версий до 4.1 (больше не используются; оставлены для чтения старых файлов настроек).
    # С версии 4.1 крупные тела классифицирует модуль classify2 (форма + светлые линии + однородность).
    sol_hi=0.91, sol_lo=0.88, circ_min=0.81, k_round=4.5, k_dense=5.75,
)


# ---------------------------------------------------------------------------
# 1. Чтение снимка и масштаб
# ---------------------------------------------------------------------------
def read_image(path):
    """Возвращает (изображение в оттенках серого как float-массив, масштаб_из_файла_или_None)."""
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.array(img).astype(float)                 # 16-бит и float: растягиваем в 0..255
        lo, hi = np.percentile(a, [0.1, 99.9])
        gray = np.clip((a - lo) / max(hi - lo, 1e-9) * 255, 0, 255)
    else:
        gray = np.array(img.convert("L")).astype(float)
    s = metadata_scale(path)
    return gray, s


def metadata_scale(path):
    """Масштаб из метаданных файла (снимки Olympus/SIS в TIFF), нм/пиксель; None — если его там нет."""
    s = None
    if tifffile is not None and path.lower().endswith((".tif", ".tiff")):
        try:
            with tifffile.TiffFile(path) as t:
                if t.is_sis and t.sis_metadata and t.sis_metadata.get("pixelsizex"):
                    s = float(t.sis_metadata["pixelsizex"]) * 1e9     # метры -> нанометры
        except Exception:
            s = None
    return s


def _box_grow(gray, box, max_px):
    """
    Дорисовывает к прямоугольнику бокса линейки то, что ещё относится к его оформлению: размытые или
    сглаженные края рамки, светлую окантовку, белое поле. Каждая сторона сдвигается наружу, пока соседняя
    строка (столбец) в основном заметно отличается от фона снимка рядом с боксом, но не больше чем на
    max_px пикселей — частицы, касающиеся рамки, так не «съедаются».
    """
    H, W = gray.shape
    y0, y1, x0, x1 = box
    ref = gray[max(0, y0 - 40):max(0, y0 - 8), x0:x1]                     # фон над боксом
    if ref.size < 200:
        ref = gray[max(0, y0 - 40):y1, max(0, x0 - 40):max(0, x0 - 8)]    # или слева от него
    if ref.size < 200:
        return box
    med = float(np.median(ref))
    thr = max(12.0, 4.0 * 1.4826 * float(np.median(np.abs(ref - med))))

    def odd(line):
        return line.size > 0 and np.mean(np.abs(line.astype(np.float32) - med) > thr) > 0.5

    for _ in range(max_px):                                               # вверх
        if y0 <= 0 or not odd(gray[y0 - 1, x0:x1]):
            break
        y0 -= 1
    for _ in range(max_px):                                               # влево
        if x0 <= 0 or not odd(gray[y0:y1, x0 - 1]):
            break
        x0 -= 1
    for _ in range(max_px):                                               # вниз
        if y1 >= H or not odd(gray[y1, x0:x1]):
            break
        y1 += 1
    for _ in range(max_px):                                               # вправо
        if x1 >= W or not odd(gray[y0:y1, x1]):
            break
        x1 += 1
    return y0, y1, x0, x1


def _box_finish(gray, box):
    """Окончательный прямоугольник исключения: оформление бокса + 2 пикс запаса; узкая полоска
    (до 6 пикс) между боксом и краем кадра тоже исключается — это такой же край."""
    H, W = gray.shape
    y0, y1, x0, x1 = _box_grow(gray, box, max(8, int(round(0.008 * max(H, W)))))
    y0, x0 = max(0, y0 - 2), max(0, x0 - 2)
    y1, x1 = min(H, y1 + 2), min(W, x1 + 2)
    if H - y1 <= 6:
        y1 = H
    if W - x1 <= 6:
        x1 = W
    return int(y0), int(y1), int(x0), int(x1)


def _scalebar_by_fill(gray):
    """
    Запасной поиск бокса, когда рамка не чёрная (серая, сглаженная при уменьшении снимка, пересохранение):
    в правой нижней части кадра ищется белое поле бокса (≥ 240) с длинной тёмной полосой линейки внутри.
    """
    H, W = gray.shape
    ys = max(0, H - max(90, int(0.1 * H))); xs = int(W * 0.45)
    sub = gray[ys:, xs:]
    lab, n = ndi.label(sub >= 240)
    if n == 0:
        return None
    best = None
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if w < 0.08 * W or h < 6:
            continue
        if best is None or w * h > best[1]:
            best = (sl, w * h)
    if best is None:
        return None
    sl = best[0]
    by0, by1, bx0, bx1 = sl[0].start + ys, sl[0].stop + ys, sl[1].start + xs, sl[1].stop + xs
    inner = gray[by0:by1, bx0:bx1]
    level = 0.5 * (255.0 + float(np.median(gray[max(0, by0 - 40):max(0, by0 - 8), bx0:bx1])
                                 if by0 > 40 else 128.0))
    dark = inner < min(level, 160.0)
    wd = dark.shape[1]
    groups = []                     # отрезки почти одинаковых концов (±2 пикс) в разных строках — края полосы линейки
    for r in range(dark.shape[0]):
        d = np.diff(np.r_[0, dark[r].astype(np.int8), 0])
        for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            if b - a <= 0.3 * wd:
                continue
            for gr in groups:
                if abs(gr["a"] - a) <= 2 and abs(gr["b"] - b) <= 2:
                    gr["rows"].append(r); gr["len"].append(b - a)
                    break
            else:
                groups.append(dict(a=int(a), b=int(b), rows=[r], len=[b - a]))
    cand = [gr for gr in groups if len(gr["rows"]) >= 2]
    if not cand:
        return None
    gr = max(cand, key=lambda t: np.median(t["len"]))
    a = gr["a"]; b = a + int(round(float(np.median(gr["len"])))); rows = gr["rows"]
    box = _box_finish(gray, (by0, by1, bx0, bx1))
    return dict(length_px=int(b - a), y=float(by0 + np.median(rows)), x0=float(a + bx0), x1=float(b + bx0), box=box)


def find_scalebar(gray):
    """
    Ищет масштабную линейку (бокс Olympus/SIS в правом нижнем углу): в нижней полосе снимка находит
    тёмные горизонтальные отрезки; самый длинный — рамка бокса, а линейка — самый длинный отрезок
    внутри рамки, встречающийся хотя бы в двух строках. Возвращает
    {length_px, y, x0, x1, box=(y0, y1, x0, x1)} или None.
    box — прямоугольник, исключаемый из анализа: бокс целиком вместе с размытыми краями рамки
    (_box_grow), без широкой «мёртвой зоны» вокруг. Если чёрной рамки нет (серая, сглаженная),
    бокс ищется по белому полю (_scalebar_by_fill).
    """
    H, W = gray.shape
    y0 = max(0, H - max(60, int(0.06 * H))); x0 = int(W * 0.45)
    dark = gray[y0:, x0:] < 60
    runs = {}
    for r in range(dark.shape[0]):
        d = np.diff(np.r_[0, dark[r].astype(np.int8), 0])
        st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        for a, b in zip(st, en):
            if b - a > 0.05 * W:
                runs.setdefault((int(a), int(b)), []).append(r)
    if not runs:
        return _scalebar_by_fill(gray)
    fa, fb = max(runs, key=lambda k: k[1] - k[0])                 # рамка бокса
    inner = [(k, rows) for k, rows in runs.items() if len(rows) >= 2 and k[0] > fa + 1 and k[1] < fb - 1]
    if inner:
        (a, b), rows = max(inner, key=lambda t: t[0][1] - t[0][0])
    else:
        cand = [(k, rows) for k, rows in runs.items() if len(rows) >= 2] or list(runs.items())
        (a, b), rows = max(cand, key=lambda t: t[0][1] - t[0][0])
    y = y0 + int(np.median(rows))
    # рамка бокса: строки с отрезками почти той же длины, что и самый длинный (верхняя и нижняя стороны)
    fr = sorted(r for k, rr in runs.items() if abs(k[0] - fa) <= 3 and abs(k[1] - fb) <= 3 for r in rr)
    top = y0 + (fr[0] if fr else min(rows))
    bot = y0 + fr[-1] + 1 if fr and fr[-1] - fr[0] > 8 else H
    box = _box_finish(gray, (top, bot, fa + x0, fb + x0 + 1))
    return dict(length_px=int(b - a), y=float(y), x0=float(a + x0), x1=float(b + x0), box=box)


def get_scale(gray, s_file, params):
    """Определяет масштаб: введённый вручную -> линейка + её длина -> метаданные файла.
    Возвращает (нм/пиксель, источник)."""
    if params.get("nm_per_px"):
        return float(params["nm_per_px"]), "введён вручную"
    if params.get("bar_nm"):
        if params.get("bar_px"):
            L = float(params["bar_px"])
            return params["bar_nm"] / L, "по линейке ({:.0f} пикс, задана вручную)".format(L)
        sb = find_scalebar(gray)
        if sb:
            return params["bar_nm"] / sb["length_px"], "по линейке ({} пикс)".format(sb["length_px"])
        if not s_file:
            raise ValueError("Линейка на снимке не найдена. Введите масштаб в нм/пиксель.")
    if s_file:
        return s_file, "из метаданных"
    raise ValueError("Не удалось определить масштаб: в файле нет данных. Укажите нм/пиксель или длину линейки.")


def valid_area(gray):
    """Маска области анализа: всё, кроме масштабной линейки и подписи к ней."""
    v = np.ones(gray.shape, bool)
    sb = find_scalebar(gray)
    if sb:
        y0, y1, x0, x1 = sb["box"]
        v[y0:y1, x0:x1] = False
    return v


# ---------------------------------------------------------------------------
# 2. Фон, шум и контур всего вещества
# ---------------------------------------------------------------------------
def _block_mean(a, f):
    H, W = a.shape; h, w = H // f, W // f
    return a[:h * f, :w * f].reshape(h, f, w, f).mean((1, 3))


def background(g, valid, S_px=70, f=4, k=4.0, iters=4):
    """
    Плавный фон под снимком. Считается на уменьшенной в f раз копии (так в ~16 раз быстрее).
    Начальная маска вещества — всё, что заметно темнее моды гистограммы (светлая плёнка), поэтому
    крупные и бледные частицы не «впитываются» в фон. Под крупными частицами фон дозаполняется
    более широким сглаживанием.
    """
    H, W = g.shape
    small = _block_mean(g, f); vs = _block_mean(valid.astype(float), f) > 0.99
    v = small[vs]
    hist, edges = np.histogram(v, bins=256)
    hist = ndi.gaussian_filter1d(hist.astype(float), 2)
    mode = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    up = v[v >= mode] - mode
    noise = (np.median(up) / 0.6745 if up.size else np.std(v)) + 1e-6
    mat = ((mode - small) / noise > k) & vs
    Ss = S_px / f
    bg = np.full(small.shape, mode)
    for _ in range(iters):
        mat = morphology.remove_small_objects(mat, 3)
        w = (vs & ~ndi.binary_dilation(mat, iterations=2)).astype(float)
        num = ndi.gaussian_filter(small * w, Ss); den = ndi.gaussian_filter(w, Ss)
        num2 = ndi.gaussian_filter(small * w, 4 * Ss); den2 = ndi.gaussian_filter(w, 4 * Ss)
        bg = np.where(den > 0.05, num / np.maximum(den, 1e-6), num2 / np.maximum(den2, 1e-6))
        r = (small - bg)[w > 0]
        if r.size < 10:
            break
        bg = bg + np.median(r)
        noise = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        mat = ((bg - small) / noise > k) & vs
    big = ndi.zoom(bg, (H / bg.shape[0], W / bg.shape[1]), order=1)
    out = np.empty_like(g, dtype=float); hh, ww = min(H, big.shape[0]), min(W, big.shape[1])
    out[:hh, :ww] = big[:hh, :ww]
    if hh < H: out[hh:, :] = out[hh - 1:hh, :]
    if ww < W: out[:, ww:] = out[:, ww - 1:ww]
    return out


def film_mask(D, M, s, valid, se_nm=15.0, k_th=4.0, k_d=0.6, amin_nm2=2000.0, wmin_nm=10.0):
    """
    Тёмные участки подложки внутри контура (ступенька толщины плёнки, край сетки): фон сглаживается
    плавно и такую ступеньку не повторяет, поэтому более тёмная часть плёнки попадает в контур.
    Признаки (проверено на 43 снимках Al с ручной разметкой и 84 снимках W):
      рельеф масштаба частицы слабый — морфологический top-hat затемнения (диск se_nm) < k_th шумов;
      затемнение ниже k_d от медианного затемнения частиц этого снимка;
      связный участок площадью не меньше amin_nm2 и шириной не меньше 2·wmin_nm (узкая бледная полоса —
      это край крупной частицы, а не ступенька плёнки).
    """
    Ds = ndi.gaussian_filter(D, 1.0 / s)
    bgp = Ds[~ndi.binary_dilation(M, iterations=5) & valid]
    if bgp.size < 100:
        return np.zeros_like(M)
    noise = 1.4826 * np.median(np.abs(bgp - np.median(bgp))) + 1e-6
    f = 2
    small = Ds[::f, ::f]
    op = ndi.zoom(ndi.grey_opening(small, footprint=morphology.disk(max(2, int(round(se_nm / s / f))))), f, order=1)
    op = np.pad(op, ((0, max(0, D.shape[0] - op.shape[0])), (0, max(0, D.shape[1] - op.shape[1]))), mode="edge")
    th = (Ds - op[:D.shape[0], :D.shape[1]]) / noise
    hi = M & (th >= 2 * k_th)
    if hi.sum() < 50:
        return np.zeros_like(M)
    F = M & (th < k_th) & (Ds < k_d * np.median(Ds[hi]))
    F = ndi.binary_opening(F, iterations=max(1, int(round(1.0 / s))))
    lab = measure.label(F)
    if lab.max() == 0:
        return np.zeros_like(M)
    area = np.bincount(lab.ravel()); keep = area * s * s >= amin_nm2; keep[0] = False
    # ступенька плёнки — широкий участок; узкая бледная полоса (край крупной частицы) — не плёнка
    Fc = ndi.binary_fill_holes(ndi.binary_closing(lab > 0, structure=morphology.disk(max(1, int(round(3 / s))))))
    edt = ndi.distance_transform_edt(Fc) * s                      # ширина — без мелких дыр от шума
    thick = np.zeros(lab.max() + 1, bool)
    if lab.max():
        thick[1:] = ndi.maximum(edt, lab, index=np.arange(1, lab.max() + 1)) >= wmin_nm
    keep &= thick
    F = keep[lab]
    return ndi.binary_dilation(F, iterations=max(1, int(round(1.5 / s)))) & M


def extend_objects(res, D32, max_nm=15.0, tol=0.01, h_noise=4.0):
    """
    Вещество вдоль края крупной частицы (модифицированной или НФ), не вошедшее в неё: край частицы
    проводится на 25 % её затемнения над окружением (refine), и бледный край крупной частицы
    (убывающая толщина шара, оболочка) остаётся снаружи полоской. Такая полоска — часть частицы,
    а прилипшие к частице мелкие частицы — нет. Различаются они так:
      - затемнение края частицы от неё наружу только убывает, а у прилипшей частицы есть своя вершина
        затемнения за «долиной» (проверка — морфологическая реконструкция от места касания: вершина
        выше достижимого от касания уровня больше чем на h_noise шумов подложки — кусок не берётся);
      - полоска вдоль края не портит выпуклость частицы (solidity падает не больше чем на tol).
    Каждой частице предлагаются соседние куски вещества не дальше max_nm (ближайшие к ней, а не к другой).
    """
    s = res["s"]; om = res["objmap"]; H, W = om.shape
    objs = [o for o in res["objects"] if o["cls"] in ("mod", "irr")]
    if not objs:
        return
    left = res["m_all"] & (om == 0) & res["valid"]
    if not left.any():
        return
    dist = ndi.distance_transform_edt(om == 0) * s
    lab = segmentation.watershed(dist, om, mask=(om > 0) | (left & (dist <= max_nm)))
    pad = int(max_nm / s) + 3
    Dsf = ndi.gaussian_filter(D32.astype(np.float32), max(0.7, 0.4 / s))
    h = h_noise * bg_stats(D32, res["m_all"], res["valid"], s)[1]

    def solidity(m):
        p = measure.regionprops(m.astype(np.uint8))
        return p[0].solidity if p else 0.0
    for o in objs:
        ys, xs = np.nonzero(lab == o["id"])
        if ys.size == 0:
            continue
        a, b = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
        c, d = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
        core = om[a:b, c:d] == o["id"]
        cand = (lab[a:b, c:d] == o["id"]) & ~core & (om[a:b, c:d] == 0)
        if not cand.any():
            continue
        pl, n = ndi.label(cand)
        sizes = np.bincount(pl.ravel())
        cur = core.copy(); sc = solidity(cur); changed = False
        Dc = Dsf[a:b, c:d]; lo = float(Dc.min()) - 1.0
        ring = ndi.binary_dilation(core, iterations=1) & ~core
        for k in np.argsort(-sizes[1:]) + 1:
            piece = pl == k
            contact = piece & ring
            if not contact.any():
                continue
            # своя вершина затемнения за «долиной» — это прилипшая частица, а не край
            marker = np.where(contact, Dc, lo); maskr = np.where(piece, Dc, lo)
            rec = morphology.reconstruction(np.minimum(marker, maskr), maskr, method="dilation")
            if (Dc - rec)[piece].max() > h:
                continue
            trial = cur | piece
            st = solidity(trial)
            if st >= sc - tol:
                cur, sc, changed = trial, max(sc, st) if st > sc else sc, True
        if not changed:
            continue
        full = np.zeros((H, W), bool); full[a:b, c:d] = cur
        f = particle_features(D32, full, s, res["lvl1"])
        f["cut"] = bool(full[:2].any() or full[-2:].any() or full[:, :2].any() or full[:, -2:].any() or
                        (ndi.binary_dilation(full, iterations=2) & ~res["valid"]).any())
        yy, xx = np.nonzero(full)
        f["pix"] = (yy.astype(np.int32), xx.astype(np.int32))
        o.update(f)
        om[full & (om == 0)] = o["id"]


def _fill_invalid(gray, valid):
    """
    Копия снимка, в которой исключённая область (бокс линейки) залита уровнем фона вокруг неё.
    Иначе сглаживание размазывает чёрную рамку, линейку и надписи бокса на соседние пиксели снимка:
    вдоль рамки появляется ложный контур, а в нём — попытки искать частицы. Чем сильнее
    шумоподавление и крупнее пиксель, тем шире эта полоса.
    """
    if valid.all():
        return gray
    ring = ndi.binary_dilation(~valid, iterations=25) & valid
    vals = gray[ring] if ring.sum() >= 50 else gray[valid]
    out = np.array(gray, dtype=float, copy=True)
    out[~valid] = float(np.median(vals)) if vals.size else float(np.median(gray))
    return out


def segment(gray, s, params, valid, info=None):
    """
    Находит всё вещество на снимке.
    Возвращает: m_all (контур по полувысоте), D (затемнение = фон - яркость), lvl1 (контраст
    одиночной первичной частицы), m0 (грубая маска).
    """
    k = params["denoise"]
    gray = _fill_invalid(gray, valid)                                  # бокс линейки не размазывается на снимок
    g = ndi.gaussian_filter(gray, k * max(0.8, 1.2 * s) / s)          # сглаживание шума
    bg = background(g, valid)
    r = (g - bg)[valid]
    up = r[r > 0]                                   # шум — по светлой половине отклонений (только плёнка)
    noise = (np.median(up) / 0.6745 if up.size > 100 else 1.4826 * np.median(np.abs(r - np.median(r)))) + 1e-6
    d = (bg - g) / noise
    m0 = filters.apply_hysteresis_threshold(d, params["k_hyst_lo"], params["k_hyst_hi"]) & valid
    m0 = morphology.remove_small_holes(m0, int((3 / s) ** 2))
    m0 = morphology.remove_small_objects(m0, int(40 / s ** 2))
    # затемнение для контуров (сглаживание помягче)
    D = bg - ndi.gaussian_filter(gray, k * max(0.5 / s, 1.0))
    e0 = ndi.distance_transform_edt(m0) * s
    sk = skeletonize(m0)
    thin = sk & (e0 > 2) & (e0 < 5)                                   # оси тонких одиночных цепочек
    lvl1 = np.median(D[thin]) if thin.sum() > 50 else (np.median(D[m0]) if m0.any() else 1.0)
    if info is not None:
        info["thin_frac"] = float(thin.sum()) / max(int(sk.sum()), 1)
    # Контур — единое правило для тонких цепочек и крупных частиц. Пиксель — вещество, если его затемнение
    # выше доли contrast (по умолчанию половины):
    #   а) от затемнения одиночной первичной частицы lvl1 (прежнее правило, подобрано по ручной разметке) ИЛИ
    #   б) от местного перепада «подложка — вещество»: наибольшего затемнения в окрестности 3 нм,
    #      если этот кусок вещества связан с найденным по (а).
    # (б) нужно крупным частицам: их край бледнее середины (убывающая толщина, оболочка), а середина
    # бывает темнее lvl1 из-за дифракционного контраста — по одному (а) край уходил внутрь. Для тонких
    # цепочек (б) почти совпадает с (а): местный перепад цепочки и есть lvl1.
    c = params["contrast"]
    Ds = ndi.gaussian_filter(D.astype(np.float32), max(0.7, 0.4 / s))
    bg0, bgn = bg_stats(D, ndi.binary_fill_holes(m0), valid, s)
    rl = max(2, int(round(3.0 / s)))
    L = ndi.grey_dilation(np.where(m0, Ds, bg0), footprint=morphology.disk(rl))
    mg = (D > c * lvl1) & ndi.binary_dilation(m0, iterations=2) & valid
    # местный перепад учитывается, только если он заметно больше шума подложки (8 шумов): пятна шума
    # и неровности подложки дают 3–5 шумов, настоящее вещество — от 10
    ml = (Ds > bg0 + c * (L - bg0)) & (L - bg0 >= 8.0 * bgn) & ndi.binary_dilation(m0, iterations=rl + 2) & valid
    # (б) только продолжает вещество, найденное по (а): отдельные бледные пятна (шум, загрязнения)
    # по одному местному перепаду в контур не попадают
    lab_, _ = ndi.label(mg | ml)
    keep_ = np.unique(lab_[mg]); m = np.isin(lab_, keep_[keep_ > 0])
    m = morphology.remove_small_holes(m, int((2 / s) ** 2))
    m = morphology.remove_small_objects(m, max(3, int(10 / s ** 2)))
    lab = measure.label(m); keep = np.unique(lab[m0 & m]); m = np.isin(lab, keep[keep > 0])
    if params.get("film", 1.0):
        F = film_mask(D, m, s, valid)
        if F.any():
            m &= ~F
            m = morphology.remove_small_objects(m, max(3, int(10 / s ** 2)))
    return m, D, float(max(lvl1, 1e-6)), m0


# ---------------------------------------------------------------------------
# 3. Модифицированные частицы
# ---------------------------------------------------------------------------
def _bopen(m, r):
    """Морфологическое "открытие" диском радиуса r: убирает всё, что тоньше 2r."""
    er = ndi.distance_transform_edt(m) > r
    return ndi.distance_transform_edt(~er) <= r


def find_candidates(D, m_all, valid, s, lvl1, params):
    """Кандидаты в крупные частицы: места, куда помещается диск, + разделение слипшихся по перешейку."""
    H, W = D.shape
    dmod = params["d_cut"]
    ro = max(2, int(round(0.36 * dmod / s)))           # радиус "открытия" (9 нм при отсечке 25 нм)
    core = _bopen(m_all, ro)
    lab = measure.label(core)
    part = np.zeros((H, W), bool)
    for p in measure.regionprops(lab):
        y0, x0, y1, x1 = p.bbox; pad = int(15 / s) + ro
        a, b, c, d = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
        comp = lab[a:b, c:d] == p.label
        level = np.percentile(D[a:b, c:d][comp], 75)
        mm = (D[a:b, c:d] > max(0.5 * level, 0.5 * lvl1)) & m_all[a:b, c:d]
        mm = _bopen(ndi.binary_fill_holes(mm), ro)
        l2 = measure.label(mm); keep = np.unique(l2[comp & mm]); mm = np.isin(l2, keep[keep > 0])
        part[a:b, c:d] |= mm
    part = ndi.binary_fill_holes(part) & m_all
    edm = ndi.distance_transform_edt(part)
    pk = feature.peak_local_max(edm, min_distance=max(3, int(6 / s)), threshold_abs=0.2 * dmod / s,
                                labels=measure.label(part), exclude_border=False)
    pk = sorted([tuple(q) for q in pk], key=lambda q: -edm[q]); keep = []

    def neck(q, k):
        L = int(np.hypot(q[0] - k[0], q[1] - k[1])) + 2
        yy = np.linspace(q[0], k[0], L).round().astype(int); xx = np.linspace(q[1], k[1], L).round().astype(int)
        return edm[yy, xx].min()
    for q in pk:
        R = edm[q]; ok = True
        for k in keep:
            if np.hypot(q[0] - k[0], q[1] - k[1]) < 0.5 * max(R, edm[k]) or neck(q, k) > 0.85 * min(R, edm[k]):
                ok = False; break
        if ok: keep.append(q)
    mk = np.zeros((H, W), np.int32)
    for i, q in enumerate(keep, 1): mk[q] = i
    return segmentation.watershed(-edm, mk, mask=part)


def bg_stats(D, matter, valid, s):
    """Уровень и шум затемнения подложки (слегка сглаженного) вдали от вещества."""
    Ds = ndi.gaussian_filter(D.astype(np.float32), max(0.7, 0.4 / s))
    far = ~ndi.binary_dilation(matter, iterations=max(3, int(round(5 / s)))) & valid
    v = Ds[far] if far.sum() >= 500 else Ds[valid]
    bg0 = float(np.median(v))
    return bg0, 1.4826 * float(np.median(np.abs(v - bg0))) + 1e-3


def refine(D, R0, others, valid, s, edge):
    """Уточнённый контур частицы: край на уровне edge (25 %) между подложкой и максимумом затемнения."""
    k = max(2, int(round(4 / s)))
    Rd = ndi.binary_dilation(R0, iterations=k) & ~others & valid
    ring = ndi.binary_dilation(R0, iterations=k + max(2, int(3 / s))) & ~ndi.binary_dilation(R0, iterations=k) & ~others
    peak = np.percentile(D[R0], 98)
    ped = max(0.0, float(np.median(D[ring]))) if ring.sum() else 0.0
    m = (D > ped + edge * (peak - ped)) & Rd
    lab = measure.label(m)
    if lab.max() == 0: return R0, ped, peak
    iy, ix = np.unravel_index(np.argmax(np.where(R0, D, -1e9)), D.shape)
    keepl = lab[iy, ix] if lab[iy, ix] else np.bincount(lab[R0 & m]).argmax()
    m = _bopen(ndi.binary_fill_holes(lab == keepl), max(1, int(round(3 / s))))
    l2 = measure.label(m)
    if l2.max() == 0: return R0, ped, peak
    return l2 == np.bincount(l2[l2 > 0]).argmax(), ped, peak


def particle_features(D, m, s, lvl1, hp=None):
    rp = measure.regionprops(m.astype(int))[0]
    return dict(deq=rp.equivalent_diameter_area * s, solidity=float(rp.solidity),
                contrast=float(np.median(D[m]) / lvl1),
                circularity=float(4 * np.pi * rp.area / max(rp.perimeter, 1) ** 2),
                feret=rp.feret_diameter_max * s, minor=rp.minor_axis_length * s,
                cy=float(rp.centroid[0]), cx=float(rp.centroid[1]), area_px=int(rp.area))


def ellipse_axes(ys, xs, s):
    """Полуоси эллипса (нм) с теми же моментами, что у проекции, подогнанного по площади проекции: a >= b."""
    A = float(len(ys))
    if A < 3:
        r = np.sqrt(max(A, 1.0) / np.pi) * s
        return r, r
    cov = np.cov(np.vstack([ys - ys.mean(), xs - xs.mean()]))
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    a, b = 2 * np.sqrt(max(ev[0], 1e-9)), 2 * np.sqrt(max(ev[1], 1e-9))
    k = np.sqrt(A / (np.pi * a * b))
    return a * k * s, b * k * s


def sphere_mass_in_field(m, s, cut, valid, deq, density, obj=None):
    """
    Масса модифицированной частицы, г.
    Целая — СПЛОШНОЙ эллипсоид ближайшей формы: полуоси a ≥ b — по эллипсу с теми же моментами и площадью,
    что у проекции, третья полуось c = b. Для круглой проекции это шар диаметром deq.
    Обрезанная краем — видимая часть шара, подогнанного по необрезанной части края.
    """
    rho = density * 1e-21                                   # г/см³ -> г/нм³
    if not cut:
        ys, xs = np.nonzero(m)
        a, b = ellipse_axes(ys, xs, s)
        if obj is not None:
            obj["axes_nm"] = (2 * a, 2 * b, 2 * b)
        return rho * 4.0 / 3.0 * np.pi * a * b * b
    e = m & ~ndi.binary_erosion(m)
    frame = np.zeros_like(m); frame[:2, :] = frame[-2:, :] = frame[:, :2] = frame[:, -2:] = True
    frame |= ndi.binary_dilation(~valid, iterations=2)
    ys, xs = np.nonzero(e & ~frame)
    if len(ys) < 10:
        return rho * np.pi * deq ** 3 / 6
    A = np.c_[2 * xs, 2 * ys, np.ones_like(xs)]; b = xs.astype(float) ** 2 + ys.astype(float) ** 2
    c, *_ = np.linalg.lstsq(A.astype(float), b, rcond=None); x0, y0 = c[0], c[1]; R = np.sqrt(c[2] + x0 ** 2 + y0 ** 2)
    yy, xx = np.nonzero(m)
    t = 2 * np.sqrt(np.clip(R * R - ((yy - y0) ** 2 + (xx - x0) ** 2), 0, None)) * s
    return rho * t.sum() * s * s


# ---------------------------------------------------------------------------
# 4-5. Агломераты и первичные частицы
# ---------------------------------------------------------------------------
def agglomerate_mask(m_all, modmask, irrmask, s, halo_nm):
    halo = max(1, int(round(halo_nm / s)))
    agg = m_all & ~ndi.binary_dilation(modmask | irrmask, iterations=halo)
    agg = _bopen(agg, max(1, int(round(1.5 / s))))
    return morphology.remove_small_objects(agg, max(2, int(30 / s ** 2)))


def agglomerate_components(agg, circles, s, valid):
    lab = measure.label(agg)
    H, W = agg.shape
    comps = []
    byc = {}
    for c in circles:
        byc.setdefault(c[3], []).append(c[2])
    for p in measure.regionprops(lab):
        y0, x0, y1, x1 = p.bbox
        cut = y0 <= 1 or x0 <= 1 or y1 >= H - 1 or x1 >= W - 1 or \
              bool((ndi.binary_dilation(lab[y0:y1, x0:x1] == p.label, iterations=2) & ~valid[y0:y1, x0:x1]).any())
        comps.append(dict(id=int(p.label), area_nm2=float(p.area * s * s), cut=bool(cut),
                          d_circles=list(byc.get(p.label, []))))
    return comps, lab


# ---------------------------------------------------------------------------
# Главная функция: обработка одного снимка
# ---------------------------------------------------------------------------
def analyze_image(path, params, progress=None):
    """
    Обрабатывает один снимок. Возвращает словарь с результатами (без масс — массы считает
    compute_masses, чтобы плотность можно было менять без повторной обработки).
    """
    P = dict(DEFAULTS); P.update(params or {})
    t0 = time.time()
    def step(msg):
        if progress: progress(msg)
    gray, s_file = read_image(path)
    s, s_src = get_scale(gray, s_file, P)
    valid = valid_area(gray)
    step("контур вещества")
    info = {}
    m_all, D, lvl1, m0 = segment(gray, s, P, valid, info)
    step("крупные частицы")
    if P.get("mode") == "no_mod":
        ws = np.zeros(m_all.shape, np.int32)          # модифицированные не ищем
    else:
        ws = find_candidates(D, m_all, valid, s, lvl1, P)
    res = dict(path=path, name=os.path.basename(path), s=s, s_src=s_src, shape=gray.shape,
               lvl1=lvl1, m_all=m_all, valid=valid, objects=[], objmap=np.zeros(ws.shape, np.int32), params=P,
               D=D.astype(np.float16),                # затемнение — для пересчёта кругов после правок
               thin_frac=info.get("thin_frac", 0.0))  # доля тонких цепочек среди осевых линий вещества (для справки)
    D32 = res["D"].astype(np.float32)
    ids = [int(i) for i in np.unique(ws) if i > 0]
    allsel = ws > 0
    for oid in ids:
        R0 = ws == oid
        if R0.sum() * s * s < np.pi * (P["d_cut"] / 2) ** 2:
            continue                                  # кандидат мельче отсечки ещё до уточнения контура
        m, ped, peak = refine(D32, R0, allsel & ~R0, valid, s, P["edge"])
        m &= res["objmap"] == 0
        if m.sum() * s * s < np.pi * (P["d_cut"] / 2) ** 2:
            continue
        add_body_parts(res, m, D32, from_circle=False)
    if P.get("extend", 1.0):
        extend_objects(res, D32)
    step("агломераты и первичные частицы")
    update_agglomerates(res)
    res["t_proc"] = time.time() - t0
    return res


def masks_from_objects(res):
    H, W = res["shape"]
    mod = np.zeros((H, W), bool); irr = np.zeros((H, W), bool)
    for o in res["objects"]:
        if o["cls"] == "mod": mod[o["pix"]] = True
        elif o["cls"] == "irr": irr[o["pix"]] = True
    return mod, irr


def deleted_mask(res):
    """Пиксели объектов, удалённых вручную (не входят ни в частицы, ни в агломераты)."""
    H, W = res["shape"]
    m = np.zeros((H, W), bool)
    for o in res["objects"]:
        if o["cls"] == "del": m[o["pix"]] = True
    return m


def disk_mask(shape, y, x, r):
    H, W = shape
    m = np.zeros((H, W), bool)
    a, b = int(max(0, np.floor(y - r))), int(min(H, np.ceil(y + r) + 1))
    c, d = int(max(0, np.floor(x - r))), int(min(W, np.ceil(x + r) + 1))
    if a < b and c < d:
        yy, xx = np.ogrid[a:b, c:d]
        m[a:b, c:d] = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return m


def circles_mask(shape, circles, s):
    """Растр (маска) всех кругов списка."""
    H, W = shape
    m = np.zeros((H, W), bool)
    for c in circles:
        y, x, r = c[0], c[1], c[2] / 2 / s
        a, b = int(max(0, np.floor(y - r))), int(min(H, np.ceil(y + r) + 1))
        cc, d = int(max(0, np.floor(x - r))), int(min(W, np.ceil(x + r) + 1))
        if a < b and cc < d:
            yy, xx = np.ogrid[a:b, cc:d]
            m[a:b, cc:d] |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return m


def _ukey(y, x):
    return (round(float(y), 1), round(float(x), 1))


def _auto_circles(res):
    """Круги, найденные автоматически (без добавленных вручную)."""
    return res["circles"][:res.get("n_auto_circles", len(res["circles"]))]


def build_agg(res):
    """
    Маска агломератов, найденных автоматически, с учётом ручных правок:
    всё вещество минус модифицированные частицы с полосой размытия, минус удалённые объекты и области.
    НФ (неправильной формы) вырезаются так же, как модифицированные: их масса считается отдельно
    (эллипсоид с плотной упаковкой шаров) и прибавляется к массе агломератов. Добавленные вручную
    первичные частицы и области хранятся отдельно (см. _finish_circles) — их площадь прибавляется к агломератам.
    """
    P = res["params"]; s = res["s"]; e = res.get("edit", {})
    mod, irr = masks_from_objects(res)
    agg = agglomerate_mask(res["m_all"], mod | irr, deleted_mask(res), s, P["halo_nm"])
    if e.get("del_mask") is not None:
        agg &= ~e["del_mask"]
    return agg


def place_circles(res, mask, cap=True):
    """
    Круги первичных частиц (метод v8) внутри маски: список (y, x, d_нм).
    cap=True — кандидаты не крупнее отсечки «первичные / модифицированные»: первичная частица
    не может быть больше модифицированной (после подгонки диаметр дополнительно ограничивается).
    """
    P = res["params"]; s = res["s"]
    if P.get("mode") == "only_mod" or not mask.any():
        return []
    D = res["D"].astype(np.float32) if "D" in res else None
    if D is None:
        return []
    dcut = float(P["d_cut"])
    dlim = min(dcut, P["prim_dmax"]) if P.get("prim_dmax") else dcut
    out = []
    for (y, x, R) in circ8.place(mask, D, res["lvl1"], s, valid=res["valid"], frame_shape=res["shape"],
                                 rmax_nm=0.47 * dlim if cap else None):
        d = 2 * R * s
        if d < (P.get("prim_dmin") or 0):
            continue
        if cap and d >= dlim:
            d = 0.99 * dlim                             # подгонка по краю чуть увеличила круг
        out.append((y, x, d))
    return out


def _add_auto_object(res, m, D, cls):
    """Новый объект (модифицированная или НФ), найденный по крупному кругу."""
    s = res["s"]
    f = particle_features(D, m, s, res["lvl1"])
    f["cut"] = bool(m[:2].any() or m[-2:].any() or m[:, :2].any() or m[:, -2:].any() or
                    (ndi.binary_dilation(m, iterations=2) & ~res["valid"]).any())
    f["cls"] = cls; f["auto_cls"] = cls; f["from_circle"] = True; f["id"] = len(res["objects"]) + 1
    ys, xs = np.nonzero(m)
    f["pix"] = (ys.astype(np.int32), xs.astype(np.int32))
    res["objmap"][m & (res["objmap"] == 0)] = f["id"]
    res["objects"].append(f)
    return f


def decompose_body(res, m, D32):
    """
    Новая классификация (v4.1, модуль classify2): тело делится на части
    (маска на весь кадр, класс mod / nf / agg / prim, причина).
    """
    P = res["params"]; s = res["s"]; H, W = res["shape"]
    ys, xs = np.nonzero(m)
    pad = int(6 / s) + 3
    a, b = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    c, d = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    mc = m[a:b, c:d]
    Dc = D32[a:b, c:d]
    ring = ndi.binary_dilation(mc, iterations=max(2, int(3 / s))) & ~mc
    ped = max(0.0, float(np.median(Dc[ring]))) if ring.any() else 0.0
    Ds = ndi.gaussian_filter(Dc, 1 / s)
    dmin = P.get("prim_dmin") or 0

    def place(q):
        M = np.zeros((H, W), bool); M[a:b, c:d] = q
        out = []
        for (y, x, R) in circ8.place(M, D32, res["lvl1"], s, valid=res["valid"], frame_shape=res["shape"], rmax_nm=None):
            if 2 * R * s >= dmin:
                out.append((y - a, x - c, R))
        return out
    B = classify2.Body(mc, Ds, ped, s, res["lvl1"], res["m_all"][a:b, c:d], place)
    CP = dict(classify2.P0, dcut=float(P["d_cut"]))
    parts = []
    for q, cls, why in classify2.classify(B, CP):
        full = np.zeros((H, W), bool); full[a:b, c:d] = q
        parts.append((full, cls, why))
    return parts


def add_body_parts(res, m, D32, from_circle):
    """
    Классифицирует тело и добавляет его части как объекты:
      mod — модифицированная (в режиме «без модифицированных» — НФ);  nf — НФ;
      agg — объект «часть агломерата» (остаётся в агломерате, класс можно сменить кликом);
      prim — ничего не добавляется: место заполнят круги первичных частиц (меньше отсечки).
    Возвращает число добавленных модифицированных и НФ.
    """
    P = res["params"]; s = res["s"]; dcut = float(P["d_cut"])
    mode = P.get("mode", "full")
    n = 0
    for q, cls, why in decompose_body(res, m, D32):
        q = q & (res["objmap"] == 0)
        if q.sum() < 4:
            continue
        if cls == "prim":
            continue
        k = {"mod": "mod" if mode != "no_mod" else "irr", "nf": "irr", "agg": "agg"}[cls]
        if q.sum() * s * s < np.pi * (dcut / 2) ** 2:
            continue              # часть мельче отсечки — первичные частицы (круги меньше отсечки)
        f = _add_auto_object(res, q, D32, k)
        f["from_circle"] = from_circle; f["why"] = why
        n += k in ("mod", "irr")
    return n


def resolve_big_circles(res, circles):
    """
    Круг не меньше отсечки «первичные / модифицированные» не может быть первичной частицей.
    Для каждого такого круга берётся тело частицы (край на уровне edge между подложкой и максимумом
    затемнения, как у модифицированных) и классифицируется (classify2): модифицированные, НФ,
    скопление (модифицированные + первичные) или агломерат; тело мельче отсечки — круг просто
    заменяется кругами меньше отсечки.
    Остальная часть круга (прилипшие частицы) затем заполняется кругами меньше отсечки.
    Возвращает список центров (y, x) крупных кругов — их агломераты надо пересчитать.
    """
    P = res["params"]; s = res["s"]; dcut = float(P["d_cut"])
    big = sorted([c for c in circles if c[2] >= dcut], key=lambda c: -c[2])
    if not big:
        return []
    D = res["D"].astype(np.float32)
    mod, irr = masks_from_objects(res)
    objmask = mod | irr | deleted_mask(res)
    Mfill = ndi.binary_fill_holes(res["m_all"])
    for (y, x, d) in big:
        R0 = disk_mask(res["shape"], y, x, d / 2 / s) & Mfill & ~objmask
        if R0.sum() * s * s < 0.5 * np.pi * (dcut / 2) ** 2:
            continue
        m, ped, peak = refine(D, R0, objmask, res["valid"], s, P["edge"])
        m = ndi.binary_fill_holes(m) & ~objmask
        if not m.any():
            continue
        lab = measure.label(m)
        m = lab == np.bincount(lab[lab > 0]).argmax()
        f = particle_features(D, m, s, res["lvl1"])
        if f["deq"] < dcut:
            continue                                  # тело мельче отсечки — это первичные частицы
        add_body_parts(res, m, D, from_circle=True)
        objmask |= m
    return [(c[0], c[1]) for c in big]


def update_agglomerates(res, region=None):
    """Пересчитывает маску агломератов и круги (после обработки или после смены класса объекта)."""
    agg = build_agg(res)
    lab = measure.label(agg)
    H, W = agg.shape

    def lab_at(lb, c):
        yi, xi = int(min(max(c[0], 0), H - 1)), int(min(max(c[1], 0), W - 1))
        return lb[yi, xi]
    if region is None or "circles" not in res:
        circles = place_circles(res, agg, cap=False)
        centers = resolve_big_circles(res, circles)
        if centers:
            # агломераты с крупными кругами (и рядом с новыми частицами) пересчитываем кругами меньше отсечки
            agg = build_agg(res); lab = measure.label(agg)
            dcut = res["params"]["d_cut"]
            touched = set()
            for c in circles:
                if c[2] >= dcut:
                    yy, xx = np.nonzero(disk_mask(agg.shape, c[0], c[1], c[2] / 2 / res["s"] + 2))
                    touched |= set(np.unique(lab[yy, xx]).tolist())
            touched.discard(0)
            keep = [c for c in circles if c[2] < dcut and lab_at(lab, c) and lab_at(lab, c) not in touched]
            new = place_circles(res, np.isin(lab, list(touched))) if touched else []
            circles = keep + new
    else:
        # пересчитываем круги только у агломератов рядом с изменённым объектом
        y0, y1, x0, x1 = region
        touched = set(np.unique(lab[y0:y1, x0:x1])) - {0}
        keep = [c for c in _auto_circles(res)
                if not (y0 <= c[0] < y1 and x0 <= c[1] < x1) and lab_at(lab, c) and lab_at(lab, c) not in touched]
        new = place_circles(res, np.isin(lab, list(touched))) if touched else []
        circles = keep + new
    _finish_circles(res, agg, lab, [(c[0], c[1], c[2]) for c in circles])
    return res


def _finish_circles(res, agg, lab, auto):
    """
    Собирает итоговый список кругов и агломератов:
      - автоматические агломераты (связные куски маски) со своими кругами;
      - каждый добавленный вручную круг первичной частицы — отдельный «агломерат» из одной частицы;
      - каждая добавленная вручную область — отдельный агломерат со своими кругами.
    К площади агломератов прибавляется только та часть ручного элемента, которая лежит ЗА ПРЕДЕЛАМИ
    уже найденного контура (и других ручных элементов). Масса ручного круга берётся с той же долей.
    """
    s = res["s"]; e = res.get("edit", {}); shape = res["shape"]
    H, W = shape
    # центр круга первичной частицы — только внутри контура агломерата (не в «дырке» и не снаружи):
    # если центр чуть вне маски (не дальше 0,25 R), он переносится в ближайшую точку агломерата, иначе круг отбрасывается
    near = None
    circles = []
    for (y, x, d) in auto:
        yi, xi = int(min(max(round(y), 0), H - 1)), int(min(max(round(x), 0), W - 1))
        k = int(lab[yi, xi]) if (0 <= y < H and 0 <= x < W) else 0
        if k == 0:
            if near is None:
                dist, idx = ndi.distance_transform_edt(lab == 0, return_indices=True)
                near = (dist, idx)
            if near[0][yi, xi] + np.hypot(y - yi, x - xi) <= 0.25 * d / 2 / s:
                y, x = float(near[1][0][yi, xi]), float(near[1][1][yi, xi])
                k = int(lab[int(y), int(x)])
        if k:
            circles.append((y, x, d, k))
    res["n_auto_circles"] = len(circles)            # круги идут так: сначала автоматические, потом ручные
    comps, _ = agglomerate_components(agg, circles, s, res["valid"])
    for c in comps:
        c["kind"] = "auto"
    nid = int(lab.max()) + 1
    taken = agg.copy()                               # уже учтённая площадь
    manual = np.zeros(shape, bool)
    for (y, x, d) in e.get("ucircles", []):
        dm = disk_mask(shape, y, x, d / 2 / s)
        new = dm & ~taken
        frac = float(new.sum()) / max(int(dm.sum()), 1)
        comps.append(dict(id=nid, area_nm2=float(new.sum() * s * s), cut=False, d_circles=[d], kind="circle", frac=frac))
        circles.append((y, x, d, nid)); nid += 1
        taken |= dm; manual |= dm
    for rg in e.get("uregions", []):
        new = rg["mask"] & ~taken
        cs = [(y, x, d) for (y, x, d) in rg["circles"] if new[int(y), int(x)]]
        comps.append(dict(id=nid, area_nm2=float(new.sum() * s * s), cut=False,
                          d_circles=[d for (y, x, d) in cs], kind="region"))
        circles += [(y, x, d, nid) for (y, x, d) in cs]; nid += 1
        taken |= rg["mask"]; manual |= rg["mask"]
    res["agg"] = agg; res["agg_manual"] = manual; res["circles"] = circles; res["comps"] = comps


def _manual_mask(res):
    e = res.get("edit", {}); s = res["s"]
    m = circles_mask(res["shape"], [(y, x, d) for (y, x, d) in e.get("ucircles", [])], s)
    for rg in e.get("uregions", []):
        m |= rg["mask"]
    return m


# ---------------------------------------------------------------------------
# Ручное редактирование (раздел «Редактировать»)
# ---------------------------------------------------------------------------
def shape_mask(shape, sh):
    """Растр фигуры: ("rect", x0, y0, x1, y1), ("circle", cx, cy, r) или ("poly", [(x, y), ...])."""
    from skimage.draw import polygon as _poly
    H, W = shape
    m = np.zeros((H, W), bool)
    if sh[0] == "rect":
        x0, y0, x1, y1 = sh[1:]
        a, b = int(max(0, min(y0, y1))), int(min(H, max(y0, y1) + 1))
        c, d = int(max(0, min(x0, x1))), int(min(W, max(x0, x1) + 1))
        m[a:b, c:d] = True
    elif sh[0] == "circle":
        m = disk_mask(shape, sh[2], sh[1], sh[3])
    elif sh[0] == "poly" and len(sh[1]) >= 3:
        pts = np.array(sh[1], float)
        rr, cc = _poly(pts[:, 1], pts[:, 0], shape)
        m[rr, cc] = True
    return m


def find_circle(res, y, x):
    """Круг, внутри которого точка (ближайший центр); None — если такого нет."""
    s = res["s"]; best = None
    for c in res["circles"]:
        d2 = (c[0] - y) ** 2 + (c[1] - x) ** 2
        r = c[2] / 2 / s
        if d2 <= (r + 1) ** 2 and (best is None or d2 < best[0]):
            best = (d2, c)
    return None if best is None else best[1]


def element_at(res, y, x):
    """Что лежит в точке: ("circle", круг) / ("object", объект) / None."""
    c = find_circle(res, y, x)
    if c is not None:
        return ("circle", c)
    H, W = res["shape"]
    if 0 <= int(y) < H and 0 <= int(x) < W:
        oid = int(res["objmap"][int(y), int(x)])
        if oid and res["objects"][oid - 1]["cls"] in ("mod", "irr"):
            return ("object", res["objects"][oid - 1])
    return None


def _new_object(res, m, cls):
    rp = measure.regionprops(m.astype(int))[0]
    s = res["s"]
    f = dict(deq=rp.equivalent_diameter_area * s, solidity=float(rp.solidity), contrast=float("nan"),
             circularity=float(4 * np.pi * rp.area / max(rp.perimeter, 1) ** 2),
             feret=rp.feret_diameter_max * s, minor=rp.minor_axis_length * s,
             cy=float(rp.centroid[0]), cx=float(rp.centroid[1]), area_px=int(rp.area))
    f["cut"] = bool(m[:2].any() or m[-2:].any() or m[:, :2].any() or m[:, -2:].any() or
                    (ndi.binary_dilation(m, iterations=2) & ~res["valid"]).any())
    f["cls"] = cls; f["auto_cls"] = "user"; f["id"] = len(res["objects"]) + 1
    ys, xs = np.nonzero(m)
    f["pix"] = (ys.astype(np.int32), xs.astype(np.int32))
    res["objmap"][m & (res["objmap"] == 0)] = f["id"]
    res["objects"].append(f)
    return f


def apply_edits(res, ops):
    """
    Вносит ручные правки в результат обработки снимка. Правки хранятся в res["edit"] и
    сохраняются при последующих щелчках смены класса.
      del_point  — удалить круг (или крупную частицу) в точке;
      del_region — удалить всё внутри фигуры (круги, частицы, контур агломерата, ручные элементы);
      add        — добавить первичную частицу (круг) / область агломерата (контур, заполняется кругами) /
                   модифицированную частицу (круг или контур).
    """
    s = res["s"]; shape = res["shape"]; P = res["params"]
    e = res.setdefault("edit", {})
    del_mask = e.get("del_mask"); del_mask = np.zeros(shape, bool) if del_mask is None else del_mask
    uc = e.setdefault("ucircles", [])
    regs = e.setdefault("uregions", [])
    dmax = 0.99 * min(P["prim_dmax"] or P["d_cut"], P["d_cut"])
    auto = [(c[0], c[1], c[2]) for c in _auto_circles(res)]
    if P.get("mode") == "only_mod":
        ops = [op for op in ops if not (op["op"] == "add" and op["cls"] != "mod")]

    for op in ops:
        kind = op["op"]
        if kind == "del_point":
            y, x = op["y"], op["x"]
            # помечаем, откуда круг: ("a", i) — автоматический, ("u", i) — ручной, ("r", j, i) — из ручной области
            # ручные круги — первыми: при совпадении центров удаляется добавленный вручную
            res["circles"] = [tuple(u) + (("u", i),) for i, u in enumerate(uc)] + \
                             [tuple(q) + (("r", j, i),) for j, rg in enumerate(regs) for i, q in enumerate(rg["circles"])] + \
                             [tuple(c) + (("a", i),) for i, c in enumerate(auto)]
            hit = element_at(res, y, x)
            if hit is None:
                continue
            if hit[0] == "circle":
                c = hit[1][:3]; tag = hit[1][3]
                if tag[0] == "u":
                    del uc[tag[1]]
                elif tag[0] == "r":
                    rg = regs[tag[1]]
                    rest = [q for i, q in enumerate(rg["circles"]) if i != tag[2]]
                    rg["mask"] &= ~(disk_mask(shape, c[0], c[1], c[2] / 2 / s) & ~circles_mask(shape, rest, s))
                    rg["circles"] = rest
                else:
                    rest = [q for i, q in enumerate(auto) if i != tag[1]]
                    del_mask |= disk_mask(shape, c[0], c[1], c[2] / 2 / s) & ~circles_mask(shape, rest, s)
                    auto = rest
            else:
                o = hit[1]
                m = np.zeros(shape, bool); m[o["pix"]] = True
                if o["cls"] == "irr":
                    del_mask |= m
                    auto = [q for q in auto if not m[int(q[0]), int(q[1])]]
                o["cls"] = "del"
        elif kind == "del_region":
            m = shape_mask(shape, op["shape"])
            del_mask |= m
            inside = lambda q: m[int(q[0]), int(q[1])]
            auto = [q for q in auto if not inside(q)]
            uc[:] = [u for u in uc if not inside(u)]
            for rg in regs:
                rg["mask"] &= ~m
                rg["circles"] = [q for q in rg["circles"] if not inside(q)]
            regs[:] = [rg for rg in regs if rg["mask"].any()]
            for o in res["objects"]:
                if o["cls"] in ("mod", "irr", "agg") and m[int(o["cy"]), int(o["cx"])]:
                    o["cls"] = "del"
        elif kind == "add":
            sh = op["shape"]; m = shape_mask(shape, sh)
            if not m.any():
                continue
            big = sh[0] == "circle" and 2 * sh[3] * s >= P["d_cut"]
            if op["cls"] == "mod" or big:
                # круг не меньше отсечки не может быть первичной частицей: он становится модифицированной
                # (в режиме «без модифицированных» — НФ)
                del_mask &= ~m
                _new_object(res, m, "irr" if (op["cls"] != "mod" and P.get("mode") == "no_mod") else "mod")
            elif sh[0] == "circle":                       # одна первичная частица
                uc.append((float(sh[2]), float(sh[1]), float(2 * sh[3] * s)))
            else:                                         # область агломерата: заполнить кругами
                e["del_mask"] = del_mask                  # кругами заполняется только часть вне контура
                free = m & ~build_agg(res) & ~_manual_mask(res)
                cs = place_circles(res, free) if free.any() else []
                if free.any() and not cs:
                    cs = [(c[0], c[1], c[2]) for c in pack_components(free, s, P["prim_dmin"] or 4.0, dmax)]
                regs.append(dict(mask=m, circles=cs))

    e["del_mask"] = del_mask
    agg = build_agg(res)
    _finish_circles(res, agg, measure.label(agg), auto)
    return res


_SNAP_KEYS = ("objmap", "edit", "circles", "agg", "agg_manual", "comps", "n_auto_circles")


def snapshot(res):
    """Копия изменяемой части результата — для отмены применённых правок (Ctrl+Z)."""
    import copy
    snap = {k: copy.deepcopy(res[k]) for k in _SNAP_KEYS if k in res}
    snap["objects"] = [dict(o) for o in res["objects"]]
    return snap


def restore(res, snap):
    for k in _SNAP_KEYS:
        if k in snap:
            res[k] = snap[k]
        else:
            res.pop(k, None)
    res["objects"] = snap["objects"]
    return res


def toggle_object(res, y, x):
    """
    Ручная правка кликом: модифицированная -> агломерат -> НФ (неправильной формы) -> модифицированная.
    Возвращает новый класс объекта или None, если в этом месте нет объекта-кандидата.
    """
    oid = int(res["objmap"][int(y), int(x)])
    if oid == 0:
        return None
    o = res["objects"][oid - 1]
    o["cls"] = {"mod": "agg", "agg": "irr", "irr": "mod"}.get(o["cls"], "mod")
    ys, xs = o["pix"]; s = res["s"]
    pad = int((res["params"]["prim_dmax"] or 25) / s) + 10
    H, W = res["shape"]
    region = (max(0, ys.min() - pad), min(H, ys.max() + pad), max(0, xs.min() - pad), min(W, xs.max() + pad))
    update_agglomerates(res, region)
    return o["cls"]


# ---------------------------------------------------------------------------
# 6. Массы и статистика
# ---------------------------------------------------------------------------
def pooled_primary_sizes(results, max_s=1.0, min_circles=50):
    """Сводные d20 и d30 первичных по снимкам с хорошим разрешением (<= max_s нм/пикс)."""
    d = [c[2] for r in results if r["s"] <= max_s for c in r["circles"]]
    if len(d) < min_circles:
        return None
    d = np.array(d)
    return float(np.sqrt((d ** 2).mean())), float(np.mean(d ** 3) ** (1 / 3))


def number_objects(res):
    """Нумерует модифицированные частицы по порядку (те же номера — на рисунке и в отчёте)."""
    n = 0
    for o in res["objects"]:
        if o["cls"] == "mod":
            n += 1; o["num"] = n
        else:
            o["num"] = None


def preview_contour(path, params):
    """Быстрая проверка параметров: только контур вещества (без поиска частиц и кругов). ~1-2 с."""
    P = dict(DEFAULTS); P.update(params or {})
    gray, s_file = read_image(path)
    s, s_src = get_scale(gray, s_file, P)
    valid = valid_area(gray)
    m_all, D, lvl1, m0 = segment(gray, s, P, valid)
    a = np.stack([gray] * 3, -1).astype(float)
    e = m_all ^ ndi.binary_erosion(m_all)
    a[e] = [40, 120, 255]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), s


def nf_mass(o, s, density, packing=0.74):
    """
    Масса тела неправильной формы (НФ), г: эллипсоид ближайшей формы, заполненный шарами
    с долей packing (плотная упаковка шаров одного размера — 0,74).
    Полуоси a ≥ b — по эллипсу с теми же моментами, что у проекции НФ, подогнанному по площади;
    третья полуось (вдоль луча) принята равной меньшей: c = b.
    """
    ys, xs = o["pix"]
    if len(ys) < 3:
        return 0.0
    a, b = ellipse_axes(ys, xs, s)                   # эллипс той же площади, что проекция
    o["nf_axes_nm"] = (2 * a, 2 * b, 2 * b)
    return density * 1e-21 * packing * 4.0 / 3.0 * np.pi * a * b * b


def size_stats(d):
    """Среднее, медиана и мода распределения диаметров (мода — максимум сглаженной гистограммы,
    ядерная оценка плотности с шириной окна по Скотту). Пустой набор — NaN."""
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    out = dict(n=int(d.size), mean=np.nan, median=np.nan, mode=np.nan)
    if d.size == 0:
        return out
    out["mean"] = float(d.mean()); out["median"] = float(np.median(d))
    if d.size < 3 or np.ptp(d) < 1e-6:
        out["mode"] = out["median"]
        return out
    from scipy.stats import gaussian_kde
    try:
        k = gaussian_kde(d)
        g = np.linspace(d.min(), d.max(), max(200, int(np.ptp(d) / 0.1)))
        out["mode"] = float(g[np.argmax(k(g))])
    except Exception:
        out["mode"] = out["median"]
    return out


def compute_masses(res, density, pooled=None):
    """
    Считает массы и статистику для одного снимка (быстро, без повторной обработки).
      - модифицированные: шары (обрезанные краем — видимая часть);
      - НФ (тела неправильной формы): эллипсоид ближайшей формы × доля заполнения плотной упаковки
        шаров (nf_packing, по умолчанию 0,74); прибавляются к массе агломератов;
      - агломераты, ПЛОСКАЯ оценка: сумма объёмов видимых кругов (частицы в глубине не учтены);
      - агломераты, ОБЪЁМНАЯ оценка: N = ka*(S/Ap)^alpha частиц размера d20/d30 (размеры — по кругам).
    """
    P = res["params"]; s = res["s"]; rho = density * 1e-21
    mode = P.get("mode", "full")
    number_objects(res)
    mods = [o for o in res["objects"] if o["cls"] == "mod" and (P["include_cut"] or not o["cut"])]
    irrs = [o for o in res["objects"] if o["cls"] == "irr" and (P["include_cut"] or not o["cut"])]
    H, W = res["shape"]
    for o in mods:
        m = np.zeros((H, W), bool); m[o["pix"]] = True
        o["mass_g"] = sphere_mass_in_field(m, s, o["cut"], res["valid"], o["deq"], density, obj=o)
    for o in irrs:
        o["mass_g"] = nf_mass(o, s, density, P.get("nf_packing", 0.74))
    M_irr = sum(o["mass_g"] for o in irrs)
    d = np.array([c[2] for c in res["circles"]])
    own = (float(np.sqrt((d ** 2).mean())), float(np.mean(d ** 3) ** (1 / 3))) if len(d) >= 50 else None
    if s > 1.2 and pooled:
        d20, d30, src = pooled[0], pooled[1], "сводные (снимки с разрешением лучше 1 нм/пикс)"
    elif own:
        d20, d30, src = own[0], own[1], "свои"
    elif pooled:
        d20, d30, src = pooled[0], pooled[1], "сводные"
    else:
        d20, d30, src = 10.0, 10.0, "нет данных, принято 10 нм"
    Ap = np.pi * d20 ** 2 / 4
    m_flat = m_3d = n3 = 0.0
    for c in res["comps"]:
        if not P["include_cut"] and c["cut"]:
            c["mass_flat_g"] = c["mass_3d_g"] = c["n3d"] = 0.0; continue
        c["mass_flat_g"] = float(np.sum(rho * np.pi * np.array(c["d_circles"]) ** 3 / 6)) if c["d_circles"] else 0.0
        if c.get("kind") == "circle":            # добавленная вручную частица: одна частица × доля вне контура
            f = c.get("frac", 1.0)
            c["n3d"] = f; c["mass_flat_g"] *= f; c["mass_3d_g"] = c["mass_flat_g"]
        else:
            c["n3d"] = P["ka"] * (c["area_nm2"] / Ap) ** P["alpha"]
            c["mass_3d_g"] = c["n3d"] * rho * np.pi * d30 ** 3 / 6
        m_flat += c["mass_flat_g"]; m_3d += c["mass_3d_g"]; n3 += c["n3d"]
    m_flat += M_irr; m_3d += M_irr                    # НФ входят в массу агломератов
    M_mod = sum(o["mass_g"] for o in mods)
    if mode == "only_mod":                             # круги не размечались: массы агломератов не считаем
        m_flat = m_3d = np.nan; n3 = np.nan
    dm = np.array([o["deq"] for o in mods if not o["cut"]]) if mods else np.array([])
    st = dict(
        name=res["name"], s=s, s_src=res["s_src"], contrast=P["contrast"], denoise=P["denoise"],
        n_mod=len(mods), n_mod_cut=sum(o["cut"] for o in mods),
        d_mod_mean=float(dm.mean()) if dm.size else np.nan, d_mod_median=float(np.median(dm)) if dm.size else np.nan,
        d_mod_sd=float(dm.std(ddof=1)) if dm.size > 1 else np.nan,
        d_mod_min=float(dm.min()) if dm.size else np.nan, d_mod_max=float(dm.max()) if dm.size else np.nan,
        mass_mod_fg=M_mod * 1e15,
        n_prim_vis=int(len(d)), n_prim_3d=float(n3),
        d_prim_median=float(np.median(d)) if d.size else np.nan, d20=d20, d30=d30, d_src=src,
        mass_agg_flat_fg=m_flat * 1e15, mass_agg_3d_fg=m_3d * 1e15,
        frac_flat=M_mod / (M_mod + m_flat) if mode == "full" and (M_mod + m_flat) > 0 else np.nan,
        frac_3d=M_mod / (M_mod + m_3d) if mode == "full" and (M_mod + m_3d) > 0 else np.nan,
        n_irr=len(irrs), mass_irr_fg=M_irr * 1e15, mode=mode,
        n_agg=len(res["comps"]),
        agg_area_nm2=float(sum(c["area_nm2"] for c in res["comps"])),
        agg_area_manual_nm2=float(sum(c["area_nm2"] for c in res["comps"] if c.get("kind") in ("circle", "region"))),
    )
    dprim = [float(x) for x in d]
    dmod = [float(x) for x in dm]
    for key, arr in (("prim", dprim), ("mod", dmod), ("all", dprim + dmod)):
        ss = size_stats(arr)
        st["d_{}_mean".format(key)] = ss["mean"]; st["d_{}_med".format(key)] = ss["median"]
        st["d_{}_mode".format(key)] = ss["mode"]
    st["_d_prim"] = dprim; st["_d_mod"] = dmod
    res["stats"] = st
    return st


def batch_summary(stats):
    """Итог по всем снимкам: средняя доля (по снимкам), ошибка среднего, медиана и доля по суммарной массе."""
    out = {}
    if not stats:
        return out
    for key, mk in [("flat", "mass_agg_flat_fg"), ("3d", "mass_agg_3d_fg")]:
        f = np.array([st["frac_" + key] for st in stats if np.isfinite(st["frac_" + key])])
        Mm = sum(st["mass_mod_fg"] for st in stats); Ma = float(np.nansum([st[mk] for st in stats]))
        out[key] = dict(mean=float(f.mean()) if f.size else np.nan,
                        se=float(f.std(ddof=1) / np.sqrt(f.size)) if f.size > 1 else np.nan,
                        median=float(np.median(f)) if f.size else np.nan,
                        pooled=Mm / (Mm + Ma) if (Mm + Ma) > 0 and f.size else np.nan)
    out["n_images"] = len(stats)
    out["n_mod"] = sum(st["n_mod"] for st in stats)
    out["n_prim_vis"] = sum(st["n_prim_vis"] for st in stats)
    out["n_prim_3d"] = float(np.nansum([st["n_prim_3d"] for st in stats]))
    out["mass_mod_fg"] = sum(st["mass_mod_fg"] for st in stats)
    out["mass_agg_flat_fg"] = float(np.nansum([st["mass_agg_flat_fg"] for st in stats]))
    out["mass_agg_3d_fg"] = float(np.nansum([st["mass_agg_3d_fg"] for st in stats]))
    out["mass_irr_fg"] = float(np.nansum([st.get("mass_irr_fg", 0) for st in stats]))
    out["n_irr"] = sum(st.get("n_irr", 0) for st in stats)
    out["mode"] = stats[0].get("mode", "full")
    dp = [x for st in stats for x in st.get("_d_prim", [])]
    dm = [x for st in stats for x in st.get("_d_mod", [])]
    out["size"] = {"prim": size_stats(dp), "mod": size_stats(dm), "all": size_stats(dp + dm)}
    return out


# ---------------------------------------------------------------------------
# Рисунок разметки
# ---------------------------------------------------------------------------
def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_overlay(res, show_circles=True, scale=1, show_contour=True, lw=1, lw_c=1):
    """
    Картинка с разметкой:
      зелёная заливка + номер — модифицированная частица;
      синий контур — агломерат;  оранжевые круги — первичные частицы;
      пурпурный контур — тело неправильной формы (НФ).
    """
    gray, _ = read_image(res["path"])
    if scale != 1:
        gray = np.kron(gray, np.ones((scale, scale)))
    a = np.stack([gray] * 3, -1).astype(float)
    k = (lambda z: np.kron(z, np.ones((scale, scale))).astype(bool)) if scale != 1 else (lambda z: z)
    mod, irr = masks_from_objects(res)
    agg = res["agg"]
    lw = max(1, int(lw))
    thick = lambda e, extra=0: ndi.binary_dilation(e, iterations=lw - 1 + extra) if lw - 1 + extra > 0 else e
    if show_contour:
        av = agg | res["agg_manual"] if res.get("agg_manual") is not None else agg
        e = av ^ ndi.binary_erosion(av); a[k(thick(e))] = [40, 120, 255]
        mm = k(mod); a[mm] = a[mm] * 0.72 + np.array([0, 200, 0]) * 0.28
        e = mod ^ ndi.binary_erosion(mod); a[k(thick(e, 1))] = [0, 200, 0]
        e = irr ^ ndi.binary_erosion(irr); a[k(thick(e, 1))] = [230, 0, 200]
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    if show_circles:
        for (y, x, dd, c) in res["circles"]:
            r = dd / 2 / res["s"] * scale
            d.ellipse([x * scale - r, y * scale - r, x * scale + r, y * scale + r], outline=(255, 140, 0),
                      width=max(1, int(lw_c) * (2 if scale > 1 else 1)))
    number_objects(res)
    if not show_contour:
        return img
    f = _font(max(12, int(18 * scale)))
    for o in res["objects"]:
        if o["cls"] == "mod":
            t = str(o["num"]); col = (255, 255, 0)
        elif o["cls"] == "irr":
            t = "НФ"; col = (230, 0, 200)
        else:
            continue
        d.text((o["cx"] * scale - 6, o["cy"] * scale - 10), t, fill=col, font=f, stroke_width=3, stroke_fill=(0, 0, 0))
    return img

