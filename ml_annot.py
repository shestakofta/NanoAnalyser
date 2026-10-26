# -*- coding: utf-8 -*-
"""
ml_annot.py — набор размеченных снимков для последующего машинного обучения (блок «Разметка для обучения (ML)»).

Разметка снимка берётся из результата обработки в том виде, в каком её оставил пользователь после правок:
  P — круги первичных частиц (с отметкой «?» для сомнительных), M — модифицированные частицы (многоугольник),
  I — тела неправильной формы, НФ (многоугольник); зона разметки и исключённые участки (линейка) — многоугольники.

Структура папки набора (её удобно переносить целиком или архивом):
  images/                   исходные снимки (копии, имена уникальны)
  labels/<разметчик>/       <имя снимка>.json — разметка одного снимка одним разметчиком
  manifest.csv              сводка по всем файлам разметки (пересоздаётся автоматически)
  README.txt                описание формата

Модуль не зависит от окна программы: его функции можно вызывать и из скриптов на другом компьютере.
"""
import os
import re
import csv
import json
import time
import shutil
import hashlib
import zipfile

import numpy as np

FORMAT = "npa-annot-1"
TZ_VERSION = "1.0"
MATCH_PX = 2.0                     # круги с центрами ближе этого (пикс.) считаются одним и тем же кругом


# --------------------------------------------------------------------------- настройки блока (разметчик, папка набора)
def config_path():
    import sys

    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get(
            "XDG_CONFIG_HOME",
            os.path.expanduser("~/.config")
        )

    return os.path.join(base, "NPA", "ml_settings.json")


def load_config():
    try:
        with open(config_path(), encoding="utf-8") as fh:
            c = json.load(fh)
        return c if isinstance(c, dict) else {}
    except Exception:
        return {}


def save_config(cfg):
    p = config_path()
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def clean_code(code):
    """Код разметчика: латинские буквы, цифры, «_» и «-», до 16 символов."""
    return re.sub(r"[^A-Za-z0-9_\-]", "", (code or "").strip())[:16]


# --------------------------------------------------------------------------- геометрия
def key(y, x):
    return (round(float(y), 1), round(float(x), 1))


def circumcircle(p1, p2, p3):
    """Окружность через три точки (x, y): (cx, cy, r) или None, если точки почти на одной прямой."""
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    span = max(np.hypot(x2 - x1, y2 - y1), np.hypot(x3 - x2, y3 - y2), np.hypot(x1 - x3, y1 - y3), 1e-9)
    if abs(d) < 1e-6 * span * span:
        return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    r = float(np.hypot(x1 - cx, y1 - cy))
    if not np.isfinite(r) or r > 20 * span:
        return None
    return float(cx), float(cy), r


def nearest(circles, y, x, tol=MATCH_PX):
    """Индекс круга (y, x, ...) с центром ближе tol к точке, или None."""
    best, bi = tol, None
    for i, c in enumerate(circles):
        d = np.hypot(c[0] - y, c[1] - x)
        if d <= best:
            best, bi = d, i
    return bi


def match_doubts(circles, doubt):
    """Отметки «?» привязываются к текущим кругам (центр мог сдвинуться на доли пикселя после правок)."""
    out = set()
    for (y, x) in doubt:
        i = nearest(circles, y, x)
        if i is not None:
            out.add(key(circles[i][0], circles[i][1]))
    return out


def compare(base, cur):
    """Сравнение кругов после обработки (base) и после правок (cur), оба — списки (y, x, r):
    сколько автоматических кругов оставлено без изменений, удалено (или заметно изменено) и добавлено."""
    used = set(); kept = 0
    for (y, x, r) in base:
        best, bj = MATCH_PX, None
        for j, (y2, x2, r2) in enumerate(cur):
            if j in used:
                continue
            d = np.hypot(y - y2, x - x2)
            if d <= best and abs(r2 - r) <= max(1.0, 0.1 * r):
                best, bj = d, j
        if bj is not None:
            used.add(bj); kept += 1
    return dict(auto=len(base), auto_kept=kept, auto_removed=len(base) - kept, added=len(cur) - kept)


def mask_polygons(mask, tol=1.0, min_area=4):
    """Внешние контуры связных участков маски как многоугольники [[x, y], ...] (координаты пикселей)."""
    from skimage import measure
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    lab = measure.label(m)
    out = []
    for rp in measure.regionprops(lab):
        if rp.area < min_area:
            continue
        r0, c0, r1, c1 = rp.bbox
        sub = np.pad(lab[r0:r1, c0:c1] == rp.label, 1)
        cs = measure.find_contours(sub.astype(float), 0.5)
        if not cs:
            continue
        cnt = max(cs, key=len)
        cnt = measure.approximate_polygon(cnt, tolerance=tol)
        pts = [[round(float(x + c0 - 1), 1), round(float(y + r0 - 1), 1)] for (y, x) in cnt]
        if len(pts) > 3 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) >= 3:
            out.append(pts)
    return out


def file_sha1(path, block=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- разметка одного снимка
def circles_px(res):
    """Круги первичных частиц результата обработки: список (y, x, r) в пикселях."""
    s = res["s"]
    return [(float(c[0]), float(c[1]), float(c[2]) / 2 / s) for c in res.get("circles", [])]


def build_annotation(res, ml, annotator, tool, extra=None):
    """
    Словарь разметки снимка (формат npa-annot-1) из результата обработки res и состояния ml:
      ml = dict(doubt=set ключей (y, x), region=(x0, y0, x1, y1) или None, mode="corrected"|"scratch",
                t_active=секунды, base=[(y, x, r)] — круги сразу после обработки).
    """
    H, W = res["shape"]; s = float(res["s"])
    P = res.get("params", {})
    cur = circles_px(res)
    doubt = match_doubts(cur, ml.get("doubt", set()))
    objs = []
    for (y, x, r) in cur:
        objs.append(dict(shape="circle", **{"class": "P"}, x=round(x, 2), y=round(y, 2), r=round(r, 2),
                         doubt=key(y, x) in doubt))
    nM = nI = 0
    for o in res.get("objects", []):
        if o.get("cls") not in ("mod", "irr"):
            continue
        m = np.zeros((H, W), bool); m[o["pix"]] = True
        for pts in mask_polygons(m):
            objs.append(dict(shape="polygon", **{"class": "M" if o["cls"] == "mod" else "I"}, points=pts))
            if o["cls"] == "mod":
                nM += 1
            else:
                nI += 1
    reg = ml.get("region")
    if reg:
        x0, y0, x1, y1 = reg
        x0, x1 = sorted((max(0.0, min(W, x0)), max(0.0, min(W, x1))))
        y0, y1 = sorted((max(0.0, min(H, y0)), max(0.0, min(H, y1))))
        region = [[round(x0, 1), round(y0, 1)], [round(x1, 1), round(y0, 1)], [round(x1, 1), round(y1, 1)], [round(x0, 1), round(y1, 1)]]
    else:
        region = [[0, 0], [W, 0], [W, H], [0, H]]
    ignore = mask_polygons(~np.asarray(res["valid"], bool)) if "valid" in res else []
    mode = ml.get("mode", "corrected")
    st = dict(n_P=len(cur), n_doubt=len(doubt), n_M=nM, n_I=nI)
    if mode != "scratch" and ml.get("base") is not None:
        st.update(compare(ml["base"], cur))
    ann = {
        "format": FORMAT, "tz_version": TZ_VERSION, "tool": tool,
        "image": None, "image_sha1": None, "source_name": res.get("name") or os.path.basename(res["path"]),
        "width": int(W), "height": int(H), "nm_per_px": round(s, 6), "scale_source": res.get("s_src"),
        "material": P.get("element"), "d_cut_nm": P.get("d_cut"), "npa_mode": P.get("mode", "full"),
        "annotator": annotator, "mode": mode, "created": None, "saved": now_iso(),
        "time_min": round(float(ml.get("t_active", 0.0)) / 60.0, 1),
        "region": region, "ignore": ignore, "stats": st, "objects": objs,
    }
    if extra:
        ann.update(extra)
    return ann


# --------------------------------------------------------------------------- папка набора
def ensure_dataset(ds):
    os.makedirs(os.path.join(ds, "images"), exist_ok=True)
    os.makedirs(os.path.join(ds, "labels"), exist_ok=True)
    rp = os.path.join(ds, "README.txt")
    if not os.path.exists(rp):
        with open(rp, "w", encoding="utf-8") as fh:
            fh.write(README)


def _image_index(ds):
    """sha1 -> имя файла в images/ (по файлам разметки, чтобы не пересчитывать хеши снимков)."""
    idx = {}
    for code, path in _label_files(ds):
        try:
            with open(path, encoding="utf-8") as fh:
                a = json.load(fh)
            if a.get("image_sha1") and a.get("image") and os.path.exists(os.path.join(ds, "images", a["image"])):
                idx[a["image_sha1"]] = a["image"]
        except Exception:
            pass
    return idx


def _label_files(ds):
    root = os.path.join(ds, "labels")
    if not os.path.isdir(root):
        return []
    out = []
    for code in sorted(os.listdir(root)):
        d = os.path.join(root, code)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".json"):
                    out.append((code, os.path.join(d, f)))
    return out


def _free_name(ds, name, sha):
    """Имя для нового снимка в images/: исходное, а если оно занято другим файлом — с началом хеша."""
    p = os.path.join(ds, "images", name)
    if not os.path.exists(p):
        return name
    stem, ext = os.path.splitext(name)
    return "{}__{}{}".format(stem, sha[:8], ext)


def label_path(ds, annotator, image_name):
    return os.path.join(ds, "labels", annotator, image_name + ".json")


def find_label(ds, annotator, sha):
    """Путь к существующей разметке этого снимка этим разметчиком (или None)."""
    idx = _image_index(ds)
    name = idx.get(sha)
    if name is None:
        return None
    p = label_path(ds, annotator, name)
    return p if os.path.exists(p) else None


def add_to_dataset(ds, src_path, ann):
    """
    Копирует снимок в images/ (если такого ещё нет) и записывает файл разметки разметчика.
    Возвращает (путь к файлу разметки, True — если разметка обновлена, False — новая).
    """
    ensure_dataset(ds)
    sha = file_sha1(src_path)
    idx = _image_index(ds)
    name = idx.get(sha)
    if name is None:
        name = _free_name(ds, os.path.basename(src_path), sha)
        shutil.copy2(src_path, os.path.join(ds, "images", name))
    ann["image"] = name; ann["image_sha1"] = sha
    lp = label_path(ds, ann["annotator"], name)
    updated = os.path.exists(lp)
    if updated:
        try:
            with open(lp, encoding="utf-8") as fh:
                ann["created"] = json.load(fh).get("created") or ann["saved"]
        except Exception:
            ann["created"] = ann["saved"]
    else:
        ann["created"] = ann["saved"]
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    tmp = lp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ann, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, lp)
    rebuild_manifest(ds)
    return lp, updated


MANIFEST_COLS = ("image", "annotator", "mode", "n_P", "n_doubt", "n_M", "n_I", "auto_kept", "auto_removed", "added",
                 "time_min", "nm_per_px", "width", "height", "material", "source_name", "created", "saved", "image_sha1")


def rebuild_manifest(ds):
    rows = []
    for code, path in _label_files(ds):
        try:
            with open(path, encoding="utf-8") as fh:
                a = json.load(fh)
        except Exception:
            continue
        st = a.get("stats", {})
        rows.append([a.get("image"), a.get("annotator", code), a.get("mode"), st.get("n_P"), st.get("n_doubt"),
                     st.get("n_M"), st.get("n_I"), st.get("auto_kept", ""), st.get("auto_removed", ""), st.get("added", ""),
                     a.get("time_min"), a.get("nm_per_px"), a.get("width"), a.get("height"), a.get("material"),
                     a.get("source_name"), a.get("created"), a.get("saved"), a.get("image_sha1")])
    with open(os.path.join(ds, "manifest.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(MANIFEST_COLS)
        w.writerows(rows)
    return len(rows)


def dataset_info(ds):
    """(число снимков, число файлов разметки, {разметчик: число файлов})."""
    if not ds or not os.path.isdir(ds):
        return 0, 0, {}
    imgs = [f for f in os.listdir(os.path.join(ds, "images"))] if os.path.isdir(os.path.join(ds, "images")) else []
    per = {}
    for code, _ in _label_files(ds):
        per[code] = per.get(code, 0) + 1
    return len(imgs), sum(per.values()), per


# --------------------------------------------------------------------------- перенос между компьютерами
def pack_dataset(ds, zip_path, progress=None):
    """Весь набор в один zip (контрольные суммы каждого файла хранятся в архиве). Возвращает число файлов."""
    rebuild_manifest(ds)
    files = []
    for root, _, fs in os.walk(ds):
        for f in fs:
            if f.endswith(".tmp"):
                continue
            p = os.path.join(root, f)
            if os.path.abspath(p) == os.path.abspath(zip_path):
                continue
            files.append(p)
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for i, p in enumerate(files):
            z.write(p, os.path.relpath(p, ds).replace(os.sep, "/"))
            if progress:
                progress(i + 1, len(files))
    os.replace(tmp, zip_path)
    return len(files)


_RX_IMG = re.compile(r"^images/([^/\\]+)$")
_RX_LBL = re.compile(r"^labels/([A-Za-z0-9_\-]{1,16})/([^/\\]+\.json)$")


def merge_archive(ds, zip_path):
    """
    Добавляет в набор содержимое архива другого набора (другой разметчик или другой компьютер).
    Снимки сравниваются по содержимому (sha1): одинаковые не дублируются, разные с одним именем получают
    новое имя. Разметка того же снимка тем же разметчиком заменяется, только если в архиве она новее.
    Возвращает словарь счётчиков.
    """
    ensure_dataset(ds)
    idx = _image_index(ds)
    cnt = dict(images_new=0, images_same=0, labels_new=0, labels_updated=0, labels_older=0, skipped=0)
    rename = {}
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        bad = z.testzip()
        if bad is not None:
            raise ValueError("Архив повреждён: " + bad)
        for n in names:
            m = _RX_IMG.match(n)
            if not m:
                continue
            data = z.read(n)
            sha = hashlib.sha1(data).hexdigest()
            if sha in idx:
                rename[m.group(1)] = idx[sha]; cnt["images_same"] += 1
                continue
            new = _free_name(ds, m.group(1), sha)
            p = os.path.join(ds, "images", new)
            if os.path.exists(p) and file_sha1(p) == sha:
                rename[m.group(1)] = new; idx[sha] = new; cnt["images_same"] += 1
                continue
            with open(p, "wb") as fh:
                fh.write(data)
            rename[m.group(1)] = new; idx[sha] = new; cnt["images_new"] += 1
        for n in names:
            m = _RX_LBL.match(n)
            if not m:
                if not (n.endswith("/") or _RX_IMG.match(n) or n in ("manifest.csv", "README.txt")):
                    cnt["skipped"] += 1
                continue
            try:
                a = json.loads(z.read(n).decode("utf-8"))
            except Exception:
                cnt["skipped"] += 1
                continue
            if a.get("format") != FORMAT:
                cnt["skipped"] += 1
                continue
            code = m.group(1)
            img = rename.get(a.get("image"), a.get("image"))
            if not img or not os.path.exists(os.path.join(ds, "images", img)):
                cnt["skipped"] += 1
                continue
            a["image"] = img
            lp = label_path(ds, code, img)
            if os.path.exists(lp):
                try:
                    with open(lp, encoding="utf-8") as fh:
                        old = json.load(fh)
                except Exception:
                    old = {}
                if str(a.get("saved", "")) <= str(old.get("saved", "")):
                    cnt["labels_older"] += 1
                    continue
                cnt["labels_updated"] += 1
            else:
                cnt["labels_new"] += 1
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            with open(lp, "w", encoding="utf-8") as fh:
                json.dump(a, fh, ensure_ascii=False, indent=1)
    rebuild_manifest(ds)
    return cnt


# --------------------------------------------------------------------------- тексты
README = """Набор размеченных снимков NPA для машинного обучения (формат npa-annot-1)

Состав папки
  images/                  исходные снимки (копии; одинаковые по содержимому снимки хранятся один раз)
  labels/<разметчик>/      разметка: <имя снимка>.json — один снимок, один разметчик
  manifest.csv             сводка по файлам разметки (разделитель «;», пересоздаётся программой)

Файл разметки (JSON)
  image, image_sha1        имя снимка в images/ и его контрольная сумма
  width, height, nm_per_px размер снимка в пикселях и масштаб
  annotator                код разметчика
  mode                     corrected — правка автоматической разметки; scratch — разметка с нуля
  time_min                 активное время правки, мин (оценка)
  region                   зона разметки: внутри неё размечены ВСЕ частицы (многоугольник [[x, y], ...])
  ignore                   участки вне анализа (масштабная линейка и подпись)
  stats                    числа объектов; для режима corrected — сколько автоматических кругов оставлено,
                           удалено и сколько кругов добавлено
  objects                  circle P — первичная частица: x, y, r (пиксели), doubt — отметка «?»;
                           polygon M — модифицированная частица; polygon I — тело неправильной формы (НФ)
Координаты — в пикселях исходного снимка: начало в левом верхнем углу, x вправо, y вниз.

Перенос
  В программе NPA: «Упаковать набор…» — один zip-архив; на другом компьютере «Добавить из архива…»
  объединяет его с местным набором (одинаковые снимки не дублируются, более новая разметка заменяет старую).
"""

MEMO = """ПАМЯТКА: РАЗМЕТКА СНИМКОВ ДЛЯ ОБУЧЕНИЯ

Как пополняется набор
• Снимки обрабатываются и правятся как обычно (раздел «Редактировать»). Когда разметка снимка проверена, нажмите «Добавить снимок в набор». Повторное нажатие обновляет разметку этого снимка.
• В набор попадает то, что видно на снимке: круги первичных частиц, модифицированные частицы и НФ (по классам программы с вашими правками).
• Перед первым добавлением укажите код разметчика (например, A1) и папку набора. Код один на человека и не меняется.

Режим разметки
• Обычно — правка автоматической разметки. Каждый оставленный круг должен быть проверен: иначе модель выучит ошибки программы, а не человека.
• «Начать с нуля» удаляет автоматическую разметку снимка (Ctrl+Z возвращает). Так размечаются тестовые снимки: два человека независимо, не видя разметку программы и друг друга.

Правила
1. Размечаются ВСЕ первичные частицы внутри зоны разметки. Пропущенная частица для модели означает «частицы нет».
2. Круг — по внешнему краю частицы, где кончается её затемнение и начинается фон, а не по середине размытого края.
3. Перекрывающиеся частицы размечаются обе; скрытая часть достраивается по видимой дуге (удобен «круг по трём точкам»).
4. Мелкая частица поверх крупной — отдельный круг (круг внутри круга допустим). Тёмная середина одной частицы отдельно не размечается.
5. Спёкшиеся частицы: есть перетяжка или более светлая линия — два круга; сплошное почти круглое тело — один круг; сплошное некруглое — НФ.
6. Частица, обрезанная краем кадра, размечается, если в кадре не меньше половины её площади.
7. Не размечаются частицы мельче 8 пикселей или 3 нм, подложка, тёмные участки плёнки, загрязнения, линейка и подписи.
8. Сомнение — не повод пропускать: поставьте круг и отметку «?» (ориентир — не больше 10 % кругов снимка).
9. Очень плотный снимок можно размечать не целиком: инструментом «Зона» выделите прямоугольник не меньше трети кадра и разметьте в нём всё.

Инструменты блока
• Круг по трём точкам — три щелчка по видимому краю частицы; круг добавляется как правка (Enter — применить).
• «?» — щелчок по кругу ставит или снимает отметку сомнения.
• «Зона» — прямоугольник, внутри которого размечено всё; «Весь кадр» — сбросить.

Перенос на другой компьютер
• «Упаковать набор…» — весь набор в один архив (снимки, разметка, сводка, описание формата).
• На другом компьютере «Добавить из архива…» объединяет архив с местным набором: одинаковые снимки не дублируются, разметка разных людей хранится отдельно, более новая разметка того же человека заменяет старую.
• Переносите архивом, а не папкой: архив проверяет целостность каждого файла.
"""
