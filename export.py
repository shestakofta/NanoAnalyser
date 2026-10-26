# -*- coding: utf-8 -*-
"""
export.py — сохранение результатов: картинки с разметкой, графики и отчёты Excel.

Отчёты (версия 4.1):
  краткий — лист «Итог» (с графиками на том же листе) и лист «Сводка»;
  полный  — «Итог», «Сводка», «Графики», «Модифицированные», «НФ», «Агломераты»,
            «Все первичные частицы», «Параметры».
Галочка «учитывать оценку глубины» (settings["use_3d"]) убирает из отчётов все величины с оценкой глубины.
"""
import io
import os
import re
import numpy as np
import nanoproc as npc
import charts


# ---------------------------------------------------------------------------- картинки
def save_image(res, path, scale=1, show_circles=True, show_contour=True, lw=1, lw_c=1):
    """Сохраняет разметку одного снимка в указанный файл (формат — по расширению)."""
    img = npc.draw_overlay(res, show_circles=show_circles, show_contour=show_contour, scale=scale, lw=lw, lw_c=lw_c)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        img.save(path, quality=92)
    else:
        img.save(path)
    return path


def save_images(results, folder, scale=1, show_circles=True, show_contour=True, fmt="png", lw=1, lw_c=1):
    """Сохраняет разметку всех снимков в папку. Возвращает список путей."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    for r in results:
        base = os.path.splitext(r["name"])[0]
        paths.append(save_image(r, os.path.join(folder, base + "_razmetka." + fmt), scale, show_circles, show_contour, lw, lw_c))
    return paths


def _safe(name):
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip() or "grafik"


def save_all_charts(results, folder, use_3d=True, dpi=200):
    """Все графики: общие по выборке и по каждому снимку — каждый рисунок целиком и каждый график отдельно (PNG)."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    figs = [("vse_snimki", charts.overall_figure(results, use_3d=use_3d))]
    figs += [(os.path.splitext(r["name"])[0], charts.image_figure(r)) for r in results]
    for base, fig in figs:
        charts.ensure_canvas(fig)
        p = os.path.join(folder, _safe(base) + "_grafiki.png")
        charts.save_figure(fig, p, dpi=dpi); paths.append(p)
        for i, (ax, title) in enumerate(charts.axes_list(fig), 1):
            p = os.path.join(folder, "{}_{}_{}.png".format(_safe(base), i, _safe(title.split(",")[0])[:40]))
            charts.save_axes(fig, ax, p, dpi=dpi); paths.append(p)
    return paths


# ---------------------------------------------------------------------------- Excel
def _num(v):
    """Значение для ячейки: numpy -> float/int, NaN -> пусто."""
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


class _Book:
    def __init__(self):
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        self.wb = Workbook(); self.Font, self.PatternFill, self.Alignment = Font, PatternFill, Alignment
        self.bold = Font(bold=True); self.fill = PatternFill("solid", fgColor="E8E2F4")
        self._first = True

    def sheet(self, title):
        if self._first:
            ws = self.wb.active; ws.title = title; self._first = False
        else:
            ws = self.wb.create_sheet(title)
        return ws

    def header(self, ws, cols, row=1):
        for j, c in enumerate(cols, 1):
            x = ws.cell(row, j, c); x.font = self.bold; x.fill = self.fill
            x.alignment = self.Alignment(wrap_text=True, vertical="center", horizontal="center")
        ws.row_dimensions[row].height = 45; ws.freeze_panes = ws.cell(row + 1, 1)
        for j in range(1, len(cols) + 1):
            ws.column_dimensions[ws.cell(row, j).column_letter].width = 14


def _mode(summary):
    return summary.get("mode", "full")


def _summary_cols(mode, use_3d):
    """(заголовок, ключ) колонок листа «Сводка» с учётом режима и оценки глубины."""
    c = [("Снимок", "name"), ("нм/пикс", "s"), ("Источник масштаба", "s_src"), ("Контраст", "contrast"),
         ("Шумоподавление", "denoise")]
    if mode != "no_mod":
        c += [("Модифицированных, шт", "n_mod"), ("из них обрезано краем", "n_mod_cut"),
              ("d модиф. среднее, нм", "d_mod_mean"), ("d модиф. медиана, нм", "d_mod_med"), ("d модиф. мода, нм", "d_mod_mode"),
              ("d модиф. ст.откл., нм", "d_mod_sd"), ("d модиф. мин, нм", "d_mod_min"), ("d модиф. макс, нм", "d_mod_max"),
              ("Масса модифицированных, фг", "mass_mod_fg")]
    if mode != "only_mod":
        c += [("Первичных НЧ размечено (круги), шт", "n_prim_vis")]
        if use_3d:
            c += [("Первичных с учётом оценки глубины, шт", "n_prim_3d")]
        c += [("d первичных среднее, нм", "d_prim_mean"), ("d первичных медиана, нм", "d_prim_med"),
              ("d первичных мода, нм", "d_prim_mode"), ("d20 первичных, нм", "d20"), ("d30 первичных, нм", "d30"),
              ("Источник d20/d30", "d_src")]
        if mode == "full":
            c += [("d всех частиц среднее, нм", "d_all_mean"), ("d всех частиц медиана, нм", "d_all_med"),
                  ("d всех частиц мода, нм", "d_all_mode")]
        c += [("Площадь агломератов, нм²", "agg_area_nm2"), ("в т.ч. добавлено вручную, нм²", "agg_area_manual_nm2"),
              ("Масса агломератов плоская (с НФ), фг", "mass_agg_flat_fg")]
        if use_3d:
            c += [("Масса агломератов с оценкой глубины (с НФ), фг", "mass_agg_3d_fg")]
    if mode == "full":
        c += [("Доля модиф. массы (плоская)", "frac_flat")]
        if use_3d:
            c += [("Доля модиф. массы (с оценкой глубины)", "frac_3d")]
    if mode != "only_mod":
        c += [("НФ, шт", "n_irr"), ("в т.ч. масса НФ (эллипсоид), фг", "mass_irr_fg"), ("Агломератов (связных кусков), шт", "n_agg")]
    return c


def _sheet_summary(B, results, summary, use_3d):
    ws = B.sheet("Сводка")
    cols = _summary_cols(_mode(summary), use_3d)
    B.header(ws, [c[0] for c in cols])
    for i, r in enumerate(results, 2):
        st = r["stats"]
        for j, (_, k) in enumerate(cols, 1):
            v = _num(st.get(k)); c = ws.cell(i, j, v)
            if k.startswith("frac"): c.number_format = "0.0%"
            elif isinstance(v, float): c.number_format = "0.00"
    ws.column_dimensions["A"].width = 28; ws.column_dimensions["C"].width = 24
    return ws


def _itog_rows(summary, use_3d):
    mode = _mode(summary)
    rows = [("Снимков обработано", summary.get("n_images"), "")]
    if mode != "no_mod":
        rows += [("Модифицированных частиц, всего", summary.get("n_mod"), ""),
                 ("Масса модифицированных, всего, фг", summary.get("mass_mod_fg"), "0.00")]
    if mode != "only_mod":
        rows += [("Первичных НЧ размечено, всего", summary.get("n_prim_vis"), "")]
        if use_3d:
            rows += [("Первичных с учётом оценки глубины, всего", summary.get("n_prim_3d"), "0")]
        rows += [("Масса агломератов плоская (с НФ), всего, фг", summary.get("mass_agg_flat_fg"), "0.00")]
        if use_3d:
            rows += [("Масса агломератов с оценкой глубины (с НФ), всего, фг", summary.get("mass_agg_3d_fg"), "0.00")]
        rows += [("НФ (неправильной формы), всего, шт", summary.get("n_irr"), ""),
                 ("   в т.ч. масса НФ (эллипсоид × плотная упаковка), фг", summary.get("mass_irr_fg"), "0.00")]
    if mode == "full":
        keys = [("flat", "плоская оценка агломератов")] + ([("3d", "с оценкой глубины")] if use_3d else [])
        for key, lab in keys:
            d = summary.get(key, {})
            rows += [("Доля модиф. массы, среднее по снимкам — " + lab, d.get("mean"), "0.0%"),
                     ("   стандартная ошибка среднего", d.get("se"), "0.0%"),
                     ("   медиана по снимкам", d.get("median"), "0.0%"),
                     ("   по суммарной массе всех снимков", d.get("pooled"), "0.0%")]
    return rows


def _size_rows(summary):
    """Таблица «среднее / медиана / мода» по всей выборке."""
    mode = _mode(summary); S = summary.get("size", {})
    groups = []
    if mode != "only_mod":
        groups.append(("Первичные", "prim"))
    if mode != "no_mod":
        groups.append(("Модифицированные", "mod"))
    if mode == "full":
        groups.append(("Все частицы", "all"))
    return [(lab, S.get(k, {})) for lab, k in groups]


def _sheet_itog(B, results, summary, use_3d, with_charts=False):
    it = B.sheet("Итог")
    rows = _itog_rows(summary, use_3d)
    for i, (a, b, fmt) in enumerate(rows, 1):
        it.cell(i, 1, a); c = it.cell(i, 2, _num(b))
        if fmt: c.number_format = fmt
    r0 = len(rows) + 2
    it.cell(r0, 1, "Распределение по размерам (вся выборка)").font = B.bold
    for j, h in enumerate(["Частицы", "Число", "Среднее, нм", "Медиана, нм", "Мода, нм"], 1):
        x = it.cell(r0 + 1, j, h); x.font = B.bold; x.fill = B.fill
    for i, (lab, d) in enumerate(_size_rows(summary), r0 + 2):
        it.cell(i, 1, lab); it.cell(i, 2, _num(d.get("n")))
        for j, k in enumerate(("mean", "median", "mode"), 3):
            c = it.cell(i, j, _num(d.get(k))); c.number_format = "0.00"
    r1 = r0 + 2 + len(_size_rows(summary)) + 1
    it.cell(r1, 1, "1 фг = 1e-15 г. Среднее по снимкам — каждый снимок учитывается одинаково; по суммарной массе — массы всех "
                   "снимков сложены, затем поделены. Мода — максимум сглаженной гистограммы (ядерная оценка плотности).")
    if not use_3d:
        it.cell(r1 + 1, 1, "Оценка глубины (частицы, скрытые в агломератах) не учитывается.")
    it.column_dimensions["A"].width = 62
    for col in "BCDE":
        it.column_dimensions[col].width = 15
    if with_charts and results:
        _put_figure(it, charts.overall_figure(results, use_3d=use_3d), "G2")
    return it


def _put_figure(ws, fig, anchor, width_px=900):
    from openpyxl.drawing.image import Image as XLImage
    charts.ensure_canvas(fig)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, facecolor="white"); buf.seek(0)
    img = XLImage(buf)
    k = width_px / img.width; img.width = int(img.width * k); img.height = int(img.height * k)
    ws.add_image(img, anchor)


def _sheet_charts(B, results, use_3d):
    ws = B.sheet("Графики")
    ws.cell(1, 1, "Графики по всей выборке").font = B.bold
    _put_figure(ws, charts.overall_figure(results, use_3d=use_3d), "A3", width_px=1000)
    return ws


def _sheet_mods(B, results):
    pt = B.sheet("Модифицированные")
    B.header(pt, ["Снимок", "№ на рисунке", "d экв., нм", "Feret макс., нм", "Малая ось, нм", "Выпуклость",
                  "Круглость", "Контраст", "Обрезана краем", "Оси эллипсоида, нм (a × b × c)", "Масса, фг (сплошной эллипсоид)",
                  "Класс задан", "Основание"])
    i = 2
    for r in results:
        npc.number_objects(r)
        for o in r["objects"]:
            if o["cls"] != "mod":
                continue
            ax = o.get("axes_nm")
            axt = "{:.1f} × {:.1f} × {:.1f}".format(*ax) if (ax and not o["cut"]) else ("видимая часть шара" if o["cut"] else "")
            how = "вручную" if o["cls"] != o["auto_cls"] else ("автоматически (по крупному кругу)" if o.get("from_circle")
                                                             else "автоматически")
            vals = [r["name"], o.get("num"), o["deq"], o["feret"], o["minor"], o["solidity"], o["circularity"],
                    o["contrast"], "да" if o["cut"] else "нет", axt, o.get("mass_g", 0) * 1e15, how, o.get("why", "")]
            for j, v in enumerate(vals, 1):
                v = _num(v); c = pt.cell(i, j, v)
                if isinstance(v, float): c.number_format = "0.00"
            i += 1
    pt.column_dimensions["A"].width = 28; pt.column_dimensions["M"].width = 26


def _sheet_nf(B, results):
    nf = B.sheet("НФ (неправ. форма)")
    B.header(nf, ["Снимок", "d экв., нм", "Выпуклость", "Контраст", "Оси эллипсоида, нм (a × b × c)",
                  "Масса, фг (эллипсоид × плотная упаковка; входит в агломераты)", "Обрезано краем", "Основание"])
    i = 2
    for r in results:
        for o in r["objects"]:
            if o["cls"] == "irr":
                ax = o.get("nf_axes_nm")
                axt = "{:.1f} × {:.1f} × {:.1f}".format(*ax) if ax else ""
                for j, v in enumerate([r["name"], o["deq"], o["solidity"], o["contrast"], axt, o.get("mass_g", 0) * 1e15,
                                       "да" if o.get("cut") else "нет", o.get("why", "")], 1):
                    c = nf.cell(i, j, _num(v))
                    if isinstance(_num(v), float): c.number_format = "0.00"
                i += 1
    nf.column_dimensions["A"].width = 28


def _sheet_aggs(B, results, use_3d):
    ag = B.sheet("Агломераты")
    cols = ["Снимок", "№ агломерата", "Тип", "Площадь проекции, нм²", "Обрезан краем", "Первичных видимых (круги)"]
    cols += (["Первичных с оценкой глубины"] if use_3d else []) + ["Масса плоская, фг"] + (["Масса с оценкой глубины, фг"] if use_3d else [])
    B.header(ag, cols)
    i = 2
    for r in results:
        for c in r["comps"]:
            vals = [r["name"], c["id"], {"circle": "вручную: частица", "region": "вручную: область"}.get(
                c.get("kind"), "автоматически"), c["area_nm2"], "да" if c["cut"] else "нет", len(c["d_circles"])]
            vals += ([c.get("n3d", 0.0)] if use_3d else []) + [c.get("mass_flat_g", 0.0) * 1e15]
            vals += [c.get("mass_3d_g", 0.0) * 1e15] if use_3d else []
            for j, v in enumerate(vals, 1):
                x = ag.cell(i, j, _num(v))
                if isinstance(_num(v), float): x.number_format = "0.00"
            i += 1
    ag.column_dimensions["A"].width = 28


def _sheet_prims(B, results):
    ws = B.sheet("Все первичные частицы")
    B.header(ws, ["№ снимка", "Снимок", "№ частицы на снимке", "Диаметр, нм", "№ агломерата", "x, пикс", "y, пикс"])
    row = 2
    for k, r in enumerate(results, 1):
        circ = sorted(r["circles"], key=lambda c: (c[3], c[0], c[1]))
        for i, (y, x, d, comp) in enumerate(circ, 1):
            ws.cell(row, 1, k); ws.cell(row, 2, r["name"]); ws.cell(row, 3, i)
            c = ws.cell(row, 4, float(d)); c.number_format = "0.0"
            ws.cell(row, 5, int(comp))
            c = ws.cell(row, 6, float(x)); c.number_format = "0"
            c = ws.cell(row, 7, float(y)); c.number_format = "0"
            row += 1
    ws.column_dimensions["B"].width = 28


def _sheet_params(B, settings):
    pr = B.sheet("Параметры")
    pr.cell(1, 1, "Параметр").font = B.bold; pr.cell(1, 2, "Значение").font = B.bold
    for i, (k, v) in enumerate(sorted(settings.items()), 2):
        pr.cell(i, 1, k)
        pr.cell(i, 2, _num(v) if isinstance(v, (int, float, str, np.number)) or v is None else str(v))
    pr.column_dimensions["A"].width = 30; pr.column_dimensions["B"].width = 30


def save_excel_short(results, summary, settings, path):
    """Краткий отчёт: «Итог» с графиками на том же листе и «Сводка»."""
    use_3d = bool(settings.get("use_3d", True))
    B = _Book()
    _sheet_itog(B, results, summary, use_3d, with_charts=True)
    _sheet_summary(B, results, summary, use_3d)
    B.wb.save(path)


def save_excel_full(results, summary, settings, path):
    """Полный отчёт: итог, сводка, графики, модифицированные, НФ, агломераты, все первичные частицы, параметры."""
    use_3d = bool(settings.get("use_3d", True))
    mode = _mode(summary)
    B = _Book()
    _sheet_itog(B, results, summary, use_3d)
    _sheet_summary(B, results, summary, use_3d)
    _sheet_charts(B, results, use_3d)
    if mode != "no_mod":
        _sheet_mods(B, results)
    if mode != "only_mod":
        _sheet_nf(B, results)
        _sheet_aggs(B, results, use_3d)
        _sheet_prims(B, results)
    _sheet_params(B, settings)
    B.wb.save(path)


save_excel = save_excel_full      # пакетный режим и старые вызовы
