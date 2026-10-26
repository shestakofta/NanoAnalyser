# -*- coding: utf-8 -*-
"""
app.py — окно программы NPA (NanoParticle Analyzer): анализ наночастиц на ПЭМ-снимках.

Запуск:  python app.py        (или двойной щелчок по собранному NanoAnalyzer.exe)

Как устроено окно:
  слева  — сворачиваемые разделы: параметры, масштаб, обработка, редактирование, сохранение;
  справа — вкладки: "Снимок", "Сводка", "Графики", "Графики <снимок>", "Журнал".

Настройки контраста, шумоподавления и масштаба хранятся ОТДЕЛЬНО ДЛЯ КАЖДОГО СНИМКА.
Кнопки «Применить к снимку / ко всем» только запоминают значения; обработка запускается
кнопкой «Начать обработку» — и обрабатывает только те снимки, у которых что-то изменилось.

Тяжёлая обработка идёт в отдельных процессах (по одному снимку на ядро процессора).
Версия 4: разметка первичных частиц методом v8 (пакет circ8).
Версия 4.1: новая классификация крупных тел (classify2), блок «Измерение», отчёты с графиками.
Версия 4.1.ml: блок «Разметка для обучения (ML)» — набор размеченных снимков для машинного обучения (модуль ml_annot).
"""
import os
import sys
import json
import time
import queue
import threading
import traceback
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw

import nanoproc as npc
import export
import ml_annot

APP_TITLE = "NPA — анализ наночастиц на ПЭМ-снимках"
APP_VERSION = "4.1.ml"
POLY_TOOLS = ("del_poly", "del_spline", "add_poly", "add_spline")
NIGERIA_URL = "https://ru.wikipedia.org/wiki/%D0%A4%D0%BB%D0%B0%D0%B3_%D0%9D%D0%B8%D0%B3%D0%B5%D1%80%D0%B8%D0%B8"
ELEMENTS = list(npc.MATERIALS.keys()) + ["Другое"]
IMG_TYPES = [("Снимки (TIFF, JPEG, PNG, BMP)", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
             ("TIFF", "*.tif *.tiff"), ("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png"), ("Все файлы", "*.*")]

# параметры, от которых зависит результат обработки (их изменение требует повторной обработки снимка)
PROC_KEYS = ("contrast", "denoise", "d_cut", "nm_per_px", "bar_nm", "bar_px", "edge", "halo_nm", "prim_dmin", "prim_dmax",
             "k_hyst_lo", "k_hyst_hi", "film", "mode")
ADV_KEYS = ("edge", "halo_nm", "prim_dmin", "prim_dmax", "ka", "alpha", "nf_packing", "k_hyst_lo", "k_hyst_hi", "film")
CONTRAST_RANGE = (0.10, 0.95)
DENOISE_RANGE = (0.2, 5.0)
ZOOM_MAX = 8.0
ZOOM_MIN = 0.05


def fmt_dur(sec):
    """Длительность: «12 с», «3 мин 05 с», «1 ч 02 мин»."""
    sec = int(round(sec))
    if sec < 60:
        return "{} с".format(sec)
    if sec < 3600:
        return "{} мин {:02d} с".format(sec // 60, sec % 60)
    return "{} ч {:02d} мин".format(sec // 3600, (sec % 3600) // 60)

# Цвета графиков и правок совпадают с цветами разметки на снимке
C_MOD = "#1baf7a"; C_MOD_LIGHT = "#9fdcc2"; C_PRIM = "#eb6834"; C_AGG = "#2a78d6"
C_DEL = "#e53935"; C_WARN = "#c25e00"
C_ML = "#00d8ff"          # отметки блока ML (сомнение «?», зона разметки, точки круга)
ACCENT = "#6a2fb0"        # фиолетовый — цвет значка NPA
DISABLED = "#b9b9b9"


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


class Tip:
    """Всплывающая подсказка при наведении мыши."""
    def __init__(self, widget, text):
        self.w, self.text, self.tw = widget, text, None
        widget.bind("<Enter>", self.show, add="+"); widget.bind("<Leave>", self.hide, add="+")

    def show(self, e=None):
        if self.tw:
            return
        self.tw = tw = tk.Toplevel(self.w); tw.wm_overrideredirect(True)
        tw.wm_geometry("+{}+{}".format(self.w.winfo_rootx() + 10, self.w.winfo_rooty() + self.w.winfo_height() + 4))
        tk.Label(tw, text=self.text, background="#ffffe8", relief="solid", borderwidth=1, justify="left",
                 padx=6, pady=3).pack()

    def hide(self, e=None):
        if self.tw:
            self.tw.destroy(); self.tw = None


class Section:
    """Сворачиваемый раздел левой панели."""
    def __init__(self, parent, title, bg, expanded=True, enabled=True, on_toggle=None):
        self.title, self.bg, self.enabled, self.on_toggle = title, bg, enabled, on_toggle
        self.frame = ttk.Frame(parent); self.frame.pack(fill=tk.X, pady=(8, 0))
        self.hdr = tk.Label(self.frame, anchor="w", bg=bg, font=("TkDefaultFont", 10, "bold"), cursor="hand2")
        self.hdr.pack(fill=tk.X)
        ttk.Separator(self.frame).pack(fill=tk.X, pady=(1, 0))
        self.body = ttk.Frame(self.frame, padding=(6, 6, 2, 2))
        self.hdr.bind("<Button-1>", lambda e: self.toggle())
        self.expanded = False
        self.expand() if expanded else self._render()

    def _render(self):
        self.hdr.config(text=("▾  " if self.expanded else "▸  ") + self.title,
                        fg=ACCENT if self.enabled else DISABLED, cursor="hand2" if self.enabled else "arrow")

    def expand(self):
        if not self.enabled:
            return
        if not self.expanded:
            self.body.pack(fill=tk.X)
        self.expanded = True; self._render()

    def collapse(self):
        if self.expanded:
            self.body.pack_forget()
        self.expanded = False; self._render()

    def toggle(self):
        if not self.enabled:
            return
        self.collapse() if self.expanded else self.expand()
        if self.on_toggle:
            self.on_toggle()

    def set_enabled(self, flag):
        self.enabled = flag
        if not flag:
            self.collapse()
        self._render()


def make_icon(kind, size, color="#333333"):
    """Пиктограммы инструментов: прямоугольник, круг (от центра к краю), произвольный контур."""
    S = size * 4
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    w = max(4, S // 12); m = S // 6
    if kind == "rect":
        d.rectangle([m, m + S // 10, S - m, S - m - S // 10], outline=color, width=w)
    elif kind == "circle":
        d.ellipse([m, m, S - m, S - m], outline=color, width=w)
        c = S / 2; d.ellipse([c - w, c - w, c + w, c + w], fill=color)
        d.line([c, c, S - m - w / 2, c], fill=color, width=max(2, w // 2))
    elif kind == "lasso":
        t = np.linspace(0, 2 * np.pi, 80)
        rr = S * (0.33 + 0.06 * np.sin(3 * t) + 0.04 * np.cos(5 * t))
        pts = [(S / 2 + r * np.cos(a), S / 2 + r * np.sin(a)) for r, a in zip(rr, t)]
        d.line(pts + [pts[0]], fill=color, width=w, joint="curve")
    elif kind == "m_pp":                        # измерение: отрезок между двумя точками
        d.line([m, S - m, S - m, m], fill=color, width=w)
        for (x, y) in ((m, S - m), (S - m, m)):
            d.ellipse([x - w * 1.3, y - w * 1.3, x + w * 1.3, y + w * 1.3], fill=color)
    elif kind == "m_ll":                        # измерение: две параллельные линии и расстояние между ними
        d.line([m, S * 0.28, S - m, S * 0.28], fill=color, width=w)
        d.line([m, S * 0.72, S - m, S * 0.72], fill=color, width=w)
        c = S / 2; a = S * 0.08
        d.line([c, S * 0.32, c, S * 0.68], fill=color, width=max(2, w // 2))
        d.polygon([(c, S * 0.32), (c - a, S * 0.32 + a * 1.4), (c + a, S * 0.32 + a * 1.4)], fill=color)
        d.polygon([(c, S * 0.68), (c - a, S * 0.68 - a * 1.4), (c + a, S * 0.68 - a * 1.4)], fill=color)
    elif kind == "m_circ":                      # измерение: диаметр круга
        d.ellipse([m, m, S - m, S - m], outline=color, width=w)
        d.line([m + w, S / 2, S - m - w, S / 2], fill=color, width=max(2, w // 2))
    elif kind in ("poly", "spline"):            # замкнутая ломаная / замкнутый сплайн по точкам
        pts = [(S * 0.22, S * 0.30), (S * 0.62, S * 0.16), (S * 0.84, S * 0.52), (S * 0.56, S * 0.84), (S * 0.18, S * 0.70)]
        if kind == "poly":
            d.line(pts + [pts[0]], fill=color, width=w, joint="curve")
        else:
            P = np.array(pts + pts[:3]); seg = []
            for i in range(len(pts)):
                p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
                for t in np.linspace(0, 1, 12, endpoint=False):
                    seg.append(tuple(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                                            (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)))
            d.line(seg + [seg[0]], fill=color, width=w, joint="curve")
            pts = [tuple(p) for p in P[1:len(pts) + 1]]
        for (x, y) in pts:
            d.rectangle([x - w, y - w, x + w, y + w], fill=color)
    elif kind == "cycle":                       # смена класса: круговые стрелки
        d.arc([m, m, S - m, S - m], 200, 340, fill=color, width=w)
        d.arc([m, m, S - m, S - m], 20, 160, fill=color, width=w)
        a = S * 0.11
        x1, y1 = S / 2 + (S / 2 - m) * np.cos(np.radians(340)), S / 2 + (S / 2 - m) * np.sin(np.radians(340))
        d.polygon([(x1 + a, y1 - a * 0.2), (x1 - a * 0.9, y1 - a * 0.6), (x1 + a * 0.1, y1 + a * 1.1)], fill=color)
        x2, y2 = S / 2 + (S / 2 - m) * np.cos(np.radians(160)), S / 2 + (S / 2 - m) * np.sin(np.radians(160))
        d.polygon([(x2 - a, y2 + a * 0.2), (x2 + a * 0.9, y2 + a * 0.6), (x2 - a * 0.1, y2 - a * 1.1)], fill=color)
    elif kind == "info":                        # классический значок информации: синий круг с белой «i»
        d.ellipse([m // 2, m // 2, S - m // 2, S - m // 2], fill=color)
        c = S / 2; w2 = max(3, S // 9)
        d.ellipse([c - w2 * 0.8, S * 0.22 - w2 * 0.8, c + w2 * 0.8, S * 0.22 + w2 * 0.8], fill="white")
        d.rectangle([c - w2 * 0.65, S * 0.38, c + w2 * 0.65, S * 0.78], fill="white")
    elif kind == "c3":                          # круг по трём точкам края
        d.ellipse([m, m, S - m, S - m], outline=color, width=max(2, w // 2))
        c, R = S / 2, S / 2 - m
        for a in (200, 300, 60):
            x, y = c + R * np.cos(np.radians(a)), c + R * np.sin(np.radians(a))
            d.ellipse([x - w * 1.4, y - w * 1.4, x + w * 1.4, y + w * 1.4], fill=color)
    elif kind == "doubt":                       # круг с вопросительным знаком — «сомнительная частица»
        d.ellipse([m, m, S - m, S - m], outline=color, width=max(2, w // 2))
        c = S / 2; q = S * 0.13
        d.arc([c - q, S * 0.27, c + q, S * 0.27 + 2 * q], 180, 90, fill=color, width=w)
        d.line([c, S * 0.27 + 2 * q, c, S * 0.60], fill=color, width=w)
        d.ellipse([c - w * 0.8, S * 0.70 - w * 0.8, c + w * 0.8, S * 0.70 + w * 0.8], fill=color)
    elif kind == "region":                      # пунктирный прямоугольник — «зона разметки»
        x0, y0, x1, y1 = m, m + S // 10, S - m, S - m - S // 10
        k = S // 9
        for t in range(int(x0), int(x1), 2 * k):
            d.line([t, y0, min(t + k, x1), y0], fill=color, width=w); d.line([t, y1, min(t + k, x1), y1], fill=color, width=w)
        for t in range(int(y0), int(y1), 2 * k):
            d.line([x0, t, x0, min(t + k, y1)], fill=color, width=w); d.line([x1, t, x1, min(t + k, y1)], fill=color, width=w)
    elif kind == "point":                       # перечёркнутый кружок — «удалить элемент»
        d.ellipse([m, m, S - m, S - m], outline=color, width=w)
        k = (S - 2 * m) * 0.5 / np.sqrt(2); c = S / 2
        d.line([c - k, c + k, c + k, c - k], fill=color, width=w)
    return ImageTk.PhotoImage(im.resize((size, size), Image.LANCZOS))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.ui = max(1.0, self.winfo_fpixels("1i") / 96.0)          # масштаб экрана: 1.0 = 100 %, 2.0 = 200 %
        self.mpl_dpi = 90 * (1.0 if sys.platform == "win32" else self.ui)   # matplotlib в Windows масштабирует сам
        self._set_icon()
        self._styles()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("{}x{}".format(min(self.px(1440), int(sw * 0.95)), min(self.px(920), int(sh * 0.9))))
        self.minsize(min(self.px(1100), int(sw * 0.8)), min(self.px(700), int(sh * 0.7)))

        self.files = []            # выбранные файлы
        self.results = {}          # путь -> результат обработки
        self.iset = {}             # путь -> настройки снимка (контраст, шумоподавление, масштаб)
        self.meta = {}             # путь -> масштаб из метаданных (или None)
        self.pending = {}          # путь -> список неприменённых правок
        self.summary = {}
        self.queue = queue.Queue()
        self.zoom = None
        self.current = None
        self.preview = None        # результат кнопки "Проверить"
        self.busy = False
        self.last_dir = ""
        self._cache = {}
        self._loading = False      # идёт подстановка значений снимка в поля (не считать это изменением)
        self._drag = None          # текущее действие мышью на снимке
        self._poly = None          # строящаяся замкнутая ломаная / сплайн (вершины в координатах снимка)
        self.barline = {}          # путь -> линия линейки (y, x0, x1), заданная вручную (пиксели снимка)
        self._sb = {}              # путь -> найденная автоматически линейка (или None)
        self._img = None           # снимок, который сейчас показан (PIL)
        self._z = 1.0; self._ox = self._oy = 0.0
        self._render_job = None
        self._t_batch = None
        self.adv = {k: npc.DEFAULTS[k] for k in ADV_KEYS}
        self.mlcfg = ml_annot.load_config()   # разметчик и папка набора для обучения (хранятся между запусками)
        self._c3 = None            # точки строящегося «круга по трём точкам» (блок ML)
        self._build_left()
        self._build_right()
        # Ctrl+Z — в любой раскладке (в русской это Ctrl+Я; keycode 90 — клавиша Z в Windows)
        self.bind_all("<Control-KeyPress>", self._ctrl_key)
        self.bind_all("<Return>", self._enter_key); self.bind_all("<KP_Enter>", self._enter_key)
        self.hist = {}             # путь -> снимки результата до последних применённых правок (до 3 шагов)
        self.after(200, self._poll_queue)

    def px(self, n):
        return int(round(n * self.ui))

    def _ctrl_key(self, e):
        ks = (e.keysym or "").lower()
        kc = getattr(e, "keycode", 0)
        if ks in ("plus", "equal", "kp_add") or (sys.platform == "win32" and kc in (187, 107)):
            self.set_zoom(1.25); return "break"
        if ks in ("minus", "underscore", "kp_subtract") or (sys.platform == "win32" and kc in (189, 109)):
            self.set_zoom(0.8); return "break"
        if ks in ("z", "cyrillic_ya") or (sys.platform == "win32" and e.keycode == 90):
            w = self.focus_get()
            if isinstance(w, (tk.Entry, ttk.Entry, ttk.Spinbox, tk.Text)) and w not in (self.txt_log, self.txt_meas):
                return                       # в полях ввода Ctrl+Z работает как обычно
            if self._meas_active():
                self.meas_undo()             # в режиме измерения — отменить последнее измерение
            else:
                self.edit_undo()
            return "break"

    def _enter_key(self, e):
        """Enter в режиме редактирования — применить изменения."""
        try:
            w = self.focus_get()
        except Exception:
            return
        if w is None or w.winfo_toplevel() is not self:
            return                                   # Enter в диалоговых окнах — их собственное действие
        if isinstance(w, (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox, tk.Text)):
            return
        if self._poly:
            self._poly_finish()
            return "break"
        if self.results.get(self.current) is not None and self.pending.get(self.current) and not self.busy:
            self.edit_apply()
            return "break"

    HIST_STEPS = 3

    def _push_hist(self, path):
        r = self.results.get(path)
        if r is None:
            return
        h = self.hist.setdefault(path, [])
        h.append(npc.snapshot(r))
        del h[:-self.HIST_STEPS]

    def _set_icon(self):
        try:
            from icon_data import ICONS
            self._icons = [tk.PhotoImage(data=ICONS[s]) for s in (256, 48, 32, 16)]
            self.iconphoto(True, *self._icons)
        except Exception:
            pass

    def _styles(self):
        st = ttk.Style(self)
        st.configure("Treeview", rowheight=self.px(22))
        st.configure("Accent.TButton", font=("TkDefaultFont", 11, "bold"), foreground=ACCENT)
        st.configure("Hint.TLabel", foreground="#777777", font=("TkDefaultFont", 8))
        st.configure("Warn.TLabel", foreground=C_WARN, font=("TkDefaultFont", 8))
        st.configure("Info.TLabel", foreground="#333333", font=("TkDefaultFont", 8))
        st.configure("Status.TLabel", foreground="#333333")
        self.panel_bg = st.lookup("TFrame", "background") or "#d9d9d9"

    # ================================================================== левая панель
    def _build_left(self):
        outer = ttk.Frame(self)
        outer.pack(side=tk.LEFT, fill=tk.Y)
        cv = tk.Canvas(outer, width=self.px(345), highlightthickness=0, bg=self.panel_bg)
        sb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y); cv.pack(side=tk.LEFT, fill=tk.Y)
        left = ttk.Frame(cv, padding=8)
        cv.create_window(0, 0, window=left, anchor="nw")
        left.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all"), width=left.winfo_reqwidth()))

        def _wheel(e):
            if cv.winfo_height() < left.winfo_reqheight():
                cv.yview_scroll(-1 if (getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4) else 1, "units")
        for w in (cv, left):
            w.bind("<Enter>", lambda e: (cv.bind_all("<MouseWheel>", _wheel), cv.bind_all("<Button-4>", _wheel),
                                         cv.bind_all("<Button-5>", _wheel)))
            w.bind("<Leave>", lambda e: (cv.unbind_all("<MouseWheel>"), cv.unbind_all("<Button-4>"),
                                         cv.unbind_all("<Button-5>")))
        wrap = self.px(305)
        bg = self.panel_bg

        # --- Файлы (список с полосой прокрутки)
        ttk.Button(left, text="Выбрать файлы", command=self.choose_files).pack(fill=tk.X)
        lf = ttk.Frame(left); lf.pack(fill=tk.X, pady=(6, 2))
        self.lst = tk.Listbox(lf, height=8, width=40, exportselection=False, activestyle="none", selectbackground=ACCENT)
        lsb = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self.lst.yview)
        self.lst.configure(yscrollcommand=lsb.set)
        lsb.pack(side=tk.RIGHT, fill=tk.Y); self.lst.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lst.bind("<<ListboxSelect>>", lambda e: self.show_selected())

        def _lwheel(e):     # колесо над списком прокручивает список, а не всю панель
            self.lst.yview_scroll(-1 if (getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4) else 1, "units")
            return "break"
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.lst.bind(ev, _lwheel)
        self.lbl_files = ttk.Label(left, text="Файлы не выбраны", style="Hint.TLabel")
        self.lbl_files.pack(anchor="w")
        ttk.Label(left, text="✓ обработан   ○ ждёт обработки   ✎ есть неприменённые правки",
                  style="Hint.TLabel").pack(anchor="w")

        # --- Параметры обработки
        self.sec_par = Section(left, "Параметры обработки", bg, expanded=True)
        box = self.sec_par.body
        self.v_contrast = tk.DoubleVar(value=npc.DEFAULTS["contrast"])
        self._slider(box, "Отсечка контраста", self.v_contrast, CONTRAST_RANGE[0], CONTRAST_RANGE[1], 0.01, "{:.2f}",
                     "Отношение границы вещества и контраста одиночной частицы. Меньше — контуры шире.", wrap, click=0.1)
        self.v_denoise = tk.DoubleVar(value=npc.DEFAULTS["denoise"])
        self._slider(box, "Шумоподавление", self.v_denoise, DENOISE_RANGE[0], DENOISE_RANGE[1], 0.1, "{:.1f}",
                     "Сила сглаживания шума фона. Больше — меньше ложных пятен на фоне, "
                     "но мелкие частицы размываются.", wrap, click=1.0)
        ttk.Button(box, text="Проверить на выбранном снимке", command=self.check_params).pack(fill=tk.X, pady=(6, 2))
        row = ttk.Frame(box); row.pack(fill=tk.X)
        ttk.Button(row, text="Применить к снимку", command=lambda: self.apply_params(False)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="Применить ко всем", command=lambda: self.apply_params(True)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.lbl_pstate = ttk.Label(box, text="", style="Info.TLabel", wraplength=wrap, justify="left")
        self.lbl_pstate.pack(anchor="w", pady=(2, 0))
        ttk.Label(box, style="Hint.TLabel", wraplength=wrap, justify="left",
                  text="«Проверить» за 1–2 с рисует контур вещества (без частиц). «Применить» запоминает значения "
                       "для выбранного снимка или для всех.").pack(anchor="w")
        for v in (self.v_contrast, self.v_denoise):
            v.trace_add("write", lambda *a: self._update_states())

        ttk.Separator(box).pack(fill=tk.X, pady=8)
        self.v_nomod = tk.BooleanVar(value=False); self.v_onlymod = tk.BooleanVar(value=False)
        cb1 = ttk.Checkbutton(box, text="Не искать модифицированные", variable=self.v_nomod,
                              command=lambda: self._mode_changed("nomod"))
        cb1.pack(anchor="w")
        Tip(cb1, "Только подсчёт первичных частиц, их распределение и масса агломератов. "
                 "Плотные однородные тела крупнее отсечки при этом считаются НФ, пёстрые — агломератом")
        cb2 = ttk.Checkbutton(box, text="Выделить только модифицированные", variable=self.v_onlymod,
                              command=lambda: self._mode_changed("onlymod"))
        cb2.pack(anchor="w", pady=(2, 2))
        Tip(cb2, "Только модифицированные частицы: круги первичных частиц не размечаются")
        self.v_3d = tk.BooleanVar(value=True)
        cb3 = ttk.Checkbutton(box, text="Учитывать оценку глубины", variable=self.v_3d, command=self._3d_changed)
        cb3.pack(anchor="w", pady=(2, 6))
        Tip(cb3, "Оценка числа и массы первичных частиц, скрытых в глубине агломератов (Köylü et al., 1995).\n"
                 "Выключено — во всех таблицах, графиках и отчётах только видимые частицы. Повторная обработка не нужна")
        row = ttk.Frame(box); row.pack(fill=tk.X)
        ttk.Label(row, text="Отсечка первичные / модифиц., нм").pack(side=tk.LEFT)
        self.v_dcut = tk.DoubleVar(value=25.0)
        self.sp_dcut = ttk.Spinbox(row, from_=5, to=500, increment=1, textvariable=self.v_dcut, width=7)
        self.sp_dcut.pack(side=tk.RIGHT)
        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(row, text="Материал").pack(side=tk.LEFT)
        self.v_elem = tk.StringVar(value="W")
        cb = ttk.Combobox(row, values=ELEMENTS, textvariable=self.v_elem, state="readonly", width=10)
        cb.pack(side=tk.RIGHT)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_element())
        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row, text="Плотность, г/см³").pack(side=tk.LEFT)
        self.v_density = tk.StringVar(value=str(npc.MATERIALS["W"]))
        self.ent_density = ttk.Entry(row, textvariable=self.v_density, width=10, state="readonly")
        self.ent_density.pack(side=tk.RIGHT)
        self.v_density.trace_add("write", lambda *a: self._density_changed())

        # --- Масштаб
        self.sec_scale = Section(left, "Масштаб", bg, expanded=False)
        sc = self.sec_scale.body
        self.v_smode = tk.StringVar(value="meta")
        self.rb_smeta = ttk.Radiobutton(sc, text="Метаданные снимка", variable=self.v_smode, value="meta")
        self.rb_smeta.pack(anchor="w")
        self.v_bar = tk.StringVar(value=""); self.v_nmpx = tk.StringVar(value="")
        for txt, val, var in (("Линейка, нм", "bar", self.v_bar), ("нм на пиксель", "nmpx", self.v_nmpx)):
            row = ttk.Frame(sc); row.pack(fill=tk.X, pady=(4, 0))
            ttk.Radiobutton(row, text=txt, variable=self.v_smode, value=val).pack(side=tk.LEFT)
            ent = ttk.Entry(row, textvariable=var, width=10); ent.pack(side=tk.RIGHT)
            ent.bind("<FocusIn>", lambda e, val=val: self.v_smode.set(val))
        row = ttk.Frame(sc); row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(row, text="Применить к снимку", command=lambda: self.apply_scale(False)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="Применить ко всем", command=lambda: self.apply_scale(True)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.lbl_sstate = ttk.Label(sc, text="", style="Info.TLabel", wraplength=wrap, justify="left")
        self.lbl_sstate.pack(anchor="w", pady=(2, 0))
        ttk.Label(sc, style="Hint.TLabel", wraplength=wrap, justify="left",
                  text="Красная линия на снимке — распознанная линейка; её концы и саму линию можно перетащить мышью."
                  ).pack(anchor="w")
        for v in (self.v_nmpx, self.v_bar, self.v_smode):
            v.trace_add("write", lambda *a: (self._update_states(), self._draw_bar()))

        # --- Обработка
        self.sec_run = Section(left, "Обработка", bg, expanded=False)
        run = self.sec_run.body
        self.v_cut = tk.BooleanVar(value=True)
        ttk.Checkbutton(run, text="Учитывать обрезанные краем объекты (видимую часть)",
                        variable=self.v_cut, command=self._density_changed).pack(anchor="w")
        ncpu = os.cpu_count() or 1
        self.v_cores = tk.DoubleVar(value=max(1, ncpu - 1))
        if ncpu > 1:
            self._slider(run, "Использовать ядер для обработки", self.v_cores, 1, ncpu, 1, "{:.0f}",
                         "Всего ядер: {}. Одно лучше оставить свободным, чтобы компьютер не тормозил.".format(ncpu), wrap)
        ttk.Button(run, text="Расширенные настройки", command=self.open_advanced).pack(fill=tk.X, pady=(8, 0))

        # --- Кнопка запуска (не сворачивается)
        go = ttk.Frame(left); go.pack(fill=tk.X, pady=(12, 0))
        self.btn_single = ttk.Button(go, text="Обработать отдельный снимок", command=lambda: self.start(single=True))
        self.btn_single.pack(fill=tk.X, pady=(0, 4))
        Tip(self.btn_single, "Обработать только выбранный в списке снимок с его настройками")
        self.btn_start = ttk.Button(go, text="▶  Начать обработку", style="Accent.TButton", command=self.start)
        self.btn_start.pack(fill=tk.X, ipady=self.px(3))
        # место под полосу прогресса зарезервировано; сама полоса видна только во время работы
        self.pb_slot = ttk.Frame(go, height=self.px(14)); self.pb_slot.pack(fill=tk.X, pady=(4, 0))
        self.pb_slot.pack_propagate(False)
        self.pb = ttk.Progressbar(self.pb_slot, mode="determinate")
        self.lbl_status = ttk.Label(go, text="Готово к работе", style="Status.TLabel", wraplength=wrap)
        self.lbl_status.pack(anchor="w", pady=(2, 0))

        # --- Измерение (работает и до обработки)
        self.sec_meas = Section(left, "Измерение", bg, expanded=False, on_toggle=self._meas_toggled)
        self._build_meas(self.sec_meas.body, wrap)

        # --- Редактировать
        self.sec_edit = Section(left, "Редактировать", bg, expanded=False, enabled=False)
        self._build_edit(self.sec_edit.body, wrap)

        # --- Разметка для обучения (ML): набор снимков для машинного обучения
        self.sec_ml = Section(left, "Разметка для обучения (ML)", bg, expanded=False, on_toggle=self._ml_refresh)
        self._build_ml(self.sec_ml.body, wrap)

        # --- Сохранение
        self.sec_save = Section(left, "Сохранение", bg, expanded=True)
        out = self.sec_save.body
        ttk.Button(out, text="Разметка выбранного снимка", command=self.save_current_image).pack(fill=tk.X)
        ttk.Button(out, text="Разметка всех снимков", command=self.save_all_images).pack(fill=tk.X, pady=(2, 0))
        self.v_x2 = tk.BooleanVar(value=False)
        ttk.Checkbutton(out, text="Сохранять картинки увеличенными в 2 раза", variable=self.v_x2).pack(anchor="w", pady=(2, 6))
        ttk.Button(out, text="Все графики (картинками)", command=self.save_all_charts).pack(fill=tk.X, pady=(0, 6))
        row = ttk.Frame(out); row.pack(fill=tk.X)
        b1 = ttk.Button(row, text="Краткий отчёт Excel", command=lambda: self.save_excel("short"))
        b1.pack(side=tk.LEFT, expand=True, fill=tk.X)
        Tip(b1, "Лист «Итог» с графиками и лист «Сводка»")
        b2 = ttk.Button(row, text="Полный отчёт Excel", command=lambda: self.save_excel("full"))
        b2.pack(side=tk.LEFT, expand=True, fill=tk.X)
        Tip(b2, "Итог, сводка, графики, модифицированные, НФ, агломераты, все первичные частицы, параметры")
        self.btn_open = ttk.Button(out, text="Открыть папку с сохранённым", command=self.open_last_dir, state="disabled")
        self.btn_open.pack(fill=tk.X, pady=(6, 6))
        row = ttk.Frame(out); row.pack(fill=tk.X)
        ttk.Button(row, text="Сохранить настройки", command=self.save_settings).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="Загрузить настройки", command=self.load_settings).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.lbl_saved = ttk.Label(out, text="", style="Info.TLabel", wraplength=wrap, justify="left")
        self.lbl_saved.pack(anchor="w", pady=(6, 0))

        # --- О программе (в самом низу)
        ttk.Separator(left).pack(fill=tk.X, pady=(14, 6))
        ttk.Button(left, text="О программе", command=self.show_about).pack(fill=tk.X, pady=(0, 6))

    def _build_edit(self, box, wrap):
        isz = self.px(22)
        self._ico = {k: make_icon(k, isz) for k in ("rect", "circle", "lasso", "poly", "spline", "cycle")}
        self._ico_del = {k: make_icon(k, isz, color=C_DEL) for k in ("rect", "circle", "lasso", "point", "poly", "spline")}
        self.v_tool = tk.StringVar(value="toggle")
        self.v_tool.trace_add("write", lambda *a: self._tool_changed())
        row = ttk.Frame(box); row.pack(fill=tk.X)
        ttk.Label(row, text="Смена класса:").pack(side=tk.LEFT)
        rb = ttk.Radiobutton(row, image=self._ico["cycle"], style="Toolbutton", variable=self.v_tool, value="toggle")
        rb.pack(side=tk.RIGHT, padx=1)
        Tip(rb, "Клик по объекту: модифицированная → агломерат → НФ")
        ttk.Label(box, style="Hint.TLabel", wraplength=wrap, justify="left",
                  text="Кликните по сложной области, чтобы автоматически изменить класс объекта "
                       "(модифицированная → агломерат → НФ).").pack(anchor="w")

        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row, text="Удалить элемент:").pack(side=tk.LEFT)
        rb = ttk.Radiobutton(row, image=self._ico_del["point"], style="Toolbutton", variable=self.v_tool, value="del_one")
        rb.pack(side=tk.RIGHT, padx=1)
        Tip(rb, "Клик по кружку или частице. Повторный клик снимает отметку")

        poly_tip = "Клик — вершина; двойной клик или клик по первой вершине — замкнуть; Esc — отмена"
        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row, text="Удалить в области:").pack(side=tk.LEFT)
        for k, val, tip in (("spline", "del_spline", "Замкнутый сплайн. " + poly_tip),
                            ("poly", "del_poly", "Замкнутая ломаная. " + poly_tip),
                            ("lasso", "del_lasso", "Произвольный контур: обведите мышью"),
                            ("circle", "del_circle", "Круг: от центра к краю"),
                            ("rect", "del_rect", "Прямоугольник: по диагонали")):
            rb = ttk.Radiobutton(row, image=self._ico_del[k], style="Toolbutton", variable=self.v_tool, value=val)
            rb.pack(side=tk.RIGHT, padx=1); Tip(rb, tip)

        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row, text="Добавить элемент:").pack(side=tk.LEFT)
        for k, val, tip in (("spline", "add_spline", "Замкнутый сплайн. " + poly_tip),
                            ("poly", "add_poly", "Замкнутая ломаная. " + poly_tip),
                            ("lasso", "add_contour", "Произвольный контур: обведите мышью"),
                            ("circle", "add_circle", "Круг: от центра к краю")):
            rb = ttk.Radiobutton(row, image=self._ico[k], style="Toolbutton", variable=self.v_tool, value=val)
            rb.pack(side=tk.RIGHT, padx=1); Tip(rb, tip)
        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(row, text="   тип:").pack(side=tk.LEFT)
        self.v_addcls = tk.StringVar(value="prim")
        ttk.Radiobutton(row, text="первичная", variable=self.v_addcls, value="prim").pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="модифицированная", variable=self.v_addcls, value="mod").pack(side=tk.LEFT)
        ttk.Label(box, style="Hint.TLabel", wraplength=wrap, justify="left",
                  text="Первичная: круг — одна частица; контур, ломаная или сплайн — область агломерата, программа "
                       "заполнит её кругами. Пока изменения не применены, новый элемент можно перетащить мышью, "
                       "а круг за край — увеличить или уменьшить.").pack(anchor="w")

        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row, text="Толщина контура на снимке").pack(side=tk.LEFT)
        self.v_lw = tk.IntVar(value=1)
        ttk.Spinbox(row, from_=1, to=5, increment=1, textvariable=self.v_lw, width=4, state="readonly",
                    command=self._lw_changed).pack(side=tk.RIGHT)
        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(row, text="Толщина линии кругов (всех)").pack(side=tk.LEFT)
        self.v_lwc = tk.IntVar(value=1)
        ttk.Spinbox(row, from_=1, to=5, increment=1, textvariable=self.v_lwc, width=4, state="readonly",
                    command=self._lw_changed).pack(side=tk.RIGHT)

        self.lbl_pending = ttk.Label(box, text="Неприменённых изменений: 0", style="Info.TLabel")
        self.lbl_pending.pack(anchor="w", pady=(6, 2))
        row = ttk.Frame(box); row.pack(fill=tk.X)
        ttk.Button(row, text="↶ Отменить шаг", command=self.edit_undo).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="Сбросить", command=self.edit_reset).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(box, text="Применить изменения (Enter)", style="Accent.TButton", command=self.edit_apply).pack(fill=tk.X, pady=(4, 0))


    # ------------------------------------------------------------------ измерение
    def _build_meas(self, box, wrap):
        self.meas = {}             # путь -> список измерений (координаты снимка)
        isz = self.px(22)
        self._ico_meas = {k: make_icon(k, isz) for k in ("m_pp", "m_ll", "m_circ")}
        self.v_meas_on = tk.BooleanVar(value=False)
        self.btn_meas = tk.Checkbutton(box, text="Измерить", variable=self.v_meas_on, indicatoron=False,
                                       relief="raised", offrelief="raised", bd=2, selectcolor="#ded2f1",
                                       font=("TkDefaultFont", 10, "bold"), fg=ACCENT, activeforeground=ACCENT,
                                       command=self._meas_toggled, cursor="hand2")
        self.btn_meas.pack(fill=tk.X, pady=(0, 6), ipady=self.px(2))
        self.v_mtool = tk.StringVar(value="pp")
        row = ttk.Frame(box); row.pack(fill=tk.X)
        ttk.Label(row, text="Режим:").pack(side=tk.LEFT)
        for k, val, tip in (("m_circ", "circ", "Диаметр круга"),
                            ("m_ll", "ll", "Расстояние между параллельными линиями"),
                            ("m_pp", "pp", "Расстояние между точками")):
            rb = ttk.Radiobutton(row, image=self._ico_meas[k], variable=self.v_mtool, value=val, style="Toolbutton")
            rb.pack(side=tk.RIGHT, padx=1); Tip(rb, tip)
        self.v_mtool.trace_add("write", lambda *a: self._meas_toggled())
        self.v_mhover = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Диаметр и масса частицы под курсором", variable=self.v_mhover,
                        command=lambda: self.canvas.delete("hover")).pack(anchor="w", pady=(4, 0))
        self.txt_meas = tk.Text(box, height=5, width=36, wrap="word", relief="flat", padx=4, pady=2,
                                font=("TkDefaultFont", 9))
        self.txt_meas.pack(fill=tk.X, pady=(4, 0)); self.txt_meas.configure(state="disabled")
        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(row, text="Убрать последнее", command=self.meas_undo).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="Очистить", command=self.meas_clear).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.lbl_mscale = ttk.Label(box, text="", style="Info.TLabel", wraplength=wrap, justify="left")
        self.lbl_mscale.pack(anchor="w", pady=(4, 0))

    def _meas_active(self):
        return self.sec_meas.expanded and bool(self.v_meas_on.get()) and self.current is not None

    def _meas_toggled(self):
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("hover")
        if self._meas_active():
            self.lbl_tool.config(text={"pp": "Измерение: точка–точка", "ll": "Измерение: линия–линия",
                                       "circ": "Измерение: круг"}[self.v_mtool.get()])
            self.canvas.config(cursor="tcross")
        else:
            self._tool_changed()
        self._meas_refresh()

    def _scale_of(self, path):
        """Масштаб снимка (нм/пикс, источник) — после обработки из результата, до неё — из настроек."""
        r = self.results.get(path)
        if r is not None:
            return r["s"], r["s_src"]
        st = self.iset.get(path, {})
        if st.get("nm_per_px"):
            return float(st["nm_per_px"]), "введён вручную"
        if st.get("bar_nm"):
            y, x0, x1, src = self._bar_line(path)
            if abs(x1 - x0) > 0:
                return st["bar_nm"] / abs(x1 - x0), "по линейке"
        if self.meta.get(path):
            return self.meta[path], "из метаданных"
        return None, None

    def _fmt_len(self, px, path):
        s, _ = self._scale_of(path)
        return "{:.1f} нм".format(px * s) if s else "{:.1f} пикс".format(px)

    @staticmethod
    def _line_dist(L1, L2):
        """Среднее расстояние от концов каждой линии до прямой другой (для почти параллельных — расстояние между ними)."""
        def d(p, L):
            (x0, y0), (x1, y1) = L; ln = np.hypot(x1 - x0, y1 - y0)
            if ln < 1e-9:
                return np.hypot(p[0] - x0, p[1] - y0)
            return abs((x1 - x0) * (y0 - p[1]) - (x0 - p[0]) * (y1 - y0)) / ln
        v = [d(L2[0], L1), d(L2[1], L1), d(L1[0], L2), d(L1[1], L2)]
        a1 = np.degrees(np.arctan2(L1[1][1] - L1[0][1], L1[1][0] - L1[0][0]))
        a2 = np.degrees(np.arctan2(L2[1][1] - L2[0][1], L2[1][0] - L2[0][0]))
        ang = abs((a1 - a2 + 90) % 180 - 90)
        return float(np.mean(v)), ang

    def _meas_end(self, x0, y0, x, y, e):
        """Конец отрезка измерения: вторая линия режима «линия–линия» строго параллельна первой;
        с Shift — только горизонталь, вертикаль или диагонали 45°."""
        lst = self.meas.get(self.current, [])
        t = self.v_mtool.get()
        if t == "ll" and lst and lst[-1]["kind"] == "ll" and len(lst[-1]["lines"]) == 1:
            (a0, a1) = lst[-1]["lines"][0]
            ux, uy = a1[0] - a0[0], a1[1] - a0[1]; ln = np.hypot(ux, uy) or 1.0; ux, uy = ux / ln, uy / ln
            p = (x - x0) * ux + (y - y0) * uy
            return x0 + p * ux, y0 + p * uy
        if t in ("pp", "ll") and (getattr(e, "state", 0) & 0x0001):
            ang = np.arctan2(y - y0, x - x0); L = np.hypot(x - x0, y - y0)
            ang = np.round(ang / (np.pi / 4)) * (np.pi / 4)
            return x0 + L * np.cos(ang), y0 + L * np.sin(ang)
        return x, y

    def _meas_value(self, m, path):
        if m["kind"] == "pp":
            (x0, y0), (x1, y1) = m["pts"]
            return "{} (точка–точка)".format(self._fmt_len(np.hypot(x1 - x0, y1 - y0), path))
        if m["kind"] == "circ":
            return "диаметр {} (круг)".format(self._fmt_len(2 * m["r"], path))
        if len(m["lines"]) < 2:
            return "первая линия проведена — проведите вторую"
        dist, ang = self._line_dist(*m["lines"])
        t = "{} (линия–линия)".format(self._fmt_len(dist, path))
        return t + ("; угол между линиями {:.0f}°".format(ang) if ang > 3 else "")

    def _meas_refresh(self):
        """Список измерений в разделе и подписи на снимке."""
        p = self.current
        s, src = self._scale_of(p) if p else (None, None)
        self.lbl_mscale.config(text=("Масштаб: {:.4f} нм/пикс ({})".format(s, src) if s else
                                     "Масштаб не задан — расстояния в пикселях.") if p else "")
        lst = self.meas.get(p, []) if p else []
        self.txt_meas.configure(state="normal"); self.txt_meas.delete("1.0", tk.END)
        self.txt_meas.insert("1.0", "\n".join("{}. {}".format(i, self._meas_value(m, p)) for i, m in enumerate(lst, 1))
                             or "Измерений нет.")
        self.txt_meas.configure(state="disabled")
        self._draw_meas()

    def _draw_meas(self):
        cv = self.canvas
        cv.delete("meas")
        p = self.current
        if p is None or self._img is None:
            return
        col = "#ffd400"
        for i, m in enumerate(self.meas.get(p, []), 1):
            if m["kind"] == "pp":
                (x0, y0), (x1, y1) = m["pts"]
                A = self._cv(x0, y0); Bp = self._cv(x1, y1)
                cv.create_line(*A, *Bp, fill=col, width=2, tags="meas")
                for X, Y in (A, Bp):
                    cv.create_oval(X - 3, Y - 3, X + 3, Y + 3, outline=col, width=2, tags="meas")
                lx, ly = (A[0] + Bp[0]) / 2, (A[1] + Bp[1]) / 2
            elif m["kind"] == "circ":
                X, Y = self._cv(m["cx"], m["cy"]); RR = m["r"] * self._z
                cv.create_oval(X - RR, Y - RR, X + RR, Y + RR, outline=col, width=2, tags="meas")
                cv.create_line(X - RR, Y, X + RR, Y, fill=col, dash=(4, 3), tags="meas")
                lx, ly = X, Y - RR - self.px(10)
            else:
                for L in m["lines"]:
                    cv.create_line(*self._cv(*L[0]), *self._cv(*L[1]), fill=col, width=2, tags="meas")
                if len(m["lines"]) == 2:
                    (a0, a1), L2 = m["lines"]
                    mx, my = (L2[0][0] + L2[1][0]) / 2, (L2[0][1] + L2[1][1]) / 2
                    dx, dy = a1[0] - a0[0], a1[1] - a0[1]; ln2 = dx * dx + dy * dy or 1e-9
                    t = ((mx - a0[0]) * dx + (my - a0[1]) * dy) / ln2
                    fx, fy = a0[0] + t * dx, a0[1] + t * dy
                    cv.create_line(*self._cv(mx, my), *self._cv(fx, fy), fill=col, dash=(4, 3), tags="meas")
                    lx, ly = self._cv((mx + fx) / 2, (my + fy) / 2)
                else:
                    lx, ly = self._cv(*m["lines"][0][1])
            txt = "{}: {}".format(i, self._meas_value(m, p).split(" (")[0].split(";")[0])
            self._label(lx, ly, txt, col, "meas")

    def _label(self, x, y, txt, col, tag, anchor="center"):
        cv = self.canvas
        t = cv.create_text(x, y, text=txt, fill=col, font=("TkDefaultFont", 9, "bold"), tags=tag, anchor=anchor)
        bb = cv.bbox(t)
        if bb:
            r = cv.create_rectangle(bb[0] - 3, bb[1] - 1, bb[2] + 3, bb[3] + 1, fill="#202020", outline="", tags=tag)
            cv.tag_lower(r, t)

    def meas_undo(self):
        lst = self.meas.get(self.current)
        if lst:
            if lst[-1]["kind"] == "ll" and len(lst[-1]["lines"]) == 2:
                lst[-1]["lines"].pop()              # сначала убирается вторая линия
            else:
                lst.pop()
            self._meas_refresh()

    def meas_clear(self):
        if self.meas.get(self.current):
            self.meas[self.current] = []; self._meas_refresh()

    def _hover(self, e):
        """Подсказка с диаметром и массой частицы под курсором (когда открыт раздел «Измерение»)."""
        cv = self.canvas
        cv.delete("hover")
        if self._poly:
            self._poly_draw(*self._img_xy(e))
            return
        if not (self._meas_active() and self.v_mhover.get()) or self._drag:
            return
        r = self.results.get(self.current)
        if r is None or not (self.v_contour.get() or self.v_circles.get()):
            return
        x, y = self._img_xy(e)
        H, W = r["shape"]
        if not (0 <= x < W and 0 <= y < H):
            return
        g = self.global_params(quiet=True)
        dens = g["density"] if g else 0.0
        txt = None
        oid = int(r["objmap"][int(y), int(x)])
        o = r["objects"][oid - 1] if oid else None
        if o is not None and o["cls"] in ("mod", "irr") and self.v_contour.get():
            mg = o.get("mass_g")
            mt = "{:.3g} фг".format(mg * 1e15) if mg else "—"
            if o["cls"] == "mod":
                txt = "Модифицированная №{}: d = {:.1f} нм, масса {}{}".format(
                    o.get("num") or "", o["deq"], mt, " (обрезана краем)" if o["cut"] else "")
            else:
                txt = "НФ: d экв. = {:.1f} нм, масса {} (эллипсоид × упаковка)".format(o["deq"], mt)
        elif self.v_circles.get():
            c = npc.find_circle(r, y, x)
            if c is not None:
                m = dens * 1e-21 * np.pi * c[2] ** 3 / 6
                txt = "Первичная: d = {:.1f} нм, масса {:.3g} фг".format(c[2], m * 1e15)
        if txt:
            X, Y = cv.canvasx(e.x), cv.canvasy(e.y)
            self._label(X + self.px(14), Y - self.px(14), txt, "#ffffff", "hover", anchor="w")
            # подпись целиком в окне
            bb = cv.bbox("hover")
            if bb and bb[2] > cv.canvasx(cv.winfo_width()):
                cv.move("hover", -(bb[2] - bb[0]) - self.px(28), 0)

    def _lw_changed(self):
        self._cache.clear(); self.redraw(keep_view=True)

    def _mode_changed(self, which):
        if which == "nomod" and self.v_nomod.get():
            self.v_onlymod.set(False)
        if which == "onlymod" and self.v_onlymod.get():
            self.v_nomod.set(False)
        self.refresh_list()
        m = self.mode()
        self.status({"full": "Режим: модифицированные и первичные частицы.",
                     "no_mod": "Режим: без поиска модифицированных — только первичные частицы и агломераты.",
                     "only_mod": "Режим: только модифицированные частицы (круги первичных не размечаются)."}[m]
                    + " Снимки нужно обработать заново.")

    def _3d_changed(self):
        on = bool(self.v_3d.get())
        self._set_columns()
        if self.results and not self.busy:
            self.recompute_all(); self.redraw(keep_view=True)
        self.status("Оценка глубины " + ("учитывается." if on else "не учитывается: только видимые частицы."))

    def mode(self):
        return "no_mod" if self.v_nomod.get() else ("only_mod" if self.v_onlymod.get() else "full")

    def _slider(self, parent, title, var, lo, hi, step, fmt, hint, wrap, click=None):
        """Ползунок. click — шаг при щелчке по линии ползунка (слева/справа от бегунка)."""
        fr = ttk.Frame(parent); fr.pack(fill=tk.X, pady=(4, 0))
        top = ttk.Frame(fr); top.pack(fill=tk.X)
        ttk.Label(top, text=title).pack(side=tk.LEFT)
        val = ttk.Label(top, text=fmt.format(var.get()), width=6, anchor="e")
        val.pack(side=tk.RIGHT)

        def on_move(v):
            x = round(float(v) / step) * step
            if not _same(round(var.get(), 6), round(x, 6)):
                var.set(x)
        var.trace_add("write", lambda *a: val.config(text=fmt.format(var.get())))
        sc = ttk.Scale(fr, from_=lo, to=hi, variable=var, command=on_move)
        sc.pack(fill=tk.X)
        if click:
            def on_click(e):
                if "slider" in str(sc.identify(e.x, e.y)):
                    return None                              # бегунок — обычное перетаскивание
                try:
                    xs = sc.coords(var.get())[0]
                except Exception:
                    return None
                v = var.get() + (click if e.x > xs else -click)
                v = min(hi, max(lo, round(round(v / step) * step, 6)))
                var.set(v)
                return "break"
            sc.bind("<Button-1>", on_click)
        ttk.Label(fr, text=hint, style="Hint.TLabel", wraplength=wrap, justify="left").pack(anchor="w")

    def _on_element(self):
        el = self.v_elem.get()
        self.ent_density.config(state="normal")
        if el == "Другое":
            self.v_density.set(""); self.ent_density.focus_set()
        else:
            self.v_density.set(str(npc.MATERIALS[el])); self.ent_density.config(state="readonly")

    # ================================================================== правая панель
    def _build_right(self):
        nb = ttk.Notebook(self)
        nb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=8)
        self.nb = nb

        t1 = ttk.Frame(nb); nb.add(t1, text="  Снимок  ")
        bar = ttk.Frame(t1, padding=(0, 2)); bar.pack(fill=tk.X)
        ttk.Button(bar, text="−", width=3, command=lambda: self.set_zoom(0.8)).pack(side=tk.LEFT)
        self.v_zoom = tk.DoubleVar(value=0.0)          # десятичный логарифм масштаба показа
        self._zsync = False
        self.sc_zoom = ttk.Scale(bar, from_=np.log10(ZOOM_MIN), to=np.log10(ZOOM_MAX), variable=self.v_zoom,
                                 length=self.px(150), command=self._zoom_slider)
        self.sc_zoom.pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="+", width=3, command=lambda: self.set_zoom(1.25)).pack(side=tk.LEFT)
        self.lbl_zoom = ttk.Label(bar, text="100 %", width=6, anchor="e")
        self.lbl_zoom.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(bar, text="Вписать в окно", command=lambda: self.set_zoom(None)).pack(side=tk.LEFT, padx=(0, 12))
        self.v_contour = tk.BooleanVar(value=True); self.v_circles = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Показать контур", variable=self.v_contour, command=self._markup_toggled).pack(side=tk.LEFT)
        ttk.Checkbutton(bar, text="Показать круги", variable=self.v_circles,
                        command=self._markup_toggled).pack(side=tk.LEFT, padx=(10, 0))
        self.lbl_tool = ttk.Label(bar, text="", style="Hint.TLabel")
        self.lbl_tool.pack(side=tk.RIGHT)
        cf = ttk.Frame(t1); cf.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(cf, background="#2b2b2b", highlightthickness=0)
        xs = ttk.Scrollbar(cf, orient=tk.HORIZONTAL, command=lambda *a: self._scroll("x", *a))
        ys = ttk.Scrollbar(cf, orient=tk.VERTICAL, command=lambda *a: self._scroll("y", *a))
        self.canvas.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
        ys.pack(side=tk.RIGHT, fill=tk.Y); xs.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        cv = self.canvas
        cv.bind("<ButtonPress-1>", self.on_press)
        cv.bind("<B1-Motion>", self.on_motion)
        cv.bind("<ButtonRelease-1>", self.on_release)
        cv.bind("<Motion>", self._hover)
        cv.bind("<Double-Button-1>", self._double_press)
        self.bind_all("<Escape>", lambda e: (self._poly_cancel(), self._c3_cancel()))
        cv.bind("<Leave>", lambda e: self.canvas.delete("hover"))
        for b_ in ("<ButtonPress-2>", "<ButtonPress-3>"):
            cv.bind(b_, self._pan_start)
        for b_ in ("<B2-Motion>", "<B3-Motion>"):
            cv.bind(b_, self._pan_move)
        cv.bind("<MouseWheel>", self._wheel)                       # Windows / macOS
        cv.bind("<Shift-MouseWheel>", lambda e: self._wheel(e, horizontal=True))
        cv.bind("<Control-MouseWheel>", lambda e: self._wheel(e, zoom=True))   # и щипок на тачпаде
        cv.bind("<Button-4>", self._wheel); cv.bind("<Button-5>", self._wheel)  # Linux
        cv.bind("<Control-Button-4>", lambda e: self._wheel(e, zoom=True))
        cv.bind("<Control-Button-5>", lambda e: self._wheel(e, zoom=True))
        cv.bind("<Shift-Button-4>", lambda e: self._wheel(e, horizontal=True))
        cv.bind("<Shift-Button-5>", lambda e: self._wheel(e, horizontal=True))
        cv.bind("<Configure>", lambda e: self.redraw(keep_view=True))
        self.lbl_img = ttk.Label(t1, text="", style="Status.TLabel")
        self.lbl_img.pack(anchor="w", pady=(4, 0))
        leg = ttk.Frame(t1); leg.pack(anchor="w", pady=(2, 2))
        for i, (col, txt) in enumerate((("#00c800", "модифицированные (с номером)"), ("#2878ff", "контур агломератов"),
                                        ("#ff8c00", "первичные частицы (круги)"),
                                        ("#e600c8", "НФ — неправильной формы (масса включается в агломераты)"))):
            tk.Label(leg, text="■", fg=col, font=("TkDefaultFont", 12)).grid(row=i // 2, column=2 * (i % 2), sticky="w")
            ttk.Label(leg, text=txt + "      ").grid(row=i // 2, column=2 * (i % 2) + 1, sticky="w")

        # --- Сводка: заголовки в две строки, ширина колонок подстраивается под окно (без прокрутки вбок)
        t2 = ttk.Frame(nb); nb.add(t2, text="  Сводка  ")
        tf = ttk.Frame(t2); tf.pack(fill=tk.BOTH, expand=True)
        self.sum_hdr = tk.Frame(tf, height=self.px(40), bg="#ece8f4")
        self.sum_hdr.pack(fill=tk.X); self.sum_hdr.pack_propagate(False)
        self.sum_hdr_lbls = []
        trf = ttk.Frame(tf); trf.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(trf, columns=("name",), show="", height=14)
        tsb = ttk.Scrollbar(trf, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set)
        tsb.pack(side=tk.RIGHT, fill=tk.Y); self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._sum_rows = []
        self._set_columns()
        self.tree.bind("<<TreeviewSelect>>", self._tree_select)
        self.tree.bind("<Configure>", lambda e: self._fit_columns())
        self.txt_sum = tk.Text(t2, height=17, wrap="word", relief="flat", padx=8, pady=6)
        self.txt_sum.pack(fill=tk.X)

        self.t3 = ttk.Frame(nb); nb.add(self.t3, text="  Графики  ")
        self.t5 = ttk.Frame(nb); nb.add(self.t5, text="  Графики снимка  ")
        for t in (self.t3, self.t5):
            ttk.Label(t, text="Графики появятся после обработки.", style="Hint.TLabel").pack(pady=20)

        t4 = ttk.Frame(nb); nb.add(t4, text="  Журнал  ")
        lf = ttk.Frame(t4); lf.pack(fill=tk.BOTH, expand=True)
        self.txt_log = tk.Text(lf, wrap="word", relief="flat", padx=8, pady=6)
        lsb = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=lsb.set)
        lsb.pack(side=tk.RIGHT, fill=tk.Y); self.txt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        from tkinter import font as tkfont
        tw = tkfont.nametofont("TkDefaultFont").measure("00:00:00") + self.px(14)
        self.txt_log.configure(tabs=(tw,), spacing1=1)
        self.txt_log.tag_configure("time", foreground="#8a8a8a")
        self.txt_log.tag_configure("msg", lmargin2=tw)

    SUM_ALL = (("name", "Снимок", False), ("s", "нм/\nпикс", False), ("contrast", "Конт-\nраст", False),
               ("denoise", "Шумо-\nподавл.", False), ("n_mod", "Модиф.,\nшт", False), ("d_mod", "d модиф.\nсред., нм", False),
               ("m_mod", "Масса\nмодиф., фг", False), ("n_vis", "Первичн.\nразмечено", False),
               ("n_3d", "Первичн. с\nглубиной", True), ("d20", "d20\nперв., нм", False), ("m_flat", "Агломер.\nплоск., фг", False),
               ("m_3d", "Агломер. с\nглубиной, фг", True), ("f_flat", "Доля\nмодиф. плоск.", False),
               ("f_3d", "Доля модиф.\nс глубиной", True), ("n_irr", "НФ,\nшт", False))

    def _set_columns(self):
        """Колонки сводки (без колонок с оценкой глубины, если она выключена)."""
        use3d = bool(self.v_3d.get()) if hasattr(self, "v_3d") else True
        cols = [c for c in self.SUM_ALL if use3d or not c[2]]
        self.sum_cols = tuple(c[0] for c in cols); self.sum_heads = tuple(c[1] for c in cols)
        for l in self.sum_hdr_lbls:
            l.destroy()
        self.sum_hdr_lbls = [tk.Label(self.sum_hdr, text=h, bg="#ece8f4", font=("TkDefaultFont", 9, "bold"), justify="center",
                                      anchor="center", relief="flat", borderwidth=0) for h in self.sum_heads]
        self.tree.configure(columns=self.sum_cols)
        for c in self.sum_cols:
            self.tree.column(c, anchor="e" if c != "name" else "w", stretch=False, minwidth=20)
        self._sum_rows = []

    def _fit_columns(self):
        """Ширины колонок сводки: по содержимому и заголовкам, растянуто на всю ширину таблицы."""
        from tkinter import font as tkfont
        W = self.tree.winfo_width()
        if W < 50:
            return
        f = tkfont.nametofont("TkDefaultFont"); fb = tkfont.Font(font=("TkDefaultFont", 9, "bold"))
        need = []
        for j, (c, h) in enumerate(zip(self.sum_cols, self.sum_heads)):
            w = max(fb.measure(line) for line in h.split("\n")) + self.px(10)
            for row in self._sum_rows:
                w = max(w, f.measure(str(row[j])) + self.px(10))
            if c == "name":
                w = min(w, self.px(240))
            need.append(w)
        tot = sum(need)
        k = W / tot if tot else 1.0
        x = 0
        hh = self.sum_hdr.winfo_height() or self.px(40)
        for j, c in enumerate(self.sum_cols):
            w = int(need[j] * k) if j < len(self.sum_cols) - 1 else max(10, W - x)
            self.tree.column(c, width=w)
            self.sum_hdr_lbls[j].place(x=x, y=0, width=w, height=hh)
            x += w

    # ================================================================== общие мелочи
    def log(self, msg):
        self.txt_log.insert(tk.END, time.strftime("%H:%M:%S"), ("time",))
        self.txt_log.insert(tk.END, "\t" + msg + "\n", ("msg",)); self.txt_log.see(tk.END)

    def status(self, msg):
        self.lbl_status.config(text=msg)

    def _begin(self, msg, indeterminate=True):
        self.busy = True
        self.btn_start.config(state="disabled"); self.btn_single.config(state="disabled")
        self.pb.pack(fill=tk.X, expand=True)
        if indeterminate:
            self.pb.config(mode="indeterminate"); self.pb.start(15)
        self.status(msg)

    def _end(self):
        self.busy = False
        self.btn_start.config(state="normal"); self.btn_single.config(state="normal")
        self.pb.stop(); self.pb.config(mode="determinate", value=0)
        self.pb.pack_forget()

    def _need_current(self):
        if self.current is None:
            messagebox.showinfo(APP_TITLE, "Выберите снимок в списке слева.", parent=self)
            return False
        return True

    # ================================================================== файлы и настройки снимков
    def choose_files(self):
        paths = filedialog.askopenfilenames(parent=self, title="Выберите снимки", filetypes=IMG_TYPES)
        if not paths:
            return
        self.files = list(paths)
        self.results = {}; self._cache = {}; self.preview = None; self.pending = {}
        self.barline = {}; self._sb = {}; self.hist = {}; self.meas = {}
        self.iset = {f: dict(contrast=npc.DEFAULTS["contrast"], denoise=npc.DEFAULTS["denoise"],
                             nm_per_px=None, bar_nm=None) for f in self.files}
        self.meta = {}
        for f in self.files:
            try:
                self.meta[f] = npc.metadata_scale(f)
            except Exception:
                self.meta[f] = None
        nometa = sum(1 for f in self.files if not self.meta[f])
        self.lbl_files.config(text="Выбрано файлов: {}{}".format(
            len(self.files), "   (без масштаба в метаданных: {})".format(nometa) if nometa else ""))
        self.log("Выбрано файлов: {}; без масштаба в метаданных: {}".format(len(self.files), nometa))
        self.sec_edit.set_enabled(False); self.sec_par.expand()
        self.current = None
        self.refresh_list()
        self.lst.selection_set(0); self.show(self.files[0])

    def refresh_list(self):
        sel = self.lst.curselection()
        top = self.lst.yview()[0] if self.files else 0
        self.lst.delete(0, tk.END)
        g = self.global_params(quiet=True)
        for f in self.files:
            if f in self.results and (g is None or not self.needs_processing(f, g)):
                mark = "✓"
            else:
                mark = "○"
            if self.pending.get(f):
                mark += "✎"
            self.lst.insert(tk.END, "{} {}".format(mark, os.path.basename(f)))
        if sel:
            self.lst.selection_set(sel[0])
        self.lst.yview_moveto(top)

    def _load_controls(self, path):
        """Показать в полях значения выбранного снимка."""
        st = self.iset.get(path)
        if not st:
            return
        self._loading = True
        try:
            self.v_contrast.set(st["contrast"]); self.v_denoise.set(st["denoise"])
            self.v_nmpx.set("" if st["nm_per_px"] is None else "{:g}".format(st["nm_per_px"]))
            self.v_bar.set("" if st["bar_nm"] is None else "{:g}".format(st["bar_nm"]))
            has_meta = bool(self.meta.get(path))
            self.rb_smeta.config(state="normal" if has_meta else "disabled")
            self.v_smode.set("nmpx" if st["nm_per_px"] else "bar" if st["bar_nm"] else ("meta" if has_meta else "bar"))
        finally:
            self._loading = False
        self._update_states()

    def _slider_vals(self):
        return round(float(self.v_contrast.get()), 2), round(float(self.v_denoise.get()), 2)

    def _scale_fields(self, quiet=True):
        def num(v):
            v = v.strip().replace(",", ".")
            return float(v) if v else None
        mode = self.v_smode.get()
        if mode == "meta":
            return None, None
        try:
            v = num((self.v_nmpx if mode == "nmpx" else self.v_bar).get())
            if v is None or v <= 0:
                raise ValueError
            return (v, None) if mode == "nmpx" else (None, v)
        except ValueError:
            if not quiet:
                messagebox.showerror(APP_TITLE, "Масштаб: введите положительное число ({}).".format(
                    "нм на пиксель, например 0.454" if mode == "nmpx" else "длина линейки в нм, например 200"), parent=self)
            return "bad"

    def params_dirty(self, path=None):
        st = self.iset.get(path or self.current)
        if not st:
            return False
        c, d = self._slider_vals()
        return not (_same(c, st["contrast"]) and _same(d, st["denoise"]))

    def scale_dirty(self, path=None):
        st = self.iset.get(path or self.current)
        if not st:
            return False
        f = self._scale_fields()
        if f == "bad":                    # выбранное поле пустое — ничего не введено, это не «неприменённое»
            return bool((self.v_nmpx if self.v_smode.get() == "nmpx" else self.v_bar).get().strip())
        return not (_same(f[0], st["nm_per_px"]) and _same(f[1], st["bar_nm"]))

    def scale_text(self, f):
        st = self.iset.get(f, {})
        if st.get("nm_per_px"):
            return "{:.4g} нм/пикс (введён вручную)".format(st["nm_per_px"])
        if st.get("bar_nm"):
            return "по линейке {:g} нм".format(st["bar_nm"])
        if self.meta.get(f):
            return "{:.4f} нм/пикс (из метаданных)".format(self.meta[f])
        return None

    def _update_states(self):
        if self._loading:
            return
        p = self.current
        st = self.iset.get(p)
        if not st:
            self.lbl_pstate.config(text="", style="Info.TLabel"); self.lbl_sstate.config(text="", style="Info.TLabel")
            return
        t = "Для этого снимка: контраст {:.2f}, шумоподавление {:.1f}".format(st["contrast"], st["denoise"])
        if self.params_dirty(p):
            self.lbl_pstate.config(text=t + "\nПараметры изменены. Примените значения.", style="Warn.TLabel")
        else:
            self.lbl_pstate.config(text=t, style="Info.TLabel")
        s = self.scale_text(p)
        t = "Масштаб снимка: " + (s if s else "НЕ ЗАДАН — введите его")
        ln = self._bar_info(p)
        if ln:
            t += "\n" + ln
        if self.scale_dirty(p):
            self.lbl_sstate.config(text=t + "\nВведённое значение не применено — нажмите «Применить».", style="Warn.TLabel")
        else:
            self.lbl_sstate.config(text=t, style="Info.TLabel" if s else "Warn.TLabel")

    def apply_params(self, to_all):
        if not self.files:
            return
        if not to_all and not self._need_current():
            return
        c, d = self._slider_vals()
        targets = self.files if to_all else [self.current]
        for f in targets:
            self.iset[f]["contrast"] = c; self.iset[f]["denoise"] = d
        msg = "Контраст {:.2f}, шумоподавление {:.1f} — {}.".format(
            c, d, "для всех снимков" if to_all else "для снимка " + os.path.basename(self.current))
        self.status(msg); self.log(msg)
        self._update_states(); self.refresh_list()

    def apply_scale(self, to_all):
        if not self.files:
            return
        if not to_all and not self._need_current():
            return
        f = self._scale_fields(quiet=False)
        if f == "bad":
            return
        if to_all:
            if f[0] or f[1]:
                targets = list(self.files)
                note = "для всех снимков ({})".format(len(targets))
            else:
                targets = [x for x in self.files if self.meta.get(x)]
                skip = len(self.files) - len(targets)
                note = "для {} снимков с масштабом в метаданных".format(len(targets)) + (
                    "; у {} снимков метаданных нет — их масштаб не изменён".format(skip) if skip else "")
        else:
            targets = [self.current]
            note = "для снимка " + os.path.basename(self.current)
            if not (f[0] or f[1]) and not self.meta.get(self.current):
                messagebox.showinfo(APP_TITLE, "У этого снимка нет масштаба в метаданных.", parent=self)
                return
        for x in targets:
            self.iset[x]["nm_per_px"], self.iset[x]["bar_nm"] = f
        what = ("{:g} нм/пикс".format(f[0]) if f[0] else "линейка {:g} нм".format(f[1]) if f[1] else "масштаб из метаданных")
        self.status("Масштаб: {} — {}.".format(what, note)); self.log("Масштаб: {} — {}".format(what, note))
        self._update_states(); self.refresh_list()

    def global_params(self, quiet=False):
        """Общие для всех снимков параметры (плотность, отсечка, расширенные)."""
        try:
            dens = float(self.v_density.get().replace(",", "."))
            if dens <= 0:
                raise ValueError
        except ValueError:
            if not quiet:
                messagebox.showerror(APP_TITLE, "Введите плотность материала (число больше нуля), г/см³.", parent=self)
            return None
        try:
            dcut = float(self.v_dcut.get())
        except (tk.TclError, ValueError):
            if not quiet:
                messagebox.showerror(APP_TITLE, "Проверьте отсечку диаметра (число, нм).", parent=self)
            return None
        p = dict(d_cut=dcut, density=dens, include_cut=bool(self.v_cut.get()), mode=self.mode())
        p.update(self.adv)
        return p

    def params_for(self, f, g):
        p = dict(g); p.update(self.iset[f])
        p["bar_px"] = None
        if p.get("bar_nm") and not p.get("nm_per_px") and f in self.barline:
            y, x0, x1 = self.barline[f]
            p["bar_px"] = round(abs(x1 - x0), 1)
        return p

    def needs_processing(self, f, g):
        r = self.results.get(f)
        if r is None:
            return True
        eff = self.params_for(f, g)
        return any(not _same(r["params"].get(k), eff.get(k)) for k in PROC_KEYS)

    # ================================================================== проверка параметров
    def check_params(self):
        if self.busy or not self._need_current():
            return
        g = self.global_params()
        if g is None:
            return
        p = self.params_for(self.current, g)
        p["contrast"], p["denoise"] = self._slider_vals()
        f = self._scale_fields(quiet=False)
        if f == "bad":
            return
        if f[0] or f[1]:
            p["nm_per_px"], p["bar_nm"] = f
        path = self.current
        self._begin("Проверка параметров…")

        def job():
            try:
                img, s = npc.preview_contour(path, p)
                self.queue.put(("preview", path, (img, p["contrast"], p["denoise"])))
            except Exception as e:
                self.queue.put(("job_error", path, str(e)))
        threading.Thread(target=job, daemon=True).start()

    # ================================================================== обработка
    def start(self, single=False):
        if self.busy:
            return
        if single and not self._need_current():
            return
        if not self.files:
            messagebox.showinfo(APP_TITLE, "Сначала нажмите «Выбрать файлы».", parent=self)
            return
        # неприменённые значения ползунков / масштаба
        if self.current and (self.params_dirty() or self.scale_dirty()):
            what = []
            if self.params_dirty(): what.append("контраст / шумоподавление")
            if self.scale_dirty(): what.append("масштаб")
            ans = messagebox.askyesnocancel(
                APP_TITLE, "Изменены, но не применены: {}.\n\nПрименить к выбранному снимку перед обработкой?\n"
                           "«Да» — применить к снимку, «Нет» — обработать без них.".format(", ".join(what)), parent=self)
            if ans is None:
                return
            if ans:
                if self.params_dirty(): self.apply_params(False)
                if self.scale_dirty():
                    if self._scale_fields(quiet=False) == "bad":
                        return
                    self.apply_scale(False)
        g = self.global_params()
        if g is None:
            return
        scope = [self.current] if single else self.files
        missing = [f for f in scope if not self.scale_text(f)]
        if missing:
            names = "\n".join(os.path.basename(f) for f in missing[:12]) + ("\n…" if len(missing) > 12 else "")
            messagebox.showwarning(APP_TITLE, "Введите масштаб снимка.\n\nНет масштаба в метаданных и он не введён:\n" + names,
                                   parent=self)
            self.sec_scale.expand()
            i = self.files.index(missing[0])
            self.lst.selection_clear(0, tk.END); self.lst.selection_set(i); self.lst.see(i)
            self.show(missing[0])
            return
        nobar = [f for f in scope if self.iset[f].get("bar_nm") and not self.iset[f].get("nm_per_px")
                 and f not in self.barline and self._detect_bar(f) is None]
        if nobar:
            names = "\n".join(os.path.basename(f) for f in nobar[:12]) + ("\n…" if len(nobar) > 12 else "")
            messagebox.showwarning(APP_TITLE, "Линейка не распознана автоматически.\n\nПеретащите красную линию на снимке "
                                              "на линейку (концы линии — на концы линейки):\n" + names, parent=self)
            self.sec_scale.expand()
            i = self.files.index(nobar[0])
            self.lst.selection_clear(0, tk.END); self.lst.selection_set(i); self.lst.see(i)
            self.show(nobar[0])
            return
        todo = [f for f in scope if self.needs_processing(f, g)]
        if not todo:
            q = ("Этот снимок уже обработан с текущими настройками.\nОбработать его заново? Ручные правки на нём будут потеряны."
                 if single else "Все снимки уже обработаны с текущими настройками.\n"
                                "Обработать все заново? Ручные правки будут потеряны.")
            if not messagebox.askyesno(APP_TITLE, q, parent=self):
                return
            todo = list(scope)
        edited = [f for f in todo if f in self.results and self._has_edits(self.results[f])]
        if edited and not messagebox.askyesno(
                APP_TITLE, "На {} снимках, которые будут обработаны заново, есть ручные правки — они будут потеряны.\n"
                           "Продолжить?".format(len(edited)), parent=self):
            return
        for f in todo:
            self.pending.pop(f, None); self._forget(f); self.hist.pop(f, None)
        self.preview = None
        plist = {f: self.params_for(f, g) for f in todo}
        workers = int(round(self.v_cores.get()))
        self._begin("Обработка…", indeterminate=False)
        self.pb.config(maximum=len(todo), value=0)
        self._todo = len(todo); self._done = 0; self._errors = 0; self._t_batch = time.time()
        mode_txt = {"full": "", "no_mod": "; режим: без модифицированных",
                    "only_mod": "; режим: только модифицированные"}[g["mode"]]
        if single:
            self.log("Старт: снимок {}{}".format(os.path.basename(todo[0]), mode_txt))
        else:
            self.log("Старт: {} из {} снимков (остальные уже обработаны с этими настройками), ядер: {}{}".format(
                len(todo), len(self.files), workers, mode_txt))
        threading.Thread(target=self._run_batch, args=(todo, plist, workers), daemon=True).start()

    @staticmethod
    def _has_edits(r):
        e = r.get("edit") or {}
        return bool(e.get("ucircles") or (e.get("del_mask") is not None and e["del_mask"].any())
                    or (e.get("add_mask") is not None and e["add_mask"].any())
                    or any(o["cls"] != o["auto_cls"] for o in r["objects"]))

    def _run_batch(self, files, plist, workers):
        try:
            if workers > 1 and len(files) > 1:
                with ProcessPoolExecutor(max_workers=min(workers, len(files))) as ex:
                    futs = {ex.submit(npc.analyze_image, f, plist[f]): f for f in files}
                    for fu in as_completed(futs):
                        f = futs[fu]
                        try:
                            self.queue.put(("done", f, fu.result()))
                        except Exception as e:
                            self.queue.put(("error", f, "".join(traceback.format_exception_only(type(e), e))))
            else:
                for f in files:
                    try:
                        self.queue.put(("done", f, npc.analyze_image(f, plist[f])))
                    except Exception as e:
                        self.queue.put(("error", f, "".join(traceback.format_exception_only(type(e), e))))
        finally:
            self.queue.put(("finished", None, None))

    def _poll_queue(self):
        try:
            while True:
                kind, f, data = self.queue.get_nowait()
                if kind == "done":
                    data["ml"] = dict(base=ml_annot.circles_px(data))      # круги сразу после обработки (для блока ML)
                    self.results[f] = data; self._forget(f)
                    self._done += 1; self.pb.step(1)
                    self.status("Готово: {} из {}".format(self._done, self._todo))
                    self.log("✓ {}  (масштаб {:.4f} нм/пикс, {}; контраст {:.2f}, шумоподавление {:.1f}; время {})".format(
                        os.path.basename(f), data["s"], data["s_src"], data["params"]["contrast"], data["params"]["denoise"],
                        fmt_dur(data.get("t_proc", 0))))
                elif kind == "error":
                    self._done += 1; self._errors += 1; self.pb.step(1)
                    self.log("✗ {}: {}".format(os.path.basename(f), data.strip()))
                elif kind == "finished":
                    self._end()
                    if self._t_batch:
                        self.log("Обработка заняла {} ({} снимков)".format(fmt_dur(time.time() - self._t_batch), self._todo))
                        self._t_batch = None
                    self.v_contour.set(True); self.v_circles.set(True)
                    self.recompute_all(); self.refresh_list()
                    if self._errors:
                        self.status("Обработка завершена: {} из {} снимков; ошибок: {} (см. «Журнал»)".format(
                            self._todo - self._errors, self._todo, self._errors))
                    else:
                        self.status("Обработка завершена, снимков: {}".format(self._todo))
                    if self.results:
                        self.sec_edit.set_enabled(True); self.sec_edit.expand(); self.sec_par.collapse()
                        cur = self.current if self.current in self.results else [p for p in self.files if p in self.results][0]
                        i = self.files.index(cur)
                        self.lst.selection_clear(0, tk.END); self.lst.selection_set(i); self.lst.see(i)
                        self.nb.select(0); self.show(cur, force=True)
                elif kind == "preview":
                    self._end()
                    img, c, d = data
                    self.preview = dict(path=f, img=img, contrast=c, denoise=d)
                    self.status("Проверка: синий контур — граница вещества при контрасте {:.2f} и шумоподавлении {:.1f}. "
                                "Подходит — «Применить к снимку» или «Применить ко всем».".format(c, d))
                    self.nb.select(0); self.redraw(keep_view=True)
                elif kind == "toggled":
                    self.busy = False
                    self._forget(f); self.recompute_all(); self.redraw(keep_view=True)
                    self.status(data)
                elif kind == "edited":
                    self._end()
                    self.pending.pop(f, None); self._forget(f)
                    self.recompute_all(); self.refresh_list(); self.redraw(keep_view=True); self._update_pending()
                    self.status(data); self.log(data)
                elif kind == "mljob":
                    self._end(); self._ml_refresh()
                    self.status(data); self.log(data)
                elif kind == "job_error":
                    self._end()
                    self.status("Ошибка — подробности на вкладке «Журнал»")
                    self.log("✗ {}: {}".format(os.path.basename(f), str(data).strip()))
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)

    def ordered_results(self):
        return [self.results[p] for p in self.files if p in self.results]

    def recompute_all(self):
        """Пересчёт масс и статистики (быстро; после обработки, правок, смены плотности)."""
        res = self.ordered_results()
        if not res:
            return
        g = self.global_params(quiet=True)
        if g is None:
            return
        for r in res:
            for k in ("include_cut", "ka", "alpha", "nf_packing"):
                r["params"][k] = g[k]
        pooled = npc.pooled_primary_sizes(res)
        stats = [npc.compute_masses(r, g["density"], pooled) for r in res]
        self.summary = npc.batch_summary(stats)
        self.fill_table(stats)
        self.draw_charts(res)
        self.draw_image_charts()

    def _density_changed(self):
        if self.results and not self.busy:
            self.recompute_all()

    def fill_table(self, stats):
        self._set_columns()
        self.tree.delete(*self.tree.get_children())
        use3d = bool(self.v_3d.get())
        f2 = lambda v: "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else "{:.2f}".format(v)
        pc = lambda v: "—" if v is None or not np.isfinite(v) else "{:.1f} %".format(100 * v)
        self._sum_rows = []
        for st in stats:
            mode = st.get("mode", "full")
            n3 = st["n_prim_3d"]
            v = dict(name=st["name"], s="{:.3f}".format(st["s"]), contrast="{:.2f}".format(st["contrast"]),
                     denoise="{:.1f}".format(st["denoise"]),
                     n_mod="—" if mode == "no_mod" else st["n_mod"], d_mod=f2(st["d_mod_mean"]),
                     m_mod="—" if mode == "no_mod" else f2(st["mass_mod_fg"]),
                     n_vis="—" if mode == "only_mod" else st["n_prim_vis"],
                     n_3d="—" if not np.isfinite(n3) else "{:.0f}".format(n3),
                     d20=f2(st["d20"]) if mode != "only_mod" else "—", m_flat=f2(st["mass_agg_flat_fg"]),
                     m_3d=f2(st["mass_agg_3d_fg"]), f_flat=pc(st["frac_flat"]), f_3d=pc(st["frac_3d"]), n_irr=st["n_irr"])
            row = tuple(v[c] for c in self.sum_cols)
            self._sum_rows.append(row)
            self.tree.insert("", tk.END, iid=st["name"], values=row)
        self._fit_columns()
        S = self.summary
        mode = S.get("mode", "full")
        fgv = lambda x: "{:.2f} фг".format(x) if x is not None and np.isfinite(x) else "—"
        t = ["Снимков: {}.  (1 фг = 10⁻¹⁵ г)".format(S["n_images"])]
        if mode != "no_mod":
            t.append("Модифицированных частиц: {};  масса модифицированных: {}.".format(S["n_mod"], fgv(S["mass_mod_fg"])))
        if mode != "only_mod":
            t.append("Первичных НЧ размечено: {}{}.".format(
                S["n_prim_vis"], ", с учётом оценки глубины: {:.0f}".format(S["n_prim_3d"]) if use3d else ""))
            t.append("Масса агломератов (вместе с НФ): плоская {}{}.".format(
                fgv(S["mass_agg_flat_fg"]), ", с учётом оценки глубины {}".format(fgv(S["mass_agg_3d_fg"])) if use3d else ""))
            if S.get("n_irr"):
                t.append("В т.ч. НФ (неправильной формы): {} шт, {} (эллипсоид × плотная упаковка шаров).".format(
                    S["n_irr"], fgv(S.get("mass_irr_fg", 0.0))))
        else:
            t.append("Режим «только модифицированные»: круги первичных частиц и массы агломератов не считаются.")
        if mode == "no_mod":
            t.append("Режим «без модифицированных»: доля модифицированной массы не считается.")
        # распределение по размерам
        sz = S.get("size", {})
        groups = []
        if mode != "only_mod": groups.append(("первичные", "prim"))
        if mode != "no_mod": groups.append(("модифицированные", "mod"))
        if mode == "full": groups.append(("все частицы", "all"))
        if groups:
            t += ["", "Распределение по размерам (вся выборка; модифицированные — без обрезанных краем):"]
            nf = lambda x: "{:.1f}".format(x) if x is not None and np.isfinite(x) else "—"
            for lab, k in groups:
                d = sz.get(k, {})
                t.append("   {}: {} шт;  среднее {} нм,  медиана {} нм,  мода {} нм".format(
                    lab, d.get("n", 0), nf(d.get("mean")), nf(d.get("median")), nf(d.get("mode"))))
            t.append("   Мода — максимум сглаженной гистограммы (ядерная оценка плотности).")
        if mode == "full":
            keys = [("flat", "ПЛОСКАЯ оценка агломератов (только видимые частицы)")]
            if use3d:
                keys.append(("3d", "оценка агломератов С УЧЁТОМ ГЛУБИНЫ (частицы в глубине)"))
            for key, lab in keys:
                d = S[key]
                se = " ± {:.1f} % (ст. ошибка)".format(100 * d["se"]) if np.isfinite(d["se"]) else ""
                t += ["", "Доля модифицированной массы — {}:".format(lab),
                      "   среднее по снимкам {:.1f} %{};  медиана {:.1f} %;  по суммарной массе {:.1f} %".format(
                          100 * d["mean"], se, 100 * d["median"], 100 * d["pooled"])]
            t += ["", "Среднее по снимкам — каждый снимок учитывается одинаково. По суммарной массе — массы всех снимков "
                      "сложены и поделены; сильно зависит от отдельных крупных частиц."]
        self.txt_sum.delete("1.0", tk.END); self.txt_sum.insert("1.0", "\n".join(t))

    # ================================================================== графики
    def _show_figure(self, fig, parent, base="grafik"):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        ttk.Label(parent, text="Правая кнопка мыши на графике — сохранить этот график или все графики.",
                  style="Hint.TLabel").pack(side=tk.BOTTOM, anchor="w")
        cv = FigureCanvasTkAgg(fig, master=parent); cv.draw()
        w = cv.get_tk_widget(); w.pack(fill=tk.BOTH, expand=True)
        self._hover_setup(fig, cv)
        w.bind("<Button-3>", lambda e: self._chart_menu(fig, cv, base, e))
        if sys.platform == "darwin":
            w.bind("<Button-2>", lambda e: self._chart_menu(fig, cv, base, e))

    def _hover_setup(self, fig, cv):
        """Подсказка со значением столбца при наведении мыши (с задержкой ~0,3 с, чтобы не мельтешило)."""
        st = dict(job=None, ann=None, patch=None)

        def hide():
            if st["ann"] is not None:
                try:
                    st["ann"].remove()
                except Exception:
                    pass
                st["ann"] = None; st["patch"] = None
                cv.draw_idle()

        def show(ev, patch, ax):
            st["job"] = None
            hide()
            txt = patch.get_gid()
            ann = ax.annotate(txt, xy=(ev.xdata, ev.ydata), xytext=(12, 12), textcoords="offset points", fontsize=8,
                              bbox=dict(boxstyle="round,pad=0.3", fc="#ffffe8", ec="#999999", lw=0.6), zorder=10)
            # подсказка у правого/верхнего края — разворачиваем внутрь графика
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            if ax.get_xscale() == "log":
                fx = (np.log(ev.xdata) - np.log(x0)) / max(np.log(x1) - np.log(x0), 1e-9)
            else:
                fx = (ev.xdata - x0) / max(x1 - x0, 1e-9)
            fy = (ev.ydata - y0) / max(y1 - y0, 1e-9)
            ann.set_horizontalalignment("right" if fx > 0.6 else "left")
            ann.set_position((-12 if fx > 0.6 else 12, -14 if fy > 0.7 else 12))
            st["ann"] = ann; st["patch"] = patch
            cv.draw_idle()

        def on_move(ev):
            if st["job"] is not None:
                cv.get_tk_widget().after_cancel(st["job"]); st["job"] = None
            if ev.inaxes is None:
                hide(); return
            hit = None
            for patch in ev.inaxes.patches:
                if patch.get_gid() and patch.contains(ev)[0]:
                    hit = patch; break
            if hit is None:
                hide(); return
            if hit is st["patch"]:
                return
            hide()
            st["job"] = cv.get_tk_widget().after(300, lambda: show(ev, hit, ev.inaxes))
        cv.mpl_connect("motion_notify_event", on_move)
        cv.mpl_connect("figure_leave_event", lambda ev: hide())

    def draw_charts(self, res):
        try:
            import charts
        except ImportError:
            return
        for w in self.t3.winfo_children():
            w.destroy()
        fig = charts.overall_figure(res, use_3d=bool(self.v_3d.get()), dpi=self.mpl_dpi)
        self._show_figure(fig, self.t3, "vse_snimki")

    def draw_image_charts(self):
        r = self.results.get(self.current)
        name = os.path.basename(self.current) if self.current else ""
        short = name if len(name) <= 28 else name[:25] + "…"
        self.nb.tab(self.t5, text="  Графики {}  ".format(short) if short else "  Графики снимка  ")
        for w in self.t5.winfo_children():
            w.destroy()
        if r is None:
            ttk.Label(self.t5, text="Этот снимок ещё не обработан.", style="Hint.TLabel").pack(pady=20)
            return
        try:
            import charts
        except ImportError:
            return
        fig = charts.image_figure(r, dpi=self.mpl_dpi)
        self._show_figure(fig, self.t5, os.path.splitext(r["name"])[0])

    def _chart_menu(self, fig, cv, base, ev):
        """Правая кнопка на графике: сохранить этот график, все графики вкладки."""
        import charts
        ax = None
        try:
            w = cv.get_tk_widget(); h = w.winfo_height()
            for a in fig.axes:
                if a.get_window_extent().contains(ev.x, h - ev.y) or a.get_tightbbox(cv.get_renderer()).contains(ev.x, h - ev.y):
                    ax = a; break
        except Exception:
            ax = None
        m = tk.Menu(self, tearoff=0)
        if ax is not None and self._is_flag(ax):
            m.add_command(label="Флаг Нигерии", command=lambda: __import__("webbrowser").open(NIGERIA_URL))
            m.add_separator()
        if ax is not None:
            title = ax.get_title(loc="left") or "график"
            m.add_command(label="Сохранить график «{}»…".format(title.split(",")[0]),
                          command=lambda: self._save_chart(fig, ax, base, title))
        m.add_command(label="Сохранить все графики этой вкладки…", command=lambda: self._save_chart(fig, None, base, "grafiki"))
        m.add_separator()
        m.add_command(label="Сохранить все графики (все снимки)…", command=self.save_all_charts)
        try:
            m.tk_popup(ev.x_root, ev.y_root)
        finally:
            m.grab_release()

    def _is_flag(self, ax):
        """На графике модифицированных снимка ровно две частицы — два равных столбца с промежутком."""
        if getattr(ax, "_npa_kind", "") != "img_mod":
            return False
        hs = [p.get_height() for p in ax.patches if hasattr(p, "get_height")]
        idx = [i for i, h in enumerate(hs) if h > 0]
        return (len(idx) == 2 and abs(hs[idx[0]] - 1) < 1e-9 and abs(hs[idx[1]] - 1) < 1e-9   # две частицы на графике
                and idx[1] - idx[0] >= 2)

    def _save_chart(self, fig, ax, base, title):
        import charts
        nm = "{}_{}.png".format(base, export._safe(title.split(",")[0]))
        path = filedialog.asksaveasfilename(parent=self, title="Сохранить график", initialdir=self.last_dir or
                                            (os.path.dirname(self.files[0]) if self.files else ""), initialfile=nm,
                                            defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf"), ("SVG", "*.svg"),
                                                       ("CSV", "*.csv")])
        if not path:
            return
        dpi = 200
        if path.lower().endswith(".csv"):
            charts.save_csv(fig.axes if ax is None else [ax], path)
        elif ax is None:
            charts.save_figure(fig, path, dpi=dpi)
        else:
            charts.save_axes(fig, ax, path, dpi=dpi)
        self._saved(path, "График")

    def save_all_charts(self):
        if self._no_results():
            return
        folder = filedialog.askdirectory(parent=self, title="Выберите папку для графиков",
                                         initialdir=self.last_dir or os.path.dirname(self.files[0]), mustexist=False)
        if not folder:
            return
        self.config(cursor="watch"); self.update_idletasks()
        try:
            paths = export.save_all_charts(self.ordered_results(), folder, use_3d=bool(self.v_3d.get()))
        finally:
            self.config(cursor="")
        self._saved(folder, "Графики ({} файлов)".format(len(paths)))

    # ================================================================== просмотр снимка
    def show_selected(self):
        sel = self.lst.curselection()
        if sel and self.files[sel[0]] != self.current:
            self.show(self.files[sel[0]])

    def _tree_select(self, e):
        sel = self.tree.selection()
        if not sel:
            return
        for p in self.files:
            if os.path.basename(p) == sel[0]:
                if p != self.current:
                    i = self.files.index(p)
                    self.lst.selection_clear(0, tk.END); self.lst.selection_set(i); self.lst.see(i)
                    self.show(p)
                break

    def show(self, path, force=False):
        if path != self.current and self.current and self.pending.get(self.current) and not self.busy:
            ans = messagebox.askyesnocancel(
                APP_TITLE, "На снимке {} есть неприменённые изменения ({}). Применить их?\n"
                           "«Нет» — отбросить.".format(os.path.basename(self.current), len(self.pending[self.current])),
                parent=self)
            if ans is None:
                i = self.files.index(self.current)
                self.lst.selection_clear(0, tk.END); self.lst.selection_set(i)
                return
            if ans:
                self._apply_now(self.current)
            else:
                self.pending.pop(self.current, None)
            self.refresh_list()
        changed = path != self.current
        if changed:
            self.preview = None
        self.current = path
        if changed or force:
            self._load_controls(path)
        self.redraw(keep_view=not changed)
        self._update_pending()
        self._meas_refresh()
        self._ml_refresh()
        if self.results:
            self.draw_image_charts()

    def _markup_toggled(self):
        self.preview = None
        self.redraw(keep_view=True)

    def _forget(self, path):
        for k in [k for k in self._cache if k[0] == path]:
            del self._cache[k]

    def _image_for(self, path, mode):
        # толщина линий задаётся в пикселях экрана: при уменьшенном показе снимка линии рисуются толще,
        # чтобы изменение толщины было видно при любом масштабе
        k = getattr(self, "_disp_k", 1)
        lw = (int(self.v_lw.get()) if hasattr(self, "v_lw") else 1) * k
        lwc = (int(self.v_lwc.get()) if hasattr(self, "v_lwc") else 1) * k
        key = (path, mode, lw, lwc)
        if key not in self._cache:
            if mode != "orig":
                img = npc.draw_overlay(self.results[path], show_contour=mode[1], show_circles=mode[2], lw=lw, lw_c=lwc)
            else:
                gray, _ = npc.read_image(path)
                img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).convert("RGB")
            if len(self._cache) > 12:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = img
        return self._cache[key]

    # ---------------------------------------------------------------- масштаб и прокрутка снимка
    def _fit_zoom(self):
        if self._img is None:
            return 1.0
        return min(max(100, self.canvas.winfo_width()) / self._img.width,
                   max(100, self.canvas.winfo_height()) / self._img.height)

    def _zoom_sync(self):
        """Ползунок и проценты — по текущему масштабу показа."""
        if not hasattr(self, "sc_zoom"):
            return
        self._zsync = True
        try:
            self.v_zoom.set(float(np.log10(min(ZOOM_MAX, max(ZOOM_MIN, self._z)))))
        finally:
            self._zsync = False
        self.lbl_zoom.config(text="{:.0f} %".format(100 * self._z))

    def _zoom_slider(self, v):
        if self._zsync or self._img is None:
            return
        z = 10 ** float(v)
        if abs(z - self._z) / max(self._z, 1e-9) > 0.01:
            self.set_zoom(z / self._z)

    def set_zoom(self, factor, at=None):
        """factor=None — вписать в окно; at=(x, y) — точка окна, которая остаётся на месте."""
        if self._img is None:
            return
        cv = self.canvas
        if at is None:
            at = (cv.winfo_width() / 2, cv.winfo_height() / 2)
        ix, iy = self._img_xy_screen(*at)
        if factor is None:
            self.zoom = None
        else:
            fit = self._fit_zoom()
            z = min(ZOOM_MAX, max(ZOOM_MIN, self._z * factor))
            self.zoom = None if abs(z - fit) < 1e-3 else z
        self.redraw()
        if factor is not None:
            # прокрутить так, чтобы под курсором осталась та же точка снимка
            sx = ix * self._z + self._ox - at[0]; sy = iy * self._z + self._oy - at[1]
            x0, y0, x1, y1 = [float(v) for v in cv.cget("scrollregion").split()]
            if x1 > 0:
                cv.xview_moveto(max(0.0, sx / x1))
            if y1 > 0:
                cv.yview_moveto(max(0.0, sy / y1))
            self._schedule_render()

    def _scroll(self, axis, *args):
        (self.canvas.xview if axis == "x" else self.canvas.yview)(*args)
        self._schedule_render()

    def _wheel(self, e, horizontal=False, zoom=False):
        if self._img is None:
            return
        up = (getattr(e, "delta", 0) > 0) or getattr(e, "num", 0) == 4
        if zoom:                       # Ctrl+колесо: вниз — увеличить, вверх — уменьшить
            self.set_zoom(1 / 1.15 if up else 1.15, at=(e.x, e.y))
            return "break"
        steps = -1 if up else 1
        if sys.platform == "darwin":
            steps = -int(e.delta) or steps
        (self.canvas.xview_scroll if horizontal else self.canvas.yview_scroll)(steps * 3, "units")
        self._schedule_render()
        return "break"

    def _pan_start(self, e):
        self.canvas.scan_mark(e.x, e.y)

    def _pan_move(self, e):
        self.canvas.scan_dragto(e.x, e.y, gain=1)
        self._schedule_render()

    def _schedule_render(self):
        if self._render_job is None:
            self._render_job = self.after_idle(self._render_view)

    def _render_view(self):
        """Рисует только видимую часть снимка в текущем масштабе (быстро при любом увеличении)."""
        self._render_job = None
        img = self._img
        if img is None:
            return
        cv = self.canvas; z = self._z
        cw, ch = max(1, cv.winfo_width()), max(1, cv.winfo_height())
        vx0, vy0 = cv.canvasx(0), cv.canvasy(0)
        ix0 = max(0.0, (vx0 - self._ox) / z); iy0 = max(0.0, (vy0 - self._oy) / z)
        ix1 = min(float(img.width), (vx0 + cw - self._ox) / z + 1); iy1 = min(float(img.height), (vy0 + ch - self._oy) / z + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        a, b, c, d = int(ix0), int(iy0), int(np.ceil(ix1)), int(np.ceil(iy1))
        part = img.crop((a, b, c, d))
        W_, H_ = max(1, int(round((c - a) * z))), max(1, int(round((d - b) * z)))
        part = part.resize((W_, H_), Image.LANCZOS if z < 1 else Image.NEAREST)
        self._tkimg = ImageTk.PhotoImage(part)
        cv.delete("img")
        cv.create_image(a * z + self._ox, b * z + self._oy, image=self._tkimg, anchor="nw", tags="img")
        cv.tag_lower("img")

    def _img_xy_screen(self, sx, sy):
        cv = self.canvas
        return (cv.canvasx(sx) - self._ox) / self._z, (cv.canvasy(sy) - self._oy) / self._z

    def _img_xy(self, e):
        return self._img_xy_screen(e.x, e.y)

    def _cv(self, x, y):
        """Координаты снимка -> координаты холста."""
        return x * self._z + self._ox, y * self._z + self._oy

    def redraw(self, keep_view=False):
        p = self.current
        if p is None:
            return
        r = self.results.get(p)
        if r is not None:
            cv_ = self.canvas; H_, W_ = r["shape"]
            z_est = self.zoom or min(max(100, cv_.winfo_width()) / W_, max(100, cv_.winfo_height()) / H_)
            self._disp_k = max(1, int(round(1.0 / z_est))) if z_est < 1 else 1
        try:
            if self.preview and self.preview["path"] == p:
                img = self.preview["img"]
                self.lbl_img.config(text="{}   ПРОВЕРКА: контур вещества при контрасте {:.2f}, шумоподавлении {:.1f}".format(
                    os.path.basename(p), self.preview["contrast"], self.preview["denoise"]))
            elif r is not None and (self.v_contour.get() or self.v_circles.get()):
                img = self._image_for(p, ("ov", self.v_contour.get(), self.v_circles.get()))
                self.lbl_img.config(text=self._image_info(r))
            else:
                img = self._image_for(p, "orig")
                self.lbl_img.config(text=os.path.basename(p) + ("   (исходный снимок)" if r is not None else "   — ещё не обработан"))
        except Exception as e:
            self.lbl_img.config(text="Не удалось открыть снимок: {}".format(e))
            return
        cv = self.canvas
        old = (cv.xview()[0], cv.yview()[0])
        self._img = img
        z = self.zoom or self._fit_zoom()
        self._z = z
        cw, ch = max(1, cv.winfo_width()), max(1, cv.winfo_height())
        vw, vh = img.width * z, img.height * z
        # снимок меньше окна — по центру
        self._ox = max(0.0, (cw - vw) / 2); self._oy = max(0.0, (ch - vh) / 2)
        cv.delete("all")
        cv.config(scrollregion=(0, 0, max(vw, cw), max(vh, ch)))
        if keep_view:
            cv.xview_moveto(old[0]); cv.yview_moveto(old[1])
        self._render_view()
        self.draw_pending()
        self._draw_bar()
        self._draw_meas()
        self._draw_ml()
        self._poly_draw()
        self._zoom_sync()

    def _image_info(self, r):
        st = r.get("stats", {}); P = r["params"]; mode = P.get("mode", "full")
        pc = lambda v: "{:.1f} %".format(100 * v) if v is not None and np.isfinite(v) else "—"
        fg = lambda v: "{:.2f}".format(v) if v is not None and np.isfinite(v) else "—"
        l1 = "{}   {:.4f} нм/пикс ({})   контраст {:.2f}, шумоподавление {:.1f}".format(
            r["name"], r["s"], r["s_src"], P["contrast"], P["denoise"])
        use3d = bool(self.v_3d.get())
        parts = []
        if mode != "no_mod":
            parts.append("модифицированных: {}".format(st.get("n_mod", 0)))
        if mode != "only_mod":
            parts.append("первичных НЧ размечено: {}{}".format(
                st.get("n_prim_vis", 0), " (с учётом оценки глубины: {:.0f})".format(st.get("n_prim_3d", 0) or 0) if use3d else ""))
        if mode == "full":
            parts.append("доля модиф. массы: плоская {}{}".format(
                pc(st.get("frac_flat")), ", с глубиной {}".format(pc(st.get("frac_3d"))) if use3d else ""))
        m = []
        if mode != "no_mod":
            m.append("модифицированные {}".format(fg(st.get("mass_mod_fg"))))
        if mode != "only_mod":
            m.append("агломераты плоская {}{}".format(fg(st.get("mass_agg_flat_fg")),
                                                     " / с глубиной {}".format(fg(st.get("mass_agg_3d_fg"))) if use3d else ""))
            if st.get("n_irr"):
                m.append("в т.ч. НФ {}".format(fg(st.get("mass_irr_fg"))))
        return l1 + "\n" + "   ".join(parts) + "\nмассы, фг: " + ";  ".join(m)

    # ---------------------------------------------------------------- линейка (красная линия)
    def _bar_mode(self, path=None):
        """Масштаб задаётся длиной линейки: введено (или применено) значение в поле линейки и нет нм/пикс."""
        path = path or self.current
        if path is None:
            return False
        if path == self.current:
            f = self._scale_fields()
            if f != "bad" and (f[0] or f[1]):
                return bool(f[1]) and not f[0]
        st = self.iset.get(path, {})
        return bool(st.get("bar_nm")) and not st.get("nm_per_px")

    def _detect_bar(self, path):
        if path not in self._sb:
            try:
                gray, _ = npc.read_image(path)
                self._sb[path] = (npc.find_scalebar(gray), gray.shape)
            except Exception:
                self._sb[path] = (None, None)
        return self._sb[path][0]

    def _bar_line(self, path):
        """(y, x0, x1, источник) — заданная вручную, найденная автоматически или по центру кадра."""
        if path in self.barline:
            y, x0, x1 = self.barline[path]
            return y, x0, x1, "user"
        sb = self._detect_bar(path)
        if sb:
            return sb["y"], sb["x0"], sb["x1"], "auto"
        shape = self._sb.get(path, (None, None))[1]
        if shape is None:
            shape = (self._img.height, self._img.width) if self._img is not None else (1032, 1376)
        H, W = shape
        return H / 2, W * 0.4, W * 0.6, "none"

    def _bar_info(self, path):
        if not self._bar_mode(path):
            return ""
        y, x0, x1, src = self._bar_line(path)
        L = abs(x1 - x0)
        t = {"user": "линия задана вручную", "auto": "линейка найдена автоматически",
             "none": "линейка НЕ распознана — перетащите красную линию на неё"}[src]
        f = self._scale_fields() if path == self.current else None
        bn = (f[1] if f and f != "bad" and f[1] else None) or self.iset.get(path, {}).get("bar_nm")
        sc = "  → {:.4f} нм/пикс".format(bn / L) if bn and L > 0 else ""
        return "Линейка: {:.0f} пикс ({}){}".format(L, t, sc)

    def _draw_bar(self):
        cv = self.canvas
        cv.delete("bar")
        if self.current is None or self._img is None or not self._bar_mode():
            return
        y, x0, x1, src = self._bar_line(self.current)
        X0, Y = self._cv(x0, y); X1, _ = self._cv(x1, y)
        cv.create_line(X0, Y, X1, Y, fill="#ff2020", width=3, tags="bar")
        h = self.px(6)
        for X in (X0, X1):
            cv.create_rectangle(X - h, Y - h, X + h, Y + h, outline="#ff2020", width=2, fill="", tags="bar")
        cv.create_text((X0 + X1) / 2, Y - self.px(14), text="{:.0f} пикс".format(abs(x1 - x0)), fill="#ff2020",
                       font=("TkDefaultFont", 9, "bold"), tags="bar")

    def _bar_hit(self, e):
        """Что под курсором у красной линии: 'x0', 'x1', 'move' или None."""
        if self.current is None or self._img is None or not self._bar_mode():
            return None
        y, x0, x1, src = self._bar_line(self.current)
        X0, Y = self._cv(x0, y); X1, _ = self._cv(x1, y)
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        tol = self.px(9)
        if abs(cx - X0) <= tol and abs(cy - Y) <= tol:
            return "x0"
        if abs(cx - X1) <= tol and abs(cy - Y) <= tol:
            return "x1"
        if min(X0, X1) <= cx <= max(X0, X1) and abs(cy - Y) <= tol:
            return "move"
        return None

    # ================================================================== редактирование
    def _tool_changed(self):
        names = {"toggle": "Инструмент: смена класса кликом", "del_one": "Инструмент: удалить элемент",
                 "del_rect": "Инструмент: удалить в прямоугольнике", "del_circle": "Инструмент: удалить в круге",
                 "del_lasso": "Инструмент: удалить внутри контура", "add_circle": "Инструмент: добавить круг",
                 "add_contour": "Инструмент: добавить контур",
                 "del_poly": "Инструмент: удалить внутри ломаной", "del_spline": "Инструмент: удалить внутри сплайна",
                 "add_poly": "Инструмент: добавить ломаную", "add_spline": "Инструмент: добавить сплайн",
                 "ml_c3": "Инструмент: круг по трём точкам края", "ml_doubt": "Инструмент: отметка «?» на круге",
                 "ml_region": "Инструмент: зона разметки (прямоугольник)"}
        self._poly_cancel(); self._c3_cancel()
        self.lbl_tool.config(text=names.get(self.v_tool.get(), ""))
        self.canvas.config(cursor="arrow" if self.v_tool.get() == "toggle" else "crosshair")

    def _ops(self):
        return self.pending.setdefault(self.current, [])

    def _update_pending(self):
        n = len(self.pending.get(self.current, []))
        self.lbl_pending.config(text="Неприменённых изменений: {}".format(n), style="Warn.TLabel" if n else "Info.TLabel")

    def _shape_hit(self, sh, x, y):
        if sh[0] == "circle":
            return (x - sh[1]) ** 2 + (y - sh[2]) ** 2 <= sh[3] ** 2
        if sh[0] == "poly":
            from matplotlib.path import Path
            return Path(sh[1]).contains_point((x, y))
        if sh[0] == "rect":
            return min(sh[1], sh[3]) <= x <= max(sh[1], sh[3]) and min(sh[2], sh[4]) <= y <= max(sh[2], sh[4])
        return False

    def on_press(self, e):
        self.canvas.focus_set()                      # чтобы Enter и Ctrl+Z относились к снимку
        # 1) красная линия линейки — в первую очередь
        hit = self._bar_hit(e)
        if hit:
            y, x0, x1, _ = self._bar_line(self.current)
            ix, iy = self._img_xy(e)
            self._drag = dict(mode="bar", part=hit, x=ix, y=iy, line=[y, x0, x1])
            return
        if self._meas_active() and self._img is not None:
            x, y = self._img_xy(e)
            self.canvas.delete("hover")
            self._drag = dict(mode="meas", x0=x, y0=y)
            return
        r = self.results.get(self.current)
        tool = self.v_tool.get()
        if r is None or self.busy or tool == "toggle":
            # перетаскивание снимка левой кнопкой; короткий клик — смена класса
            self.canvas.scan_mark(e.x, e.y)
            self._drag = dict(mode="pan", sx=e.x, sy=e.y, moved=False)
            return
        x, y = self._img_xy(e)
        H, W = r["shape"]
        if not (0 <= x < W and 0 <= y < H):
            return
        if self.preview:
            self.preview = None; self.redraw(keep_view=True)
        if not (self.v_contour.get() or self.v_circles.get()):
            self.v_contour.set(True); self.v_circles.set(True); self.redraw(keep_view=True)
        if tool.startswith("ml_"):
            self._ml_press(r, tool, x, y)
            return
        if tool == "del_one":
            ops = self._ops()
            # повторный клик по уже отмеченному элементу снимает отметку
            for i in range(len(ops) - 1, -1, -1):
                op = ops[i]
                if op["op"] == "del_point":
                    _, cx, cy, R = op["vis"]
                    if (x - cx) ** 2 + (y - cy) ** 2 <= max(R, 3) ** 2:
                        del ops[i]
                        self.draw_pending(); self._update_pending(); self.refresh_list()
                        self.status("Отметка удаления снята.")
                        return
            hit = npc.element_at(r, y, x)
            if hit is None:
                self.status("Здесь нет элемента: кликните по кружку или по частице.")
                return
            if hit[0] == "circle":
                c = hit[1]; vis = ("circle", c[1], c[0], c[2] / 2 / r["s"])
            else:
                o = hit[1]; vis = ("circle", o["cx"], o["cy"], o["deq"] / 2 / r["s"])
            self._ops().append(dict(op="del_point", y=y, x=x, vis=vis))
            self._after_op()
            return
        if tool in POLY_TOOLS and self._poly:
            P = self._poly["pts"]
            X0, Y0 = self._cv(*P[0]); cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
            if len(P) >= 3 and np.hypot(cx - X0, cy - Y0) <= self.px(8):
                self._poly_finish()
            elif np.hypot(x - P[-1][0], y - P[-1][1]) * self._z > 2:
                P.append((x, y)); self._poly_draw(x, y)
            return
        if tool.startswith("add"):
            tol = self.px(6) / self._z
            for op in reversed(self._ops()):
                if op["op"] != "add":
                    continue
                sh = op["shape"]
                if sh[0] == "circle":
                    d = np.hypot(x - sh[1], y - sh[2])
                    if abs(d - sh[3]) <= max(tol, 0.12 * sh[3]) and d > 0.5 * sh[3]:
                        self._drag = dict(mode="resize", op=op)        # за край — изменить размер
                        return
                if self._shape_hit(sh, x, y):
                    self._drag = dict(mode="move", op=op, x=x, y=y)    # внутри — перетащить
                    return
        if tool in POLY_TOOLS:
            self._poly = dict(tool=tool, pts=[(x, y)])
            self._poly_draw(x, y)
            self.status("Ставьте вершины кликами; двойной клик или клик по первой вершине — замкнуть, Esc — отмена.")
            return
        self._drag = dict(mode="draw", tool=tool, x0=x, y0=y, pts=[(x, y)])

    def _double_press(self, e):
        """Двойной щелчок замыкает ломаную / сплайн; для инструментов блока ML это просто ещё один щелчок
        (точки круга по трём точкам и отметки «?» часто ставятся быстро и рядом)."""
        if self._poly:
            self._poly_finish()
        elif self.v_tool.get() in ("ml_c3", "ml_doubt"):
            self.on_press(e)

    # ---------------------------------------------------------------- замкнутая ломаная / сплайн
    @staticmethod
    def _spline_closed(P, n=12):
        """Замкнутый сплайн Катмулла–Рома через вершины (кривая проходит через все точки)."""
        P = np.asarray(P, float); k = len(P); out = []
        for i in range(k):
            p0, p1, p2, p3 = P[(i - 1) % k], P[i], P[(i + 1) % k], P[(i + 2) % k]
            for t in np.linspace(0, 1, n, endpoint=False):
                out.append(tuple(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                                        (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)))
        return out

    def _poly_draw(self, mx=None, my=None):
        cv = self.canvas
        cv.delete("polytmp")
        if not self._poly:
            return
        t = self._poly["tool"]; P = list(self._poly["pts"])
        col = C_DEL if t.startswith("del") else (C_MOD if self.v_addcls.get() == "mod" else C_PRIM)
        if mx is not None:
            P = P + [(mx, my)]
        if len(P) >= 2:
            Q = self._spline_closed(P) + [self._spline_closed(P)[0]] if (t.endswith("spline") and len(P) >= 3) else P
            cv.create_line(*[c for p in Q for c in self._cv(*p)], fill=col, width=2, dash=(4, 2), tags="polytmp")
        for (px_, py_) in self._poly["pts"]:
            X, Y = self._cv(px_, py_); h = self.px(3)
            cv.create_rectangle(X - h, Y - h, X + h, Y + h, outline=col, fill="white", tags="polytmp")

    def _poly_cancel(self):
        if self._poly:
            self._poly = None
            self.canvas.delete("polytmp")
            self.status("Построение отменено.")

    def _poly_finish(self):
        if not self._poly:
            return
        t = self._poly["tool"]; P = self._poly["pts"]
        self._poly = None; self.canvas.delete("polytmp")
        if len(P) < 3:
            self.status("Нужно не меньше трёх вершин.")
            return
        sh = ("poly", self._spline_closed(P) if t.endswith("spline") else list(P))
        if t.startswith("del"):
            self._ops().append(dict(op="del_region", shape=sh))
        else:
            self._ops().append(dict(op="add", cls=self.v_addcls.get(), shape=sh))
        self._after_op()

    def on_motion(self, e):
        if not self._drag:
            return
        d = self._drag
        if d["mode"] == "pan":
            if abs(e.x - d["sx"]) + abs(e.y - d["sy"]) > 3:
                d["moved"] = True
            if d["moved"]:
                self.canvas.scan_dragto(e.x, e.y, gain=1); self._schedule_render()
            return
        x, y = self._img_xy(e)
        if d["mode"] == "meas":
            self.canvas.delete("tmp")
            if self.v_mtool.get() != "circ":
                x, y = self._meas_end(d["x0"], d["y0"], x, y, e)
            X0, Y0 = self._cv(d["x0"], d["y0"]); X, Y = self._cv(x, y)
            p = self.current; col = "#ffd400"
            if self.v_mtool.get() == "circ":
                R = np.hypot(X - X0, Y - Y0)
                self.canvas.create_oval(X0 - R, Y0 - R, X0 + R, Y0 + R, outline=col, width=2, dash=(4, 2), tags="tmp")
                t = "d = " + self._fmt_len(2 * np.hypot(x - d["x0"], y - d["y0"]), p)
            else:
                self.canvas.create_line(X0, Y0, X, Y, fill=col, width=2, dash=(4, 2), tags="tmp")
                t = self._fmt_len(np.hypot(x - d["x0"], y - d["y0"]), p)
                lst = self.meas.get(p, [])
                if self.v_mtool.get() == "ll" and lst and lst[-1]["kind"] == "ll" and len(lst[-1]["lines"]) == 1:
                    dist, ang = self._line_dist(lst[-1]["lines"][0], ((d["x0"], d["y0"]), (x, y)))
                    t = "между линиями " + self._fmt_len(dist, p)
            self._label(X + self.px(12), Y - self.px(12), t, col, "tmp", anchor="w")
            return
        if d["mode"] == "bar":
            if self._img is not None:                       # линия не выходит за пределы снимка
                x = min(max(x, 0.0), float(self._img.width)); y = min(max(y, 0.0), float(self._img.height - 1))
            ln = d["line"]; dx, dy = x - d["x"], y - d["y"]
            if d["part"] == "move" and self._img is not None:
                dx = min(max(dx, -min(ln[1], ln[2])), self._img.width - max(ln[1], ln[2]))
                dy = min(max(dy, -ln[0]), self._img.height - 1 - ln[0])
            if d["part"] == "x0":
                ln[1] = x; ln[0] = y
            elif d["part"] == "x1":
                ln[2] = x; ln[0] = y
            else:
                ln[0] += dy; ln[1] += dx; ln[2] += dx
                d["x"], d["y"] = x, y
            self.barline[self.current] = (ln[0], min(ln[1], ln[2]), max(ln[1], ln[2])) if d["part"] == "move" else \
                (ln[0], ln[1], ln[2])
            self._draw_bar()
            return
        if d["mode"] == "move":
            dx, dy = x - d["x"], y - d["y"]; d["x"], d["y"] = x, y
            sh = d["op"]["shape"]
            if sh[0] == "circle":
                d["op"]["shape"] = ("circle", sh[1] + dx, sh[2] + dy, sh[3])
            elif sh[0] == "poly":
                d["op"]["shape"] = ("poly", [(px + dx, py + dy) for px, py in sh[1]])
            self.draw_pending()
            return
        if d["mode"] == "resize":
            sh = d["op"]["shape"]
            d["op"]["shape"] = ("circle", sh[1], sh[2], max(1.5, float(np.hypot(x - sh[1], y - sh[2]))))
            self.draw_pending()
            return
        if d["mode"] == "mlregion":
            self.canvas.delete("tmp")
            self.canvas.create_rectangle(*self._cv(d["x0"], d["y0"]), *self._cv(x, y), outline=C_ML, width=2,
                                         dash=(6, 3), tags="tmp")
            return
        d["pts"].append((x, y))
        self.canvas.delete("tmp")
        t = d["tool"]
        col = C_DEL if t.startswith("del") else (C_MOD if self.v_addcls.get() == "mod" else C_PRIM)
        X0, Y0 = self._cv(d["x0"], d["y0"]); X, Y = self._cv(x, y)
        if t == "del_rect":
            self.canvas.create_rectangle(X0, Y0, X, Y, outline=col, width=2, dash=(4, 2), tags="tmp")
        elif t in ("del_circle", "add_circle"):
            R = np.hypot(X - X0, Y - Y0)
            self.canvas.create_oval(X0 - R, Y0 - R, X0 + R, Y0 + R, outline=col, width=2, dash=(4, 2), tags="tmp")
            self.canvas.create_line(X0, Y0, X, Y, fill=col, tags="tmp")
        else:
            pts = [c for p in d["pts"] for c in self._cv(*p)]
            if len(pts) >= 4:
                self.canvas.create_line(*pts, fill=col, width=2, tags="tmp")

    def on_release(self, e):
        d, self._drag = self._drag, None
        self.canvas.delete("tmp")
        if not d:
            return
        if d["mode"] == "pan":
            r = self.results.get(self.current)
            if not d["moved"] and r is not None and not self.busy and self.v_tool.get() == "toggle":
                x, y = self._img_xy(e)
                H, W = r["shape"]
                if 0 <= x < W and 0 <= y < H:
                    if self.preview:
                        self.preview = None; self.redraw(keep_view=True)
                    if not self.v_contour.get():
                        self.status("Чтобы менять класс частиц, включите «Показать контур».")
                        return
                    self.toggle_click(r, x, y)
            return
        if d["mode"] == "meas":
            x, y = self._img_xy(e)
            if self.v_mtool.get() != "circ":
                x, y = self._meas_end(d["x0"], d["y0"], x, y, e)
            p = self.current; lst = self.meas.setdefault(p, [])
            L = float(np.hypot(x - d["x0"], y - d["y0"]))
            if L * self._z < 3:
                return                                     # просто щелчок — не измерение
            t = self.v_mtool.get()
            if t == "pp":
                lst.append(dict(kind="pp", pts=((d["x0"], d["y0"]), (x, y))))
            elif t == "circ":
                lst.append(dict(kind="circ", cx=d["x0"], cy=d["y0"], r=L))
            else:
                seg = ((d["x0"], d["y0"]), (x, y))
                if lst and lst[-1]["kind"] == "ll" and len(lst[-1]["lines"]) == 1:
                    lst[-1]["lines"].append(seg)
                else:
                    lst.append(dict(kind="ll", lines=[seg]))
            self._meas_refresh()
            self.status("Измерение: " + self._meas_value(lst[-1], p))
            return
        if d["mode"] == "bar":
            y, x0, x1 = self.barline[self.current]
            self.barline[self.current] = (y, min(x0, x1), max(x0, x1))
            self._draw_bar(); self._update_states(); self.refresh_list()
            self.status("Линейка: {:.0f} пикс. Значение учтётся при обработке снимка.".format(abs(x1 - x0)))
            return
        if d["mode"] == "move":
            self.status("Элемент перемещён (изменение ещё не применено).")
            return
        if d["mode"] == "resize":
            self.status("Размер круга изменён (изменение ещё не применено).")
            return
        if d["mode"] == "mlregion":
            x, y = self._img_xy(e)
            r = self.results.get(self.current)
            if r is None or abs(x - d["x0"]) < 5 or abs(y - d["y0"]) < 5:
                return
            H, W = r["shape"]
            x0, x1 = sorted((min(max(d["x0"], 0.0), W), min(max(x, 0.0), W)))
            y0, y1 = sorted((min(max(d["y0"], 0.0), H), min(max(y, 0.0), H)))
            ml = self._ml(r); ml["region"] = (x0, y0, x1, y1); self._ml_tick()
            self._draw_ml(); self._ml_refresh()
            frac = (x1 - x0) * (y1 - y0) / float(W * H)
            self.status("Зона разметки: {:.0f} % кадра. Внутри неё должны быть размечены все частицы.".format(100 * frac))
            return
        x, y = self._img_xy(e)
        t = d["tool"]
        if t == "del_rect":
            if abs(x - d["x0"]) < 2 or abs(y - d["y0"]) < 2:
                return
            sh = ("rect", d["x0"], d["y0"], x, y)
        elif t in ("del_circle", "add_circle"):
            R = float(np.hypot(x - d["x0"], y - d["y0"]))
            if R < 1.5:
                self.status("Тяните мышью от центра круга к его краю.")
                return
            sh = ("circle", d["x0"], d["y0"], R)
        else:
            pts = d["pts"][::max(1, len(d["pts"]) // 300)]
            if len(pts) < 3:
                self.status("Обведите контур, удерживая кнопку мыши.")
                return
            sh = ("poly", pts)
        if t.startswith("del"):
            self._ops().append(dict(op="del_region", shape=sh))
        else:
            self._ops().append(dict(op="add", cls=self.v_addcls.get(), shape=sh))
        self._after_op()

    def _after_op(self):
        self._ml_tick()
        self.draw_pending(); self._update_pending(); self.refresh_list()
        self.status("Изменение добавлено. Нажмите «Применить изменения», чтобы учесть его в статистике.")

    def draw_pending(self):
        cv = self.canvas
        cv.delete("edit")
        z = self._z
        for op in self.pending.get(self.current, []):
            if op["op"] == "del_point":
                _, cx, cy, R = op["vis"]; R = max(R, 3)
                X, Y = self._cv(cx, cy); RR = R * z; k = RR * 0.7
                cv.create_oval(X - RR, Y - RR, X + RR, Y + RR, outline=C_DEL, width=2, tags="edit")
                cv.create_line(X - k, Y - k, X + k, Y + k, fill=C_DEL, width=2, tags="edit")
                cv.create_line(X - k, Y + k, X + k, Y - k, fill=C_DEL, width=2, tags="edit")
                continue
            sh = op["shape"]
            if op["op"] == "del_region":
                col, dash = C_DEL, (6, 3)
            else:
                col, dash = (C_MOD if op["cls"] == "mod" else C_PRIM), (3, 2)
            if sh[0] == "rect":
                cv.create_rectangle(*self._cv(sh[1], sh[2]), *self._cv(sh[3], sh[4]), outline=col, width=2, dash=dash, tags="edit")
            elif sh[0] == "circle":
                X, Y = self._cv(sh[1], sh[2]); RR = sh[3] * z
                cv.create_oval(X - RR, Y - RR, X + RR, Y + RR, outline=col, width=max(2, int(self.v_lwc.get())),
                               dash=dash, tags="edit")
            else:
                pts = [c for p in sh[1] + [sh[1][0]] for c in self._cv(*p)]
                cv.create_line(*pts, fill=col, width=2, dash=dash, tags="edit")
            if op["op"] == "del_region":
                if sh[0] == "rect":
                    cx, cy = (sh[1] + sh[3]) / 2, (sh[2] + sh[4]) / 2
                elif sh[0] == "circle":
                    cx, cy = sh[1], sh[2]
                else:
                    a = np.array(sh[1]); cx, cy = a[:, 0].mean(), a[:, 1].mean()
                cv.create_text(*self._cv(cx, cy), text="удалить", fill=C_DEL, font=("TkDefaultFont", 9, "bold"), tags="edit")

    def edit_undo(self):
        """Ctrl+Z: сначала снимаются неприменённые отметки, затем отменяются до 3 последних применённых
        изменений (правки и смена класса кликом)."""
        if self.busy:
            return
        ops = self.pending.get(self.current)
        if ops:
            ops.pop(); self.draw_pending(); self._update_pending(); self.refresh_list()
            self.status("Последнее неприменённое изменение отменено.")
            return
        h = self.hist.get(self.current)
        r = self.results.get(self.current)
        if h and r is not None:
            npc.restore(r, h.pop())
            if "ml" in r:
                r["ml"]["dirty"] = True
            self._forget(self.current); self.recompute_all(); self.redraw(keep_view=True); self._ml_refresh()
            msg = "Отменено применённое изменение. Можно отменить ещё: {}.".format(len(h))
            self.status(msg); self.log(msg + " ({})".format(os.path.basename(self.current)))
        else:
            self.status("Отменять нечего (хранятся 3 последних применённых изменения).")

    def edit_reset(self):
        if self.pending.get(self.current):
            self.pending.pop(self.current, None)
            self.draw_pending(); self._update_pending(); self.refresh_list()
            self.status("Неприменённые изменения сброшены.")

    def edit_apply(self):
        r = self.results.get(self.current)
        ops = self.pending.get(self.current)
        if r is None or not ops or self.busy:
            if not ops:
                self.status("Нет неприменённых изменений.")
            return
        path = self.current
        ops = [dict(o) for o in ops]
        self._push_hist(path)
        self._begin("Применяю изменения…")

        def job():
            try:
                npc.apply_edits(r, ops)
                self.queue.put(("edited", path, "Изменения применены ({}): {}".format(len(ops), os.path.basename(path))))
            except Exception as e:
                self.queue.put(("job_error", path, "".join(traceback.format_exception_only(type(e), e))))
        threading.Thread(target=job, daemon=True).start()

    def _apply_now(self, path):
        """Применить правки сразу (при переходе на другой снимок)."""
        r = self.results.get(path); ops = self.pending.pop(path, None)
        if r is None or not ops:
            return
        self._push_hist(path)
        self.config(cursor="watch"); self.update_idletasks()
        try:
            npc.apply_edits(r, ops); self._forget(path); self.recompute_all()
            self.log("Изменения применены ({}): {}".format(len(ops), os.path.basename(path)))
        finally:
            self.config(cursor="")

    def toggle_click(self, r, x, y):
        if r["objmap"][int(y), int(x)] == 0:
            self.status("В этом месте нет крупного объекта-кандидата (кликайте по самой частице).")
            return
        self.status("Пересчитываю агломераты рядом с объектом…")
        path = self.current
        self._push_hist(path)
        self._ml_tick()
        self.busy = True

        def job():
            try:
                cls = npc.toggle_object(r, y, x)
                names = {"mod": "модифицированная", "agg": "часть агломерата", "irr": "НФ (неправильной формы)"}
                self.queue.put(("toggled", path, "Объект отмечен как: " + names.get(cls, "?")))
            except Exception as e:
                self.busy = False
                self.queue.put(("job_error", path, str(e)))
        threading.Thread(target=job, daemon=True).start()

    # ================================================================== разметка для обучения (ML)
    def _build_ml(self, box, wrap):
        isz = self.px(22)
        self._ico_ml = {k: make_icon(k, isz) for k in ("c3", "doubt", "region")}
        row = ttk.Frame(box); row.pack(fill=tk.X)
        ttk.Label(row, text="Разметчик:").pack(side=tk.LEFT)
        self.v_ml_annot = tk.StringVar(value=self.mlcfg.get("annotator", ""))
        ent = ttk.Entry(row, textvariable=self.v_ml_annot, width=8)
        ent.pack(side=tk.LEFT, padx=(4, 0))
        ent.bind("<FocusOut>", lambda e: self._ml_save_annot())
        Tip(ent, "Код разметчика латиницей, например A1 (один на человека)")
        ttk.Button(row, text="Памятка", command=self.show_ml_memo).pack(side=tk.RIGHT)
        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row, text="Набор:").pack(side=tk.LEFT)
        self.lbl_ml_dir = ttk.Label(row, text="", style="Info.TLabel")
        self.lbl_ml_dir.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(row, text="Папка…", command=self._ml_choose_dir).pack(side=tk.RIGHT)
        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row, text="Инструменты:").pack(side=tk.LEFT)
        for k, val, tip in (("region", "ml_region", "Зона разметки: прямоугольник по диагонали"),
                            ("doubt", "ml_doubt", "Отметка «?»: щелчок по кругу ставит или снимает"),
                            ("c3", "ml_c3", "Круг по трём точкам края частицы")):
            rb = ttk.Radiobutton(row, image=self._ico_ml[k], style="Toolbutton", variable=self.v_tool, value=val)
            rb.pack(side=tk.RIGHT, padx=1); Tip(rb, tip)
        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(row, text="Зона: весь кадр", command=self.ml_region_reset).pack(side=tk.LEFT, expand=True, fill=tk.X)
        b = ttk.Button(row, text="Начать с нуля", command=self.ml_scratch)
        b.pack(side=tk.LEFT, expand=True, fill=tk.X)
        Tip(b, "Удалить автоматическую разметку снимка и разметить его вручную")
        ttk.Button(box, text="Добавить снимок в набор", style="Accent.TButton",
                   command=self.ml_add_current).pack(fill=tk.X, pady=(6, 0))
        self.lbl_ml = ttk.Label(box, text="", style="Info.TLabel", wraplength=wrap, justify="left")
        self.lbl_ml.pack(anchor="w", pady=(4, 0))
        row = ttk.Frame(box); row.pack(fill=tk.X, pady=(4, 0))
        b = ttk.Button(row, text="Упаковать набор…", command=self.ml_pack)
        b.pack(side=tk.LEFT, expand=True, fill=tk.X)
        Tip(b, "Весь набор — в один zip-архив для переноса на другой компьютер")
        b = ttk.Button(row, text="Добавить из архива…", command=self.ml_merge)
        b.pack(side=tk.LEFT, expand=True, fill=tk.X)
        Tip(b, "Объединить с набором из архива другого компьютера или разметчика")
        ttk.Button(box, text="Открыть папку набора", command=self.ml_open_dir).pack(fill=tk.X, pady=(2, 0))
        self._ml_refresh()

    @staticmethod
    def _short(path, n=30):
        return path if len(path) <= n else "…" + path[-(n - 1):]

    def _ml(self, r):
        """Состояние разметки для обучения у результата снимка (хранится вместе с результатом)."""
        ml = r.setdefault("ml", {})
        ml.setdefault("base", None); ml.setdefault("doubt", set()); ml.setdefault("region", None)
        ml.setdefault("t_active", 0.0); ml.setdefault("t_last", None)
        ml["mode"] = "scratch" if r.get("edit", {}).get("ml_scratch") else "corrected"
        return ml

    def _ml_tick(self, path=None):
        """Учёт активного времени правки снимка: паузы длиннее 3 минут считаются как 30 с."""
        r = self.results.get(path or self.current)
        if r is None:
            return
        ml = self._ml(r); t = time.time()
        ml["dirty"] = True                             # разметка изменилась после добавления в набор
        if ml["t_last"] is not None:
            gap = t - ml["t_last"]
            ml["t_active"] += gap if gap <= 180 else 30.0
        ml["t_last"] = t

    def _ml_press(self, r, tool, x, y):
        if tool == "ml_region":
            self._drag = dict(mode="mlregion", x0=x, y0=y)
            return
        if tool == "ml_doubt":
            c = npc.find_circle(r, y, x)
            if c is None:
                self.status("Щёлкните по кругу первичной частицы (неприменённые круги сначала примените Enter).")
                return
            ml = self._ml(r)
            ml["doubt"] = ml_annot.match_doubts(r["circles"], ml["doubt"])
            k = ml_annot.key(c[0], c[1])
            if k in ml["doubt"]:
                ml["doubt"].discard(k); msg = "Отметка «?» снята."
            else:
                ml["doubt"].add(k); msg = "Круг отмечен как сомнительный («?»)."
            self._ml_tick(); self._draw_ml(); self._ml_refresh(); self.status(msg)
            return
        if tool == "ml_c3":
            self._c3 = (self._c3 or []) + [(x, y)]
            if len(self._c3) < 3:
                self._draw_ml(); self.status("Точка {} из 3 на краю частицы (Esc — отмена).".format(len(self._c3)))
                return
            pts, self._c3 = self._c3, None
            self._draw_ml()
            cc = ml_annot.circumcircle(*pts)
            if cc is None or cc[2] < 1.5:
                self.status("Точки почти на одной прямой — поставьте их по краю частицы заново.")
                return
            self._ops().append(dict(op="add", cls="prim", shape=("circle", cc[0], cc[1], cc[2])))
            self._after_op()
            self.status("Круг по трём точкам добавлен (d = {}). Enter — применить.".format(self._fmt_len(2 * cc[2], self.current)))

    def _c3_cancel(self):
        if self._c3:
            self._c3 = None
            if hasattr(self, "canvas"):
                self.canvas.delete("mltmp")

    def _draw_ml(self):
        if not hasattr(self, "canvas"):
            return
        cv = self.canvas
        cv.delete("ml"); cv.delete("mltmp")
        r = self.results.get(self.current) if self.current else None
        if r is not None and "ml" in r and not (self.preview and self.preview["path"] == self.current):
            ml = self._ml(r); z = self._z
            if ml["doubt"] and self.v_circles.get():
                ml["doubt"] = ml_annot.match_doubts(r["circles"], ml["doubt"])
                s = r["s"]
                for c in r["circles"]:
                    if ml_annot.key(c[0], c[1]) in ml["doubt"]:
                        X, Y = self._cv(c[1], c[0]); RR = c[2] / 2 / s * z
                        cv.create_oval(X - RR, Y - RR, X + RR, Y + RR, outline=C_ML, width=2, dash=(4, 3), tags="ml")
                        cv.create_text(X, Y, text="?", fill=C_ML, font=("TkDefaultFont", max(8, min(16, int(RR))), "bold"),
                                       tags="ml")
            if ml["region"]:
                x0, y0, x1, y1 = ml["region"]
                cv.create_rectangle(*self._cv(x0, y0), *self._cv(x1, y1), outline=C_ML, width=2, dash=(8, 4), tags="ml")
                X, Y = self._cv(x0, y0)
                cv.create_text(X + 4, Y + 4, text="зона разметки", fill=C_ML, anchor="nw",
                               font=("TkDefaultFont", 9, "bold"), tags="ml")
        for (px_, py_) in (self._c3 or []):
            X, Y = self._cv(px_, py_); h = self.px(4)
            cv.create_oval(X - h, Y - h, X + h, Y + h, outline=C_ML, width=2, tags="mltmp")

    def _ml_refresh(self):
        """Строки состояния блока ML (о наборе — только когда блок раскрыт)."""
        if not hasattr(self, "lbl_ml"):
            return
        ds = self.mlcfg.get("dataset", "")
        self.lbl_ml_dir.config(text=self._short(ds, 28) if ds else "не выбран")
        if not self.sec_ml.expanded:
            return
        lines = []
        r = self.results.get(self.current) if self.current else None
        if r is not None:
            ml = self._ml(r)
            t = "Снимок: " + ("разметка с нуля" if ml["mode"] == "scratch" else "правка автоматической разметки")
            nd = len(ml_annot.match_doubts(r["circles"], ml["doubt"])) if ml["doubt"] else 0
            if nd:
                t += "; «?»: {}".format(nd)
            if ml["region"]:
                t += "; зона задана"
            if ml.get("exported"):
                t += ("; изменён после добавления в набор ({}) — добавьте снова" if ml.get("dirty") else
                      "; добавлен в набор в {}").format(ml["exported"])
            elif ds and os.path.isdir(ds):
                code = ml_annot.clean_code(self.v_ml_annot.get())
                try:
                    sha = self._ml_sha.get(r["path"]) if hasattr(self, "_ml_sha") else None
                    if sha is None:
                        sha = ml_annot.file_sha1(r["path"]); self._ml_sha = getattr(self, "_ml_sha", {}); self._ml_sha[r["path"]] = sha
                    if code and ml_annot.find_label(ds, code, sha):
                        t += "; уже есть в наборе ({})".format(code)
                except Exception:
                    pass
            lines.append(t)
        if ds and os.path.isdir(ds):
            ni, nl, per = ml_annot.dataset_info(ds)
            lines.append("В наборе: снимков {}, файлов разметки {}{}".format(
                ni, nl, " (" + ", ".join("{} — {}".format(k, v) for k, v in per.items()) + ")" if per else ""))
        self.lbl_ml.config(text="\n".join(lines))

    def _ml_save_annot(self):
        code = ml_annot.clean_code(self.v_ml_annot.get())
        if code != self.v_ml_annot.get():
            self.v_ml_annot.set(code)
        if code != self.mlcfg.get("annotator", ""):
            self.mlcfg["annotator"] = code; ml_annot.save_config(self.mlcfg); self._ml_refresh()

    def _ml_choose_dir(self):
        d = filedialog.askdirectory(parent=self, title="Папка набора для обучения (новая пустая или уже существующий набор)",
                                    initialdir=self.mlcfg.get("dataset") or self.last_dir or None)
        if not d:
            return False
        try:
            ml_annot.ensure_dataset(d)
        except Exception as e:
            messagebox.showerror(APP_TITLE, "Не удалось подготовить папку набора:\n{}".format(e), parent=self)
            return False
        self.mlcfg["dataset"] = d; ml_annot.save_config(self.mlcfg)
        self.sec_ml.expand(); self._ml_refresh()
        self.log("Папка набора для обучения: " + d)
        return True

    def _ml_ready(self, need_code=True):
        """Код разметчика и папка набора заданы (если нет — спросить). Возвращает (код, папка) или None."""
        from tkinter import simpledialog
        code = ml_annot.clean_code(self.v_ml_annot.get())
        if need_code and not code:
            ans = simpledialog.askstring(APP_TITLE, "Код разметчика латиницей (например, A1):", parent=self)
            code = ml_annot.clean_code(ans or "")
            if not code:
                return None
            self.v_ml_annot.set(code); self._ml_save_annot()
        ds = self.mlcfg.get("dataset", "")
        if not ds or not os.path.isdir(ds):
            if not self._ml_choose_dir():
                return None
            ds = self.mlcfg["dataset"]
        return code, ds

    def ml_add_current(self):
        r = self.results.get(self.current) if self.current else None
        if r is None:
            messagebox.showinfo(APP_TITLE, "Сначала обработайте снимок.", parent=self); return
        if r["params"].get("mode") == "only_mod":
            messagebox.showinfo(APP_TITLE, "В режиме «Выделить только модифицированные» первичные частицы не размечаются — "
                                           "такой снимок не подходит для набора.", parent=self); return
        if self.busy:
            return
        if self.pending.get(self.current):
            if not messagebox.askyesno(APP_TITLE, "На снимке есть неприменённые изменения. Применить их и добавить снимок?",
                                       parent=self):
                return
            self._apply_now(self.current); self.refresh_list(); self.redraw(keep_view=True); self._update_pending()
        ready = self._ml_ready()
        if not ready:
            return
        code, ds = ready
        ml = self._ml(r)
        self.config(cursor="watch"); self.update_idletasks()
        try:
            ann = ml_annot.build_annotation(r, ml, code, "NPA " + APP_VERSION, extra=dict(material=self.v_elem.get()))
            lp, upd = ml_annot.add_to_dataset(ds, r["path"], ann)
        except Exception as e:
            messagebox.showerror(APP_TITLE, "Не удалось добавить снимок в набор:\n{}".format(e), parent=self); return
        finally:
            self.config(cursor="")
        ml["exported"] = time.strftime("%H:%M"); ml["dirty"] = False
        st = ann["stats"]
        msg = "{} {} в набор: кругов {}{}, модифицированных {}, НФ {}".format(
            r["name"], "обновлён" if upd else "добавлен", st["n_P"], ", «?» {}".format(st["n_doubt"]) if st["n_doubt"] else "",
            st["n_M"], st["n_I"])
        self.status(msg); self.log(msg + " — " + lp)
        self.sec_ml.expand(); self._ml_refresh()

    def ml_scratch(self):
        r = self.results.get(self.current) if self.current else None
        if r is None:
            messagebox.showinfo(APP_TITLE, "Сначала обработайте снимок.", parent=self); return
        if self.busy:
            return
        if not messagebox.askyesno(APP_TITLE, "Удалить автоматическую разметку снимка {} и разметить его с нуля?\n"
                                              "Контур, круги и крупные частицы будут удалены; вернуть — Ctrl+Z.".format(r["name"]),
                                   parent=self):
            return
        H, W = r["shape"]
        self.pending.pop(self.current, None)
        self._push_hist(self.current)
        npc.apply_edits(r, [dict(op="del_region", shape=("rect", -1, -1, W + 1, H + 1))])
        r["edit"]["ml_scratch"] = True                 # хранится вместе с правками: Ctrl+Z возвращает и режим
        ml = self._ml(r); ml["doubt"] = set(); ml["t_active"] = 0.0; ml["t_last"] = None; ml["dirty"] = True
        self._forget(self.current); self.recompute_all(); self.refresh_list(); self.redraw(keep_view=True)
        self._update_pending(); self._ml_refresh()
        self.sec_edit.expand(); self.v_addcls.set("prim"); self.v_tool.set("add_circle")
        self.status("Автоматическая разметка удалена. Ставьте круги (круг от центра к краю или по трём точкам) и применяйте Enter.")
        self.log("Разметка с нуля: " + r["name"])

    def ml_region_reset(self):
        r = self.results.get(self.current) if self.current else None
        if r is None:
            return
        ml = self._ml(r); ml["region"] = None; ml["dirty"] = True
        self._draw_ml(); self._ml_refresh()
        self.status("Зона разметки — весь кадр.")

    def ml_pack(self):
        ready = self._ml_ready(need_code=False)
        if not ready or self.busy:
            return
        _, ds = ready
        code = ml_annot.clean_code(self.v_ml_annot.get()) or "nabor"
        path = filedialog.asksaveasfilename(parent=self, title="Упаковать набор для переноса", defaultextension=".zip",
                                            initialdir=self.last_dir or os.path.dirname(ds),
                                            initialfile="NPA_nabor_{}_{}.zip".format(code, time.strftime("%Y-%m-%d")),
                                            filetypes=[("Архив ZIP", "*.zip")])
        if not path:
            return
        if os.path.abspath(path).startswith(os.path.abspath(ds) + os.sep):
            messagebox.showinfo(APP_TITLE, "Архив нужно сохранить вне папки набора.", parent=self); return
        self._begin("Упаковываю набор…")

        def job():
            try:
                n = ml_annot.pack_dataset(ds, path)
                mb = os.path.getsize(path) / 1e6
                self.queue.put(("mljob", None, "Набор упакован: {} файлов, {:.1f} МБ — {}".format(n, mb, path)))
            except Exception as e:
                self.queue.put(("job_error", path, "Упаковка набора: {}".format(e)))
        threading.Thread(target=job, daemon=True).start()
        self.last_dir = os.path.dirname(path); self.btn_open.config(state="normal")

    def ml_merge(self):
        ready = self._ml_ready(need_code=False)
        if not ready or self.busy:
            return
        _, ds = ready
        path = filedialog.askopenfilename(parent=self, title="Добавить в набор из архива", initialdir=self.last_dir or None,
                                          filetypes=[("Архив ZIP", "*.zip")])
        if not path:
            return
        self._begin("Добавляю из архива…")

        def job():
            try:
                c = ml_annot.merge_archive(ds, path)
                msg = ("Из архива {}: снимков новых {}, уже были {}; разметки новых {}, обновлено {}, "
                       "старее имеющихся {}{}").format(os.path.basename(path), c["images_new"], c["images_same"], c["labels_new"],
                                                       c["labels_updated"], c["labels_older"],
                                                       "; пропущено файлов {}".format(c["skipped"]) if c["skipped"] else "")
                self.queue.put(("mljob", None, msg))
            except Exception as e:
                self.queue.put(("job_error", path, "Добавление из архива: {}".format(e)))
        threading.Thread(target=job, daemon=True).start()

    def ml_open_dir(self):
        ds = self.mlcfg.get("dataset", "")
        if not ds or not os.path.isdir(ds):
            self._ml_choose_dir(); return
        self._open_dir(ds)

    def show_ml_memo(self):
        win = tk.Toplevel(self); win.title("Памятка: разметка для обучения"); win.transient(self)
        win.geometry("{}x{}".format(self.px(640), self.px(600)))
        fr = ttk.Frame(win, padding=(12, 12, 12, 0)); fr.pack(fill=tk.BOTH, expand=True)
        txt = tk.Text(fr, wrap="word", relief="flat", padx=8, pady=6, font=("TkDefaultFont", 10), spacing1=2)
        sb = ttk.Scrollbar(fr, orient=tk.VERTICAL, command=txt.yview); txt.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y); txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        txt.tag_configure("h", font=("TkDefaultFont", 11, "bold"), spacing1=8, foreground=ACCENT)
        for i, line in enumerate(ml_annot.MEMO.strip().split("\n")):
            head = i == 0 or (line and not line.startswith(("•", " ")) and not line[:1].isdigit())
            txt.insert(tk.END, line + "\n", "h" if head else ())
        txt.configure(state="disabled")
        ttk.Button(win, text="OK", command=win.destroy).pack(pady=8)
        win.bind("<Escape>", lambda e: win.destroy())

    # ================================================================== сохранение
    def _saved(self, path, what):
        self.last_dir = path if os.path.isdir(path) else os.path.dirname(path)
        self.btn_open.config(state="normal")
        self.lbl_saved.config(text="Сохранено — {} ({}):\n{}".format(what, time.strftime("%H:%M"), path))
        self.sec_save.expand()
        self.log("Сохранено ({}): {}".format(what, path))

    def _no_results(self):
        if not self.results:
            messagebox.showinfo(APP_TITLE, "Сначала обработайте снимки.", parent=self)
            return True
        if self.pending.get(self.current):
            if messagebox.askyesno(APP_TITLE, "На текущем снимке есть неприменённые изменения. Применить их перед сохранением?",
                                   parent=self):
                self._apply_now(self.current); self.refresh_list(); self.redraw(keep_view=True); self._update_pending()
        return False

    def _save_layers(self):
        c, k = self.v_contour.get(), self.v_circles.get()
        if not (c or k):
            c = k = True
        return dict(show_contour=c, show_circles=k)

    def save_current_image(self):
        if self._no_results():
            return
        r = self.results.get(self.current)
        if r is None:
            messagebox.showinfo(APP_TITLE, "Выбранный снимок ещё не обработан.", parent=self); return
        path = filedialog.asksaveasfilename(
            parent=self, title="Сохранить разметку снимка", initialdir=self.last_dir or os.path.dirname(r["path"]),
            initialfile=os.path.splitext(r["name"])[0] + "_razmetka.png", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"), ("JPEG", "*.jpg")])
        if not path:
            return
        export.save_image(r, path, scale=2 if self.v_x2.get() else 1, lw=int(self.v_lw.get()), lw_c=int(self.v_lwc.get()),
                          **self._save_layers())
        self._saved(path, "Разметка")

    def save_all_images(self):
        if self._no_results():
            return
        folder = filedialog.askdirectory(parent=self, title="Выберите папку для картинок разметки",
                                         initialdir=self.last_dir or os.path.dirname(self.files[0]), mustexist=False)
        if not folder:
            return
        paths = export.save_images(self.ordered_results(), folder, scale=2 if self.v_x2.get() else 1,
                                   lw=int(self.v_lw.get()), lw_c=int(self.v_lwc.get()), **self._save_layers())
        self._saved(folder, "Картинки разметки ({} шт)".format(len(paths)))

    def save_excel(self, kind):
        if self._no_results():
            return
        settings = self.global_params()
        if settings is None:
            return
        settings["element"] = self.v_elem.get()
        settings["use_3d"] = bool(self.v_3d.get())
        name = "otchet_kratkiy.xlsx" if kind == "short" else "otchet_polnyy.xlsx"
        path = filedialog.asksaveasfilename(
            parent=self, title="Сохранить " + ("краткий" if kind == "short" else "полный") + " отчёт Excel",
            initialdir=self.last_dir or os.path.dirname(self.files[0]), initialfile=name,
            defaultextension=".xlsx", filetypes=[("Книга Excel", "*.xlsx")])
        if not path:
            return
        self.config(cursor="watch"); self.update_idletasks()
        try:
            (export.save_excel_short if kind == "short" else export.save_excel_full)(
                self.ordered_results(), self.summary, settings, path)
        except PermissionError:
            self.config(cursor="")
            messagebox.showerror(APP_TITLE, "Не удалось записать файл. Возможно, он открыт в Excel — закройте его и повторите.",
                                 parent=self)
            return
        finally:
            self.config(cursor="")
        self._saved(path, "Краткий отчёт" if kind == "short" else "Полный отчёт")

    def open_last_dir(self):
        if self.last_dir:
            self._open_dir(self.last_dir)

    def _open_dir(self, path):
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            self.log("Не удалось открыть папку: {}".format(e))

    def save_settings(self):
        g = self.global_params()
        if g is None:
            return
        f = self._scale_fields(quiet=False)
        if f == "bad":
            return
        g["contrast"], g["denoise"] = self._slider_vals()
        g["nm_per_px"], g["bar_nm"] = f
        g["element"] = self.v_elem.get()
        g["use_3d"] = bool(self.v_3d.get())
        path = filedialog.asksaveasfilename(parent=self, title="Сохранить настройки", defaultextension=".json",
                                            initialdir=self.last_dir, filetypes=[("Настройки NPA", "*.json")],
                                            initialfile="nano_settings.json")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(g, fh, ensure_ascii=False, indent=2)
            self._saved(path, "Настройки")

    def load_settings(self):
        path = filedialog.askopenfilename(parent=self, title="Загрузить настройки", initialdir=self.last_dir,
                                          filetypes=[("Настройки NPA", "*.json")])
        if not path:
            return
        with open(path, encoding="utf-8") as fh:
            p = json.load(fh)
        self.v_contrast.set(p.get("contrast", 0.5)); self.v_denoise.set(p.get("denoise", 1.0))
        self.v_dcut.set(p.get("d_cut", 25.0)); self.v_cut.set(p.get("include_cut", True))
        self.v_nmpx.set("" if p.get("nm_per_px") is None else str(p["nm_per_px"]))
        self.v_bar.set("" if p.get("bar_nm") is None else str(p["bar_nm"]))
        self.v_smode.set("nmpx" if p.get("nm_per_px") else "bar" if p.get("bar_nm") else "meta")
        el = p.get("element", "W"); self.v_elem.set(el if el in ELEMENTS else "Другое"); self._on_element()
        md = p.get("mode", "full")
        self.v_nomod.set(md == "no_mod"); self.v_onlymod.set(md == "only_mod")
        self.v_3d.set(bool(p.get("use_3d", True))); self._set_columns()
        if el == "Другое" or el not in npc.MATERIALS:
            self.v_density.set(str(p.get("density", "")))
        for k in self.adv:
            if k in p:
                self.adv[k] = p[k]
        if self.files:            # загруженные контраст, шумоподавление и масштаб — для всех снимков
            self.apply_params(True)
            if p.get("nm_per_px") or p.get("bar_nm"):
                self.apply_scale(True)
        self.log("Настройки загружены: " + path)

    ADV_INFO = {
        "edge": ("Край крупной частицы (доля контраста над подложкой)",
                 "Где проводится граница модифицированной частицы и НФ. Берётся уровень между затемнением подложки рядом "
                 "с частицей (0) и максимумом затемнения самой частицы (1); 0,25 — граница на четверти высоты. "
                 "Значение подобрано на модельных сферах с размытием, типичным для ПЭМ-снимков: при нём эквивалентный "
                 "диаметр совпадает с истинным. Меньше — контур шире, диаметр и масса больше; больше — наоборот. "
                 "Вступает в силу при следующей обработке."),
        "halo_nm": ("Полоса размытия вокруг частицы, нм",
                    "Узкая полоса вокруг модифицированных частиц и НФ, которая не считается агломератом: это размытый край "
                    "самой частицы, а не налипшее вещество. Больше — меньше площадь агломератов возле крупных частиц. "
                    "Вступает в силу при следующей обработке."),
        "prim_dmin": ("Мин. диаметр круга первичной частицы, нм",
                      "Круги первичных частиц мельче этого диаметра отбрасываются как шум. При размере пикселя 0,2–1 нм "
                      "4 нм — это 4–20 пикселей. Вступает в силу при следующей обработке."),
        "prim_dmax": ("Макс. диаметр круга первичной частицы, нм (пусто = до отсечки)",
                      "Верхний предел размера первичной частицы. Пусто — до отсечки «первичные / модифицированные»; "
                      "больше отсечки задать нельзя: первичная частица не бывает крупнее модифицированной. "
                      "Вступает в силу при следующей обработке."),
        "ka": ("k_a (оценка глубины)",
               "Коэффициент формулы числа частиц агломерата по площади его проекции: N = k_a · (S / A_p)^α, где S — площадь "
               "контура, A_p — проекция средней первичной частицы (Köylü и др., 1995). Учитывает частицы, скрытые за другими "
               "в проекции. 1,15 — значение для рыхлых агрегатов сажи (D_f ≈ 1,8). Пересчитывается сразу."),
        "alpha": ("α (оценка глубины)",
                  "Показатель той же формулы N = k_a · (S / A_p)^α. α = 1 означало бы, что скрытых частиц нет; чем плотнее "
                  "агрегат, тем больше α. 1,09 — значение Köylü и др. (1995) для D_f ≈ 1,8. Пересчитывается сразу."),
        "nf_packing": ("Доля заполнения НФ шарами",
                       "Масса тела неправильной формы (НФ) = плотность × эта доля × объём эллипсоида ближайшей формы. "
                       "0,74 — плотнейшая упаковка шаров одного размера; 0,64 — случайная плотная упаковка. "
                       "Пересчитывается сразу."),
        "k_hyst_lo": ("Нижний порог поиска вещества (в шумах фона)",
                      "Поиск вещества идёт с двумя порогами (гистерезис): участок считается веществом, если он темнее фона "
                      "на верхний порог хотя бы в одной точке, а затем расширяется по всем соседним точкам, темнее фона на "
                      "нижний порог. Меньше — находятся более бледные частицы, но появляются ложные пятна на фоне. "
                      "Вступает в силу при следующей обработке."),
        "k_hyst_hi": ("Верхний порог поиска вещества (в шумах фона)",
                      "Верхний порог гистерезиса: насколько темнее фона (в шумах) должна быть хотя бы одна точка участка, "
                      "чтобы он считался веществом. Больше — меньше ложных пятен, но бледные одиночные частицы теряются. "
                      "Вступает в силу при следующей обработке."),
        "film": ("Убирать из контура тёмные участки плёнки (1 — да, 0 — нет)",
                 "Если плёнка-подложка в части кадра толще (ступенька толщины, край сетки), эта часть темнее остального фона "
                 "и без этой проверки целиком попадает в контур агломерата. Вырезаются крупные (от 2000 нм²) участки без "
                 "рельефа частиц и заметно светлее частиц снимка. Проверено на ручной разметке: настоящее вещество почти "
                 "не задевается. Вступает в силу при следующей обработке."),
    }

    def open_advanced(self):
        win = tk.Toplevel(self); win.title("Расширенные настройки"); win.transient(self)
        if not hasattr(self, "_ico_info"):
            self._ico_info = make_icon("info", self.px(18), color="#2a78d6")
        frm = ttk.Frame(win, padding=(8, 8, 8, 0)); frm.pack(fill=tk.BOTH, expand=True)
        frm.columnconfigure(0, weight=1)
        vars_ = {}
        wrap = self.px(520)
        for i, (k, (lab, info)) in enumerate(self.ADV_INFO.items()):
            r = 2 * i
            ttk.Label(frm, text=lab).grid(row=r, column=0, sticky="w", pady=2)
            v = tk.StringVar(value="" if self.adv[k] is None else str(self.adv[k]))
            ttk.Entry(frm, textvariable=v, width=10).grid(row=r, column=1, padx=(8, 4))
            vars_[k] = v
            box = tk.Label(frm, text=info, justify="left", anchor="w", wraplength=wrap, bg="#eef3fb", fg="#222222",
                           padx=8, pady=6, relief="flat")
            box.grid(row=r + 1, column=0, columnspan=3, sticky="ew", pady=(0, 4))
            box.grid_remove()
            ib = tk.Label(frm, image=self._ico_info, cursor="hand2")
            ib.grid(row=r, column=2, padx=(2, 0))

            def toggle(e=None, box=box):
                if box.winfo_ismapped():
                    box.grid_remove()
                else:
                    box.grid()
            ib.bind("<Button-1>", toggle)
            Tip(ib, "Что это за параметр")

        def ok():
            try:
                for k, v in vars_.items():
                    t = v.get().strip().replace(",", ".")
                    self.adv[k] = None if (t == "" and k == "prim_dmax") else float(t)
            except ValueError:
                messagebox.showerror(APP_TITLE, "Проверьте числа в настройках.", parent=win); return
            win.destroy(); self.refresh_list(); self._density_changed()

        def reset():
            for k, v in vars_.items():
                d = npc.DEFAULTS[k]; v.set("" if d is None else str(d))
        bot = ttk.Frame(win, padding=8); bot.pack(fill=tk.X)
        ttk.Button(bot, text="По умолчанию", command=reset).pack(side=tk.LEFT)
        ttk.Button(bot, text="OK", command=ok).pack(side=tk.RIGHT)


    # ------------------------------------------------------------------ о программе
    @staticmethod
    def about_text():
        """Текст окна «О программе». При запуске из исходников — из файла !infoabout.txt рядом с app.py;
        в собранной программе — только встроенный при сборке текст (модуль about_text), правка файла не действует."""
        if not getattr(sys, "frozen", False):
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "!infoabout.txt")
            if os.path.exists(p):
                try:
                    with open(p, encoding="utf-8-sig") as fh:
                        return fh.read()
                except Exception:
                    pass
        try:
            from about_text import TEXT
            return TEXT
        except Exception:
            return "NPA — анализ наночастиц на ПЭМ-снимках. Версия {}.".format(APP_VERSION)

    def show_about(self):
        win = tk.Toplevel(self); win.title("О программе"); win.transient(self)
        win.geometry("{}x{}".format(self.px(620), self.px(560)))
        top = ttk.Frame(win, padding=(12, 12, 12, 6)); top.pack(fill=tk.X)
        if getattr(self, "_icons", None):
            tk.Label(top, image=self._icons[1]).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(top, text="NPA — NanoParticle Analyzer", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(top, text="Анализ наночастиц на ПЭМ-снимках", style="Info.TLabel").pack(anchor="w")
        fr = ttk.Frame(win, padding=(12, 0, 12, 0)); fr.pack(fill=tk.BOTH, expand=True)
        txt = tk.Text(fr, wrap="word", relief="flat", padx=8, pady=6, font=("TkDefaultFont", 10))
        sb = ttk.Scrollbar(fr, orient=tk.VERTICAL, command=txt.yview); txt.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y); txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        txt.insert("1.0", self.about_text()); txt.configure(state="disabled")
        ttk.Button(win, text="OK", command=win.destroy).pack(pady=8)
        win.bind("<Escape>", lambda e: win.destroy())


# ---------------------------------------------------------------------- запуск
def windows_setup():
    """Для Windows (до создания окна): учитывать масштаб экрана 125-200 % и показывать значок NPA на панели задач."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("NPA.NanoparticleAnalyzer")
        except Exception:
            pass
    except Exception:
        pass


def batch_mode(argv):
    """Режим без окна:  NanoAnalyzer.exe --batch ПАПКА_РЕЗУЛЬТАТОВ снимок1.tif снимок2.tif ...
    Настройки — по умолчанию или из nano_settings.json рядом с программой."""
    import glob
    out = argv[0]
    files = []
    for a in argv[1:]:
        files += sorted(glob.glob(a)) if any(ch in a for ch in "*?") else [a]
    os.makedirs(out, exist_ok=True)
    params = dict(npc.DEFAULTS); params["element"] = "W"
    sfile = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "nano_settings.json")
    if os.path.exists(sfile):
        with open(sfile, encoding="utf-8") as fh:
            params.update(json.load(fh))
    log = open(os.path.join(out, "batch_log.txt"), "w", encoding="utf-8")
    res = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        futs = {ex.submit(npc.analyze_image, f, params): f for f in files}
        for fu in as_completed(futs):
            try:
                res.append(fu.result()); log.write("OK " + futs[fu] + "\n")
            except Exception as e:
                log.write("ERROR {}: {}\n".format(futs[fu], e))
            log.flush()
    order = {f: i for i, f in enumerate(files)}
    res.sort(key=lambda r: order.get(r["path"], 0))
    if res:
        pooled = npc.pooled_primary_sizes(res)
        stats = [npc.compute_masses(r, params["density"], pooled) for r in res]
        summ = npc.batch_summary(stats)
        export.save_excel_short(res, summ, params, os.path.join(out, "otchet_kratkiy.xlsx"))
        export.save_excel_full(res, summ, params, os.path.join(out, "otchet_polnyy.xlsx"))
        export.save_images(res, out)
    log.write("done: {} of {}\n".format(len(res), len(files))); log.close()


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--batch":
        batch_mode(sys.argv[2:])
        return
    windows_setup()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    mp.freeze_support()      # обязательно для параллельной обработки в собранном .exe
    main()
