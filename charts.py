# -*- coding: utf-8 -*-
"""
charts.py — графики программы (одни и те же для окна, сохранения картинок и отчёта Excel).

  overall_figure(results, use_3d)  — по всей выборке: первичные, модифицированные, доли, массы;
  image_figure(res)                — по одному снимку: модифицированные, первичные, все частицы.

У столбцов задан gid — текст подсказки при наведении мыши в окне программы.
"""
import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter, NullFormatter

C_MOD = "#1baf7a"; C_MOD_LIGHT = "#9fdcc2"; C_PRIM = "#eb6834"; C_AGG = "#2a78d6"; C_AGG_LIGHT = "#9cc3ee"


def _tidy(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=10, loc="left")
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8); ax.set_axisbelow(True)
    if ylabel == "Число":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def _gid_hist(patches, unit="шт"):
    for p in patches:
        if not hasattr(p, "get_height"):
            continue
        h = p.get_height()
        if h > 0:
            x0 = p.get_x(); p.set_gid("{:.3g}–{:.3g} нм: {:.0f} {}".format(x0, x0 + p.get_width(), h, unit))


def _gid_bars(bars, names, fmt):
    for p, n in zip(bars, names):
        h = p.get_height()
        if np.isfinite(h):
            p.set_gid("{}: {}".format(n, fmt.format(h)))


def _hist_data(ax, bins, counts, names, raw):
    """Данные гистограммы для сохранения в CSV: границы столбцов, число в каждом ряду и исходные значения."""
    rows = [[float(bins[i]), float(bins[i + 1])] + [float(c[i]) for c in counts] for i in range(len(bins) - 1)]
    ax._npa_data = dict(cols=["Диаметр от, нм", "Диаметр до, нм"] + list(names), rows=rows,
                        raw=[(n, [float(v) for v in vals]) for n, vals in raw])


def _bar_data(ax, names, cols, vals):
    """Данные столбчатого графика по снимкам для CSV."""
    if not cols:
        return
    ax._npa_data = dict(cols=["Снимок"] + list(cols),
                        rows=[[n] + [float(v[i]) for v in vals] for i, n in enumerate(names)], raw=[])


def _fmt_csv(v):
    if isinstance(v, str):
        return '"' + v.replace('"', '""') + '"'
    if v is None or not np.isfinite(v):
        return ""
    return ("{:.6g}".format(v)).replace(".", ",")


def save_csv(axes, path):
    """Текстовые данные графиков (CSV: разделитель «;», десятичная запятая, UTF-8) — для построения в другой программе."""
    lines = []
    for ax in axes:
        d = getattr(ax, "_npa_data", None)
        title = ax.get_title(loc="left") or "График"
        lines.append(_fmt_csv(title))
        if not d:
            lines += ["нет данных", ""]
            continue
        lines.append(";".join(_fmt_csv(c) for c in d["cols"]))
        for r in d["rows"]:
            lines.append(";".join(_fmt_csv(v) for v in r))
        for name, vals in d.get("raw", []):
            lines += ["", _fmt_csv("Исходные значения — " + name)]
            lines += [_fmt_csv(v) for v in vals]
        lines.append("")
    with open(path, "w", encoding="utf-8-sig", newline="\r\n") as fh:
        fh.write("\n".join(lines))


def _short(name):
    b = os.path.splitext(name)[0]
    return b if len(b) <= 10 else "…" + b[-9:]


def _note(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="#777", fontsize=9)


def overall_figure(results, use_3d=True, dpi=90, figsize=(8, 6)):
    """Графики по всей выборке (вкладка «Графики», лист «Графики» отчёта)."""
    fig = Figure(figsize=figsize, dpi=dpi, layout="constrained")
    res = results
    mode = res[0]["params"].get("mode", "full") if res else "full"
    dm = [o["deq"] for r in res for o in r["objects"] if o["cls"] == "mod" and not o["cut"]]
    dp = [c[2] for r in res if r["s"] <= 1.0 for c in r["circles"]] or [c[2] for r in res for c in r["circles"]]
    names = [_short(r["name"]) for r in res]
    full = [r["name"] for r in res]
    x = np.arange(len(res))
    step = max(1, int(np.ceil(len(res) / 30)))
    ax = fig.add_subplot(2, 2, 1)
    if dp:
        cn, bn, pt = ax.hist(dp, bins=np.arange(0, max(dp) + 1.5, 1.0), color=C_PRIM, edgecolor="white", linewidth=0.5)
        _gid_hist(pt); _hist_data(ax, bn, [cn], ["Число"], [("Диаметр первичных частиц, нм", dp)])
    elif mode == "only_mod":
        _note(ax, "режим «только модифицированные»")
    _tidy(ax, "Первичные частицы (круги), {} шт".format(len(dp)), "Диаметр, нм", "Число")
    ax = fig.add_subplot(2, 2, 2)
    if dm:
        cn, bn, pt = ax.hist(dm, bins=min(30, max(5, len(dm) // 5)), color=C_MOD, edgecolor="white", linewidth=0.5)
        _gid_hist(pt); _hist_data(ax, bn, [cn], ["Число"], [("Диаметр модифицированных частиц, нм", dm)])
    elif mode == "no_mod":
        _note(ax, "режим «без модифицированных»")
    _tidy(ax, "Модифицированные частицы, {} шт".format(len(dm)), "Диаметр, нм", "Число")
    ax = fig.add_subplot(2, 2, 3)
    if mode == "full":
        if use_3d:
            b1 = ax.bar(x - 0.2, [100 * r["stats"]["frac_flat"] for r in res], 0.4, label="плоская", color=C_MOD_LIGHT)
            b2 = ax.bar(x + 0.2, [100 * r["stats"]["frac_3d"] for r in res], 0.4, label="с оценкой глубины", color=C_MOD)
            _gid_bars(b1, full, "доля модиф. (плоская) {:.1f} %"); _gid_bars(b2, full, "доля модиф. (с глубиной) {:.1f} %")
            _bar_data(ax, full, ["Доля модифицированной массы (плоская), %", "Доля модифицированной массы (с оценкой глубины), %"],
                      [[100 * r["stats"]["frac_flat"] for r in res], [100 * r["stats"]["frac_3d"] for r in res]])
            ax.legend(fontsize=7, frameon=False, loc="upper right")
        else:
            b1 = ax.bar(x, [100 * r["stats"]["frac_flat"] for r in res], 0.6, color=C_MOD)
            _gid_bars(b1, full, "доля модиф. {:.1f} %")
            _bar_data(ax, full, ["Доля модифицированной массы, %"], [[100 * r["stats"]["frac_flat"] for r in res]])
        ax.set_ylim(0, 130); ax.set_yticks(range(0, 101, 20))
    else:
        _note(ax, "доля модифицированной массы\nв этом режиме не считается")
    _tidy(ax, "Доля модифицированной массы, %")
    ax.set_xticks(x[::step]); ax.set_xticklabels(names[::step], rotation=90, fontsize=7)
    ax = fig.add_subplot(2, 2, 4)
    vals = []
    two = mode == "full"
    mcols, mvals = [], []
    if mode != "no_mod":
        v = [r["stats"]["mass_mod_fg"] for r in res]; vals += v
        b1 = ax.bar(x - 0.2 if two else x, v, 0.4, label="модифицированные", color=C_MOD)
        _gid_bars(b1, full, "модифицированные {:.2f} фг"); mcols.append("Масса модифицированных, фг"); mvals.append(v)
    if mode != "only_mod":
        key, lab = ("mass_agg_3d_fg", "агломераты (с глубиной)") if use_3d else ("mass_agg_flat_fg", "агломераты")
        v = [r["stats"][key] for r in res]; vals += v
        b2 = ax.bar(x + 0.2 if two else x, v, 0.4, label=lab, color=C_AGG)
        _gid_bars(b2, full, lab + " {:.2f} фг"); mcols.append("Масса: " + lab + ", фг"); mvals.append(v)
    _bar_data(ax, full, mcols, mvals)
    _tidy(ax, "Массы, фг")
    top = max([v for v in vals if np.isfinite(v)] + [1e-9])
    ax.set_ylim(0, top * 1.35)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_xticks(x[::step]); ax.set_xticklabels(names[::step], rotation=90, fontsize=7)
    return fig


def image_figure(r, dpi=90, figsize=(8, 6)):
    """Графики одного снимка (вкладка «Графики <снимок>»)."""
    fig = Figure(figsize=figsize, dpi=dpi, layout="constrained")
    dm = np.array([o["deq"] for o in r["objects"] if o["cls"] == "mod" and not o["cut"]])
    dp = np.array([c[2] for c in r["circles"]])
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    ax._npa_kind = "img_mod"
    if dm.size:
        cn, bn, pt = ax.hist(dm, bins=min(20, max(4, dm.size // 2)), color=C_MOD, edgecolor="white", linewidth=0.5)
        _gid_hist(pt); _hist_data(ax, bn, [cn], ["Число"], [("Диаметр модифицированных частиц, нм", dm)])
    else:
        _note(ax, "нет модифицированных\n(не обрезанных краем)")
    _tidy(ax, "Модифицированные, {} шт".format(dm.size), "Диаметр, нм", "Число")
    ax = fig.add_subplot(gs[0, 1])
    if dp.size:
        cn, bn, pt = ax.hist(dp, bins=np.arange(0, dp.max() + 1.5, 1.0), color=C_PRIM, edgecolor="white", linewidth=0.5)
        _gid_hist(pt); _hist_data(ax, bn, [cn], ["Число"], [("Диаметр первичных частиц, нм", dp)])
    _tidy(ax, "Первичные, {} шт".format(dp.size), "Диаметр, нм", "Число")
    ax = fig.add_subplot(gs[1, :])
    allv = np.concatenate([dp, dm]) if (dp.size + dm.size) else np.array([])
    if allv.size:
        bins = np.logspace(np.log10(max(1.0, allv.min() * 0.9)), np.log10(allv.max() * 1.1), 60)
        data = [a for a in (dp, dm) if a.size]
        cols = [c for a, c in ((dp, C_PRIM), (dm, C_MOD)) if a.size]
        labs = [l for a, l in ((dp, "первичные"), (dm, "модифицированные")) if a.size]
        cn, bn, pts = ax.hist(data, bins=bins, stacked=True, color=cols, label=labs, edgecolor="white", linewidth=0.4)
        cn = [cn] if len(data) == 1 else list(cn)
        cn = [cn[0]] + [cn[i] - cn[i - 1] for i in range(1, len(cn))]      # stacked -> отдельные ряды
        _hist_data(ax, bn, cn, ["Число: " + l for l in labs], [("Диаметр, нм: " + l, a) for l, a in zip(labs, data)])
        groups = pts if len(data) > 1 else [pts]
        for lab, g in zip(labs, groups):
            for p in g:
                h = p.get_height()
                if h > 0:
                    p.set_gid("{}: {:.3g}–{:.3g} нм, {:.0f} шт".format(lab, p.get_x(), p.get_x() + p.get_width(), h))
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=8, frameon=False)
    _tidy(ax, "Все частицы, {} шт".format(allv.size), "Диаметр, нм (логарифмическая шкала)", "Число")
    return fig


def save_axes(fig, ax, path, dpi=200):
    """Сохраняет один график (оси с подписями) из общего рисунка."""
    ensure_canvas(fig)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = ax.get_tightbbox(r).expanded(1.02, 1.04).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, dpi=dpi, bbox_inches=bb, facecolor="white")


def save_figure(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, facecolor="white")


def ensure_canvas(fig):
    """Рисунку без окна нужен «холст» для отрисовки в файл."""
    if fig.canvas is None or type(fig.canvas).__name__ == "FigureCanvasBase":
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
    return fig


def axes_list(fig):
    """Графики рисунка с названиями (для сохранения по отдельности)."""
    out = []
    for i, ax in enumerate(fig.axes, 1):
        t = ax.get_title(loc="left") or "график {}".format(i)
        out.append((ax, t))
    return out
