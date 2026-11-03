"""
battery_app.py - Battery Cycling Analyzer (Python version of the MATLAB app)

Run it:   in Spyder press F5,   or in Miniforge Prompt:
          conda activate battery
          python battery_app.py

Needs battery_core.py in the same folder.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy import QtCore, QtGui, QtWidgets as W
from qtpy.QtCore import Qt

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as Toolbar
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent))
import battery_core as bc                                     # noqa: E402

HEART, FLOW, FLOW2 = "\u2665", "\u273f", "\u2740"
TABS = [("vq", f"{HEART} Profiles"), ("plat", f"{FLOW} Hysteresis"), ("cap", f"{FLOW} Capacity & CE"),
        ("rate", f"{HEART} Rate"), ("dqdv", f"{HEART} dQ/dV"), ("cv", f"{FLOW2} CV"),
        ("cvk", f"{FLOW} CV kinetics"), ("ps", f"{HEART} Plating/Stripping"), ("eis", f"{FLOW} Impedance"),
        ("vt", f"{FLOW2} V vs time"), ("table", f"{HEART} Table")]
THEMES = {
    "Blossom (pink)": dict(bg="#FFF6FA", panel="#FFFFFF", field="#FFFFFF", text="#000000", title="#7A2E5C",
                           btn="#FCE6F0", btntext="#000000", prim="#EBA9CA", primtext="#000000",
                           axbg="#FFFFFF", axfg="#000000"),
    "Blossom (deep pink)": dict(bg="#FFEBF5", panel="#FFFFFF", field="#FFFFFF", text="#000000", title="#6B0047",
                                btn="#FFD1E6", btntext="#000000", prim="#C20066", primtext="#FFFFFF",
                                axbg="#FFFFFF", axfg="#000000"),
    "Lavender": dict(bg="#F2EDFF", panel="#FFFFFF", field="#FFFFFF", text="#000000", title="#4D147F",
                     btn="#E0D1FF", btntext="#000000", prim="#6B1FA8", primtext="#FFFFFF",
                     axbg="#FFFFFF", axfg="#000000"),
    "Mint": dict(bg="#E8FAF2", panel="#FFFFFF", field="#FFFFFF", text="#000000", title="#00543D",
                 btn="#C2F0DB", btntext="#000000", prim="#007352", primtext="#FFFFFF",
                 axbg="#FFFFFF", axfg="#000000"),
    "Plain (white)": dict(bg="#FFFFFF", panel="#FFFFFF", field="#FFFFFF", text="#000000", title="#000000",
                          btn="#D9EBFF", btntext="#000000", prim="#003399", primtext="#FFFFFF",
                          axbg="#FFFFFF", axfg="#000000"),
    "Night (dark)": dict(bg="#000000", panel="#121212", field="#1F1F1F", text="#FFFFFF", title="#FFB8D9",
                         btn="#4D1A38", btntext="#FFFFFF", prim="#FF8CC7", primtext="#000000",
                         axbg="#000000", axfg="#FFFFFF"),
}
FONTS = ["Verdana", "Atkinson Hyperlegible", "Tahoma", "Segoe UI", "Arial", "Calibri", "Helvetica",
         "Lexend", "Trebuchet MS", "Georgia", "DejaVu Sans"]
LEG_PLACES = ["Draggable (drag it anywhere)", "Best", "Top right", "Top left", "Bottom right", "Bottom left",
              "Outside right", "Outside below", "Hidden"]
LOC = {"Best": "best", "Top right": "upper right", "Top left": "upper left", "Bottom right": "lower right",
       "Bottom left": "lower left"}


def _float(text):
    try:
        v = float(str(text).replace(",", "."))
        return v if np.isfinite(v) else np.nan
    except ValueError:
        return np.nan


class App(W.QMainWindow):
    # ------------------------------------------------------------------ setup
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{HEART} Battery Cycling Analyzer {FLOW}")
        self.DS: list[bc.Dataset] = []
        self.next_id = 1
        self.sel: list[int] = []
        self.avail: list[int] = []
        self.lim = {k: {"x": [0, 1, True], "y": [0, 1, True], "y2": [0, 1, True]} for k, _ in TABS}
        self.legpos = {}
        self.spec = None
        self.axes = []
        self.axes_key = None
        self.export_list = []
        self.last_dir = str(Path.home())
        self.reg = {}                                  # widgets saved in sessions
        self.ready = False

        root = W.QWidget(objectName="root")
        self.setCentralWidget(root)
        lay = W.QHBoxLayout(root)
        self.split = W.QSplitter(Qt.Horizontal)
        lay.addWidget(self.split)
        self.split.addWidget(self._build_left())
        self.split.addWidget(self._build_center())
        self.split.addWidget(self._build_right())
        self.split.setStretchFactor(0, 9)
        self.split.setStretchFactor(1, 18)
        self.split.setStretchFactor(2, 5)

        self.ready = True
        self.apply_theme()
        self.resize(1700, 1000)
        self.split.setSizes([470, 860, 370])      # left / plot / right (drag the dividers to change)
        self.redraw()

    # small widget factories ------------------------------------------------
    def _combo(self, name, items, data=None, cb=None, value=None):
        w = W.QComboBox()
        w.setSizeAdjustPolicy(W.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        w.setMinimumContentsLength(8)
        w.setSizePolicy(W.QSizePolicy.Preferred, W.QSizePolicy.Fixed)
        w.setMinimumWidth(70)
        for i, it in enumerate(items):
            w.addItem(it, data[i] if data else it)
        if value is not None:
            w.setCurrentText(value)
        w.currentIndexChanged.connect(cb or (lambda *_: self.redraw()))
        self.reg[name] = w
        return w

    def _check(self, name, text, value=True, cb=None):
        w = W.QCheckBox(text)
        w.setChecked(value)
        w.toggled.connect(cb or (lambda *_: self.redraw()))
        self.reg[name] = w
        return w

    def _spin(self, name, lo, hi, value, step=1, decimals=None, cb=None):
        w = W.QDoubleSpinBox() if decimals is not None else W.QSpinBox()
        if decimals is not None:
            w.setDecimals(decimals)
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setValue(value)
        w.setKeyboardTracking(False)
        w.setMinimumWidth(60)
        w.setSizePolicy(W.QSizePolicy.Preferred, W.QSizePolicy.Fixed)
        w.valueChanged.connect(cb or (lambda *_: self.redraw()))
        self.reg[name] = w
        return w

    def _button(self, text, cb, primary=False):
        b = W.QPushButton(text)
        b.clicked.connect(lambda *_: cb())
        if primary:
            b.setProperty("primary", True)
        return b

    @staticmethod
    def _row(*widgets, stretch_last=True):
        w = W.QWidget()
        h = W.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for x in widgets:
            h.addWidget(W.QLabel(x) if isinstance(x, str) else x)
        if not stretch_last:
            h.addStretch(1)
        return w

    @staticmethod
    def _orow(*widgets, stretch_last=False):
        """compact row for the option bars above the plot"""
        w = W.QWidget()
        h = W.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        for x in widgets:
            if isinstance(x, str):
                x = W.QLabel(x)
            elif isinstance(x, W.QComboBox):
                x.setMinimumContentsLength(16)
                x.setSizePolicy(W.QSizePolicy.Preferred, W.QSizePolicy.Fixed)
            elif isinstance(x, (W.QSpinBox, W.QDoubleSpinBox)):
                x.setSizePolicy(W.QSizePolicy.Fixed, W.QSizePolicy.Fixed)
                x.setFixedWidth(95)
            h.addWidget(x, 0)
        h.addStretch(1)
        return w

    # left: data sets + details ---------------------------------------------
    def _build_left(self):
        w = W.QWidget()
        v = W.QVBoxLayout(w)
        g = W.QGridLayout()
        g.addWidget(self._button(f"{HEART} Load files", self.on_load, True), 0, 0)
        g.addWidget(self._button("Remove", self.on_remove), 0, 1)
        g.addWidget(self._button("Clear all", self.on_clear), 0, 2)
        g.addWidget(self._button(f"{FLOW} Open session", self.on_open_session), 1, 0)
        g.addWidget(self._button(f"{FLOW} Save session", self.on_save_session), 1, 1)
        v.addLayout(g)

        box = W.QGroupBox(f"{FLOW} Data sets   (tick Show, type Load, Unit, Area)")
        bv = W.QVBoxLayout(box)
        bv.addWidget(self._row(self._button(f"{HEART} Select all", lambda: self.on_show_all(True)),
                               self._button("Deselect all", lambda: self.on_show_all(False))))
        self.tbl = W.QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(["Show", "Name", "Type", "Cyc", "Load", "Unit", "Area (cm\u00b2)"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.itemChanged.connect(self.on_set_item)
        self.tbl.currentCellChanged.connect(lambda r, *_: self.on_set_row(r))
        bv.addWidget(self.tbl)
        v.addWidget(box, 3)

        box = W.QGroupBox(f"{FLOW2} Details")
        f = W.QFormLayout(box)
        self.dd_info = W.QComboBox()
        self.dd_info.currentIndexChanged.connect(lambda *_: self.on_info_changed())
        f.addRow("Details for:", self.dd_info)
        self.lbl_big = W.QLabel(f"{HEART} 0 cycles", objectName="big")
        f.addRow(self.lbl_big)
        self.info_keys = ["Cycle range", "Loading", "First step", "1st discharge", "1st charge", "1st CE",
                          "Last discharge", "Retention"]
        self.info_val = {}
        for k in self.info_keys:
            lab = W.QLabel("-")
            lab.setWordWrap(True)
            self.info_val[k] = lab
            f.addRow(k + ":", lab)
        v.addWidget(box, 2)
        return w

    # centre: tabs, options, plot -------------------------------------------
    def _build_center(self):
        w = W.QWidget()
        v = W.QVBoxLayout(w)
        self.banner = W.QLabel(f"{FLOW} {HEART}  Battery Cycling Analyzer  {HEART} {FLOW}", objectName="banner")
        self.banner.setAlignment(Qt.AlignCenter)
        bm, bp = self._button("A -", lambda: self.font_step(-1)), self._button("A +", lambda: self.font_step(+1))
        for b in (bm, bp):
            b.setFixedWidth(60)
        v.addWidget(self._row(bm, self.banner, bp))
        self.tabs = W.QTabBar()
        for _, title in TABS:
            self.tabs.addTab(title.replace("&", "&&"))
        self.tabs.setExpanding(False)
        self.tabs.currentChanged.connect(lambda *_: self.on_tab_changed())
        v.addWidget(self.tabs)

        self.opt_stack = W.QStackedWidget()
        self.opt_stack.setSizePolicy(W.QSizePolicy.Preferred, W.QSizePolicy.Maximum)
        for key, _ in TABS:
            self.opt_stack.addWidget(self._options_page(key))
        v.addWidget(self.opt_stack)

        self.view_stack = W.QStackedWidget()
        pw = W.QWidget()
        pv = W.QVBoxLayout(pw)
        pv.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(layout="constrained")
        self.canvas = Canvas(self.fig)
        self.toolbar = Toolbar(self.canvas, self)
        self.toolbar.setIconSize(QtCore.QSize(18, 18))
        pv.addWidget(self.toolbar)
        pv.addWidget(self.canvas, 1)
        self.view_stack.addWidget(pw)
        self.cyc_tbl = W.QTableWidget()
        self.view_stack.addWidget(self.cyc_tbl)
        v.addWidget(self.view_stack, 1)

        self.status = W.QLabel("Load Arbin or PAT-Tester files to begin.")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        return w

    def _options_page(self, key):
        page = W.QWidget()
        g = W.QVBoxLayout(page)
        g.setContentsMargins(0, 0, 0, 0)
        add = g.addWidget
        if key == "vq":
            add(self._orow("Capacity in:", self._combo("cap_unit", ["mAh/g  (specific)", "mAh/cm\u00b2  (areal)",
                                                                    "mAh  (absolute)"], ["g", "cm2", "abs"])))
            add(self._orow("Smoothing:", self._combo("vq_smooth", bc.SMOOTH), "window",
                          self._spin("vq_win", 3, 301, 11, 2)))
        elif key == "plat":
            add(self._orow(self._check("hy_mirror", "Mirrored charge (top axis)"), self._check("hy_arrows", "Arrows"),
                          self._check("hy_lis", "Li-S labels"), self._check("hy_dv", "\u0394V at 50 %"),
                          stretch_last=False))
            add(self._orow(self._check("hy_blackred", "Black discharge / red charge"),
                          self._check("hy_right", "Right axis numbers", False),
                          self._check("hy_align", "Start each charge at its own discharge end", False),
                          stretch_last=False))
        elif key == "cap":
            add(self._orow("Show:", self._combo("cap_mode", [
                "Discharge & charge (3 panels)", "Discharge + charge (1 panel)", "Discharge only", "Charge only",
                "Energy, average voltage & energy efficiency", "Hysteresis dV (50 %) vs cycle"],
                cb=lambda *_: self.view_changed("cap"))))
        elif key == "rate":
            add(self._orow("Show:", self._combo("rate_view", ["Capacity vs cycle (labelled rates)",
                                                             "Average capacity per rate"],
                                               cb=lambda *_: self.view_changed("rate"))))
            add(self._orow("Label rates as:", self._combo("rate_label", ["Specific current (A/g)", "C-rate"],
                                                         cb=lambda *_: self.view_changed("rate")),
                          "1C (mAh/g) =", self._spin("rate_ref", 1, 100000, 1675, 25, 1)))
        elif key == "dqdv":
            add(self._orow("Points:", self._spin("dq_bins", 20, 2000, 200, 20), "Smoothing:",
                          self._spin("dq_smooth", 1, 51, 5, 2), stretch_last=False))
        elif key == "cv":
            add(self._orow("Y axis:", self._combo("cv_unit", ["Current (mA)", "Specific current (A/g)",
                                                             "Specific current (mA/mg)", "Specific current (mA/g)",
                                                             "Current density (mA/cm\u00b2)"],
                                                 ["mA", "Ag", "mAmg", "mAg", "mAcm2"]),
                          self._check("cv_rate_legend", "Scan rate in legend")))
            add(self._orow("Smoothing:", self._combo("cv_smooth", bc.SMOOTH), "window",
                          self._spin("cv_win", 3, 301, 5, 2)))
        elif key == "cvk":
            add(self._orow("Show:", self._combo("k_view", ["Peak current vs sqrt(scan rate)  (Randles-Sevcik)",
                                                          "log|i_p| vs log(scan rate)  (b-value)",
                                                          "Peak potential vs scan rate"],
                                               cb=lambda *_: self.view_changed("cvk")),
                          "Cycle per rate:", self._combo("k_pick", ["Last", "First", "All"])))
            add(self._orow("Cathodic peak between", self._spin("k_c1", -10, 10, 0, 0.05, 3), "and",
                          self._spin("k_c2", -10, 10, 5, 0.05, 3), "V", stretch_last=False))
            add(self._orow("Anodic peak between", self._spin("k_a1", -10, 10, 0, 0.05, 3), "and",
                          self._spin("k_a2", -10, 10, 5, 0.05, 3), "V", stretch_last=False))
            add(self._orow("Electrons n =", self._spin("k_n", 0.01, 100, 1, 1, 2),
                          "Li\u207a concentration (mol/L) =", self._spin("k_conc", 1e-6, 100, 1, 0.1, 4),
                          stretch_last=False))
        elif key == "ps":
            add(self._orow("Show:", self._combo("ps_view", ["Voltage vs time", "Voltage vs areal capacity",
                                                           "Overpotential vs cycle", "Coulombic efficiency (Li||Cu)"],
                                               cb=lambda *_: self.view_changed("ps"))))
            add(self._orow(self._check("ps_sel", "Selected cycles only", False),
                          self._check("ps_max", "Max (nucleation) overpotential"), stretch_last=False))
        elif key == "eis":
            add(self._orow("Show:", self._combo("eis_view", ["Nyquist", "Bode: |Z| + phase", "Bode: |Z|",
                                                            "Bode: phase", "Resistances vs cycle (Rs, Rct)",
                                                            "C12 vs cycle"],
                                               cb=lambda *_: self.view_changed("eis")),
                          "Z:", self._combo("eis_z", ["Z12 (cell)", "Z1 (working vs ref.)", "Z2 (counter vs ref.)"],
                                            [0, 1, 2])))
            add(self._orow(self._check("eis_area", "\u03a9 cm\u00b2 (area-normalized)", False),
                          self._check("eis_equal", "Equal axes (Nyquist)"), stretch_last=False))
        else:
            add(W.QLabel(""))
        return page

    # right: cycles / axes & look / export ------------------------------------
    def _scroll(self, inner):
        s = W.QScrollArea()
        s.setFrameShape(W.QFrame.NoFrame)
        inner.layout().setContentsMargins(10, 8, 10, 8)
        s.setWidgetResizable(True)
        s.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        s.setWidget(inner)
        return s

    def _build_right(self):
        tw = W.QTabWidget()
        tw.setUsesScrollButtons(False)
        tw.tabBar().setExpanding(True)
        tw.setMinimumWidth(340)                      # everything in Cycles / Look / Export fits
        # --- cycles
        w = W.QWidget()
        v = W.QVBoxLayout(w)
        self.lbl_avail = W.QLabel("Available: -")
        v.addWidget(self.lbl_avail)
        self.lst = W.QListWidget()
        self.lst.setSelectionMode(W.QAbstractItemView.ExtendedSelection)
        self.lst.itemSelectionChanged.connect(self.on_cycle_list)
        v.addWidget(self.lst, 1)
        self.ed_quick = W.QLineEdit("1 2 3")
        self.ed_quick.returnPressed.connect(self.on_quick)
        self.reg["quick"] = self.ed_quick
        v.addWidget(self._row(self.ed_quick, self._button("Apply", self.on_quick, True)))
        hint = W.QLabel("e.g. 1 2 3 | 1:10 | 1 10:10:100 | end | all")
        hint.setWordWrap(True)
        v.addWidget(hint)
        v.addWidget(self._row(self._button("All", lambda: self.set_selection(self.avail)),
                              self._button("None", lambda: self.set_selection([])),
                              self._button("First + last", lambda: self.set_selection(
                                  sorted({self.avail[0], self.avail[-1]}) if self.avail else []))))
        self.sld = W.QSlider(Qt.Horizontal)
        self.sld.valueChanged.connect(self.on_scrub)
        v.addWidget(self._row("Scrub:", self.sld))
        self.lbl_sel = W.QLabel("0 selected")
        v.addWidget(self._row(self._check("show_dis", "Discharge"), self._check("show_ch", "Charge"),
                              self.lbl_sel))
        tw.addTab(self._scroll(w), f"{HEART} Cycles")

        # --- axes & look
        w = W.QWidget()
        v = W.QVBoxLayout(w)
        box = W.QGroupBox(f"{HEART} Axis limits   (current plot tab)")
        g = W.QGridLayout(box)
        for c, t in enumerate(["", "Lower", "Upper", "Auto"]):
            g.addWidget(W.QLabel(t), 0, c)
        self.limw = {}
        for r, (dim, txt) in enumerate((("x", "X"), ("y", "Y"), ("y2", "Y2")), start=1):
            lo, hi = W.QDoubleSpinBox(), W.QDoubleSpinBox()
            for s in (lo, hi):
                s.setRange(-1e9, 1e9)
                s.setDecimals(3)
                s.setKeyboardTracking(False)
                s.setMinimumWidth(60)
                s.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Fixed)
                s.valueChanged.connect(lambda *_, d=dim: self.on_lim_edit(d))
            au = W.QCheckBox()
            au.setChecked(True)
            au.toggled.connect(lambda *_, d=dim: self.on_lim_auto(d))
            lab = W.QLabel(txt)
            for c, x in enumerate((lab, lo, hi, au)):
                g.addWidget(x, r, c)
            self.limw[dim] = (lo, hi, au, lab)
        g.addWidget(self._button("All auto (this tab)", self.on_all_auto), 4, 1, 1, 3)
        v.addWidget(box)

        box = W.QGroupBox(f"{FLOW2} Display")
        f = W.QFormLayout(box)
        f.addRow("Window colors", self._combo("theme", list(THEMES), cb=lambda *_: self.apply_theme()))
        fams = self._fonts()
        f.addRow("Font", self._combo("font", fams, cb=lambda *_: self.apply_theme()))
        f.addRow("Text size", self._spin("font_size", 7, 32, 9, 1, cb=lambda *_: self.apply_theme()))
        f.addRow("Plot text size", self._spin("plot_fs", 5, 32, 8, 1))
        self.sld_w, self.sld_h = W.QSlider(Qt.Horizontal), W.QSlider(Qt.Horizontal)
        for name, s in (("chart_w", self.sld_w), ("chart_h", self.sld_h)):
            s.setMinimumWidth(60)
            s.setSizePolicy(W.QSizePolicy.Ignored, W.QSizePolicy.Fixed)
            s.setRange(40, 100)
            s.setValue(90)
            s.valueChanged.connect(lambda *_: self.redraw())
            self.reg[name] = s
        f.addRow(f"{HEART} Chart width", self.sld_w)
        f.addRow(f"{HEART} Chart height", self.sld_h)
        f.addRow("Line width", self._spin("lw", 0.5, 6, 2, 0.5, 1))
        f.addRow("Plot colors", self._combo("scheme", bc.SCHEMES, cb=lambda *_: self.on_scheme()))
        f.addRow("Legend place", self._combo("leg_place", LEG_PLACES))
        f.addRow("Legend size", self._combo("leg_size", ["Auto"] + [str(i) for i in (8, 9, 10, 11, 12, 14, 16, 18,
                                                                                      20, 24, 28)]))
        f.addRow("Tick direction", self._combo("tickdir", ["in", "out"]))
        f.addRow("", self._check("grid", "Grid", False))
        v.addWidget(box)
        v.addStretch(1)
        tw.addTab(self._scroll(w), f"{FLOW} Look")

        # --- export
        w = W.QWidget()
        v = W.QVBoxLayout(w)
        lab = W.QLabel("Exports what is plotted in the current tab.")
        lab.setWordWrap(True)
        v.addWidget(lab)
        v.addWidget(self._check("crop", "Crop to axis limits", False, cb=lambda *_: None))
        v.addWidget(self._combo("hdr", ["Origin headers (3 rows)", "Simple header (1 row)"], cb=lambda *_: None))
        v.addWidget(self._button(f"{HEART} Copy to clipboard", self.on_copy, True))
        v.addWidget(self._button("Save to Excel...", self.on_save_excel))
        v.addWidget(self._button("Add to export list", self.on_add_list))
        v.addWidget(self._button("Write list to Excel...", self.on_write_list))
        self.lbl_list = W.QLabel("Export list: empty")
        v.addWidget(self.lbl_list)
        v.addWidget(self._button("Clear list", self.on_clear_list))
        v.addWidget(self._button("Save PNG...", self.on_save_png))
        v.addWidget(self._button(f"{HEART} Journal figure...", self.on_journal, True))
        v.addStretch(1)
        tw.addTab(self._scroll(w), f"{FLOW2} Export")
        return tw

    def _fonts(self):
        try:
            have = set(QtGui.QFontDatabase.families())
        except TypeError:
            have = set(QtGui.QFontDatabase().families())
        f = [x for x in FONTS if x in have]
        return f or [QtGui.QFont().family()]

    # ------------------------------------------------------------------ theme
    def apply_theme(self):
        if not self.ready:
            return
        t = THEMES[self.reg["theme"].currentText()]
        fs = self.reg["font_size"].value()
        fam = self.reg["font"].currentText()
        px = int(max(14, min(28, fs * 2)))
        self.toolbar.setIconSize(QtCore.QSize(px, px))
        self.setStyleSheet(f"""
            QWidget {{ font-family: "{fam}"; font-size: {fs}pt; color: {t['text']}; }}
            QMainWindow, QWidget#root, QSplitter {{ background: {t['bg']}; }}
            QScrollArea, QStackedWidget, QTabWidget::pane {{ background: {t['panel']}; }}
            QGroupBox {{ background: {t['panel']}; border: 2px solid {t['prim']}; border-radius: 6px;
                         margin-top: 1.3em; font-weight: bold; padding: 3px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {t['title']}; }}
            QLabel, QCheckBox {{ font-weight: bold; color: {t['text']}; background: transparent; }}
            QLabel#big {{ font-size: {fs + 10}pt; color: {t['title']}; }}
            QLabel#banner {{ font-size: {fs + 6}pt; color: {t['title']}; }}
            QPushButton {{ background: {t['btn']}; color: {t['btntext']}; font-weight: bold;
                           border: 1px solid {t['prim']}; border-radius: 6px; padding: 4px 10px; }}
            QPushButton[primary="true"] {{ background: {t['prim']}; color: {t['primtext']}; }}
            QTabBar::tab {{ background: {t['btn']}; color: {t['btntext']}; padding: 5px 10px; margin: 1px;
                            border-radius: 4px; }}
            QTabBar::tab:selected {{ background: {t['prim']}; color: {t['primtext']}; }}
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QListWidget, QTableWidget, QHeaderView::section
                {{ background: {t['field']}; color: {t['text']}; }}
            QListWidget::item:selected, QTableWidget::item:selected
                {{ background: {t['prim']}; color: {t['primtext']}; }}
        """)
        self.redraw()

    def font_step(self, d):
        p = self.reg["plot_fs"]
        p.blockSignals(True)
        p.setValue(min(max(p.value() + d, p.minimum()), p.maximum()))
        p.blockSignals(False)
        s = self.reg["font_size"]
        s.setValue(min(max(s.value() + d, s.minimum()), s.maximum()))

    def style(self, fs=None, lw=None, chart=True):
        t = THEMES[self.reg["theme"].currentText()]
        return dict(fs=fs or self.reg["plot_fs"].value(), family=self.reg["font"].currentText(),
                    lw=lw or self.reg["lw"].value(), fg=t["axfg"], bg=t["axbg"],
                    dark=t["axbg"] == "#000000", grid=self.reg["grid"].isChecked(),
                    tickdir=self.reg["tickdir"].currentText(), leg_place=self.reg["leg_place"].currentText(),
                    leg_size=self.reg["leg_size"].currentText(),
                    cw=self.sld_w.value() if chart else 100, ch=self.sld_h.value() if chart else 100,
                    right_numbers=self.reg["hy_right"].isChecked())

    # ---------------------------------------------------------------- options
    def key(self):
        return TABS[self.tabs.currentIndex()][0]

    def opts(self):
        r = self.reg
        cur = lambda n: r[n].currentData()
        txt = lambda n: r[n].currentText()
        chk = lambda n: r[n].isChecked()
        val = lambda n: r[n].value()
        return dict(cycles=list(self.sel), scheme=txt("scheme"), cap_unit=cur("cap_unit"),
                    show_dis=chk("show_dis"), show_ch=chk("show_ch"), vq_smooth=txt("vq_smooth"), vq_win=val("vq_win"),
                    dq_bins=val("dq_bins"), dq_smooth=val("dq_smooth"),
                    hy_mirror=chk("hy_mirror"), hy_align=chk("hy_align"), hy_blackred=chk("hy_blackred"),
                    hy_lis=chk("hy_lis"), hy_arrows=chk("hy_arrows"), hy_dv=chk("hy_dv"),
                    cap_mode=txt("cap_mode"), rate_view=txt("rate_view"), rate_label=txt("rate_label"),
                    rate_ref=val("rate_ref"), cv_unit=cur("cv_unit"), cv_rate_legend=chk("cv_rate_legend"),
                    cv_smooth=txt("cv_smooth"), cv_win=val("cv_win"), k_view=txt("k_view"), k_pick=txt("k_pick"),
                    k_cwin=(val("k_c1"), val("k_c2")), k_awin=(val("k_a1"), val("k_a2")), k_n=val("k_n"),
                    k_conc=val("k_conc"), ps_view=txt("ps_view"), ps_sel=chk("ps_sel"), ps_max=chk("ps_max"),
                    eis_view=txt("eis_view"), eis_z=cur("eis_z"), eis_area=chk("eis_area"),
                    eis_equal=chk("eis_equal"))

    def build_spec(self, key):
        o, D = self.opts(), self.DS
        return {"vq": lambda: bc.build_profiles(D, o, "vq"), "dqdv": lambda: bc.build_profiles(D, o, "dqdv"),
                "vt": lambda: bc.build_profiles(D, o, "vt"), "plat": lambda: bc.build_hysteresis(D, o),
                "cap": lambda: bc.build_capacity(D, o), "rate": lambda: bc.build_rate(D, o),
                "cv": lambda: bc.build_cv(D, o), "cvk": lambda: bc.build_cvk(D, o),
                "ps": lambda: bc.build_ps(D, o), "eis": lambda: bc.build_eis(D, o)}[key]()

    # ---------------------------------------------------------------- drawing
    def redraw(self):
        if not self.ready:
            return
        key = self.key()
        cur = self.tabs.currentIndex()
        if self.opt_stack.currentIndex() != cur or not getattr(self, "_opt_sized", False):
            for i in range(self.opt_stack.count()):
                pol = W.QSizePolicy.Preferred if i == cur else W.QSizePolicy.Ignored
                self.opt_stack.widget(i).setSizePolicy(pol, pol)
            self.opt_stack.setCurrentIndex(cur)
            self.opt_stack.adjustSize()
            self._opt_sized = True
        self._remember_legends()
        if key == "table":
            self.view_stack.setCurrentIndex(1)
            self.fill_cycle_table()
            self.show_limits(key)
            return
        self.view_stack.setCurrentIndex(0)
        try:
            spec = self.build_spec(key)
        except Exception as e:                                    # keep the app alive, show the reason
            self.set_status(f"Could not build this plot: {e}")
            return
        self.spec = spec
        self.axes = draw_spec(self.fig, spec, self.style(), self.legpos, key)
        self.axes_key = key
        self.apply_limits(key)
        self.canvas.draw_idle()
        self.show_limits(key)
        n_curves = sum(1 for s in spec.series if s.show and s.legend and s.label)
        msg = (f"{n_curves} curve(s)  |  {len(self.sel)} cycle(s) selected  |  "
               f"{sum(d.show for d in self.DS)} data set(s) shown")
        if spec.warn:
            msg += "  |  enter loading / area for: " + ", ".join(dict.fromkeys(spec.warn))
        if spec.info:
            msg += "  |  " + spec.info
        self.set_status(msg)

    def _remember_legends(self):
        for i, ax in enumerate(self.axes):
            leg = ax.get_legend()
            if leg is not None and isinstance(getattr(leg, "_loc", None), tuple):
                self.legpos[(self.axes_key, i)] = leg._loc

    def set_status(self, msg):
        self.status.setText(f"{HEART}  {msg}")

    # ---------------------------------------------------------------- limits
    def _lim_axes(self, key):
        """x axes, y axes, y2 axes for the limit rows"""
        A = self.axes
        if not A:
            return [], [], []
        if len(A) == 1:
            return A, A, []
        if len(A) == 3:
            return A, A[:2], [A[2]]
        return A, [A[0]], [A[-1]]

    def apply_limits(self, key):
        if key == "table":
            return
        xa, ya, y2a = self._lim_axes(key)
        L = self.lim[key]
        for axs, dim in ((xa[-1:], "x"), (ya, "y"), (y2a, "y2")):
            lo, hi, auto = L[dim]
            if auto or not lo < hi:
                continue
            for ax in axs:
                (ax.set_xlim if dim == "x" else ax.set_ylim)(lo, hi)

    def show_limits(self, key):
        xa, ya, y2a = self._lim_axes(key)
        for dim, axs in (("x", xa[-1:]), ("y", ya), ("y2", y2a)):
            lo, hi, au, lab = self.limw[dim]
            visible = bool(axs) and key != "table"
            for x in (lo, hi, au, lab):
                x.setVisible(visible if dim == "y2" else True)
                x.setEnabled(visible)
            if not visible:
                continue
            L = self.lim[key][dim]
            cur = axs[0].get_xlim() if dim == "x" else axs[0].get_ylim()
            for x in (lo, hi, au):
                x.blockSignals(True)
            au.setChecked(bool(L[2]))
            lo.setValue(cur[0] if L[2] else L[0])
            hi.setValue(cur[1] if L[2] else L[1])
            for x in (lo, hi, au):
                x.blockSignals(False)
        self.limw["y2"][3].setText("Phase" if key == "eis" else "CE / Y2")

    def on_lim_edit(self, dim):
        key = self.key()
        lo, hi, au, _ = self.limw[dim]
        self.lim[key][dim] = [lo.value(), hi.value(), False]
        au.blockSignals(True)
        au.setChecked(False)
        au.blockSignals(False)
        if lo.value() >= hi.value():
            self.set_status("Lower limit must be smaller than upper limit.")
            return
        self.redraw()

    def on_lim_auto(self, dim):
        key = self.key()
        lo, hi, au, _ = self.limw[dim]
        self.lim[key][dim] = [lo.value(), hi.value(), au.isChecked()]
        self.redraw()

    def on_all_auto(self):
        for d in ("x", "y", "y2"):
            self.lim[self.key()][d][2] = True
        self.redraw()

    def view_changed(self, key):
        """a different view uses different units -> automatic limits again"""
        for d in ("x", "y", "y2"):
            self.lim[key][d][2] = True
        for k in list(self.legpos):
            if k[0] == key:
                del self.legpos[k]
        self.redraw()

    def on_scheme(self):
        # a colour scheme should be visible everywhere, also on the hysteresis plot
        cb = self.reg["hy_blackred"]
        if cb.isChecked():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self.redraw()

    def on_tab_changed(self):
        self.redraw()

    # ---------------------------------------------------------------- data sets
    def on_load(self):
        files, _ = W.QFileDialog.getOpenFileNames(
            self, "Select one or more cycling / CV / impedance files", self.last_dir,
            "Arbin and PAT-Tester files (*.xls *.xlsx *.txt *.csv);;All files (*.*)")
        if not files:
            return
        self.last_dir = str(Path(files[0]).parent)
        prog = W.QProgressDialog("Reading files...", "Cancel", 0, len(files), self)
        prog.setCancelButton(None)
        prog.setWindowModality(Qt.WindowModal)
        prog.show()
        errs, n_new, n_cache = [], 0, 0
        for k, fn in enumerate(files):
            prog.setLabelText(f"Reading {Path(fn).name}  ({k + 1} of {len(files)})")
            prog.setValue(k)
            W.QApplication.processEvents()
            try:
                entries, cached = bc.read_file_cached(fn)
                n_cache += cached
                for label, nick, cur in entries:
                    self.DS.append(bc.Dataset(self.next_id, label, nick, fn, cur, kind=cur["kind"]))
                    self.next_id += 1
                    n_new += 1
            except Exception as e:
                errs.append(f"{Path(fn).name}: {e}")
        prog.close()
        self.refresh_sets()
        self.refresh_info_list(select_last=n_new > 0)
        self.refresh_cycle_list()
        if n_cache:
            self.set_status(f"{n_cache} file(s) loaded from the fast cache")
        if errs:
            W.QMessageBox.warning(self, "Some files could not be read", "\n".join(errs))

    def on_remove(self):
        r = self.tbl.currentRow()
        if r < 0 or r >= len(self.DS):
            W.QMessageBox.information(self, "Nothing selected", "Click a row in the data set table first.")
            return
        del self.DS[r]
        self.refresh_sets()
        self.refresh_info_list()
        self.refresh_cycle_list()

    def on_clear(self):
        if self.DS and W.QMessageBox.question(self, "Clear all", "Remove all data sets?") == W.QMessageBox.Yes:
            self.DS.clear()
            self.refresh_sets()
            self.refresh_info_list()
            self.refresh_cycle_list()

    def on_show_all(self, tf):
        for d in self.DS:
            d.show = tf
        self.refresh_sets()
        self.refresh_cycle_list()

    def refresh_sets(self):
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(self.DS))
        for r, d in enumerate(self.DS):
            it = W.QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if d.show else Qt.Unchecked)
            self.tbl.setItem(r, 0, it)
            self.tbl.setItem(r, 1, W.QTableWidgetItem(d.nick))
            cb = W.QComboBox()
            cb.addItems(bc.KINDS)
            cb.setCurrentText(d.kind)
            cb.currentTextChanged.connect(lambda t, did=d.id: self.on_kind(did, t))
            self.tbl.setCellWidget(r, 2, cb)
            it = W.QTableWidgetItem(str(len(d.cycles)))
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.tbl.setItem(r, 3, it)
            self.tbl.setItem(r, 4, W.QTableWidgetItem("" if np.isnan(d.mass) else f"{d.mass:g}"))
            cu = W.QComboBox()
            cu.addItems(["mg", "g"])
            cu.setCurrentText(d.unit)
            cu.currentTextChanged.connect(lambda t, did=d.id: self.on_unit(did, t))
            self.tbl.setCellWidget(r, 5, cu)
            self.tbl.setItem(r, 6, W.QTableWidgetItem("" if np.isnan(d.area) else f"{d.area:g}"))
        self.tbl.resizeColumnsToContents()
        self.tbl.blockSignals(False)

    def _ds(self, did):
        return next(d for d in self.DS if d.id == did)

    def on_kind(self, did, t):
        self._ds(did).kind = t
        self.refresh_info()
        self.redraw()

    def on_unit(self, did, t):
        self._ds(did).unit = t
        self.refresh_info()
        self.redraw()

    def on_set_item(self, item):
        r, c = item.row(), item.column()
        if r >= len(self.DS):
            return
        d = self.DS[r]
        if c == 0:
            d.show = item.checkState() == Qt.Checked
            if d.show:
                self.select_info(d.id)                          # details follow the ticked set
            self.refresh_cycle_list()
            return
        if c == 1 and item.text().strip():
            d.nick = item.text().strip()
            self.refresh_info_list()
        elif c == 4:
            d.mass = _float(item.text()) if _float(item.text()) > 0 else np.nan
        elif c == 6:
            d.area = _float(item.text()) if _float(item.text()) > 0 else np.nan
        self.refresh_info()
        self.redraw()

    def on_set_row(self, r):
        if 0 <= r < len(self.DS):
            self.select_info(self.DS[r].id)

    # details -----------------------------------------------------------------
    def refresh_info_list(self, select_last=False):
        cur = self.dd_info.currentData()
        self.dd_info.blockSignals(True)
        self.dd_info.clear()
        for d in self.DS:
            self.dd_info.addItem(d.nick, d.id)
        if select_last and self.DS:
            cur = self.DS[-1].id
        i = self.dd_info.findData(cur)
        self.dd_info.setCurrentIndex(max(i, 0))
        self.dd_info.blockSignals(False)
        self.refresh_info()

    def select_info(self, did):
        i = self.dd_info.findData(did)
        if i >= 0:
            self.dd_info.setCurrentIndex(i)

    def info_ds(self):
        did = self.dd_info.currentData()
        return next((d for d in self.DS if d.id == did), None)

    def on_info_changed(self):
        self.refresh_info()
        if self.key() == "table":
            self.redraw()

    def refresh_info(self):
        d = self.info_ds()
        vals = {k: "-" for k in self.info_keys}
        if d is None:
            self.lbl_big.setText(f"{HEART} 0 cycles")
        else:
            self.lbl_big.setText(f"{HEART} {len(d.cycles)} cycles")
            if d.cycles:
                vals["Cycle range"] = f"{min(d.cycles)}  to  {max(d.cycles)}"
            vals["Loading"] = "(enter in table)" if np.isnan(d.mass) else f"{d.mass:g} {d.unit}"
            ex = d.cur["extra"]
            if d.kind == "EIS" and "spectra" in ex:
                vals["First step"] = f"Impedance  ({len(ex['spectra'])} spectra)"
            elif d.kind == "C12" and "c12" in ex:
                vals["First step"] = f"C12 output  ({len(ex['c12']['C'])} points)"
            elif d.kind == "Plating/Stripping":
                vals["First step"] = "Plating/stripping  " + bc.ps_text(*bc.ps_rates(d), d.area)
            elif d.kind == "CV":
                r = np.unique(d.cur["scanRate"][np.isfinite(d.cur["scanRate"])])
                vals["First step"] = "CV  (" + ", ".join(f"{x:g}" for x in r) + " mV/s)"
            else:
                fd = d.cur["disFirst"][np.isfinite(d.cur["disFirst"])]
                if len(fd):
                    vals["First step"] = "Discharge (anode / half cell)" if fd[0] else "Charge (cathode)"
            f = bc.spec_factor(d)
            if np.isfinite(f):
                dis, ch, ce = bc.summary(d, f)
                iD, iC, iE = (np.where(np.isfinite(a))[0] for a in (dis, ch, ce))
                if len(iD):
                    vals["1st discharge"] = f"{dis[iD[0]]:.1f} mAh/g" + ("   check unit!" if dis[iD[0]] > 3000 else "")
                    vals["Last discharge"] = f"{dis[iD[-1]]:.1f} mAh/g  (cycle {d.cycles[iD[-1]]})"
                    if len(iD) > 1:
                        vals["Retention"] = f"{100 * dis[iD[-1]] / dis[iD[1]]:.1f} %  (last vs 2nd)"
                if len(iC):
                    vals["1st charge"] = f"{ch[iC[0]]:.1f} mAh/g"
                if len(iE):
                    vals["1st CE"] = f"{ce[iE[0]]:.2f} %"
        for k, lab in self.info_val.items():
            lab.setText(vals[k])
            lab.setStyleSheet("color: #CC0000;" if "check unit" in vals[k] else "")

    def fill_cycle_table(self):
        d = self.info_ds()
        self.cyc_tbl.clear()
        if d is None:
            self.cyc_tbl.setRowCount(0)
            return
        df = bc.cycle_table(d)
        self.cyc_tbl.setColumnCount(len(df.columns))
        self.cyc_tbl.setRowCount(len(df))
        self.cyc_tbl.setHorizontalHeaderLabels(list(df.columns))
        for r, row in enumerate(df.itertuples(index=False)):
            for c, v in enumerate(row):
                self.cyc_tbl.setItem(r, c, W.QTableWidgetItem("" if not np.isfinite(v) else f"{v:.6g}"))
        self.cyc_tbl.resizeColumnsToContents()
        self.set_status(f"Cycle table for {d.nick}.")

    # ---------------------------------------------------------------- cycles
    def refresh_cycle_list(self):
        self.avail = sorted({c for d in self.DS if d.show for c in d.cycles})
        self.sel = [c for c in self.sel if c in self.avail]
        if not self.sel and self.avail:
            self.sel = self.avail[:3]
        self.lst.blockSignals(True)
        self.lst.clear()
        self.lst.addItems([str(c) for c in self.avail])
        self.lst.blockSignals(False)
        self.lbl_avail.setText(f"Available: {self.avail[0]} to {self.avail[-1]}   ({len(self.avail)} cycles)"
                               if self.avail else "Available: -")
        self.sld.blockSignals(True)
        self.sld.setRange(0, max(len(self.avail) - 1, 0))
        self.sld.blockSignals(False)
        self.sync_cycles()
        self.redraw()

    def sync_cycles(self):
        self.lst.blockSignals(True)
        sel = set(self.sel)
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            it.setSelected(int(it.text()) in sel)
        self.lst.blockSignals(False)
        self.lbl_sel.setText(f"{len(self.sel)} selected")
        if len(self.sel) == 1 and self.sel[0] in self.avail:
            self.sld.blockSignals(True)
            self.sld.setValue(self.avail.index(self.sel[0]))
            self.sld.blockSignals(False)

    def on_cycle_list(self):
        self.sel = sorted(int(it.text()) for it in self.lst.selectedItems())
        self.lbl_sel.setText(f"{len(self.sel)} selected")
        self.redraw()

    def set_selection(self, cyc):
        self.sel = sorted(cyc)
        self.sync_cycles()
        self.redraw()

    def on_quick(self):
        try:
            sel, missing = bc.parse_cycles(self.ed_quick.text(), self.avail)
        except ValueError as e:
            W.QMessageBox.warning(self, "Cycle selection", str(e))
            return
        self.set_selection(sel)
        if missing:
            self.set_status(f"Cycle(s) {missing[:10]} do not exist and were ignored.")

    def on_scrub(self, v):
        if self.avail:
            self.set_selection([self.avail[min(v, len(self.avail) - 1)]])

    # ---------------------------------------------------------------- export
    def current_columns(self):
        key = self.key()
        if key == "table":
            d = self.info_ds()
            return (bc.table_columns(bc.cycle_table(d), d.nick) if d else []), f"Table {d.nick if d else ''}"
        if self.spec is None:
            return [], ""
        limits = [(ax.get_xlim(), ax.get_ylim()) for ax in self.axes]
        cols = bc.export_columns(self.spec, limits, self.reg["crop"].isChecked())
        title = {"vq": "Profiles", "plat": "Hysteresis", "cap": "Capacity_CE", "rate": "Rate_capability",
                 "dqdv": "dQdV", "cv": "CV", "cvk": "CV_kinetics", "ps": "Plating_Stripping",
                 "eis": "Impedance", "vt": "V_vs_t"}[key]
        if key in ("vq", "plat", "dqdv", "cv", "eis", "vt") and self.sel:
            title += " C" + "-".join(map(str, self.sel[:6])) + ("..." if len(self.sel) > 6 else "")
        return cols, title

    def origin(self):
        return self.reg["hdr"].currentIndex() == 0

    def on_copy(self):
        cols, _ = self.current_columns()
        if not cols:
            self.set_status("Nothing to copy - select cycles / data first.")
            return
        W.QApplication.clipboard().setText(bc.to_clipboard_text(cols, self.origin()))
        self.set_status(f"Copied {len(cols)} columns. Click a cell in Excel or Origin and paste (Ctrl+V).")

    def _write_sheet(self, path, name, cols, mode):
        used = []
        if mode == "a":
            from openpyxl import load_workbook
            used = load_workbook(path, read_only=True).sheetnames
        sheet = bc.sheet_name(name, used)
        kw = {"if_sheet_exists": "new"} if mode == "a" else {}
        with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kw) as w:
            bc.to_frame(cols, self.origin()).to_excel(w, sheet_name=sheet, header=False, index=False)
        return sheet

    def on_save_excel(self):
        cols, title = self.current_columns()
        if not cols:
            self.set_status("Nothing to save - select cycles / data first.")
            return
        path, _ = W.QFileDialog.getSaveFileName(self, "Save plotted data", str(Path(self.last_dir) / "Battery_data.xlsx"),
                                                "Excel (*.xlsx)")
        if not path:
            return
        mode = "w"
        if Path(path).exists():
            box = W.QMessageBox(self)
            box.setText(f"{Path(path).name} already exists.")
            add = box.addButton("Add as new sheet", W.QMessageBox.AcceptRole)
            rep = box.addButton("Replace file", W.QMessageBox.DestructiveRole)
            box.addButton("Cancel", W.QMessageBox.RejectRole)
            box.exec_() if hasattr(box, "exec_") else box.exec()
            if box.clickedButton() == add:
                mode = "a"
            elif box.clickedButton() != rep:
                return
        try:
            sheet = self._write_sheet(path, title, cols, mode)
            self.set_status(f'Saved sheet "{sheet}" to {path}')
        except Exception as e:
            W.QMessageBox.warning(self, "Could not write Excel file", f"{e}\n\nIs the file open in Excel?")

    def on_add_list(self):
        cols, title = self.current_columns()
        if not cols:
            self.set_status("Nothing to add - select cycles / data first.")
            return
        name, ok = W.QInputDialog.getText(self, "Add to export list", "Sheet name for this plot:", text=title)
        if not ok:
            return
        self.export_list.append((name.strip() or title, cols))
        self.lbl_list.setText(f"Export list: {len(self.export_list)} plot(s)")

    def on_write_list(self):
        if not self.export_list:
            W.QMessageBox.information(self, "Export list", 'The export list is empty. Use "Add to export list" first.')
            return
        path, _ = W.QFileDialog.getSaveFileName(self, "Write export list",
                                                str(Path(self.last_dir) / "Battery_export.xlsx"), "Excel (*.xlsx)")
        if not path:
            return
        try:
            used = []
            with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
                for name, cols in self.export_list:
                    sheet = bc.sheet_name(name, used + ["Info"])
                    used.append(sheet)
                    bc.to_frame(cols, self.origin()).to_excel(w, sheet_name=sheet, header=False, index=False)
                pd.DataFrame([{"Data set": d.nick, "File": d.file, "Loading": d.mass, "Unit": d.unit,
                               "Area (cm2)": d.area, "Type": d.kind, "Cycles": len(d.cycles)} for d in self.DS]
                             ).to_excel(w, sheet_name="Info", index=False)
            self.set_status(f"Wrote {len(self.export_list)} sheet(s) to {path}")
        except Exception as e:
            W.QMessageBox.warning(self, "Could not write Excel file", f"{e}\n\nIs the file open in Excel?")

    def on_clear_list(self):
        self.export_list.clear()
        self.lbl_list.setText("Export list: empty")

    def on_save_png(self):
        if self.key() == "table":
            return
        path, _ = W.QFileDialog.getSaveFileName(self, "Save plot", str(Path(self.last_dir) / "plot.png"),
                                                "PNG (*.png)")
        if path:
            self.fig.savefig(path, dpi=300, facecolor=self.fig.get_facecolor())
            self.set_status(f"Saved {path}")

    def on_journal(self):
        if self.key() == "table" or self.spec is None:
            W.QMessageBox.information(self, "No plot", "Switch to a plot tab first.")
            return
        dlg = W.QDialog(self)
        dlg.setWindowTitle("Journal figure")
        f = W.QFormLayout(dlg)
        wcm, hcm, fpt, lpt = (W.QDoubleSpinBox() for _ in range(4))
        for s, (lo, hi, v) in zip((wcm, hcm, fpt, lpt), ((3, 30, 8.5), (3, 30, 7), (5, 20, 8), (0.25, 4, 1))):
            s.setRange(lo, hi)
            s.setValue(v)
            s.setSingleStep(0.5)
        fmt = W.QComboBox()
        fmt.addItems(["TIFF, 600 dpi", "PNG, 600 dpi", "PDF (vector)", "SVG (vector)", "EPS (vector)"])
        f.addRow("Width (cm)  [8.5 single, 11.4 1.5-column, 17.8 double]", wcm)
        f.addRow("Height (cm)", hcm)
        f.addRow("Font size (pt)", fpt)
        f.addRow("Line width (pt)", lpt)
        f.addRow("Format", fmt)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        f.addRow(bb)
        res = dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()
        if not res:
            return
        ext = [".tif", ".png", ".pdf", ".svg", ".eps"][fmt.currentIndex()]
        path, _ = W.QFileDialog.getSaveFileName(self, "Save journal figure",
                                                str(Path(self.last_dir) / f"{self.key()}_figure{ext}"),
                                                f"{fmt.currentText()} (*{ext})")
        if not path:
            return
        jf = Figure(figsize=(wcm.value() / 2.54, hcm.value() / 2.54), layout="constrained")
        st = self.style(fs=fpt.value(), lw=lpt.value(), chart=False)
        st.update(bg="#FFFFFF", fg="#000000", dark=False)
        axes = draw_spec(jf, self.spec, st, self.legpos, self.key())
        for a_new, a_old in zip(axes, self.axes):                # same limits as on screen
            a_new.set_xlim(a_old.get_xlim())
            a_new.set_ylim(a_old.get_ylim())
        try:
            jf.savefig(path, dpi=600 if ext in (".tif", ".png") else None, facecolor="white")
            self.set_status(f"Journal figure saved: {path}  ({wcm.value():g} x {hcm.value():g} cm, {fpt.value():g} pt)")
        except Exception as e:
            W.QMessageBox.warning(self, "Could not export", str(e))

    # ---------------------------------------------------------------- sessions
    def get_settings(self):
        out = {}
        for n, w in self.reg.items():
            if isinstance(w, W.QComboBox):
                out[n] = w.currentText()
            elif isinstance(w, W.QCheckBox):
                out[n] = w.isChecked()
            elif isinstance(w, (W.QSpinBox, W.QDoubleSpinBox, W.QSlider)):
                out[n] = w.value()
            elif isinstance(w, W.QLineEdit):
                out[n] = w.text()
        return out

    def set_settings(self, s):
        self.ready = False
        for n, v in s.items():
            w = self.reg.get(n)
            if w is None:
                continue
            w.blockSignals(True)
            try:
                if isinstance(w, W.QComboBox):
                    w.setCurrentText(v)
                elif isinstance(w, W.QCheckBox):
                    w.setChecked(bool(v))
                elif isinstance(w, (W.QSpinBox, W.QDoubleSpinBox, W.QSlider)):
                    w.setValue(v)
                elif isinstance(w, W.QLineEdit):
                    w.setText(v)
            finally:
                w.blockSignals(False)
        self.ready = True

    def on_save_session(self):
        path, _ = W.QFileDialog.getSaveFileName(self, "Save session", str(Path(self.last_dir) / "BatterySession.pkl"),
                                                "Battery session (*.pkl)")
        if not path:
            return
        sess = dict(version=1, DS=self.DS, next_id=self.next_id, sel=self.sel, lim=self.lim, legpos=self.legpos,
                    export_list=self.export_list, settings=self.get_settings(), tab=self.tabs.currentIndex())
        try:
            with open(path, "wb") as fh:
                pickle.dump(sess, fh, protocol=pickle.HIGHEST_PROTOCOL)
            self.set_status(f"Session saved: {path}")
        except Exception as e:
            W.QMessageBox.warning(self, "Could not save session", str(e))

    def on_open_session(self):
        path, _ = W.QFileDialog.getOpenFileName(self, "Open session", self.last_dir, "Battery session (*.pkl)")
        if not path:
            return
        try:
            with open(path, "rb") as fh:
                sess = pickle.load(fh)
        except Exception as e:
            W.QMessageBox.warning(self, "Could not open session", str(e))
            return
        self.DS, self.next_id, self.sel = sess["DS"], sess["next_id"], list(sess["sel"])
        self.lim.update(sess.get("lim", {}))
        self.legpos = sess.get("legpos", {})
        self.export_list = sess.get("export_list", [])
        self.set_settings(sess.get("settings", {}))
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(sess.get("tab", 0))
        self.tabs.blockSignals(False)
        self.lbl_list.setText(f"Export list: {len(self.export_list)} plot(s)" if self.export_list
                              else "Export list: empty")
        self.refresh_sets()
        self.refresh_info_list()
        self.apply_theme()
        self.refresh_cycle_list()
        self.set_status(f"Session opened: {Path(path).name}")


# =============================================================================
# Drawing a recipe on any matplotlib figure (screen, journal export)
# =============================================================================
def _lighten(col, dark):
    c = np.array(matplotlib.colors.to_rgb(col))
    if not dark:
        return tuple(c)
    if c.sum() < 0.3:
        return (1.0, 1.0, 1.0)
    return tuple(c + (1 - c) * 0.35)


def draw_spec(fig, spec, st, legpos=None, key=None):
    fs, lw, fg, bg, dark = st["fs"], st["lw"], st["fg"], st["bg"], st["dark"]
    fig.clear()
    fig.set_facecolor(bg)
    W_in, H_in = fig.get_size_inches()
    fig.set_layout_engine("constrained", w_pad=max(0.04, (1 - st["cw"] / 100) * W_in / 2),
                          h_pad=max(0.04, (1 - st["ch"] / 100) * H_in / 2))
    n = len(spec.panels)
    gs = fig.add_gridspec(n, 1, height_ratios=[p["h"] for p in spec.panels])
    axes = []
    for i, p in enumerate(spec.panels):
        ax = fig.add_subplot(gs[i], sharex=axes[0] if axes and n > 1 else None)
        axes.append(ax)
        ax.set_facecolor(bg)
        ax.set_xscale(p["xscale"])
        ax.set_yscale(p["yscale"])
        for sp_ in ax.spines.values():
            sp_.set_color(fg)
            sp_.set_linewidth(max(0.6, 0.65 * lw))
        ax.tick_params(direction=st["tickdir"], colors=fg, labelsize=fs, width=max(0.6, 0.65 * lw),
                       length=fs * 0.45, top=True, right=True, which="major")
        ax.tick_params(direction=st["tickdir"], colors=fg, top=True, right=True, which="minor")
        yl = p["ylabel"]
        if n > 1 and len(yl) > 16 and " (" in yl:
            yl = yl.replace(" (", "\n(", 1)              # stacked panels: title on two lines
        ax.set_ylabel(yl, fontsize=fs, color=fg, family=st["family"])
        if i == n - 1 or not p["xlabel"] == "":
            ax.set_xlabel(p["xlabel"], fontsize=fs, color=fg, family=st["family"])
        if n > 1 and i < n - 1:
            ax.tick_params(labelbottom=False)
        if p["equal"]:
            ax.set_aspect("equal", adjustable="datalim")
        if st["grid"]:
            ax.grid(True, color=fg, alpha=0.25)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_family(st["family"])

    if n > 1:
        fig.align_ylabels(axes)
    handles = {i: [] for i in range(n)}
    for s in spec.series:
        if not s.show or len(s.x) == 0:
            continue
        ax = axes[min(s.panel, n - 1)]
        col = _lighten(s.color, dark)
        mfc = col if s.mfc is None else ("none" if s.mfc == "none" else _lighten(s.mfc, dark))
        h, = ax.plot(s.x, s.y, color=col, ls=s.ls if s.ls != "none" else "none", marker=s.marker or None,
                     mfc=mfc, ms=s.ms, lw=s.lw if s.lw is not None else lw, mew=max(1.0, 0.6 * lw),
                     label=s.label)
        err = getattr(s, "err", None)
        if err is not None:
            ax.errorbar(s.x, s.y, yerr=err, fmt="none", ecolor=col, elinewidth=max(1, 0.6 * lw), capsize=fs * 0.3)
        if s.legend and s.label:
            handles[min(s.panel, n - 1)].append(h)

    for a in spec.ann:
        ax = axes[min(a.get("panel", 0), n - 1)]
        col = _lighten(a["color"], dark)
        if a["type"] == "arrow":
            ax.annotate("", xy=(a["x2"], a["y"]), xytext=(a["x1"], a["y"]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=max(1.2, lw), mutation_scale=fs * 1.4))
        else:
            ax.text(a["x"], a["y"], a["text"], color=col, fontsize=max(6, fs - 2), fontweight="bold",
                    ha=a["ha"], va=a["va"], family=st["family"])

    if spec.top:
        xr = spec.top["xref"]
        col = _lighten(spec.top["color"], dark)
        ax0 = axes[0]
        sec = ax0.secondary_xaxis("top", functions=(lambda x: xr - x, lambda x: xr - x))
        sec.set_xlabel(spec.top["label"], color=col, fontsize=fs, family=st["family"])
        sec.tick_params(direction=st["tickdir"], colors=col, labelsize=fs, width=max(0.6, 0.65 * lw),
                        length=fs * 0.45)
        sec.spines["top"].set_color(col)
        sec.spines["top"].set_linewidth(max(1.2, 1.3 * lw))
        ax0.spines["top"].set_visible(False)
        ax0.tick_params(top=False, labelright=st.get("right_numbers", False))
        for lab in sec.get_xticklabels():
            lab.set_family(st["family"])

    place = st["leg_place"]
    for i, ax in enumerate(axes):
        hs = handles[i]
        if not hs or place == "Hidden" or len(hs) > 40:
            continue
        if st["leg_size"] == "Auto":
            longest = max(len(h.get_label()) for h in hs)
            lf = max(7, round(fs - max(0, len(hs) - 5) * 0.5 - max(0, longest - 14) * 0.25))
        else:
            lf = float(st["leg_size"])
        kw = dict(fontsize=lf, frameon=True, edgecolor=fg, facecolor=bg, labelcolor=fg, framealpha=1,
                  ncols=1 + (len(hs) > 10), prop={"family": st["family"], "size": lf})
        kw.pop("fontsize")
        if place.startswith("Draggable"):
            loc = (legpos or {}).get((key, i), "best")
            leg = ax.legend(handles=hs, loc=loc, **kw)
        elif place == "Outside right":
            leg = ax.legend(handles=hs, loc="center left", bbox_to_anchor=(1.02, 0.5), **kw)
        elif place == "Outside below":
            kw["ncols"] = min(len(hs), 4)
            leg = ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.18), **kw)
        else:
            leg = ax.legend(handles=hs, loc=LOC[place], **kw)
        leg.get_frame().set_linewidth(max(0.6, 0.65 * lw))
        leg.set_draggable(True)                  # every legend can be dragged with the mouse
    return axes


def main():
    app = W.QApplication.instance()
    created = app is None
    if created:
        app = W.QApplication(sys.argv)
    win = App()
    win.show()
    if created:
        app.exec_() if hasattr(app, "exec_") else app.exec()
    return win


if __name__ == "__main__":
    window = main()
