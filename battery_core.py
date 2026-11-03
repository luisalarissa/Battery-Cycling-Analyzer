"""
battery_core.py - data reading, analysis and plot recipes for the
Battery Cycling Analyzer (Python version of the MATLAB app).

No windows in here: battery_app.py builds the GUI on top of this module.
"""
from __future__ import annotations

import os
import pickle
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

CACHE_VERSION = "py1"
_trapz = getattr(np, "trapezoid", None) or np.trapz
KINDS = ["Cycling", "CV", "Plating/Stripping", "EIS", "C12"]
SMOOTH = ["None", "Moving average", "Savitzky-Golay", "Gaussian"]


# =============================================================================
# Data sets
# =============================================================================
@dataclass
class Dataset:
    id: int
    label: str
    nick: str
    file: str
    cur: dict                       # per-cycle curves (see prep_curves)
    mass: float = np.nan            # loading
    unit: str = "mg"                # 'mg' or 'g'
    area: float = np.nan            # cm2
    show: bool = True
    kind: str = "Cycling"

    @property
    def cycles(self):
        return self.cur["cycles"]

    def idx(self, cycle):
        """index of a cycle number, or None"""
        try:
            return self.cur["cycles"].index(cycle)
        except ValueError:
            return None


def mass_g(ds):
    if not (ds.mass > 0):
        return np.nan
    return ds.mass / 1000 if ds.unit == "mg" else ds.mass


def spec_factor(ds):
    """Ah -> mAh/g"""
    m = mass_g(ds)
    return 1000 / m if m > 0 else np.nan


def cap_factor(ds, unit):
    """Ah -> mAh/g ('g'), mAh/cm2 ('cm2') or mAh ('abs')"""
    if unit == "cm2":
        return 1000 / ds.area if ds.area > 0 else np.nan
    if unit == "abs":
        return 1000.0
    return spec_factor(ds)


CAP_LABEL = {"g": "Specific capacity (mAh g$^{-1}$)",
             "cm2": "Areal capacity (mAh cm$^{-2}$)",
             "abs": "Capacity (mAh)"}
CAP_UNIT = {"g": "mAh/g", "cm2": "mAh/cm2", "abs": "mAh"}
VLABEL = "Voltage (V vs. Li/Li$^+$)"


# =============================================================================
# Reading files
# =============================================================================
def _num(s):
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def segment_ah(qcol, vcol, rows, I, t):
    """one half-cycle, capacity from zero (Ah); integrates I if no counter"""
    if len(rows) == 0:
        return np.array([]), np.array([])
    v = vcol[rows]
    q = None
    if qcol is not None:
        first = rows[0]
        base = qcol[first - 1] if first > 0 and qcol[first - 1] <= qcol[first] else 0.0
        q = qcol[rows] - base
    if q is None or not np.all(np.isfinite(q)) or not (np.nanmax(q) > 0):
        Ii, ti = np.abs(I[rows]), t[rows]
        dt = np.diff(ti)
        dt[(np.diff(rows) > 1) | (dt < 0)] = 0
        q = np.concatenate([[0.0], np.cumsum(0.5 * (Ii[1:] + Ii[:-1]) * dt)]) / 3600
    return q, v


def prep_curves(df):
    """df columns: time, cycle, current (A), voltage, qch (Ah), qdis (Ah), [sweep]"""
    df = df.dropna(subset=["time", "cycle", "current", "voltage"])
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No data rows found.")
    t, cyc, I, V = (df[c].to_numpy(float) for c in ("time", "cycle", "current", "voltage"))
    qch = df["qch"].to_numpy(float) if "qch" in df else None
    qdis = df["qdis"].to_numpy(float) if "qdis" in df else None
    sweep = df["sweep"].to_numpy(bool) if "sweep" in df else np.zeros(len(df), bool)
    tol = 1e-3 * np.nanmax(np.abs(I)) or np.finfo(float).eps
    active = (np.abs(I) > tol) | sweep

    cur = {k: [] for k in ("cycles", "qD", "vD", "qC", "vC", "tAll", "vAll", "aV", "aI")}
    rates, disfirst = [], []
    groups = pd.Series(np.arange(len(df))).groupby(cyc, sort=False).indices
    for c, rows in groups.items():
        rows = np.sort(rows)
        ar = rows[active[rows]]
        if len(ar) == 0:
            continue                          # e.g. initial rest
        dis, chg = rows[I[rows] < -tol], rows[I[rows] > tol]
        qd, vd = segment_ah(qdis, V, dis, I, t)
        qc, vc = segment_ah(qch, V, chg, I, t)
        cur["cycles"].append(int(c))
        for k, val in (("qD", qd), ("vD", vd), ("qC", qc), ("vC", vc), ("tAll", t[rows]),
                       ("vAll", V[rows]), ("aV", V[ar]), ("aI", I[ar] * 1000)):
            cur[k].append(val)
        r = np.nan
        if len(ar) > 3:
            dv, dt = np.diff(V[ar]), np.diff(t[ar])
            ok = dt > 0
            if ok.any():
                r = float(np.median(np.abs(dv[ok] / dt[ok])) * 1000)
                r = float(f"{r:.2g}")
        rates.append(r)
        disfirst.append(np.nan if (len(dis) == 0 or len(chg) == 0) else float(dis[0] < chg[0]))
    cur["scanRate"] = np.array(rates)
    cur["disFirst"] = np.array(disfirst)
    cur["extra"] = {}
    kind = "Cycling"
    if active.any():
        va = V[active]
        if sweep[active].mean() > 0.5:
            kind = "CV"
        elif va.min() < -0.002 and va.max() < 1.05:
            kind = "Plating/Stripping"
    cur["kind"] = kind
    return cur


def empty_cur(cycles, kind, extra):
    n = len(cycles)
    cur = {k: [np.array([])] * n for k in ("qD", "vD", "qC", "vC", "tAll", "vAll", "aV", "aI")}
    cur.update(cycles=[int(c) for c in cycles], scanRate=np.full(n, np.nan),
               disFirst=np.full(n, np.nan), kind=kind, extra=extra)
    return cur


def _pat_table(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        df = pd.read_excel(path, header=0, skiprows=[1])
    else:
        with open(path, "r", errors="replace") as fh:
            head = fh.readline()
        sep = "\t" if "\t" in head else (";" if ";" in head else ",")
        df = pd.read_csv(path, sep=sep, header=0, skiprows=[1], low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_pat(path):
    """EL-Cell PAT-Tester export: main file, Output-Impedance or Output-C12"""
    df = _pat_table(path)
    names = set(df.columns)
    if "freq" in names and "Re(Z12)" in names:                       # impedance
        f, cyc = _num(df["freq"]), _num(df["cycle"])
        st = _num(df["step"]) if "step" in names else np.zeros(len(df))
        ok = np.isfinite(f) & (f > 0) & np.isfinite(cyc)
        f, cyc, st = f[ok], cyc[ok], st[ok]
        Z = []
        for tag in ("12", "1", "2"):
            if f"Re(Z{tag})" in names and f"Im(Z{tag})" in names:
                Z.append((_num(df[f"Re(Z{tag})"])[ok], _num(df[f"Im(Z{tag})"])[ok]))
            else:
                Z.append(None)
        new = np.r_[True, (np.diff(cyc) != 0) | (np.diff(st) != 0) | (f[1:] > 1.5 * f[:-1])]
        sid = np.cumsum(new)
        spectra = []
        for q in range(1, sid.max() + 1):
            m = sid == q
            c = int(cyc[m][0])
            k = sum(1 for s in spectra if s["cycle"] == c) + 1
            spectra.append({"cycle": c, "k": k, "f": f[m],
                            "re": [None if z is None else z[0][m] for z in Z],
                            "im": [None if z is None else z[1][m] for z in Z]})
        if not spectra:
            raise ValueError("No impedance spectra found.")
        return empty_cur(sorted({s["cycle"] for s in spectra}), "EIS", {"spectra": spectra})

    if "C12" in names:                                               # C12 output
        c12 = {k: _num(df[col]) if col in names else np.array([])
               for k, col in (("t", "t"), ("cycle", "cycle"), ("V", "V12"), ("dQ", "dQ12"), ("C", "C12"))}
        ok = np.isfinite(c12["cycle"]) & np.isfinite(c12["C"])
        c12 = {k: (v[ok] if len(v) else v) for k, v in c12.items()}
        if len(c12["dQ"]) == 0:
            c12["dQ"] = np.sign(c12["C"])
        return empty_cur(sorted(set(c12["cycle"].astype(int))), "C12", {"c12": c12})

    need = {"t", "cycle", "V12", "I12"}
    if not need <= names:
        raise ValueError("Not a PAT-Tester export (columns t, cycle, V12, I12 not found).")
    d = pd.DataFrame({"time": _num(df["t"]), "cycle": _num(df["cycle"]),
                      "current": _num(df["I12"]) / 1000, "voltage": _num(df["V12"]),
                      "qch": _num(df["Qc"]) / 1000 if "Qc" in names else np.nan,
                      "qdis": _num(df["Qd"]) / 1000 if "Qd" in names else np.nan})
    if "type" in names:
        d["sweep"] = df["type"].astype(str).str.strip().eq("VS").to_numpy()
    return prep_curves(d)


def read_arbin(path):
    """Arbin export -> list of (label, nick, cur), one per channel"""
    path = Path(path)
    with open(path, "rb") as fh:
        engine = "openpyxl" if fh.read(2) == b"PK" else None    # xlsx saved as .xls
    sheets = pd.read_excel(path, sheet_name=None, engine=engine)
    ch = [n for n in sheets if str(n).lower().startswith("channel")]
    if not ch:
        return None
    base = {n: re.sub(r"^(Channel_\d+-\d+)_\d+$", r"\1", str(n), flags=re.I) for n in ch}
    out = []
    channels = list(dict.fromkeys(base.values()))
    for chan in channels:
        data = pd.concat([sheets[n] for n in ch if base[n] == chan], ignore_index=True)

        def col(start):
            for c in data.columns:
                if str(c).lower().startswith(start):
                    return _num(data[c])
            raise ValueError(f'Column starting with "{start}" not found. Is this an Arbin export?')

        d = pd.DataFrame({"time": col("test_time"), "cycle": col("cycle_index"), "current": col("current"),
                          "voltage": col("voltage"), "qch": col("charge_capacity"),
                          "qdis": col("discharge_capacity")})
        cur = prep_curves(d)
        label, nick = path.name, path.stem
        if len(channels) > 1:
            label = f"{path.name} [{chan}]"
            nick = f"{path.stem}_{re.sub('^Channel_', 'Ch', chan, flags=re.I)}"
        out.append((label, nick, cur))
    return out


def read_file(path):
    """any supported file -> list of (label, nick, cur)"""
    path = Path(path)
    if path.suffix.lower() in (".txt", ".csv", ".dat"):
        return [(path.name, path.stem, read_pat(path))]
    entries = read_arbin(path)
    if entries is None:                                   # a PAT export saved as Excel
        entries = [(path.name, path.stem, read_pat(path))]
    return entries


def cache_file(path):
    try:
        st = os.stat(path)
        name = re.sub(r"\W", "_", Path(path).stem)[:60]
        folder = Path(tempfile.gettempdir()) / "BatteryCyclingApp_cache"
        folder.mkdir(exist_ok=True)
        return folder / f"{CACHE_VERSION}_{name}_{st.st_size}_{int(st.st_mtime)}.pkl"
    except OSError:
        return None


def read_file_cached(path):
    """read_file with a fast pickle cache in the temp folder; returns (entries, from_cache)"""
    cf = cache_file(path)
    if cf is not None and cf.exists():
        try:
            with open(cf, "rb") as fh:
                return pickle.load(fh), True
        except Exception:
            pass
    entries = read_file(path)
    if cf is not None:
        try:
            with open(cf, "wb") as fh:
                pickle.dump(entries, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return entries, False


# =============================================================================
# Small analysis helpers
# =============================================================================
def parse_cycles(text, available):
    """'1 2 3', '1:10', '1 10:10:100', 'end', 'all' -> (selected, missing)"""
    available = sorted(available)
    text = str(text).strip().lower()
    if not available:
        return [], []
    if text == "all":
        return available, []
    text = text.replace("end", str(max(available)))
    wanted = []
    for tok in re.split(r"[\s,;]+", text):
        if not tok:
            continue
        parts = [int(round(float(p))) for p in tok.split(":")]
        if len(parts) == 1:
            wanted += parts
        elif len(parts) == 2:
            wanted += range(parts[0], parts[1] + 1)
        elif len(parts) == 3:
            wanted += range(parts[0], parts[2] + 1, parts[1])
        else:
            raise ValueError(f'Could not understand "{tok}"')
    w = set(wanted)
    return [c for c in available if c in w], sorted(w - set(available))


def smooth(y, method, w):
    y = np.asarray(y, float)
    if method == "None" or len(y) < 4:
        return y
    w = int(max(3, min(round(w), len(y))))
    if method == "Moving average":
        return pd.Series(y).rolling(w, center=True, min_periods=1).mean().to_numpy()
    if method == "Savitzky-Golay":
        w = w if w % 2 else w - 1
        return savgol_filter(y, max(w, 5), 2) if len(y) >= 5 else y
    if method == "Gaussian":
        return gaussian_filter1d(y, w / 4)
    return y


def summary(ds, f):
    n = len(ds.cycles)
    dis, ch, ce = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if len(ds.cur["qD"][i]):
            dis[i] = np.nanmax(ds.cur["qD"][i]) * f
        if len(ds.cur["qC"][i]):
            ch[i] = np.nanmax(ds.cur["qC"][i]) * f
        dfst = ds.cur["disFirst"][i]
        if np.isfinite(dfst):
            ce[i] = 100 * ch[i] / dis[i] if dfst else 100 * dis[i] / ch[i]
    return dis, ch, ce


def dqdv(q, v, nbins, sm):
    q, v = np.asarray(q, float), np.asarray(v, float)
    if len(v) < 5:
        return np.array([]), np.array([])
    if sm > 1:
        v = pd.Series(v).rolling(int(sm), center=True, min_periods=1).mean().to_numpy()
    vs, ia = np.unique(v, return_index=True)
    if len(vs) < 3:
        return np.array([]), np.array([])
    qs = q[ia]
    edges = np.linspace(vs[0], vs[-1], int(nbins) + 1)
    qi = np.interp(edges, vs, qs)
    d = np.diff(qi) / np.diff(edges)
    vm = 0.5 * (edges[1:] + edges[:-1])
    if sm > 1:
        d = pd.Series(d).rolling(int(sm), center=True, min_periods=1).mean().to_numpy()
    return vm, d


def hyst50(qD, vD, qC, vC):
    """discharge V at 50 % of Q_dis and charge V after re-charging the same amount"""
    qD, qC = np.asarray(qD, float), np.asarray(qC, float)
    if len(qD) < 2 or len(qC) < 2:
        return np.nan, np.nan
    QE = np.nanmax(qD)
    qd, id_ = np.unique(qD, return_index=True)
    qc, ic = np.unique(qC, return_index=True)
    if len(qd) < 2 or len(qc) < 2 or not QE > 0:
        return np.nan, np.nan
    vd = np.interp(0.5 * QE, qd, np.asarray(vD)[id_])
    vc = np.interp(0.5 * QE, qc, np.asarray(vC)[ic]) if 0.5 * QE <= qc.max() else np.nan
    return float(vd), float(vc)


def rate_steps(ds, as_c, ref):
    """rate of every cycle (A/g or C-rate) and its step number (1..)"""
    m = mass_g(ds)
    n = len(ds.cycles)
    Iq = np.full(n, np.nan)
    for i, a in enumerate(ds.cur["aI"]):
        a = np.asarray(a)
        d = np.abs(a[a < 0])
        if len(d) == 0:
            d = np.abs(a[a > 0])
        if len(d):
            Iq[i] = np.nanmedian(d)                         # mA
    rv = (Iq / m) / ref if as_c else Iq / 1000 / m
    stp, cur, prev = np.zeros(n, int), 0, np.nan
    for i in range(n):
        if np.isfinite(rv[i]) and (np.isnan(prev) or abs(rv[i] - prev) / prev > 0.10):
            cur, prev = cur + 1, rv[i]
        stp[i] = max(cur, 1)
    return rv, stp, int(stp.max()) if n else 0


def rate_label(r, as_c):
    r = float(f"{r:.2g}")
    return f"{r:g}C" if as_c else f"{r:g} A g$^{{-1}}$"


def eis_res(f, zr, zim):
    """Rs (high-frequency intercept) and Rct (first semicircle width)"""
    o = np.argsort(-np.asarray(f))
    zr, zim = np.asarray(zr)[o], np.asarray(zim)[o]
    ok = np.isfinite(zr) & np.isfinite(zim)
    zr, zim = zr[ok], zim[ok]
    n = len(zr)
    if n < 4:
        return np.nan, np.nan
    k = np.where((zim[:-1] <= 0) & (zim[1:] > 0))[0]
    if len(k):
        k = k[0]
        Rs = zr[k] + (zr[k + 1] - zr[k]) * (0 - zim[k]) / (zim[k + 1] - zim[k])
        st = k + 1
    else:
        Rs, st = zr[0], 0
    zs = pd.Series(zim).rolling(3, center=True, min_periods=1).mean().to_numpy()
    ap = st + int(np.argmax(zs[st:]))
    mn = None
    for j in range(ap + 1, n - 1):
        if zs[j] <= zs[j - 1] and zs[j] < zs[j + 1]:
            mn = j
            break
    if mn is None:
        mn = n - 1
    return float(Rs), float(zr[mn] - Rs)


def envelope(t, v, nbins=20000):
    """min/max per bin: same look, far fewer points"""
    t, v = np.asarray(t), np.asarray(v)
    n = len(v)
    k = int(np.ceil(n / nbins))
    nb = int(np.ceil(n / k))
    V = np.r_[v, np.full(nb * k - n, np.nan)].reshape(nb, k)
    imn, imx = np.nanargmin(V, 1), np.nanargmax(V, 1)
    base = np.arange(nb) * k
    idx = np.stack([np.minimum(imn, imx) + base, np.maximum(imn, imx) + base], 1).ravel()
    idx = idx[idx < n]
    _, u = np.unique(idx, return_index=True)
    idx = idx[np.sort(u)]
    return t[idx], v[idx]


def ps_rates(ds):
    I = [np.nanmedian(np.abs(a)) for a in ds.cur["aI"] if len(a)]
    Q = [np.nanmax(q) for q in ds.cur["qD"] if len(q)]
    return (np.nanmedian(I) if I else np.nan), (np.nanmedian(Q) * 1000 if Q else np.nan)


def ps_text(j, q, area):
    if area > 0:
        return f"{j / area:.3g} mA/cm2, {q / area:.3g} mAh/cm2 per half-cycle"
    return f"{j:.3g} mA, {q:.3g} mAh per half-cycle (enter area for /cm2)"


# =============================================================================
# Colours
# =============================================================================
OKABE = ["#0072B2", "#D55E00", "#000000", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
ANCHORS = {
    "High contrast": [(0.10, 0.10, 0.45), (0.00, 0.35, 0.80), (0.00, 0.50, 0.50), (0.15, 0.50, 0.10),
                      (0.75, 0.40, 0.00), (0.80, 0.10, 0.10), (0.60, 0.00, 0.50)],
    "Rainbow (dark)": [(0.35, 0, 0.55), (0, 0.30, 0.80), (0, 0.50, 0.55), (0.15, 0.50, 0.05),
                       (0.75, 0.40, 0), (0.80, 0.05, 0.05)],
    "Berry": [(0.30, 0, 0.40), (0.55, 0, 0.55), (0.80, 0, 0.45), (0.90, 0.25, 0.35)],
    "Ocean": [(0.05, 0.10, 0.40), (0, 0.35, 0.75), (0, 0.50, 0.55), (0.05, 0.45, 0.20)],
    "Sunset": [(0.30, 0, 0.45), (0.70, 0, 0.35), (0.90, 0.30, 0), (0.70, 0.45, 0)],
}
SCHEMES = list(ANCHORS) + ["tab10", "viridis", "plasma", "turbo"]


def colors(n, scheme):
    import matplotlib as mpl
    n = max(int(n), 1)
    if scheme == "High contrast" and n <= len(OKABE):
        return [mpl.colors.to_rgb(c) for c in OKABE[:n]]
    if scheme in ANCHORS:
        A = np.array(ANCHORS[scheme])
        if n == 1:
            return [tuple(A[0])]
        t = np.linspace(0, 1, n)
        xp = np.linspace(0, 1, len(A))
        return [tuple(np.interp(ti, xp, A[:, j]) for j in range(3)) for ti in t]
    cmap = mpl.colormaps.get(scheme, mpl.colormaps["viridis"])
    if scheme == "tab10":
        return [cmap(i % 10)[:3] for i in range(n)]
    return [cmap(x)[:3] for x in np.linspace(0, 0.85, n)]


# =============================================================================
# Plot recipes: everything a tab needs, independent of the GUI
# =============================================================================
@dataclass
class S:
    """one plotted (or export-only) series"""
    label: str
    x: np.ndarray
    y: np.ndarray
    color: tuple = (0, 0, 0)
    ls: str = "-"
    marker: str = ""
    mfc: object = None            # None = same as color, 'none' = open marker
    ms: float = 6
    lw: float | None = None
    panel: int = 0
    legend: bool = True
    export: bool = True
    show: bool = True
    crop: bool = True
    xname: str = ""
    xunit: str = ""
    yname: str = ""
    yunit: str = ""

    def __post_init__(self):
        self.x = np.asarray(self.x, float).ravel()
        self.y = np.asarray(self.y, float).ravel()


@dataclass
class Spec:
    panels: list                                  # [{'xlabel','ylabel','xscale','yscale','equal','h'}]
    series: list = field(default_factory=list)
    ann: list = field(default_factory=list)       # {'type':'text'|'arrow', ...}
    top: dict | None = None                       # mirrored top axis {'xref','label','color'}
    info: str = ""
    warn: list = field(default_factory=list)


def panel(xlabel, ylabel, h=1.0, xscale="linear", yscale="linear", equal=False):
    return {"xlabel": xlabel, "ylabel": ylabel, "h": h, "xscale": xscale, "yscale": yscale,
            "equal": equal}


def _shown(DS, kinds):
    return [d for d in DS if d.show and d.kind in kinds]


def _cycle_colors(nS, ncyc, scheme):
    return colors(nS, scheme), colors(ncyc, scheme)


STYLES = ["-", "--", ":", "-."]


def build_profiles(DS, o, kind="vq"):
    """voltage profiles ('vq'), dQ/dV ('dqdv') or V vs time ('vt')"""
    if kind == "vq":
        sp = Spec([panel(CAP_LABEL[o["cap_unit"]], VLABEL)])
    elif kind == "dqdv":
        sp = Spec([panel(VLABEL, "dQ/dV (mAh g$^{-1}$ V$^{-1}$)")])
    else:
        sp = Spec([panel("Test time (h)", "Voltage (V)")])
    S_ = _shown(DS, ["Cycling"] if kind != "vt" else ["Cycling", "CV", "Plating/Stripping"])
    cyc = o["cycles"]
    cS, cC = _cycle_colors(len(S_), len(cyc), o["scheme"])
    for s, ds in enumerate(S_):
        f = cap_factor(ds, o["cap_unit"]) if kind == "vq" else spec_factor(ds)
        if np.isnan(f) and kind != "vt":
            sp.warn.append(ds.nick)
            continue
        pre = f"{ds.nick} " if len(S_) > 1 else ""
        for k, c in enumerate(cyc):
            i = ds.idx(c)
            if i is None:
                continue
            if len(S_) == 1:
                col, ls, lbl = cC[k], "-", f"Cycle {c}"
            else:
                col, ls, lbl = cS[s], STYLES[k % 4], f"{ds.nick}  C{c}"
            tag = f"{pre}C{c}"
            if kind == "vt":
                if len(ds.cur["tAll"][i]):
                    sp.series.append(S(lbl, ds.cur["tAll"][i] / 3600, ds.cur["vAll"][i], col, ls,
                                       xname=f"{tag} Test Time", xunit="h", yname=f"{tag} Voltage", yunit="V"))
                continue
            first = True
            for half, use in (("Discharge", o["show_dis"]), ("Charge", o["show_ch"])):
                if not use:
                    continue
                q = ds.cur["qD" if half == "Discharge" else "qC"][i]
                v = ds.cur["vD" if half == "Discharge" else "vC"][i]
                if len(q) == 0:
                    continue
                if kind == "vq":
                    x, y = q * f, smooth(v, o["vq_smooth"], o["vq_win"])
                    xn, xu, yn, yu = f"{tag} {half} Capacity", CAP_UNIT[o["cap_unit"]], f"{tag} {half} Voltage", "V"
                else:
                    x, y = dqdv(q * f, v, o["dq_bins"], o["dq_smooth"])
                    xn, xu, yn, yu = f"{tag} {half} Voltage", "V", f"{tag} {half} dQ/dV", "mAh/(g V)"
                sp.series.append(S(lbl, x, y, col, ls, legend=first, xname=xn, xunit=xu, yname=yn, yunit=yu))
                first = False
    return sp


def build_hysteresis(DS, o):
    unit = o["cap_unit"]
    sp = Spec([panel(CAP_LABEL[unit], VLABEL)])
    S_ = _shown(DS, ["Cycling"])
    mirror, align, axis_col = o["hy_mirror"], o["hy_align"], o["hy_blackred"]
    RED = (0.85, 0, 0)
    H = []
    for ds in S_:
        f = cap_factor(ds, unit)
        if np.isnan(f):
            sp.warn.append(ds.nick)
            continue
        pre = f"{ds.nick} " if len(S_) > 1 else ""
        for c in o["cycles"]:
            i = ds.idx(c)
            if i is None:
                continue
            qD, qC = ds.cur["qD"][i] * f, ds.cur["qC"][i] * f
            vD = smooth(ds.cur["vD"][i], o["vq_smooth"], o["vq_win"]) if len(qD) else np.array([])
            vC = smooth(ds.cur["vC"][i], o["vq_smooth"], o["vq_win"]) if len(qC) else np.array([])
            if len(qD) == 0 and len(qC) == 0:
                continue
            xe = np.nanmax(qD) if len(qD) else np.nanmax(qC)
            name = f"{ds.nick} C{c}" if len(S_) > 1 else f"Cycle {c}"
            H.append(dict(tag=f"{pre}C{c}", name=name, qD=qD, vD=vD, qC=qC, vC=vC, xEnd=xe))
    if not H:
        return sp
    single = len(H) == 1
    xref = max(h["xEnd"] for h in H)
    cmap = colors(len(H), o["scheme"])
    chcol = RED
    if not axis_col:
        chcol = colors(2, o["scheme"])[1] if single else (0, 0, 0)
    allV, info = [], []
    for k, h in enumerate(H):
        ls = STYLES[k % 4]
        if axis_col:
            shade = (k // 4) / max(1, int(np.ceil(len(H) / 4)))
            colD = tuple(np.clip(np.array([0, 0, 0]) + shade * np.array([.25, .25, .55]), 0, 1))
            colC = tuple(np.clip(np.array(RED) + shade * np.array([-.25, 0, .35]), 0, 1))
        elif single:
            colD, colC = colors(2, o["scheme"])
            ls = "-"
        else:
            colD = colC = cmap[k]
            ls = "-"
        if single:
            lD, lC = "Discharge", "Charge"
        elif axis_col:
            lD, lC = f"{h['name']} discharge", f"{h['name']} charge"
        else:
            lD, lC = h["name"], f"{h['name']} charge"
        if len(h["qD"]):
            sp.series.append(S(lD, h["qD"], h["vD"], colD, ls, xname=f"{h['tag']} Discharge Capacity",
                               xunit=CAP_UNIT[unit], yname=f"{h['tag']} Discharge Voltage", yunit="V"))
            allV += list(h["vD"])
        if len(h["qC"]):
            xc = h["qC"] if not mirror else ((h["xEnd"] if align else xref) - h["qC"])
            lsC = "--" if (not axis_col and not single) else ls
            sp.series.append(S(lC, xc, h["vC"], colC, lsC, legend=axis_col or single,
                               xname=f"{h['tag']} Charge plot-x (Qref - Qch)", xunit=CAP_UNIT[unit],
                               yname=f"{h['tag']} Charge Voltage", yunit="V"))
            sp.series.append(S("", h["qC"], h["vC"], colC, show=False, legend=False, crop=False,
                               xname=f"{h['tag']} Charge Capacity", xunit=CAP_UNIT[unit],
                               yname=f"{h['tag']} Charge Voltage", yunit="V"))
            allV += list(h["vC"])
        if len(h["qD"]) and len(h["qC"]):
            vd, vc = hyst50(h["qD"], h["vD"], h["qC"], h["vC"])
            if np.isfinite(vd) and np.isfinite(vc):
                info.append(f"{h['tag']}: dV(50%) = {1000 * (vc - vd):.0f} mV")
    sp.info = "   ".join(info[:6])
    if not allV:
        return sp
    lo, hi = np.nanmin(allV), np.nanmax(allV)
    rng = max(hi - lo, 0.1)
    if mirror and (single or not align):
        sp.top = {"xref": xref, "label": CAP_LABEL[unit], "color": chcol}
    h = H[0]
    xc0 = xref if (mirror and not align) else h["xEnd"]
    if len(h["qD"]):
        qT = h["xEnd"]
        qu, iu = np.unique(h["qD"], return_index=True)
        vAt = lambda qq: float(np.interp(qq, qu, h["vD"][iu]))
        off = 0.04 * rng
        if o["hy_lis"] and single:
            for frac, txt, ha in ((0.05, "Li$_2$S$_8$", "left"), (0.28, "Li$_2$S$_6$", "left"),
                                  (0.45, "Li$_2$S$_4$", "center"), (0.70, "Li$_2$S$_2$", "center"),
                                  (1.0, "Li$_2$S", "center")):
                sp.ann.append(dict(type="text", x=frac * qT, y=vAt(frac * qT) - off, text=txt,
                                   color=(0, 0, 0), ha=ha, va="top"))
            if len(h["qC"]) and mirror:
                sp.ann.append(dict(type="text", x=xc0 - np.nanmax(h["qC"]), y=h["vC"][-1] + off,
                                   text="Starting with S$_8$", color=chcol, ha="left", va="bottom"))
        if o["hy_arrows"]:
            y0 = lo + 0.07 * rng
            sp.ann.append(dict(type="arrow", x1=0.30 * qT, x2=0.52 * qT, y=y0, color=(0, 0, 0)))
            sp.ann.append(dict(type="text", x=0.41 * qT, y=y0 + 0.015 * rng, text="Discharge",
                               color=(0, 0, 0), ha="center", va="bottom"))
            if len(h["qC"]) and mirror:
                cu, ic = np.unique(xc0 - h["qC"], return_index=True)
                xa1, xa2 = 0.84 * qT, 0.62 * qT
                yC = max(np.interp([xa1, xa2], cu, h["vC"][ic])) + 0.09 * rng
                sp.ann.append(dict(type="arrow", x1=xa1, x2=xa2, y=yC, color=chcol))
                sp.ann.append(dict(type="text", x=(xa1 + xa2) / 2, y=yC + 0.015 * rng, text="Charge",
                                   color=chcol, ha="center", va="bottom"))
        if o["hy_dv"] and single and len(h["qC"]) and mirror:
            vd, vc = hyst50(h["qD"], h["vD"], h["qC"], h["vC"])
            if np.isfinite(vd) and np.isfinite(vc):
                xm = 0.5 * qT
                if align or abs((xref - 0.5 * qT) - xm) < 1e-9:
                    sp.series.append(S("", [xm, xm], [vd, vc], (0.1, 0.3, 0.85), ":", lw=2,
                                       legend=False, export=False, crop=False))
                sp.ann.append(dict(type="text", x=xm, y=(vd + vc) / 2,
                                   text=f"  $\\Delta$V = {1000 * (vc - vd):.0f} mV",
                                   color=(0.1, 0.3, 0.85), ha="left", va="center"))
    # export-only summary per cycle
    for ds in S_:
        f = cap_factor(ds, unit)
        if np.isnan(f):
            continue
        pre = f"{ds.nick} " if len(S_) > 1 else ""
        M = np.full((len(ds.cycles), 5), np.nan)
        for i in range(len(ds.cycles)):
            qd, qc = ds.cur["qD"][i] * f, ds.cur["qC"][i] * f
            if len(qd) and len(qc):
                vd, vc = hyst50(qd, ds.cur["vD"][i], qc, ds.cur["vC"][i])
                M[i] = [np.nanmax(qd), np.nanmax(qc), vd, vc, 1000 * (vc - vd)]
        for j, (nm, un) in enumerate((("Discharge Capacity", CAP_UNIT[unit]), ("Charge Capacity", CAP_UNIT[unit]),
                                      ("V dis @50%", "V"), ("V ch @50%", "V"), ("Hysteresis dV @50%", "mV"))):
            sp.series.append(S("", ds.cycles, M[:, j], show=False, legend=False, crop=False,
                               xname=f"{pre}Cycle", yname=f"{pre}{nm}", yunit=un))
    return sp


def _marker(lbl, x, y, col, filled=True, mk="o", panel_=0, **kw):
    return S(lbl, x, y, col, "none", mk, col if filled else "none", panel=panel_, lw=1, **kw)


def build_capacity(DS, o):
    mode = o["cap_mode"]
    S_ = _shown(DS, ["Cycling", "CV"])
    cS = colors(len(S_), o["scheme"])
    CE = "Coulombic efficiency (%)"
    if mode.startswith("Energy"):
        sp = Spec([panel("", "Specific energy (Wh kg$^{-1}$)"), panel("", "Average voltage (V)"),
                   panel("Cycle number", "Energy efficiency (%)", 0.8)])
        for s, ds in enumerate(_shown(DS, ["Cycling"])):
            f = spec_factor(ds)
            if np.isnan(f):
                sp.warn.append(ds.nick)
                continue
            pre = f"{ds.nick} " if len(S_) > 1 else ""
            n = len(ds.cycles)
            Ed, Ec, Qd, Qc = (np.full(n, np.nan) for _ in range(4))
            for i in range(n):
                if len(ds.cur["qD"][i]) > 1:
                    q = ds.cur["qD"][i] * f
                    Ed[i], Qd[i] = _trapz(ds.cur["vD"][i], q), np.nanmax(q)
                if len(ds.cur["qC"][i]) > 1:
                    q = ds.cur["qC"][i] * f
                    Ec[i], Qc[i] = _trapz(ds.cur["vC"][i], q), np.nanmax(q)
            col, x = cS[s], ds.cycles
            for y, lbl, nm, un, p, filled, mk in (
                    (Ed, "discharge energy", "Discharge energy", "Wh/kg", 0, True, "o"),
                    (Ec, "charge energy", "Charge energy", "Wh/kg", 0, False, "o"),
                    (Ed / Qd, "discharge", "Average discharge voltage", "V", 1, True, "o"),
                    (Ec / Qc, "charge", "Average charge voltage", "V", 1, False, "o"),
                    (100 * Ed / Ec, "energy efficiency", "Energy efficiency (Edis/Ech)", "%", 2, False, "^")):
                sp.series.append(_marker(pre + lbl, x, y, col, filled, mk, p, xname=f"{pre}Cycle",
                                         yname=pre + nm, yunit=un))
        return sp
    if mode.startswith("Hysteresis"):
        sp = Spec([panel("", "Hysteresis $\\Delta$V$_{50\\%}$ (mV)", 2), panel("Cycle number", CE)])
        for s, ds in enumerate(_shown(DS, ["Cycling"])):
            f = spec_factor(ds)
            if np.isnan(f):
                sp.warn.append(ds.nick)
                continue
            pre = f"{ds.nick} " if len(S_) > 1 else ""
            dv = np.full(len(ds.cycles), np.nan)
            for i in range(len(ds.cycles)):
                if len(ds.cur["qD"][i]) and len(ds.cur["qC"][i]):
                    vd, vc = hyst50(ds.cur["qD"][i] * f, ds.cur["vD"][i], ds.cur["qC"][i] * f, ds.cur["vC"][i])
                    dv[i] = 1000 * (vc - vd)
            ce = summary(ds, f)[2]
            sp.series.append(_marker(pre + "$\\Delta$V", ds.cycles, dv, cS[s], True, "o", 0,
                                     xname=f"{pre}Cycle", yname=f"{pre}Hysteresis dV at 50%", yunit="mV"))
            sp.series.append(_marker(pre + "CE", ds.cycles, ce, cS[s], False, "^", 1,
                                     xname=f"{pre}Cycle", yname=f"{pre}Coulombic Efficiency", yunit="%"))
        return sp
    three = mode.startswith("Discharge & charge")
    lab1 = {"Discharge only": "Discharge capacity (mAh g$^{-1}$)",
            "Charge only": "Charge capacity (mAh g$^{-1}$)",
            "Discharge + charge (1 panel)": "Specific capacity (mAh g$^{-1}$)"}.get(
        mode, "Discharge capacity (mAh g$^{-1}$)")
    if three:
        sp = Spec([panel("", lab1), panel("", "Charge capacity (mAh g$^{-1}$)"), panel("Cycle number", CE, 0.8)])
        pc, pce = 1, 2
    else:
        sp = Spec([panel("", lab1, 2), panel("Cycle number", CE)])
        pc, pce = 0, 1
    showD, showC = mode != "Charge only", mode != "Discharge only"
    ring_done = set()
    for s, ds in enumerate(S_):
        f = spec_factor(ds)
        if np.isnan(f):
            sp.warn.append(ds.nick)
            continue
        dis, ch, ce = summary(ds, f)
        x = np.array(ds.cycles)
        col = cS[s]
        pre = f"{ds.nick} " if len(S_) > 1 else ""
        m = np.isin(x, o["cycles"])
        if showD:
            sp.series.append(_marker(pre + "discharge", x, dis, col, True, "o", 0, xname=f"{pre}Cycle",
                                     yname=f"{pre}Discharge Capacity", yunit="mAh/g"))
        if showC:
            sp.series.append(_marker(pre + "charge", x, ch, col, False, "o", pc if three else 0,
                                     xname=f"{pre}Cycle", yname=f"{pre}Charge Capacity", yunit="mAh/g"))
        sp.series.append(_marker(pre + "CE", x, ce, col, False, "^", pce, xname=f"{pre}Cycle",
                                 yname=f"{pre}Coulombic Efficiency", yunit="%"))
        if m.any():
            rings = [(0, dis if showD else ch)]
            if three and showD and showC:
                rings.append((pc, ch))
            for p_, yy in rings:
                sp.series.append(S("selected cycles", x[m], yy[m], (0, 0, 0), "none", "o", "none", 12, 1.5,
                                   panel=p_, legend=p_ not in ring_done, export=False))
                ring_done.add(p_)
    if any(d.show and d.kind == "Plating/Stripping" for d in DS):
        sp.info = "plating/stripping data is shown in the Plating/Stripping tab"
    return sp


def build_rate(DS, o):
    as_c, ref = o["rate_label"] == "C-rate", o["rate_ref"]
    avg = o["rate_view"].startswith("Average")
    xl = ("C-rate (h$^{-1}$)" if as_c else "Specific current (A g$^{-1}$)") if avg else "Cycle number"
    sp = Spec([panel(xl, "Specific capacity (mAh g$^{-1}$)")])
    S_ = _shown(DS, ["Cycling"])
    if not S_:
        sp.info = "No cycling data shown"
        return sp
    cS = colors(len(S_), o["scheme"])
    allx, vmax, vmin = [], -np.inf, np.inf
    for s, ds in enumerate(S_):
        f = spec_factor(ds)
        if np.isnan(f):
            sp.warn.append(ds.nick)
            continue
        pre = f"{ds.nick} " if len(S_) > 1 else ""
        rv, stp, nst = rate_steps(ds, as_c, ref)
        dis, ch, _ = summary(ds, f)
        x = np.array(ds.cycles)
        labs = [rate_label(np.nanmedian(rv[stp == k]), as_c) for k in range(1, nst + 1)]
        plain = [l.replace(" g$^{-1}$", "/g").replace("$^{-1}$", "-1") for l in labs]
        if s == 0:
            sp.info = f"{nst} rate steps: " + ", ".join(plain)
        vmax = np.nanmax([vmax, np.nanmax(dis), np.nanmax(ch)])
        vmin = np.nanmin([vmin, np.nanmin(dis), np.nanmin(ch)])
        if not avg:
            cm = colors(nst, o["scheme"])
            for k in range(1, nst + 1):
                m = stp == k
                col, lbl = (cm[k - 1], labs[k - 1]) if len(S_) == 1 else (cS[s], ds.nick)
                leg = len(S_) == 1 or k == 1
                if o["show_dis"]:
                    sp.series.append(_marker(lbl, x[m], dis[m], col, True, legend=leg, xname=f"{pre}Cycle",
                                             yname=f"{pre}{plain[k - 1]} Discharge Capacity", yunit="mAh/g"))
                if o["show_ch"]:
                    sp.series.append(_marker(lbl, x[m], ch[m], col, False, legend=leg and not o["show_dis"],
                                             xname=f"{pre}Cycle", yname=f"{pre}{plain[k - 1]} Charge Capacity",
                                             yunit="mAh/g"))
                if s == 0:
                    sp.ann.append(dict(type="text", x=float(np.mean(x[m])),
                                       y=float(np.nanmax(np.r_[dis[m], ch[m]])), text=labs[k - 1],
                                       color=(0, 0, 0), ha="center", va="bottom", lift=True))
        else:
            rx = np.array([np.nanmedian(rv[stp == k]) for k in range(1, nst + 1)])
            mu = np.array([np.nanmean(dis[stp == k]) for k in range(1, nst + 1)])
            sd = np.array([np.nanstd(dis[stp == k], ddof=1) if (stp == k).sum() > 1 else 0
                           for k in range(1, nst + 1)])
            muC = np.array([np.nanmean(ch[stp == k]) for k in range(1, nst + 1)])
            nn = np.array([(stp == k).sum() for k in range(1, nst + 1)])
            col = cS[s]
            sp.series.append(S("discharge" if len(S_) == 1 else ds.nick, rx, mu, col, "none", "o", col, 8,
                               export=False))
            sp.series.append(S("", rx, mu, col, "none", "", legend=False, export=False, crop=False))
            sp.series[-1].err = sd                       # error bars
            allx += list(rx)
            ru = "C" if as_c else "A/g"
            for y, nm, un in ((rx, "Rate", ru), (mu, "Mean discharge capacity", "mAh/g"),
                              (sd, "Std discharge capacity", "mAh/g"), (muC, "Mean charge capacity", "mAh/g"),
                              (nn, "Cycles", "")):
                sp.series.append(S("", np.arange(1, nst + 1), y, show=False, legend=False, crop=False,
                                   xname=f"{pre}Step", yname=f"{pre}{nm}", yunit=un))
    if sp.ann and np.isfinite(vmax):
        off = 0.03 * max(vmax - vmin, 1)
        for a in sp.ann:
            a["y"] += off
        sp.series.append(S("", [sp.ann[0]["x"]], [vmax + 6 * off], (1, 1, 1), "none", "",
                           legend=False, export=False, crop=False))
    allx = np.array([a for a in allx if np.isfinite(a) and a > 0])
    if len(allx) and allx.max() / allx.min() > 20:
        sp.panels[0]["xscale"] = "log"
    return sp


def build_cv(DS, o):
    yu = o["cv_unit"]
    ylab = {"Ag": "Specific current (A g$^{-1}$)", "mAmg": "Specific current (mA mg$^{-1}$)",
            "mAg": "Specific current (mA g$^{-1}$)", "mAcm2": "Current density (mA cm$^{-2}$)"}.get(yu, "Current (mA)")
    sp = Spec([panel(VLABEL, ylab)])
    S_ = _shown(DS, ["CV"])
    if not S_:
        sp.info = "No CV data shown: load a PAT CV file or set Type = CV in the table"
        return sp
    cyc = o["cycles"]
    cS, cC = _cycle_colors(len(S_), len(cyc), o["scheme"])
    for s, ds in enumerate(S_):
        scale, unit = 1.0, "mA"
        if yu in ("Ag", "mAmg", "mAg"):
            m = mass_g(ds)
            if not m > 0:
                sp.warn.append(ds.nick)
                continue
            scale, unit = {"mAg": (1 / m, "mA/g"), "Ag": (1 / (1000 * m), "A/g"),
                           "mAmg": (1 / (1000 * m), "mA/mg")}[yu]
        elif yu == "mAcm2":
            if not ds.area > 0:
                sp.warn.append(ds.nick)
                continue
            scale, unit = 1 / ds.area, "mA/cm2"
        pre = f"{ds.nick} " if len(S_) > 1 else ""
        for k, c in enumerate(cyc):
            i = ds.idx(c)
            if i is None or len(ds.cur["aV"][i]) == 0:
                continue
            col, ls, lbl = (cC[k], "-", f"Cycle {c}") if len(S_) == 1 else (cS[s], STYLES[k % 4], f"{ds.nick}  C{c}")
            rate = ds.cur["scanRate"][i]
            tag = f"{pre}C{c}"
            if np.isfinite(rate):
                if o["cv_rate_legend"]:
                    lbl += f"  ({rate:g} mV/s)"
                tag += f" ({rate:g} mV/s)"
            I = smooth(ds.cur["aI"][i] * scale, o["cv_smooth"], o["cv_win"])
            sp.series.append(S(lbl, ds.cur["aV"][i], I, col, ls, xname=f"{tag} Potential", xunit="V",
                               yname=f"{tag} Current", yunit=unit))
    return sp


def build_cvk(DS, o):
    view = o["k_view"]
    wc, wa = sorted(o["k_cwin"]), sorted(o["k_awin"])
    nE, Cm = o["k_n"], o["k_conc"] / 1000
    if view.startswith("Peak current"):
        xl, yl = "$\\nu^{1/2}$ (V$^{1/2}$ s$^{-1/2}$)", "Peak current (mA)"
    elif view.startswith("Peak potential"):
        xl, yl = "Scan rate (mV s$^{-1}$)", "Peak potential (V vs. Li/Li$^+$)"
    else:
        xl, yl = "log($\\nu$ / mV s$^{-1}$)", "log(|$i_p$| / mA)"
    sp = Spec([panel(xl, yl)])
    S_ = _shown(DS, ["CV"])
    if not S_:
        sp.info = "No CV data shown"
        return sp
    cS = colors(max(len(S_), 2), o["scheme"])
    info = []
    for s, ds in enumerate(S_):
        pre = f"{ds.nick} " if len(S_) > 1 else ""
        colA, colC = (cS[0], cS[1]) if len(S_) == 1 else (cS[s], cS[s])
        rates = np.unique(ds.cur["scanRate"][np.isfinite(ds.cur["scanRate"])])
        P = []
        for r in rates:
            idx = np.where(ds.cur["scanRate"] == r)[0]
            idx = {"Last": idx[-1:], "First": idx[:1]}.get(o["k_pick"], idx)
            for i in idx:
                v = ds.cur["aV"][i]
                I = smooth(ds.cur["aI"][i], o["cv_smooth"], o["cv_win"])
                if len(v) == 0:
                    continue
                pk = [np.nan] * 4
                mc = (v >= wc[0]) & (v <= wc[1]) & (I < 0)
                if mc.any():
                    j = np.argmin(I[mc]); pk[1], pk[3] = I[mc][j], v[mc][j]
                ma = (v >= wa[0]) & (v <= wa[1]) & (I > 0)
                if ma.any():
                    j = np.argmax(I[ma]); pk[0], pk[2] = I[ma][j], v[ma][j]
                P.append([r] + pk + [ds.cycles[i]])
        if not P:
            continue
        P = np.array(P)
        nu = P[:, 0]
        if view.startswith("Peak current"):
            x, ya, yc = np.sqrt(nu / 1000), P[:, 1], P[:, 2]
        elif view.startswith("Peak potential"):
            x, ya, yc = nu, P[:, 3], P[:, 4]
        else:
            x, ya, yc = np.log10(nu), np.log10(np.abs(P[:, 1])), np.log10(np.abs(P[:, 2]))
        sp.series.append(S(pre + "anodic", x, ya, colA, "none", "o", colA, 8, export=False))
        sp.series.append(S(pre + "cathodic", x, yc, colC, "none", "s", "none", 8, 1.5, export=False))
        if not view.startswith("Peak potential"):
            for nm, y, col in (("anodic", ya, colA), ("cathodic", yc, colC)):
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 2:
                    continue
                p = np.polyfit(x[ok], y[ok], 1)
                yf = np.polyval(p, x[ok])
                r2 = 1 - np.sum((y[ok] - yf) ** 2) / max(np.sum((y[ok] - y[ok].mean()) ** 2), 1e-30)
                xf = np.array([0 if view.startswith("Peak current") else x[ok].min(), x[ok].max()])
                sp.series.append(S(f"{pre}{nm} fit", xf, np.polyval(p, xf), col, "--", lw=1.5,
                                   export=False, crop=False))
                if view.startswith("Peak current"):
                    D = "enter area for D"
                    if ds.area > 0:
                        D = f"D = {(abs(p[0]) / 1000 / (2.69e5 * nE ** 1.5 * ds.area * Cm)) ** 2:.2e} cm2/s"
                    info.append(f"{pre}{nm}: slope {p[0]:.3g} mA (V/s)^-0.5, R2 {r2:.3f}, {D}")
                else:
                    info.append(f"{pre}{nm}: b = {p[0]:.2f} (R2 {r2:.3f})")
        for y, nm, un in ((np.sqrt(nu / 1000), "Scan rate^0.5", "(V/s)^0.5"), (P[:, 1], "Anodic peak current", "mA"),
                          (P[:, 2], "Cathodic peak current", "mA"), (P[:, 3], "Anodic peak potential", "V"),
                          (P[:, 4], "Cathodic peak potential", "V"), (P[:, 5], "Cycle", "")):
            sp.series.append(S("", nu, y, show=False, legend=False, crop=False, xname=f"{pre}Scan rate",
                               xunit="mV/s", yname=pre + nm, yunit=un))
    sp.info = "   ".join(info)
    if view.startswith("log"):
        sp.info += "   (b = 0.5 diffusion-controlled, b = 1 capacitive)"
    return sp


def build_ps(DS, o):
    view = o["ps_view"]
    sp = Spec([panel("Time (h)", "Voltage (V)")])
    S_ = _shown(DS, ["Plating/Stripping"])
    if not S_:
        sp.info = "No plating/stripping data shown (set Type = Plating/Stripping in the table)"
        return sp
    cyc = o["cycles"]
    cS, cC = _cycle_colors(len(S_), len(cyc), o["scheme"])
    allT, allV, rate_txt, info = [], [], "", []
    for s, ds in enumerate(S_):
        pre = f"{ds.nick} " if len(S_) > 1 else ""
        col = cS[s]
        use = [i for i, c in enumerate(ds.cycles) if (c in cyc or not o["ps_sel"])]
        full = any(len(v) for v in ds.cur["vAll"])
        c12 = ds.cur["extra"].get("c12") if not full else None
        if not full and c12 is None:
            info.append(f"{ds.nick}: no voltage data in this file")
            continue
        if full:
            j, q = ps_rates(ds)
            if not rate_txt and np.isfinite(j):
                rate_txt = f"{j / ds.area:.3g} mA cm$^{{-2}}$" if ds.area > 0 else f"{j:.3g} mA"
            info.append(pre + ps_text(j, q, ds.area))
        else:
            mC = np.isin(c12["cycle"], [ds.cycles[i] for i in use])
            info.append(f"{ds.nick}: only the C12 output points - load the main PAT file for the full trace")
        if view == "Voltage vs time":
            if full:
                t = np.concatenate([ds.cur["tAll"][i] for i in use]) if use else np.array([])
                v = np.concatenate([ds.cur["vAll"][i] for i in use]) if use else np.array([])
            else:
                t, v = c12["t"][mC], c12["V"][mC]
            if len(t) == 0:
                continue
            lbl = "Voltage" if len(S_) == 1 else ds.nick
            if len(t) > 60000:
                td, vd = envelope(t, v)
                sp.series.append(S(lbl, td / 3600, vd, col, export=False))
                sp.series.append(S("", t / 3600, v, col, show=False, legend=False, xname=f"{pre}Time",
                                   xunit="h", yname=f"{pre}Voltage", yunit="V"))
            else:
                sp.series.append(S(lbl, t / 3600, v, col, "-", "" if full else "o", col, 4,
                                   xname=f"{pre}Time", xunit="h", yname=f"{pre}Voltage", yunit="V"))
            allT += [t.min() / 3600, t.max() / 3600]
            allV += [np.nanmin(v), np.nanmax(v)]
        elif view == "Voltage vs areal capacity":
            sp.panels[0]["xlabel"] = "Areal capacity (mAh cm$^{-2}$)"
            if not full:
                continue
            fA, un = cap_factor(ds, "cm2"), "mAh/cm2"
            if np.isnan(fA):
                sp.warn.append(ds.nick)
                fA, un = 1000, "mAh"
                sp.panels[0]["xlabel"] = "Capacity (mAh)"
            for k, c in enumerate(cyc):
                i = ds.idx(c)
                if i is None:
                    continue
                cc, lbl = (cC[k], f"Cycle {c}") if len(S_) == 1 else (col, f"{ds.nick}  C{c}")
                first = True
                for half, q, v, ls in (("Plating", "qD", "vD", "-"), ("Stripping", "qC", "vC", "--")):
                    if len(ds.cur[q][i]):
                        sp.series.append(S(lbl, ds.cur[q][i] * fA, ds.cur[v][i], cc, ls, legend=first,
                                           xname=f"{pre}C{c} {half} Capacity", xunit=un,
                                           yname=f"{pre}C{c} {half} Voltage", yunit="V"))
                        first = False
        elif view == "Overpotential vs cycle":
            sp.panels[0].update(xlabel="Cycle number", ylabel="Overpotential (mV)")
            n = len(ds.cycles)
            avgP, maxP, avgS, maxS = (np.full(n, np.nan) for _ in range(4))
            for i in range(n):
                if full:
                    vp, vs = np.abs(ds.cur["vD"][i]), np.abs(ds.cur["vC"][i])
                else:
                    m = c12["cycle"] == ds.cycles[i]
                    vp, vs = np.abs(c12["V"][m & (c12["V"] < 0)]), c12["V"][m & (c12["V"] > 0)]
                if len(vp):
                    avgP[i], maxP[i] = 1000 * np.nanmean(vp), 1000 * np.nanmax(vp)
                if len(vs):
                    avgS[i], maxS[i] = 1000 * np.nanmean(vs), 1000 * np.nanmax(vs)
            x = np.array(ds.cycles)[use]
            for y, lbl, nm, filled, mk, ls in ((avgS, "stripping", "Stripping overpotential (mean)", True, "o", "none"),
                                               (avgP, "plating", "Plating overpotential (mean)", False, "o", "none")):
                sp.series.append(_marker(pre + lbl, x, y[use], col, filled, mk, xname=f"{pre}Cycle",
                                         yname=pre + nm, yunit="mV"))
            if o["ps_max"]:
                for y, lbl, nm, filled, mk in ((maxS, "stripping max", "Stripping overpotential (max)", True, "^"),
                                               (maxP, "plating max", "Plating overpotential (max)", False, "v")):
                    s_ = _marker(pre + lbl, x, y[use], col, filled, mk, xname=f"{pre}Cycle", yname=pre + nm, yunit="mV")
                    s_.ls, s_.ms = ":", 5
                    sp.series.append(s_)
        else:                                                         # CE (Li||Cu)
            sp.panels[0].update(xlabel="Cycle number", ylabel="Coulombic efficiency (%)")
            if not full:
                continue
            ce = np.array([100 * np.nanmax(qc) / np.nanmax(qd) if len(qd) and len(qc) and np.nanmax(qd) > 0
                           else np.nan for qd, qc in zip(ds.cur["qD"], ds.cur["qC"])])
            sp.series.append(_marker(pre + "CE", np.array(ds.cycles)[use], ce[use], col, xname=f"{pre}Cycle",
                                     yname=f"{pre}Coulombic Efficiency (strip/plate)", yunit="%"))
    if view == "Voltage vs time" and allT and rate_txt:
        r = max(allV) - min(allV)
        xm, ym = (min(allT) + max(allT)) / 2, min(allV) - 0.12 * r
        sp.ann.append(dict(type="text", x=xm, y=ym, text=rate_txt, color=(0, 0, 0), ha="center", va="top"))
        sp.series.append(S("", [xm, xm], [ym - 0.15 * r, max(allV) + 0.05 * r], (1, 1, 1), "none",
                           legend=False, export=False, crop=False))
    sp.info = "   ".join(info[:4])
    return sp


def build_eis(DS, o):
    view, zi = o["eis_view"], o["eis_z"]
    normA = o["eis_area"]
    zu, zt = ("$\\Omega$ cm$^2$", "Ohm cm2") if normA else ("$\\Omega$", "Ohm")
    if view == "C12 vs cycle":
        sp = Spec([panel("Cycle number", "C12 (mAh V$^{-1}$)")])
        S_ = _shown(DS, ["C12"])
        if not S_:
            sp.info = "No C12 file shown (load an Output-C12.txt)"
        cS = colors(len(S_), o["scheme"])
        for s, ds in enumerate(S_):
            c = ds.cur["extra"]["c12"]
            pre = f"{ds.nick} " if len(S_) > 1 else ""
            m = np.isin(c["cycle"], o["cycles"])
            pos, neg = m & (c["dQ"] > 0), m & (c["dQ"] < 0)
            sp.series.append(_marker(pre + "charge", c["cycle"][pos], c["C"][pos], cS[s], True, export=False))
            sp.series.append(_marker(pre + "discharge", c["cycle"][neg], c["C"][neg], cS[s], False, export=False))
            for key, nm, un in (("V", "V12", "V"), ("dQ", "dQ12", "uAh"), ("C", "C12", "mAh/V")):
                sp.series.append(S("", c["cycle"][m], c[key][m], show=False, legend=False, crop=False,
                                   xname=f"{pre}Cycle", yname=pre + nm, yunit=un))
        return sp
    if view == "Nyquist":
        sp = Spec([panel(f"Z' ({zu})", f"-Z'' ({zu})", equal=o["eis_equal"])])
    elif view == "Bode: |Z| + phase":
        sp = Spec([panel("", f"|Z| ({zu})", 1, "log", "log"), panel("Frequency (Hz)", "-Phase (°)", 1, "log")])
    elif view == "Bode: |Z|":
        sp = Spec([panel("Frequency (Hz)", f"|Z| ({zu})", 1, "log", "log")])
    elif view == "Bode: phase":
        sp = Spec([panel("Frequency (Hz)", "-Phase (°)", 1, "log")])
    else:
        sp = Spec([panel("Cycle number", f"Resistance ({zu})")])
    S_ = _shown(DS, ["EIS"])
    if not S_:
        sp.info = "No impedance file shown (load an Output-Impedance.txt)"
        return sp
    lst = [(s, q) for s, ds in enumerate(S_) for q, spc in enumerate(ds.cur["extra"]["spectra"])
           if spc["cycle"] in o["cycles"]]
    if not lst:
        sp.info = "None of the selected cycles has a spectrum"
        return sp
    if view.startswith("Resistances"):
        cS = colors(max(len(S_), 2), o["scheme"])
        info = []
        for s, ds in enumerate(S_):
            rows = [q for s2, q in lst if s2 == s]
            sc = ds.area if normA else 1.0
            if normA and not ds.area > 0:
                sp.warn.append(ds.nick)
                continue
            cy, Rs, Rct = [], [], []
            for q in rows:
                spc = ds.cur["extra"]["spectra"][q]
                if spc["re"][zi] is None:
                    continue
                a, b = eis_res(spc["f"], spc["re"][zi] * sc, -spc["im"][zi] * sc)
                cy.append(spc["cycle"]); Rs.append(a); Rct.append(b)
            pre = f"{ds.nick} " if len(S_) > 1 else ""
            c1, c2 = (cS[0], cS[1]) if len(S_) == 1 else (cS[s], cS[s])
            sp.series.append(S(pre + "R$_s$", cy, Rs, c1, "-", "o", c1, 7, xname=f"{pre}Cycle", yname=f"{pre}Rs", yunit=zt))
            sp.series.append(S(pre + "R$_{ct}$", cy, Rct, c2, "-", "s", "none", 7, xname=f"{pre}Cycle",
                               yname=f"{pre}Rct", yunit=zt))
            if cy:
                info.append(f"{pre}Rs {Rs[0]:.3g} -> {Rs[-1]:.3g}, Rct {Rct[0]:.3g} -> {Rct[-1]:.3g} {zt}")
        sp.info = "   ".join(info)
        return sp
    cmap = colors(len(lst), o["scheme"])
    missing = False
    for r, (s, q) in enumerate(lst):
        ds, spc = S_[s], S_[s].cur["extra"]["spectra"][q]
        re_, im_ = spc["re"][zi], spc["im"][zi]
        if re_ is None or not np.isfinite(re_).any():
            missing = True
            continue
        sc = 1.0
        if normA:
            if not ds.area > 0:
                sp.warn.append(ds.nick)
                continue
            sc = ds.area
        zr, zim = re_ * sc, -im_ * sc
        mag, ph = np.hypot(re_, im_) * sc, -np.degrees(np.arctan2(im_, re_))
        lbl = f"{ds.nick}  C{spc['cycle']}" if len(S_) > 1 else f"Cycle {spc['cycle']}"
        if spc["k"] > 1:
            lbl += f" #{spc['k']}"
        col = cmap[r]
        if view == "Nyquist":
            sp.series.append(S(lbl, zr, zim, col, "-", "o", col, 5, export=False))
        elif view == "Bode: |Z| + phase":
            sp.series.append(S(lbl, spc["f"], mag, col, "-", "o", col, 4, export=False))
            sp.series.append(S(lbl, spc["f"], ph, col, "-", "o", col, 4, panel=1, legend=False, export=False))
        elif view == "Bode: |Z|":
            sp.series.append(S(lbl, spc["f"], mag, col, "-", "o", col, 4, export=False))
        else:
            sp.series.append(S(lbl, spc["f"], ph, col, "-", "o", col, 4, export=False))
        tag = lbl.replace("  ", " ")
        for y, nm, un in ((zr, "Re(Z)", zt), (zim, "-Im(Z)", zt), (mag, "|Z|", zt), (ph, "-Phase", "deg")):
            sp.series.append(S("", spc["f"], y, show=False, legend=False, crop=False, xname=f"{tag} Frequency",
                               xunit="Hz", yname=f"{tag} {nm}", yunit=un))
    if missing:
        sp.info = "This file has no data for the chosen Z (needs a reference electrode)"
    return sp


def cycle_table(ds):
    f = spec_factor(ds)
    dis, ch, ce = summary(ds, f)
    ref = dis[np.isfinite(dis)][0] if np.isfinite(dis).any() else np.nan
    return pd.DataFrame({"Cycle": ds.cycles, "Discharge (mAh/g)": dis, "Charge (mAh/g)": ch,
                         "CE (%)": ce, "Retention (%)": 100 * dis / ref})


# =============================================================================
# Export (Origin: Long Name / Units / Comments)
# =============================================================================
def axis_qty(label):
    q = re.sub(r"\s*\(.*\)\s*$", "", label or "").strip()
    q = q.replace("$\\Omega$", "Ohm").replace("$\\nu$", "nu").replace("$\\Delta$", "d")
    return re.sub(r"[$^_{}\\]", "", q)


def _split_common(a, b):
    wa, wb = a.split(), b.split()
    k = 0
    while k < min(len(wa), len(wb)) - 1 and wa[k] == wb[k]:
        k += 1
    return " ".join(wa[:k]), " ".join(wa[k:]), " ".join(wb[k:])


def export_columns(spec, limits=None, crop=False):
    """-> list of dicts {name, long, unit, comment, data}; limits = [(xlim, ylim) per panel]"""
    cols, last_x = [], None
    for s in spec.series:
        if not s.export or len(s.x) == 0:
            continue
        x, y = s.x, s.y
        if crop and s.crop and limits:
            (x0, x1), (y0, y1) = limits[min(s.panel, len(limits) - 1)]
            m = (x >= min(x0, x1)) & (x <= max(x0, x1)) & (y >= min(y0, y1)) & (y <= max(y0, y1))
            x, y = x[m], y[m]
        if len(x) == 0:
            continue
        cmt, xr, yr = _split_common(s.xname, s.yname)
        xq, yq = xr, yr
        if s.show:
            p = spec.panels[min(s.panel, len(spec.panels) - 1)]
            xl = p["xlabel"] or spec.panels[-1]["xlabel"]
            xq, yq = axis_qty(xl) or xr, axis_qty(p["ylabel"]) or yr
            cmt = cmt or s.label
        if "plot-x" in xr:
            cmt = (cmt + " (mirrored x = Qdis,end - Qch)").strip()
        cmt = re.sub(r"^C(\d+)(?=\s|$)", r"Cycle \1", cmt).replace("$", "")
        if last_x is None or len(last_x) != len(x) or not np.array_equal(last_x, x, equal_nan=True):
            cols.append({"name": s.xname, "long": xq, "unit": s.xunit, "comment": cmt, "data": x})
            last_x = x
        cols.append({"name": s.yname, "long": yq, "unit": s.yunit, "comment": cmt, "data": y})
    return cols


def table_columns(df, comment=""):
    out = []
    for c in df.columns:
        m = re.match(r"(.*?)\s*\((.*)\)$", c)
        long, unit = (m.group(1), m.group(2)) if m else (c, "")
        out.append({"name": long, "long": long, "unit": unit, "comment": comment, "data": df[c].to_numpy(float)})
    return out


def header_rows(cols, origin):
    if origin:
        return [[c["long"] for c in cols], [c["unit"] for c in cols], [c["comment"] for c in cols]]
    return [[f"{c['name']} ({c['unit']})" if c["unit"] else c["name"] for c in cols]]


def to_frame(cols, origin):
    """header rows + numbers as one object table (blank cells for shorter columns)"""
    n = max(len(c["data"]) for c in cols)
    body = np.full((n, len(cols)), np.nan)
    for j, c in enumerate(cols):
        body[:len(c["data"]), j] = c["data"]
    data = pd.DataFrame(body).astype(object).where(pd.notna(pd.DataFrame(body)), None)
    head = pd.DataFrame(header_rows(cols, origin))
    return pd.concat([head, data], ignore_index=True)


def to_clipboard_text(cols, origin):
    lines = ["\t".join(r) for r in header_rows(cols, origin)]
    n = max(len(c["data"]) for c in cols)
    for i in range(n):
        lines.append("\t".join(f"{c['data'][i]:.10g}" if i < len(c["data"]) and np.isfinite(c["data"][i])
                               else "" for c in cols))
    return "\n".join(lines)


def sheet_name(raw, used):
    name = re.sub(r"[\[\]:*?/\\]", "_", str(raw)).strip("'").strip() or "Data"
    name, base, k = name[:31], name[:31], 2
    while name.lower() in {u.lower() for u in used}:
        suf = f"_{k}"
        name, k = base[:31 - len(suf)] + suf, k + 1
    return name
