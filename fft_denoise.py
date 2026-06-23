#!/usr/bin/env python3
"""
FFT-based noise removal for 1-D data (time series or spectra).

Three methods are provided, because "remove noise" means different things
depending on what your data looks like:

  psd_threshold  Keep only frequency components whose power exceeds a threshold,
                 zero the rest. Best when the true signal is built from a few
                 dominant frequencies (oscillatory / periodic data). The
                 threshold can be found automatically from the noise floor.

  lowpass        Zero every frequency above a cutoff. Best for a smooth signal
                 or spectrum sitting under broadband high-frequency noise
                 (most lab traces, spectra, slowly varying time series).

  notch          Zero a narrow band around one or more frequencies. Best for
                 discrete periodic interference, e.g. 50/60 Hz mains hum.

Quick start
-----------
    python fft_denoise.py                 # runs a self-contained synthetic demo
    python fft_denoise.py data.csv        # denoise your own CSV

Pipeline order: load -> (optional) resample to a uniform grid -> (optional)
background/baseline correction -> FFT denoise -> preview -> save. The first
column is the x-axis; every other column is treated as a replicate.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Prefer an interactive backend so plt.show() works on a desktop, but fall
# back cleanly to the non-interactive Agg backend on a headless machine
# (we always save the plot to a PNG regardless). A candidate backend is only
# accepted if it can actually create a figure -- matplotlib.use() alone selects
# lazily and won't reveal a missing GUI library until later.
import matplotlib
import matplotlib.pyplot as plt


def _ensure_working_backend():
    # If a backend was requested explicitly (e.g. MPLBACKEND=Agg, as the
    # Streamlit app and headless servers set), honour it and don't probe for a
    # GUI -- probing can raise on machines with no display libraries installed.
    if os.environ.get("MPLBACKEND"):
        return
    for name in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
        try:
            plt.switch_backend(name)
            fig = plt.figure()
            plt.close(fig)
            return
        except Exception:
            continue
    plt.switch_backend("Agg")


_ensure_working_backend()

# Seaborn is used purely for styling (theme + palette); the actual plotting is
# done with matplotlib for speed on long signals. If seaborn isn't installed we
# fall back to plain matplotlib so the script still runs.
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
    _PALETTE = sns.color_palette("deep")
    C_NOISY, C_CLEAN, C_SPEC, C_KEPT = (
        _PALETTE[7], _PALETTE[0], _PALETTE[7], _PALETTE[3])
    _HAVE_SEABORN = True
except ImportError:
    sns = None
    C_NOISY, C_CLEAN, C_SPEC, C_KEPT = ("0.45", "C0", "0.45", "C3")
    _HAVE_SEABORN = False


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #
def load_csv(path, x_col=None, y_col=None):
    """Load a CSV and return (x, Y, names).

    The first column is taken as the x-axis (wavelength, time, etc.); every
    remaining column is treated as a separate replicate of the signal. So a file
    laid out as  wavelength, intensity, intensity_rep2, intensity_rep3  yields
    one x array and three replicate columns, each denoised independently.

    x      1-D array of the independent variable, or None if the file has a
           single column (then a sample index is used and Y holds that column).
    Y      2-D array, shape (n_samples, n_replicates).
    names  list of the replicate column names (for the output header / legend).
    x_name name of the x column (or None for a single-column file).

    --x-col overrides which column is the x-axis. --y-col, if given, restricts
    processing to that single column instead of "all columns after x".
    """
    try:
        import pandas as pd
        df = pd.read_csv(path)
        cols = list(df.columns)
        arr = df.to_numpy(dtype=float)
    except ImportError:
        data = np.genfromtxt(path, delimiter=",", names=True)
        cols = list(data.dtype.names)
        arr = np.vstack([data[c] for c in cols]).T

    def resolve(col):
        if isinstance(col, int):
            return col
        lowered = [c.lower() for c in cols]
        if col.lower() in lowered:
            return lowered.index(col.lower())
        raise ValueError(f"Column '{col}' not found. Available: {cols}")

    n_cols = arr.shape[1]
    if n_cols == 1:
        # Single column: no x-axis, one replicate.
        return None, arr[:, [0]], [cols[0]], None

    xi = resolve(x_col) if x_col is not None else 0
    x = np.asarray(arr[:, xi], dtype=float)

    if y_col is not None:
        yidx = [resolve(y_col)]
    else:
        yidx = [j for j in range(n_cols) if j != xi]

    Y = np.asarray(arr[:, yidx], dtype=float)
    names = [cols[j] for j in yidx]
    return x, Y, names, cols[xi]


def estimate_fs(x):
    """Estimate the sampling frequency (Hz) from an x (time) array.

    Returns None if x is None. Assumes roughly uniform sampling and warns if
    the spacing is irregular.
    """
    if x is None:
        return None
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("x (time) values must be strictly increasing.")
    spread = np.std(dx) / np.mean(dx)
    if spread > 1e-3:
        print(f"  [warning] sampling looks non-uniform "
              f"(relative spacing std = {spread:.2e}). FFT assumes uniform "
              f"spacing; results may be unreliable. Consider resampling first.")
    return 1.0 / np.mean(dx)


def resample_uniform(x, Y, method="pchip", step=None):
    """Interpolate (x, Y) onto an evenly-spaced x grid so the FFT is valid.

    Y may be 1-D (one signal) or 2-D with shape (n_samples, n_replicates); all
    replicates share the x-axis and are resampled onto the same uniform grid.

    Use this when the instrument samples at slightly irregular intervals. The
    new grid spacing defaults to the median of the native spacing (preserving
    resolution); pass `step` to override it.

    method
        'pchip'  shape-preserving cubic (smooth, won't overshoot near peaks).
                 Requires SciPy; falls back to 'linear' if SciPy is missing.
        'linear' most conservative -- cannot invent features between points.

    Returns (x_uniform, Y_uniform, step_used) with Y_uniform matching the input
    dimensionality.
    """
    x = np.asarray(x, dtype=float)
    Y = np.asarray(Y, dtype=float)
    was_1d = Y.ndim == 1
    if was_1d:
        Y = Y[:, None]

    # Sort by x and drop duplicate x values (interpolators need x strictly
    # increasing); keep the first occurrence, applied across all replicates.
    order = np.argsort(x)
    x, Y = x[order], Y[order, :]
    ux, idx = np.unique(x, return_index=True)
    if ux.size != x.size:
        print(f"  [note] dropped {x.size - ux.size} duplicate x value(s) "
              f"before resampling.")
        x, Y = ux, Y[idx, :]

    if step is None:
        step = float(np.median(np.diff(x)))
    if step <= 0:
        raise ValueError("resample step must be positive.")

    n = int(round((x.max() - x.min()) / step)) + 1
    x_u = np.linspace(x.min(), x.max(), n)

    if method == "pchip":
        try:
            from scipy.interpolate import PchipInterpolator
            Y_u = PchipInterpolator(x, Y, axis=0)(x_u)
        except ImportError:
            print("  [note] SciPy not installed; using linear interpolation "
                  "instead of PCHIP. Install scipy for shape-preserving "
                  "resampling.")
            Y_u = np.column_stack([np.interp(x_u, x, Y[:, j])
                                   for j in range(Y.shape[1])])
    elif method == "linear":
        Y_u = np.column_stack([np.interp(x_u, x, Y[:, j])
                               for j in range(Y.shape[1])])
    else:
        raise ValueError(f"Unknown resample method: {method!r}")

    if was_1d:
        Y_u = Y_u[:, 0]
    return x_u, Y_u, step


# --------------------------------------------------------------------------- #
#  Core denoising
# --------------------------------------------------------------------------- #
def denoise(y, fs, method="psd_threshold",
            threshold="auto", n_sigma=4.0,
            cutoff=None, notch_freqs=None, notch_width=1.0,
            detrend=True):
    """Return (y_clean, info) where info holds spectrum data for plotting.

    Parameters
    ----------
    y           1-D noisy signal.
    fs          sampling frequency in Hz. If None, a normalised frequency axis
                (cycles per sample) is used and absolute cutoffs are interpreted
                in those units.
    method      'psd_threshold' | 'lowpass' | 'notch'.
    threshold   for psd_threshold: a number, or 'auto' to derive it from the
                noise floor (median + n_sigma * robust spread of the power).
    n_sigma     robustness multiplier for the automatic threshold.
    cutoff      for lowpass: cutoff frequency in Hz (or cycles/sample if fs None).
    notch_freqs for notch: list of centre frequencies to remove.
    notch_width for notch: half-width (Hz) of each removed band.
    detrend     subtract the mean before transforming and add it back after,
                so denoising never silently shifts the DC level of the signal.
    """
    y = np.asarray(y, dtype=float)
    n = y.size

    # Optionally remove (and later restore) the mean so the DC component is
    # never accidentally zeroed by a power threshold.
    offset = y.mean() if detrend else 0.0
    yw = y - offset

    fhat = np.fft.rfft(yw)
    if fs is None:
        freqs = np.fft.rfftfreq(n, d=1.0)   # cycles per sample
        funit = "cycles/sample"
    else:
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)  # Hz
        funit = "Hz"

    psd = np.abs(fhat) ** 2  # power per component (relative; absolute scale irrelevant for masking)
    mask = np.ones_like(fhat, dtype=bool)
    used_threshold = None

    if method == "psd_threshold":
        if threshold == "auto":
            med = np.median(psd)
            mad = np.median(np.abs(psd - med))
            robust_sigma = 1.4826 * mad  # MAD -> std for normal data
            used_threshold = med + n_sigma * robust_sigma
        else:
            used_threshold = float(threshold)
        mask = psd >= used_threshold
        mask[0] = True  # always keep DC

    elif method == "lowpass":
        if cutoff is None:
            raise ValueError("lowpass requires --cutoff")
        mask = freqs <= cutoff

    elif method == "notch":
        if not notch_freqs:
            raise ValueError("notch requires --notch-freqs")
        mask = np.ones_like(freqs, dtype=bool)
        for f0 in notch_freqs:
            mask &= np.abs(freqs - f0) > notch_width

    else:
        raise ValueError(f"Unknown method: {method!r}")

    fhat_clean = fhat * mask
    y_clean = np.fft.irfft(fhat_clean, n=n) + offset

    # Diagnostics
    power_total = psd.sum()
    power_kept = psd[mask].sum()
    info = {
        "freqs": freqs,
        "funit": funit,
        "psd": psd,
        "mask": mask,
        "threshold": used_threshold,
        "cutoff": cutoff,
        "notch_freqs": notch_freqs,
        "n_kept": int(mask.sum()),
        "n_total": int(mask.size),
        "power_kept_frac": power_kept / power_total if power_total else 1.0,
    }
    return y_clean, info


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #
def plot_results(x, Y, Y_clean, names, info, save_path=None, show=True):
    """Three stacked panels: raw replicates, denoised replicates, spectrum.

    Y and Y_clean are 2-D, shape (n_samples, n_replicates). The two signal
    panels share both axes so the before/after amplitude scale is identical.
    Each replicate keeps one colour across both panels so it can be tracked;
    the spectrum panel is shown for the first replicate (the frequency mask is
    the same across replicates for lowpass/notch).
    """
    Y = np.atleast_2d(Y.T).T if Y.ndim == 1 else Y
    Y_clean = np.atleast_2d(Y_clean.T).T if Y_clean.ndim == 1 else Y_clean
    k = Y.shape[1]
    multi = k > 1

    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.15], hspace=0.32)
    ax_before = fig.add_subplot(gs[0])
    ax_after = fig.add_subplot(gs[1], sharex=ax_before, sharey=ax_before)
    ax_f = fig.add_subplot(gs[2])

    xt = x if x is not None else np.arange(Y.shape[0])
    xlabel = "time (s)" if x is not None else "sample index"

    # Single replicate keeps the clean grey/blue look; multiple replicates get
    # one palette colour each, reused across both panels.
    if multi and _HAVE_SEABORN:
        colors = sns.color_palette("husl", k)
    elif multi:
        colors = [f"C{j}" for j in range(k)]
    else:
        colors = None

    for j in range(k):
        c_raw = colors[j] if multi else C_NOISY
        c_clean = colors[j] if multi else C_CLEAN
        lbl = names[j] if names else f"col {j}"
        ax_before.plot(xt, Y[:, j], lw=0.8,
                       alpha=0.7 if multi else 1.0,
                       color=c_raw, label=lbl)
        ax_after.plot(xt, Y_clean[:, j], lw=1.3, color=c_clean, label=lbl)

    ax_before.set_ylabel("amplitude")
    ax_before.set_title("Before  (raw replicates)" if multi
                        else "Before  (raw signal)",
                        loc="left", fontweight="bold")
    ax_after.set_xlabel(xlabel)
    ax_after.set_ylabel("amplitude")
    ax_after.set_title("After  (denoised)", loc="left", fontweight="bold")
    if multi:
        ax_before.legend(loc="best", frameon=True, fontsize=8, ncol=2)

    # --- Power spectrum (first replicate) ---
    freqs, psd, mask = info["freqs"], info["psd"], info["mask"]
    ax_f.semilogy(freqs, psd, lw=0.8, alpha=0.55, color=C_SPEC,
                  label="full spectrum")
    ax_f.semilogy(freqs[mask], psd[mask], "o", ms=3.0, color=C_KEPT,
                  label="kept")
    if info["threshold"] is not None:
        ax_f.axhline(info["threshold"], color="0.2", ls="--", lw=1.2,
                     label="threshold")
    if info["cutoff"] is not None:
        ax_f.axvline(info["cutoff"], color="0.2", ls="--", lw=1.2,
                     label="cutoff")
    for f0 in (info["notch_freqs"] or []):
        ax_f.axvline(f0, color=C_KEPT, ls=":", lw=1.2)
    ax_f.set_xlabel(f"frequency ({info['funit']})")
    ax_f.set_ylabel("power")
    spec_title = ("Power spectrum  (replicate 1; check the cut is where you "
                  "expect)" if multi
                  else "Power spectrum  (check the cut is where you expect)")
    ax_f.set_title(spec_title, loc="left", fontweight="bold")
    ax_f.legend(loc="best", frameon=True)

    if _HAVE_SEABORN:
        sns.despine(fig=fig)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  saved plot -> {save_path}")
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Synthetic demo (runs when no input file is given)
# --------------------------------------------------------------------------- #
def make_demo():
    fs = 1000.0                      # 1000 Hz sampling
    t = np.arange(0, 2.0, 1 / fs)    # 2 seconds
    clean = (np.sin(2 * np.pi * 7 * t)
             + 0.5 * np.sin(2 * np.pi * 23 * t))
    rng = np.random.default_rng(0)
    noisy = clean + 1.2 * rng.standard_normal(t.size)
    return t, noisy, clean, fs


# --------------------------------------------------------------------------- #
#  Baseline / background correction (pybaselines)
# --------------------------------------------------------------------------- #
# The top three methods for SERS / Raman fluorescence backgrounds:
#   arpls     asymmetrically reweighted penalized least squares -- robust,
#             low-tuning; smoothness set by `lam` (larger = stiffer baseline).
#   airpls    adaptive iteratively reweighted PLS -- also `lam`-controlled,
#             a popular Raman choice.
#   imodpoly  improved modified polynomial -- the classic Raman fluorescence
#             remover; controlled by `poly_order` instead of a smoothness.
BASELINE_METHODS = ("arpls", "airpls", "imodpoly")


def baseline_param_label(method, lam, poly_order):
    """Human-readable description of the active parameter for the method."""
    if method in ("arpls", "airpls"):
        return f"lam={lam:g}"
    return f"poly_order={poly_order}"


def apply_baseline(x, Y, method, lam=1e5, poly_order=5):
    """Fit and subtract a background from every replicate.

    Returns (Y_corrected, B) where B holds the fitted baselines (same shape as
    Y), so the caller can plot raw, baseline, and corrected together.
    """
    from pybaselines import Baseline
    fitter = Baseline(x_data=x if x is not None else np.arange(Y.shape[0]))
    Y_corr = np.empty_like(Y)
    B = np.empty_like(Y)
    for j in range(Y.shape[1]):
        col = Y[:, j]
        if method == "arpls":
            bkg, _ = fitter.arpls(col, lam=lam)
        elif method == "airpls":
            bkg, _ = fitter.airpls(col, lam=lam)
        elif method == "imodpoly":
            bkg, _ = fitter.imodpoly(col, poly_order=poly_order)
        else:
            raise ValueError(f"Unknown baseline method: {method!r}")
        B[:, j] = bkg
        Y_corr[:, j] = col - bkg
    return Y_corr, B


def plot_baseline(x, Y, B, Y_corr, names, method, param_label,
                  save_path=None, show=True):
    """Two panels: raw spectra with the fitted baseline overlaid, and the
    baseline-corrected result. Lets you judge the fit before committing."""
    k = Y.shape[1]
    multi = k > 1
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    ax_fit = fig.add_subplot(gs[0])
    ax_corr = fig.add_subplot(gs[1], sharex=ax_fit)

    xt = x if x is not None else np.arange(Y.shape[0])
    xlabel = "x (wavelength / Raman shift)" if x is not None else "sample index"

    if multi and _HAVE_SEABORN:
        colors = sns.color_palette("husl", k)
    elif multi:
        colors = [f"C{j}" for j in range(k)]
    else:
        colors = [C_NOISY]

    for j in range(k):
        c = colors[j] if multi else C_NOISY
        lbl = names[j] if names else f"col {j}"
        ax_fit.plot(xt, Y[:, j], lw=0.8, alpha=0.6, color=c, label=lbl)
        ax_fit.plot(xt, B[:, j], lw=1.5, ls="--", color=c)
        ax_corr.plot(xt, Y_corr[:, j], lw=1.0,
                     color=colors[j] if multi else C_CLEAN, label=lbl)

    ax_fit.set_title(f"Fitted baseline  ({method}, {param_label})   "
                     f"dashed = baseline", loc="left", fontweight="bold")
    ax_fit.set_ylabel("intensity")
    if multi:
        ax_fit.legend(loc="best", fontsize=8, ncol=2)

    ax_corr.axhline(0, color="0.3", lw=0.8, ls=":")
    ax_corr.set_title("Baseline-corrected  (this is what goes into the FFT)",
                      loc="left", fontweight="bold")
    ax_corr.set_xlabel(xlabel)
    ax_corr.set_ylabel("intensity")

    if _HAVE_SEABORN:
        sns.despine(fig=fig)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  saved baseline plot -> {save_path}")
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Cosmic-ray / spike removal (despiking)
# --------------------------------------------------------------------------- #
# Cosmic rays (muon hits) appear as 1-3 pixel outliers with an abrupt rise and
# fall. They are detected via the modified z-score of the first difference of
# the spectrum: a real Raman peak rises over several points (each step modest),
# whereas a spike jumps in a single step (a huge difference). Flagged spikes are
# replaced by linear interpolation across the gap from their unflagged
# neighbours. A max-width guard ensures only narrow features are ever removed,
# so genuine peaks are preserved even if mis-tuned.
def _modified_z(a):
    """Modified z-score (median/MAD based) of a 1-D array."""
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return 0.6745 * (a - med) / (mad if mad > 0 else 1e-12)


def despike_signal(y, threshold=7.0, max_width=3, prominence=6.0):
    """Remove cosmic-ray spikes from one spectrum.

    A cosmic ray is a 1-`max_width`-pixel feature with an abrupt rise and an
    abrupt fall. Candidates are found as a pair of large opposite-sign steps in
    the first difference (modified z-score above `threshold`) lying within
    `max_width` of each other; genuine peaks rise gradually and so never form
    such a narrow pair, and nothing wider than `max_width` is ever removed --
    that width cap is the main thing protecting real peaks. Each candidate must
    also rise more than `prominence` robust-noise sigmas above the line through
    its bracketing neighbours, which simply rejects noise wiggles (real cosmic
    rays clear this by a wide margin). Removed points are filled by linear
    interpolation from their neighbours.

    Returns (y_clean, flag) where flag marks the removed sample positions.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    flag = np.zeros(n, dtype=bool)
    if n < 3:
        return y.copy(), flag

    d = np.diff(y)
    z = np.abs(_modified_z(d))
    sign = np.sign(d)
    jumps = np.where(z > threshold)[0]
    mad_d = np.median(np.abs(d - np.median(d)))
    sigma = 1.4826 * mad_d / np.sqrt(2) or 1e-12   # robust noise scale

    # Collect candidate spike regions [a, b) bracketed by opposite-sign jumps.
    regions = []
    p = 0
    while p < jumps.size:
        k1 = jumps[p]
        partner, pq = -1, -1
        q = p + 1
        while q < jumps.size and (jumps[q] - k1) <= max_width:
            if sign[jumps[q]] != sign[k1] and sign[k1] != 0:
                partner, pq = jumps[q], q
            q += 1
        if partner >= 0:
            regions.append((k1 + 1, partner + 1))
            p = pq + 1
        else:
            p += 1                           # lone step (real edge), not a spike

    # Confirm each candidate by prominence above its bracketing neighbours.
    for a, b in regions:
        left, right = a - 1, b
        if 0 <= left and right < n:
            interp_vals = np.interp(np.arange(a, b), [left, right],
                                    [y[left], y[right]])
            prom = np.max(np.abs(y[a:b] - interp_vals))
        else:
            prom = np.max(np.abs(y[a:b] - np.median(y)))
        if prom > prominence * sigma:
            flag[a:b] = True

    y_out = y.copy()
    good = ~flag
    if flag.any() and good.sum() >= 2:
        xi = np.arange(n)
        y_out[flag] = np.interp(xi[flag], xi[good], y[good])
    return y_out, flag


def apply_despike(Y, threshold=7.0, max_width=3, prominence=6.0):
    """Despike every replicate independently (cosmic rays are per-acquisition).

    Returns (Y_clean, flags, counts) where flags is a 2-D boolean mask and
    counts is the number of points removed per replicate.
    """
    Y_out = np.empty_like(Y)
    flags = np.zeros(Y.shape, dtype=bool)
    counts = []
    for j in range(Y.shape[1]):
        yj, fj = despike_signal(Y[:, j], threshold, max_width, prominence)
        Y_out[:, j] = yj
        flags[:, j] = fj
        counts.append(int(fj.sum()))
    return Y_out, flags, counts


def plot_despike(x, Y_raw, flags, Y_desp, names, threshold, max_width,
                 save_path=None, show=True):
    """Two panels: raw spectra with removed spikes marked, and the result."""
    k = Y_raw.shape[1]
    multi = k > 1
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    ax_raw = fig.add_subplot(gs[0])
    ax_ds = fig.add_subplot(gs[1], sharex=ax_raw)

    xt = x if x is not None else np.arange(Y_raw.shape[0])
    xlabel = "x (wavelength / Raman shift)" if x is not None else "sample index"

    if multi and _HAVE_SEABORN:
        colors = sns.color_palette("husl", k)
    elif multi:
        colors = [f"C{j}" for j in range(k)]
    else:
        colors = [C_NOISY]

    for j in range(k):
        c = colors[j] if multi else C_NOISY
        lbl = names[j] if names else f"col {j}"
        ax_raw.plot(xt, Y_raw[:, j], lw=0.8, alpha=0.7, color=c, label=lbl)
        fj = flags[:, j]
        if fj.any():
            ax_raw.plot(np.asarray(xt)[fj], Y_raw[fj, j], "x", ms=9,
                        mew=2, color=C_KEPT)
        ax_ds.plot(xt, Y_desp[:, j], lw=0.9,
                   color=colors[j] if multi else C_CLEAN, label=lbl)

    ax_raw.set_title(f"Detected cosmic-ray spikes  "
                     f"(threshold={threshold:g}, max_width={max_width})   "
                     f"x = removed", loc="left", fontweight="bold")
    ax_raw.set_ylabel("intensity")
    if multi:
        ax_raw.legend(loc="best", fontsize=8, ncol=2)
    ax_ds.set_title("After spike removal  (spikes interpolated over)",
                    loc="left", fontweight="bold")
    ax_ds.set_xlabel(xlabel)
    ax_ds.set_ylabel("intensity")

    if _HAVE_SEABORN:
        sns.despine(fig=fig)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  saved despike plot -> {save_path}")
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Preview / confirm / output naming
# --------------------------------------------------------------------------- #
def print_preview(x, Y_clean, names, x_name, n=5):
    """Print the first few rows of the cleaned data as a quick sanity check."""
    cols = ([x_name or "x"] if x is not None else []) + list(names)
    w = max(13, max(len(c) for c in cols) + 2)
    nshow = min(n, Y_clean.shape[0])
    print(f"  Preview of cleaned data (first {nshow} of "
          f"{Y_clean.shape[0]} rows):")
    print("  " + "".join(f"{c:>{w}}" for c in cols))
    for i in range(nshow):
        vals = ([x[i]] if x is not None else []) + list(Y_clean[i, :])
        print("  " + "".join(f"{v:>{w}.5g}" for v in vals))


def confirm(question, assume_yes=False):
    """Ask a yes/no question on the command line. Returns True for yes.

    With assume_yes (the --yes flag) or when stdin isn't interactive (e.g. the
    script is piped), it proceeds without blocking.
    """
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(f"  {question} -> non-interactive session, proceeding with yes.")
        return True
    try:
        ans = input(f"  {question} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def prompt_new_value(name, current, cast=float):
    """Ask for a new numeric value, blank keeps the current one."""
    try:
        raw = input(f"  New {name} (current {current:g}, blank = keep): ").strip()
    except EOFError:
        return current
    if raw == "":
        return current
    try:
        return cast(float(raw)) if cast is int else cast(raw)
    except ValueError:
        print("  not a number; keeping the current value.")
        return current


def default_out_path(csv_path, out_dir=None):
    """Build '<input-stem>_Clean.csv', in out_dir if given else beside input."""
    stem = Path(csv_path).stem if csv_path else "denoise_demo"
    folder = Path(out_dir) if out_dir else (
        Path(csv_path).parent if csv_path else Path("."))
    return folder / f"{stem}_Clean.csv"


def resolve_png(png_dir, plot_path):
    """Place a plot file inside png_dir unless an absolute path was given."""
    p = Path(plot_path)
    return p if p.is_absolute() else Path(png_dir) / p


# --------------------------------------------------------------------------- #
#  Peak detection & integration
# --------------------------------------------------------------------------- #
# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def estimate_noise(y):
    """Robust noise standard deviation from the median abs. deviation of the
    first difference (insensitive to peaks and spikes)."""
    d = np.diff(np.asarray(y, dtype=float))
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2)) or 1e-12


def detect_peaks(x, y, sensitivity=8.0, min_dist_cm=8.0,
                 min_width_cm=3.0, min_rel_height=0.03, max_peaks=40):
    """Find peaks in a (cleaned) spectrum and propose integration windows.

    sensitivity    peak prominence required, in units of the noise sigma.
    min_dist_cm    minimum separation between peaks, in x units.
    min_width_cm   minimum peak width, in x units.
    min_rel_height keep only peaks at least this fraction as prominent as the
                   tallest peak (0-1). Suppresses noise / FFT ripple sitting in
                   the shadow of a dominant band. Set 0 to keep everything.

    Returns (windows, sigma) where windows is a list of dicts {center, xl, xr}
    sorted by centre, with edges at each peak's feet.
    """
    from scipy.signal import find_peaks, peak_widths

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = estimate_noise(y)
    dx = float(np.median(np.diff(x)))
    if dx <= 0:
        dx = 1.0
    distance = max(1, int(round(min_dist_cm / dx)))
    width = max(1, int(round(min_width_cm / dx)))

    idx, props = find_peaks(y, prominence=sensitivity * sigma,
                            distance=distance, width=width)
    if idx.size == 0:
        return [], sigma

    proms = props["prominences"]
    if min_rel_height > 0 and proms.size:
        keep = proms >= min_rel_height * proms.max()
        idx, proms = idx[keep], proms[keep]
    if idx.size > max_peaks:
        order = np.argsort(proms)[::-1][:max_peaks]
        idx = np.sort(idx[order])

    _, _, left_ips, right_ips = peak_widths(y, idx, rel_height=0.9)
    grid = np.arange(x.size)
    windows = []
    for k, i in enumerate(idx):
        xl = float(np.interp(left_ips[k], grid, x))
        xr = float(np.interp(right_ips[k], grid, x))
        lo, hi = min(xl, xr), max(xl, xr)
        windows.append({"center": float(x[i]), "xl": lo, "xr": hi})
    windows.sort(key=lambda w: w["center"])
    return windows, sigma


def integrate_window(x, y, xl, xr, baseline="local_linear"):
    """Integrate one spectrum between xl and xr.

    baseline 'local_linear' subtracts the straight line joining the two window
    endpoints (removes residual offset/tilt); 'zero' integrates above zero.

    Returns (area, height, xs, ys, base) — area is the trapezoidal net area,
    height is the tallest point above the baseline, and xs/ys/base describe the
    window for plotting the shaded region.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo, hi = (xl, xr) if xl <= xr else (xr, xl)
    m = (x >= lo) & (x <= hi)
    if m.sum() < 2:
        return 0.0, 0.0, x[m], y[m], np.zeros(int(m.sum()))
    xs, ys = x[m], y[m]
    if baseline == "local_linear":
        base = np.interp(xs, [xs[0], xs[-1]], [ys[0], ys[-1]])
    else:
        base = np.zeros_like(xs)
    area = float(_trapz(ys - base, xs))
    height = float(np.max(ys - base))
    return area, height, xs, ys, base


def integrate_replicates(x, Y, windows, baseline="local_linear"):
    """Integrate every window in every replicate over the same x ranges.

    Returns a dict with 2-D arrays areas/heights (n_peaks x n_replicates) and
    per-peak summary stats (mean, sd, rsd %) across replicates.
    """
    Y = Y if Y.ndim == 2 else Y[:, None]
    n_peaks, k = len(windows), Y.shape[1]
    areas = np.zeros((n_peaks, k))
    heights = np.zeros((n_peaks, k))
    for pi, w in enumerate(windows):
        for j in range(k):
            a, h, *_ = integrate_window(x, Y[:, j], w["xl"], w["xr"], baseline)
            areas[pi, j] = a
            heights[pi, j] = h
    mean = areas.mean(axis=1) if n_peaks else np.array([])
    if k > 1 and n_peaks:
        sd = areas.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rsd = np.where(mean != 0, 100.0 * sd / np.abs(mean), np.nan)
    else:
        sd = np.zeros(n_peaks)
        rsd = np.full(n_peaks, np.nan)
    return {"areas": areas, "heights": heights,
            "mean": mean, "sd": sd, "rsd": rsd}


def plot_integration(x, Y, names, windows, baseline="local_linear",
                     title=None, save_path=None, show=True):
    """Publication-style figure: the cleaned spectra with each integration
    window shaded above its local baseline, peaks labelled by centre."""
    Y = Y if Y.ndim == 2 else Y[:, None]
    k = Y.shape[1]
    multi = k > 1
    fig, ax = plt.subplots(figsize=(11, 5.5))

    if multi and _HAVE_SEABORN:
        colors = sns.color_palette("husl", k)
    elif multi:
        colors = [f"C{j}" for j in range(k)]
    else:
        colors = [C_CLEAN]

    for j in range(k):
        ax.plot(x, Y[:, j], lw=1.0, alpha=0.9 if multi else 1.0,
                color=colors[j], label=names[j] if names else f"col {j}",
                zorder=3)

    # Reference spectrum for the shaded fills (mean across replicates).
    yref = Y.mean(axis=1)
    ymax = float(np.max(Y)) if Y.size else 1.0
    for w in windows:
        _, _, xs, _, _ = integrate_window(x, yref, w["xl"], w["xr"], baseline)
        if xs.size < 2:
            continue
        _, _, xs, ys_ref, base = integrate_window(
            x, yref, w["xl"], w["xr"], baseline)
        ax.fill_between(xs, base, ys_ref, color=C_KEPT, alpha=0.22, zorder=2)
        if baseline == "local_linear":
            ax.plot([xs[0], xs[-1]], [base[0], base[-1]], color="0.45",
                    lw=1.0, ls="--", zorder=2)
        ax.annotate(f"{w['center']:.0f}",
                    xy=(w["center"], ymax * 1.02),
                    ha="center", va="bottom", fontsize=9, rotation=0,
                    color="0.25")

    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("intensity (a.u.)")
    ax.set_title(title or "Integrated peaks", loc="left", fontweight="bold")
    ax.margins(x=0.01)
    ax.set_ylim(top=ymax * 1.10)
    if multi:
        ax.legend(loc="best", fontsize=8, ncol=2, frameon=True)
    if _HAVE_SEABORN:
        sns.despine(fig=fig)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Remove noise from 1-D data with an FFT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("csv", nargs="?", help="input CSV; omit to run a demo")
    p.add_argument("--x-col", help="name or index of the x/time column")
    p.add_argument("--y-col", help="name or index of the signal column")
    p.add_argument("--fs", type=float,
                   help="sampling frequency in Hz (overrides value inferred "
                        "from the x column)")
    p.add_argument("--method", default="psd_threshold",
                   choices=["psd_threshold", "lowpass", "notch"])
    p.add_argument("--threshold", default="auto",
                   help="psd_threshold: 'auto' or a numeric power level")
    p.add_argument("--n-sigma", type=float, default=4.0,
                   help="aggressiveness of the automatic threshold "
                        "(higher = removes more)")
    p.add_argument("--cutoff", type=float,
                   help="lowpass cutoff frequency in Hz")
    p.add_argument("--notch-freqs", type=float, nargs="+",
                   help="notch: centre frequencies to remove (Hz)")
    p.add_argument("--notch-width", type=float, default=1.0,
                   help="notch: half-width of each removed band (Hz)")
    p.add_argument("--baseline", default="none",
                   choices=["none"] + list(BASELINE_METHODS),
                   help="background-correction method applied (per replicate) "
                        "before the FFT")
    p.add_argument("--baseline-lam", type=float, default=1e5,
                   help="smoothness for arpls/airpls (larger = stiffer, "
                        "flatter baseline)")
    p.add_argument("--baseline-poly-order", type=int, default=5,
                   help="polynomial order for imodpoly")
    p.add_argument("--baseline-plot", default="baseline_result.png",
                   help="filename for the baseline diagnostic plot (saved "
                        "inside --png-dir)")
    p.add_argument("--out",
                   help="explicit output CSV path; if omitted, the result is "
                        "saved as <input-name>_Clean.csv")
    p.add_argument("--out-dir",
                   help="folder for the auto-named <input>_Clean.csv "
                        "(created if needed; default: next to the input file)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="skip the save confirmation prompt and save directly")
    p.add_argument("--resample", action="store_true",
                   help="interpolate onto an evenly-spaced x grid before the "
                        "FFT (use when the instrument samples non-uniformly)")
    p.add_argument("--resample-method", default="pchip",
                   choices=["pchip", "linear"],
                   help="interpolation used by --resample")
    p.add_argument("--resample-step", type=float,
                   help="uniform grid spacing for --resample "
                        "(default: median of the native spacing)")
    p.add_argument("--despike", action="store_true",
                   help="remove cosmic-ray / muon spikes (1-3 pixel outliers) "
                        "before baseline correction and the FFT")
    p.add_argument("--despike-threshold", type=float, default=7.0,
                   help="sensitivity for spike detection; lower removes more "
                        "(modified z-score of the point-to-point difference)")
    p.add_argument("--despike-max-width", type=int, default=3,
                   help="widest spike (in points) that will ever be removed; "
                        "guards genuine peaks (default 3)")
    p.add_argument("--despike-prominence", type=float, default=6.0,
                   help="a spike must rise this many robust-noise sigmas above "
                        "its neighbours (noise-rejection floor; default 6)")
    p.add_argument("--despike-plot", default="despike_result.png",
                   help="filename for the despike diagnostic plot (saved "
                        "inside --png-dir)")
    p.add_argument("--plot", default="denoise_result.png",
                   help="filename for the FFT diagnostic plot (saved inside "
                        "--png-dir)")
    p.add_argument("--png-dir", default="png_outputs",
                   help="folder for all PNG plots (created if needed)")
    p.add_argument("--no-show", action="store_true",
                   help="don't open an interactive window")
    args = p.parse_args(argv)

    threshold = args.threshold
    if threshold != "auto":
        threshold = float(threshold)

    baseline_plot_path = resolve_png(args.png_dir, args.baseline_plot)
    fft_plot_path = resolve_png(args.png_dir, args.plot)
    despike_plot_path = resolve_png(args.png_dir, args.despike_plot)

    if args.csv:
        print(f"Loading {args.csv} ...")
        x, Y, names, x_name = load_csv(args.csv, args.x_col, args.y_col)
        clean = None
        print(f"  {Y.shape[1]} replicate column(s): {', '.join(names)}")
        if args.despike:
            thr = args.despike_threshold
            mw = args.despike_max_width
            prom = args.despike_prominence
            while True:
                Y_ds, flags, counts = apply_despike(Y, thr, mw, prom)
                total = int(sum(counts))
                print(f"  despike: removed {total} point(s) across "
                      f"{Y.shape[1]} replicate(s) {counts} "
                      f"(threshold={thr:g}, max_width={mw}).")
                plot_despike(x, Y, flags, Y_ds, names, thr, mw,
                             save_path=despike_plot_path,
                             show=not args.no_show)
                print(f"  despike plot: {despike_plot_path}  "
                      f"(x = removed and interpolated over)")
                if args.yes or not sys.stdin.isatty():
                    print("  spike removal accepted automatically "
                          "(non-interactive / --yes).")
                    Y = Y_ds
                    break
                if confirm("Accept this spike removal? "
                           "(n = adjust the threshold)"):
                    Y = Y_ds
                    break
                thr = prompt_new_value(
                    "despike threshold (lower removes more)", thr, float)
        if args.resample:
            if x is None:
                print("  [warning] --resample needs an x column; skipping "
                      "(nothing to resample against).")
            else:
                x, Y, step = resample_uniform(
                    x, Y, method=args.resample_method,
                    step=args.resample_step)
                print(f"  resampled onto a uniform grid: step = {step:.6g}, "
                      f"{Y.shape[0]} points ({args.resample_method}).")
        fs = args.fs if args.fs is not None else estimate_fs(x)
    else:
        print("No CSV given -> running synthetic demo "
              "(7 Hz + 23 Hz sinusoids under Gaussian noise).")
        x, y, clean, fs = make_demo()
        Y = y[:, None]
        names = ["signal"]
        x_name = "time"
        fs = args.fs if args.fs is not None else fs

    if fs is not None:
        print(f"  sampling frequency: {fs:.4g} Hz, N = {Y.shape[0]} samples")
    else:
        print(f"  no sampling rate available; frequency axis is in "
              f"cycles/sample. N = {Y.shape[0]} samples")

    # ----- Step 1: background correction, with a refine-until-happy loop ----
    if args.baseline != "none":
        try:
            import pybaselines  # noqa: F401  (checked here for a clean message)
        except ImportError:
            print("  [error] --baseline needs the pybaselines package. "
                  "Install it with:  pip install pybaselines")
            return 1

        lam = args.baseline_lam
        poly_order = args.baseline_poly_order
        print(f"  baseline method: {args.baseline}")
        while True:
            plabel = baseline_param_label(args.baseline, lam, poly_order)
            print(f"  fitting baseline ({plabel}) ...")
            Y_bc, B = apply_baseline(x, Y, args.baseline,
                                     lam=lam, poly_order=poly_order)
            plot_baseline(x, Y, B, Y_bc, names, args.baseline, plabel,
                          save_path=baseline_plot_path, show=not args.no_show)
            print_preview(x, Y_bc, names, x_name)
            print(f"  baseline plot: {baseline_plot_path}  (dashed line should "
                  f"thread the background, under the peaks)")

            if args.yes or not sys.stdin.isatty():
                print("  baseline accepted automatically "
                      "(non-interactive / --yes).")
                break
            if confirm("Accept this baseline? (n = adjust the parameter)"):
                break
            if args.baseline in ("arpls", "airpls"):
                lam = prompt_new_value("lam (larger = smoother/flatter)",
                                       lam, float)
            else:
                poly_order = prompt_new_value("polynomial order",
                                              poly_order, int)
        Y = Y_bc  # the FFT now runs on the baseline-corrected data

    print(f"  FFT method: {args.method}")

    # ----- Step 2: FFT denoise each replicate (on corrected data) -----------
    Y_clean = np.empty_like(Y)
    first_info = None
    for j in range(Y.shape[1]):
        y_clean, info = denoise(
            Y[:, j], fs, method=args.method,
            threshold=threshold, n_sigma=args.n_sigma,
            cutoff=args.cutoff,
            notch_freqs=args.notch_freqs, notch_width=args.notch_width)
        Y_clean[:, j] = y_clean
        if first_info is None:
            first_info = info
        tag = names[j] if names else f"col {j}"
        print(f"    [{tag}] kept {info['n_kept']}/{info['n_total']} components "
              f"({100 * info['power_kept_frac']:.2f}% power retained)")

    if clean is not None:
        rmse_before = np.sqrt(np.mean((Y[:, 0] - clean) ** 2))
        rmse_after = np.sqrt(np.mean((Y_clean[:, 0] - clean) ** 2))
        print(f"  [demo check] RMSE vs truth: "
              f"{rmse_before:.3f} -> {rmse_after:.3f}")

    plot_results(x, Y, Y_clean, names, first_info,
                 save_path=fft_plot_path, show=not args.no_show)

    # --- Preview, then ask before writing the file ---
    print()
    print_preview(x, Y_clean, names, x_name)
    print(f"  Diagnostic plot: {fft_plot_path}  "
          f"(open it to confirm the cut sits past your peaks and before the "
          f"noise floor)")

    out_path = Path(args.out) if args.out else default_out_path(
        args.csv, args.out_dir)

    if confirm(f"Cutoff looks good? Save cleaned data to '{out_path}'?",
               assume_yes=args.yes):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if x is not None:
            header = ",".join([x_name or "x"] + list(names))
            out_arr = np.column_stack([x, Y_clean])
        else:
            header = ",".join(names)
            out_arr = Y_clean
        np.savetxt(out_path, out_arr, delimiter=",", header=header,
                   comments="", fmt="%.8g")
        print(f"  saved denoised data -> {out_path} "
              f"({Y_clean.shape[1]} replicate column(s))")
    else:
        print("  not saved. Re-run with a different --cutoff to adjust the "
              "filter, then confirm.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
