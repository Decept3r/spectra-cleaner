"""
Streamlit web interface for the SERS / Raman spectra cleaner.

This is a thin UI layer on top of fft_denoise.py -- all of the science
(cosmic-ray despiking, resampling, baseline correction, FFT denoising) lives in
that module and is imported here unchanged. The command-line tool's interactive
y/n checkpoints become live sidebar controls: adjust a slider and the diagnostic
plots update immediately.

Run locally with:   streamlit run streamlit_app.py
"""

# Force matplotlib's non-interactive backend BEFORE importing fft_denoise, so it
# never tries to open a GUI window on a headless server (Streamlit Cloud). Our
# edit to fft_denoise._ensure_working_backend honours this.
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import tempfile

import numpy as np
import pandas as pd
import streamlit as st

import fft_denoise as fd


# --------------------------------------------------------------------------- #
#  Processing (no Streamlit calls in here, so it can be tested on its own)
# --------------------------------------------------------------------------- #
def process(x, Y, names, p, plot_dir):
    """Run the enabled pipeline stages in order.

    Returns (x_work, Y_final, names, plots, messages) where `plots` maps a stage
    label to (png_path, caption) and `messages` is a list of (level, text).
    """
    messages = []
    plots = {}
    x_work = x
    Y_cur = Y

    # 1. Despike -- must run first, on the native grid, before any interpolation.
    if p["despike"]:
        Y_ds, flags, counts = fd.apply_despike(
            Y_cur, p["d_threshold"], p["d_maxwidth"], p["d_prominence"])
        path = os.path.join(plot_dir, "despike.png")
        fd.plot_despike(x_work, Y_cur, flags, Y_ds, names,
                        p["d_threshold"], p["d_maxwidth"],
                        save_path=path, show=False)
        per_col = ", ".join(f"{n}: {c}" for n, c in zip(names, counts))
        plots["1 · Despike"] = (
            path, f"Removed {int(sum(counts))} spike point(s) — {per_col}.")
        Y_cur = Y_ds

    # 2. Resample onto a uniform grid (needed for a valid FFT).
    if p["resample"] and x_work is not None:
        step = p["r_step"] if (p["r_step"] and p["r_step"] > 0) else None
        x_work, Y_cur, step_used = fd.resample_uniform(
            x_work, Y_cur, method=p["r_method"], step=step)
        messages.append(
            ("info", f"Resampled onto a uniform grid "
                     f"(spacing = {step_used:.4g}, {len(x_work)} points)."))

    # Sampling frequency + uniformity check for the FFT stage.
    fs = None
    if x_work is not None:
        dx = np.diff(x_work)
        if np.all(dx > 0):
            fs = 1.0 / np.mean(dx)
            spread = np.std(dx) / np.mean(dx)
            if spread > 1e-3 and not p["resample"]:
                messages.append(
                    ("warning", "The x-axis spacing is **non-uniform**, but the "
                     "FFT assumes uniform spacing. Turn on **Resample** "
                     "(left) for reliable denoising."))
        else:
            messages.append(
                ("warning", "The x-axis is not strictly increasing — turn on "
                 "**Resample**, which sorts and regularises it."))

    # 3. Baseline correction.
    if p["baseline"]:
        Y_corr, B = fd.apply_baseline(
            x_work, Y_cur, p["b_method"],
            lam=p["b_lam"], poly_order=p["b_poly"])
        label = fd.baseline_param_label(p["b_method"], p["b_lam"], p["b_poly"])
        path = os.path.join(plot_dir, "baseline.png")
        fd.plot_baseline(x_work, Y_cur, B, Y_corr, names,
                         p["b_method"], label, save_path=path, show=False)
        plots["3 · Baseline"] = (
            path, f"Method: {p['b_method']} ({label}). The dashed line should "
                  f"thread the background, under the peaks.")
        Y_cur = Y_corr

    # 4. FFT denoise (per replicate; the frequency mask is shared for lowpass/notch).
    if p["fft"]:
        notch = p["f_notch_freqs"] if p["f_method"] == "notch" else None
        Y_clean = np.empty_like(Y_cur)
        first_info = None
        for j in range(Y_cur.shape[1]):
            yj, info = fd.denoise(
                Y_cur[:, j], fs, method=p["f_method"],
                threshold="auto", n_sigma=p["f_nsigma"],
                cutoff=p["f_cutoff"] if p["f_method"] == "lowpass" else None,
                notch_freqs=notch, notch_width=p["f_notchwidth"])
            Y_clean[:, j] = yj
            if j == 0:
                first_info = info
        path = os.path.join(plot_dir, "denoise.png")
        fd.plot_results(x_work, Y_cur, Y_clean, names, first_info,
                        save_path=path, show=False)
        plots["4 · FFT denoise"] = (
            path, f"Kept {first_info['n_kept']}/{first_info['n_total']} "
                  f"frequency components (replicate 1). The bottom panel's "
                  f"x-axis is your wavenumber; 'frequency' is spectral "
                  f"frequency (cycles per point).")
        Y_cur = Y_clean

    return x_work, Y_cur, names, plots, messages


# --------------------------------------------------------------------------- #
#  User interface
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Spectra Cleaner", page_icon="🔬",
                       layout="wide")
    st.title("🔬 SERS / Raman spectra cleaner")
    st.markdown(
        "Upload a spectra CSV and clean it in four optional steps: **despike** "
        "(remove cosmic-ray / muon hits), **resample** onto a uniform grid, "
        "**baseline**-correct, and **FFT denoise**. Adjust the controls on the "
        "left and the diagnostic plots update live. Download the cleaned data "
        "at the bottom.")

    sb = st.sidebar
    sb.header("Data")
    up = sb.file_uploader("Spectra CSV", type=["csv", "txt"])
    sb.caption("First column = x-axis (wavenumber); every other column = a "
               "spectrum / replicate.")

    # ---- Stage controls (rendered up front; harmless when no file yet) ----
    sb.header("Steps")

    with sb.expander("1 · Despike (cosmic rays)", expanded=True):
        despike = st.checkbox("Enable despike", value=True)
        d_threshold = st.slider("Detection sensitivity (lower removes more)",
                                1.0, 30.0, 7.0, 0.5,
                                help="Modified z-score of the point-to-point "
                                     "difference. Lower catches subtler spikes.")
        d_maxwidth = st.slider("Max spike width (points)", 1, 7, 3,
                               help="Widest feature that can ever be removed — "
                                    "guards genuine peaks.")
        d_prominence = st.slider("Noise-rejection floor (σ)", 0.0, 30.0, 6.0,
                                 0.5, help="A spike must rise this many noise "
                                           "sigmas above its neighbours.")

    with sb.expander("2 · Resample (uniform grid)", expanded=True):
        resample = st.checkbox("Enable resample", value=True,
                               help="Recommended when the instrument samples at "
                                    "slightly uneven wavenumber steps.")
        r_method = st.selectbox("Interpolation", ["pchip", "linear"], index=0,
                                help="pchip is smooth and won't overshoot peaks; "
                                     "linear is the most conservative.")
        r_step = st.number_input("Grid spacing (0 = auto / median spacing)",
                                 min_value=0.0, value=0.0, step=0.1)

    with sb.expander("3 · Baseline correction", expanded=True):
        baseline = st.checkbox("Enable baseline", value=True)
        b_method = st.selectbox("Method", ["arpls", "airpls", "imodpoly"],
                                index=0)
        b_lam = st.number_input("Smoothness λ (arpls / airpls)",
                                min_value=1.0, value=100000.0, step=10000.0,
                                format="%.0f")
        b_poly = st.slider("Polynomial order (imodpoly)", 1, 10, 5)

    with sb.expander("4 · FFT denoise", expanded=True):
        fft = st.checkbox("Enable FFT denoise", value=True)
        f_method = st.selectbox("Method",
                                ["lowpass", "psd_threshold", "notch"], index=0)
        f_cutoff = st.number_input("Low-pass cutoff (cycles per point)",
                                   min_value=0.0, value=0.05, step=0.01,
                                   format="%.3f",
                                   help="Keep frequencies below this. Check the "
                                        "spectrum plot: the cut should sit past "
                                        "your peaks and before the noise floor.")
        f_nsigma = st.slider("Threshold strictness n·σ (psd_threshold)",
                             1.0, 10.0, 4.0, 0.5)
        f_notch_raw = st.text_input("Notch frequencies (comma-separated)", "")
        f_notchwidth = st.number_input("Notch half-width", min_value=0.0,
                                       value=1.0, step=0.5)

    # Parse notch frequencies safely.
    f_notch_freqs = []
    if f_notch_raw.strip():
        try:
            f_notch_freqs = [float(v) for v in f_notch_raw.split(",") if v.strip()]
        except ValueError:
            sb.error("Notch frequencies must be numbers separated by commas.")

    if up is None:
        st.info("⬅️ Upload a CSV in the sidebar to begin.")
        st.markdown("**Expected format**")
        st.code("Wavelength,S1,S2,S3,S4\n139.19,2368,1547,1492,1578\n"
                "141.24,2344,1549,1502,1569\n...", language="text")
        return

    # ---- Load the file ----
    try:
        df = pd.read_csv(up)
    except Exception as e:                      # noqa: BLE001
        st.error(f"Could not read the CSV: {e}")
        return
    df.columns = [str(c).strip() for c in df.columns]
    all_cols = list(df.columns)
    if len(all_cols) < 2:
        st.error("The file needs at least two columns (x-axis + one spectrum).")
        return

    with st.expander("Column mapping (defaults are usually right)"):
        x_col = st.selectbox("X-axis column", all_cols, index=0)
        spec_cols = st.multiselect(
            "Spectrum columns", [c for c in all_cols if c != x_col],
            default=[c for c in all_cols if c != x_col])
    if not spec_cols:
        st.warning("Select at least one spectrum column.")
        return

    try:
        x = df[x_col].to_numpy(dtype=float)
        Y = df[spec_cols].to_numpy(dtype=float)
    except ValueError as e:
        st.error(f"Non-numeric data in the selected columns: {e}")
        return
    names = list(spec_cols)

    params = dict(
        despike=despike, d_threshold=d_threshold, d_maxwidth=d_maxwidth,
        d_prominence=d_prominence,
        resample=resample, r_method=r_method, r_step=r_step,
        baseline=baseline, b_method=b_method, b_lam=b_lam, b_poly=b_poly,
        fft=fft, f_method=f_method, f_cutoff=f_cutoff, f_nsigma=f_nsigma,
        f_notch_freqs=f_notch_freqs, f_notchwidth=f_notchwidth,
    )

    plot_dir = tempfile.mkdtemp()
    try:
        x_work, Y_final, names, plots, messages = process(
            x, Y, names, params, plot_dir)
    except Exception as e:                      # noqa: BLE001
        st.error(f"Processing failed: {e}")
        st.stop()

    for level, text in messages:
        getattr(st, level)(text)

    # ---- Diagnostic plots ----
    if not plots:
        st.warning("All steps are disabled — enable at least one on the left.")
    for stage, (path, caption) in plots.items():
        st.subheader(stage)
        st.image(path, use_container_width=True)
        st.caption(caption)

    # ---- Result preview + download ----
    st.subheader("Cleaned data")
    out = pd.DataFrame({x_col: x_work})
    for j, n in enumerate(names):
        out[n] = Y_final[:, j]
    st.dataframe(out.head(15), use_container_width=True)

    stem = os.path.splitext(up.name)[0]
    st.download_button(
        "⬇️ Download cleaned CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_Clean.csv",
        mime="text/csv")


if __name__ == "__main__":
    main()
