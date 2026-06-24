"""
Streamlit web interface for the SERS / Raman spectra toolkit.

Two tabs share one cleaned spectrum:
  1. Clean & preprocess  -- despike, resample, baseline, FFT denoise.
  2. Peak integration    -- auto-detect peaks, adjust integration windows by
                            dragging a box on the plot or editing the table,
                            and export areas + reproducibility statistics.

All of the science lives in fft_denoise.py; this file is the UI only.
Run locally with:   streamlit run streamlit_app.py
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")   # headless rendering before fd import

import base64
import datetime as _dt
import io
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# streamlit-drawable-canvas calls streamlit.elements.image.image_to_url, whose
# location and signature changed in recent Streamlit, which breaks the canvas
# background image. Replace it with a small self-contained version that returns
# a base64 data URI, so the component keeps working across Streamlit versions.
import streamlit.elements.image as _st_image


def _image_to_url(image, *args, **kwargs):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


_st_image.image_to_url = _image_to_url

from streamlit_drawable_canvas import st_canvas   # noqa: E402

import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.ticker import MaxNLocator          # noqa: E402

import fft_denoise as fd                           # noqa: E402

BAND = "rgba(55,138,221,0.13)"        # integration band fill (works light/dark)
BAND_LINE = "rgba(55,138,221,0.55)"


# --------------------------------------------------------------------------- #
#  Cleaning pipeline (no Streamlit calls -> independently testable)
# --------------------------------------------------------------------------- #
def process(x, Y, names, p, plot_dir):
    """Run the enabled cleaning stages. Returns
    (x_work, Y_final, names, plots, messages)."""
    messages, plots = [], {}
    x_work, Y_cur = x, Y

    if p["despike"]:
        Y_ds, flags, counts = fd.apply_despike(
            Y_cur, p["d_threshold"], p["d_maxwidth"], p["d_prominence"])
        path = os.path.join(plot_dir, "despike.png")
        fd.plot_despike(x_work, Y_cur, flags, Y_ds, names,
                        p["d_threshold"], p["d_maxwidth"],
                        save_path=path, show=False)
        per_col = ", ".join(f"{n}: {c}" for n, c in zip(names, counts))
        plots["1 · Despike"] = (path, f"Removed {int(sum(counts))} spike "
                                      f"point(s) — {per_col}.")
        Y_cur = Y_ds

    if p["resample"] and x_work is not None:
        step = p["r_step"] if (p["r_step"] and p["r_step"] > 0) else None
        x_work, Y_cur, step_used = fd.resample_uniform(
            x_work, Y_cur, method=p["r_method"], step=step)
        messages.append(("info", f"Resampled onto a uniform grid "
                                 f"(spacing = {step_used:.4g}, "
                                 f"{len(x_work)} points)."))

    fs = None
    if x_work is not None:
        dx = np.diff(x_work)
        if np.all(dx > 0):
            fs = 1.0 / np.mean(dx)
            if np.std(dx) / np.mean(dx) > 1e-3 and not p["resample"]:
                messages.append(("warning", "The x-axis spacing is "
                                 "**non-uniform**; turn on **Resample** for a "
                                 "reliable FFT."))
        else:
            messages.append(("warning", "The x-axis is not strictly "
                             "increasing — turn on **Resample**."))

    if p["baseline"]:
        Y_corr, B = fd.apply_baseline(
            x_work, Y_cur, p["b_method"], lam=p["b_lam"],
            poly_order=p["b_poly"])
        label = fd.baseline_param_label(p["b_method"], p["b_lam"], p["b_poly"])
        path = os.path.join(plot_dir, "baseline.png")
        fd.plot_baseline(x_work, Y_cur, B, Y_corr, names, p["b_method"],
                         label, save_path=path, show=False)
        plots["3 · Baseline"] = (path, f"Method: {p['b_method']} ({label}). "
                                       f"The dashed line should thread the "
                                       f"background, under the peaks.")
        Y_cur = Y_corr

    if p["fft"]:
        notch = p["f_notch_freqs"] if p["f_method"] == "notch" else None
        Y_clean = np.empty_like(Y_cur)
        first_info = None
        for j in range(Y_cur.shape[1]):
            yj, info = fd.denoise(
                Y_cur[:, j], fs, method=p["f_method"], threshold="auto",
                n_sigma=p["f_nsigma"],
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
                  f"frequency components (replicate 1).")
        Y_cur = Y_clean

    return x_work, Y_cur, names, plots, messages


# --------------------------------------------------------------------------- #
#  Integration helpers
# --------------------------------------------------------------------------- #
def build_combined_csv(x_col, x, Y, names, peaks, result, baseline_mode, params):
    """One CSV holding processing metadata, the integration table, and the full
    cleaned spectra, in clearly-marked sections."""
    lines = []
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append("# SERS / Raman processed export")
    lines.append(f"# generated: {now}")
    steps = []
    if params["despike"]:
        steps.append(f"despike(threshold={params['d_threshold']:g},"
                     f"max_width={params['d_maxwidth']},"
                     f"prominence={params['d_prominence']:g})")
    if params["resample"]:
        steps.append(f"resample({params['r_method']})")
    if params["baseline"]:
        steps.append(f"baseline({params['b_method']},lam={params['b_lam']:g})")
    if params["fft"]:
        steps.append(f"fft({params['f_method']})")
    lines.append("# processing: " + (", ".join(steps) if steps else "none"))
    lines.append(f"# integration baseline: {baseline_mode}")
    lines.append("")

    lines.append("# === PEAK INTEGRATION (areas in intensity*cm^-1) ===")
    if peaks:
        hdr = ["peak_cm-1", "range_start", "range_end"]
        hdr += [f"area_{n}" for n in names]
        hdr += ["area_mean", "area_sd", "area_rsd_pct"]
        hdr += [f"height_{n}" for n in names]
        lines.append(",".join(hdr))
        for i, w in enumerate(peaks):
            row = [f"{w['center']:.2f}", f"{w['xl']:.2f}", f"{w['xr']:.2f}"]
            row += [f"{a:.4f}" for a in result["areas"][i]]
            row += [f"{result['mean'][i]:.4f}", f"{result['sd'][i]:.4f}"]
            rsd = result["rsd"][i]
            row += ["" if not np.isfinite(rsd) else f"{rsd:.2f}"]
            row += [f"{h:.4f}" for h in result["heights"][i]]
            lines.append(",".join(row))
    else:
        lines.append("# (no peaks defined)")
    lines.append("")

    lines.append("# === CLEANED SPECTRA ===")
    spec = pd.DataFrame({x_col: x})
    for j, n in enumerate(names):
        spec[n] = Y[:, j]
    lines.append(spec.to_csv(index=False).rstrip("\n"))
    return ("\n".join(lines) + "\n").encode("utf-8")


def make_plotly(x, Y, names, peaks):
    fig = go.Figure()
    for j, n in enumerate(names):
        fig.add_trace(go.Scatter(x=x, y=Y[:, j], mode="lines", name=n,
                                 line=dict(width=1.3)))
    ymax = float(np.max(Y)) if Y.size else 1.0
    for w in peaks:
        fig.add_vrect(x0=w["xl"], x1=w["xr"], fillcolor=BAND, opacity=1.0,
                      line_width=1, line_color=BAND_LINE, layer="below")
        fig.add_annotation(x=w["center"], y=ymax, text=f"{w['center']:.0f}",
                           showarrow=False, yshift=10, font=dict(size=11))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10), height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Raman shift (cm⁻¹)", yaxis_title="intensity (a.u.)")
    return fig


# ---- drawable-canvas helpers --------------------------------------------- #
CANVAS_W, CANVAS_H = 760, 380


def _data_to_px(v, lo, hi, size):
    return 0.0 if hi <= lo else (float(v) - lo) / (hi - lo) * size


def _px_to_data(px, lo, hi, size):
    return lo if size <= 0 else lo + (float(px) / size) * (hi - lo)


def render_spectrum_png(x, Y, names, xmin, xmax, ymin, ymax):
    """Render the spectra to a CANVAS_W x CANVAS_H image whose plot area fills
    the whole frame, so canvas pixels map linearly to data coordinates. Used as
    the (static) background the editable boxes sit on."""
    dpi = 100
    fig = plt.figure(figsize=(CANVAS_W / dpi, CANVAS_H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    for g in MaxNLocator(8).tick_values(xmin, xmax):
        if xmin <= g <= xmax:
            ax.axvline(g, color="0.88", lw=0.8, zorder=0)
            ax.text(g, ymin + 0.015 * (ymax - ymin), f"{g:.0f}", fontsize=7,
                    color="0.55", ha="center", va="bottom", zorder=1)
    ax.axhline(0, color="0.82", lw=0.8, zorder=0)
    k = Y.shape[1]
    if k > 1 and getattr(fd, "_HAVE_SEABORN", False):
        colors = fd.sns.color_palette("husl", k)
    else:
        colors = [f"C{j}" for j in range(k)]
    for j in range(k):
        ax.plot(x, Y[:, j], lw=0.9, color=colors[j], zorder=2)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def canvas_rects(peaks, xmin, xmax):
    """Fabric.js JSON of one full-height, horizontally-editable rectangle per
    window (edges resize the x-range, body moves it; rotation/vertical locked)."""
    objects = []
    for w in peaks:
        left = _data_to_px(w["xl"], xmin, xmax, CANVAS_W)
        right = _data_to_px(w["xr"], xmin, xmax, CANVAS_W)
        objects.append({
            "type": "rect", "left": round(left, 2), "top": 0.0,
            "width": round(max(2.0, right - left), 2), "height": CANVAS_H,
            "fill": "rgba(55,138,221,0.16)", "stroke": "#185FA5",
            "strokeWidth": 1, "scaleX": 1, "scaleY": 1, "angle": 0,
            "selectable": True, "lockMovementY": True, "lockScalingY": True,
            "lockRotation": True, "hasRotatingPoint": False})
    return {"version": "4.4.0", "objects": objects}


def canvas_to_peaks(json_data, x, yref, xmin, xmax):
    """Read rectangle objects back into windows (handles Fabric resize via
    scaleX; recomputes each centre; ignores slivers)."""
    peaks = []
    for obj in (json_data or {}).get("objects", []):
        if obj.get("type") != "rect":
            continue
        left = float(obj.get("left", 0.0))
        width = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1) or 1)
        x0 = _px_to_data(left, xmin, xmax, CANVAS_W)
        x1 = _px_to_data(left + width, xmin, xmax, CANVAS_W)
        x0, x1 = max(xmin, min(x0, x1)), min(xmax, max(x0, x1))
        if x1 - x0 < (xmax - xmin) * 0.002:
            continue
        m = (x >= x0) & (x <= x1)
        center = float(x[m][np.argmax(yref[m])]) if m.any() else 0.5 * (x0 + x1)
        peaks.append({"center": center, "xl": x0, "xr": x1})
    peaks.sort(key=lambda w: w["center"])
    return peaks


# --------------------------------------------------------------------------- #
#  Integration tab
# --------------------------------------------------------------------------- #
def integration_tab(x, Y, names, x_col, clean_params):
    st.caption("Auto-detected peaks appear as boxes on the spectrum. **Click a "
               "box to select it, drag its side handles to resize, drag its "
               "body to move it, and use the toolbar's 🗑 to delete it.** "
               "Switch to *Add* to draw a new box.")

    c1, c2, c3, c4, c5 = st.columns([1.2, 1.1, 1.0, 1.2, 0.9])
    sens = c1.slider("Sensitivity (σ)", 3.0, 30.0, 20.0, 0.5,
                     help="Peak prominence required, in noise sigmas. "
                          "Higher = fewer peaks.")
    min_height = c2.slider("Min height (%)", 0, 50, 15,
                           help="Ignore peaks shorter than this % of the "
                                "tallest peak — suppresses noise/ripple.")
    min_dist = c3.number_input("Min gap (cm⁻¹)", 1.0, 300.0, 8.0, 1.0)
    baseline_mode = c4.selectbox(
        "Integration baseline", ["zero", "local_linear"],
        format_func=lambda v: "From zero" if v == "zero" else "Local linear")
    detect = c5.button("Auto-detect", use_container_width=True)

    yref = Y.mean(axis=1)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin = float(min(0.0, np.min(Y))) if Y.size else 0.0
    ymax = float(np.max(Y)) * 1.05 if Y.size else 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    if "peaks" not in st.session_state or detect:
        wins, _ = fd.detect_peaks(x, yref, sensitivity=sens,
                                  min_dist_cm=min_dist,
                                  min_rel_height=min_height / 100.0)
        st.session_state.peaks = wins
        st.session_state.canvas_ver = st.session_state.get("canvas_ver", 0) + 1

    peaks = st.session_state.peaks

    tool = st.radio("Tool", ["Edit boxes (move / resize / delete)",
                             "Add a new box"], horizontal=True,
                    label_visibility="collapsed")
    draw_mode = "transform" if tool.startswith("Edit") else "rect"

    canvas = st_canvas(
        background_image=render_spectrum_png(x, Y, names, xmin, xmax,
                                             ymin, ymax),
        fill_color="rgba(55,138,221,0.16)", stroke_color="#185FA5",
        stroke_width=1, height=CANVAS_H, width=CANVAS_W,
        drawing_mode=draw_mode, display_toolbar=True, update_streamlit=True,
        initial_drawing=canvas_rects(peaks, xmin, xmax),
        key=f"canvas_{st.session_state.canvas_ver}")
    st.caption("Only a box's left/right edges set the integration range "
               "(height is ignored). In **Edit** mode: click a box, then drag a "
               "side handle to resize or its body to move; the 🗑 in the toolbar "
               "beneath the plot deletes the selected box.")

    if canvas is not None and canvas.json_data is not None:
        st.session_state.peaks = canvas_to_peaks(
            canvas.json_data, x, yref, xmin, xmax)
        peaks = st.session_state.peaks

    # precise numeric entry (discrete -> re-seeds the canvas)
    if peaks:
        with st.expander("Type exact bounds for a window"):
            labels = [f"{w['center']:.0f} cm⁻¹  ({w['xl']:.0f}–{w['xr']:.0f})"
                      for w in peaks]
            q1, q2, q3, q4 = st.columns([2.4, 1, 1, 0.9])
            sel = q1.selectbox("Window", range(len(peaks)),
                               format_func=lambda i: labels[i], key="exact_sel")
            sel = min(sel, len(peaks) - 1)
            nxl = q2.number_input("Start", value=round(peaks[sel]["xl"], 1),
                                  step=1.0, key=f"exl_{sel}")
            nxr = q3.number_input("End", value=round(peaks[sel]["xr"], 1),
                                  step=1.0, key=f"exr_{sel}")
            q4.markdown("<div style='height:1.7em'></div>",
                        unsafe_allow_html=True)
            if q4.button("Apply", use_container_width=True) and nxr > nxl:
                peaks[sel]["xl"], peaks[sel]["xr"] = float(nxl), float(nxr)
                m = (x >= nxl) & (x <= nxr)
                if m.any():
                    peaks[sel]["center"] = float(x[m][np.argmax(yref[m])])
                st.session_state.peaks = peaks
                st.session_state.canvas_ver += 1
                st.rerun()

    # --- integrate ---
    result = fd.integrate_replicates(x, Y, peaks, baseline=baseline_mode)

    if peaks and Y.shape[1] > 1:
        valid = np.isfinite(result["rsd"])
        mean_rsd = float(np.nanmean(result["rsd"])) if valid.any() else 0.0
        m1, m2, _ = st.columns(3)
        m1.metric("Peaks", len(peaks))
        m2.metric("Mean %RSD (area)", f"{mean_rsd:.1f}%")
    else:
        st.metric("Peaks", len(peaks))
        if Y.shape[1] == 1:
            st.caption("Load multiple replicate columns to get %RSD.")

    # --- results table ---
    if peaks:
        st.markdown("**Peak areas**")
        tbl = pd.DataFrame({
            "Peak (cm⁻¹)": [round(w["center"], 1) for w in peaks],
            "Range": [f"{w['xl']:.0f}–{w['xr']:.0f}" for w in peaks]})
        for j, n in enumerate(names):
            tbl[f"{n}"] = np.round(result["areas"][:, j], 1)
        tbl["Mean"] = np.round(result["mean"], 1)
        tbl["SD"] = np.round(result["sd"], 1)
        tbl["%RSD"] = [("" if not np.isfinite(v) else round(v, 1))
                       for v in result["rsd"]]
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        # --- area ratio tool ---
        with st.expander("Area ratio between two peaks"):
            labels = [f"{w['center']:.0f} cm⁻¹" for w in peaks]
            r1, r2 = st.columns(2)
            num = r1.selectbox("Numerator", labels, index=0, key="ratio_num")
            den = r2.selectbox("Denominator", labels,
                               index=min(1, len(labels) - 1), key="ratio_den")
            i, j = labels.index(num), labels.index(den)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = result["areas"][i] / result["areas"][j]
            rdf = pd.DataFrame({"Replicate": list(names),
                                f"{num} / {den}": np.round(ratios, 3)})
            st.dataframe(rdf, use_container_width=True, hide_index=True)
            if Y.shape[1] > 1 and np.all(np.isfinite(ratios)):
                st.caption(f"Mean ratio {np.mean(ratios):.3f} ± "
                           f"{np.std(ratios, ddof=1):.3f}")
    else:
        st.info("No peaks yet — click **Auto-detect**, or switch to **Add a new "
                "box** and draw one on the spectrum.")

    # --- zoomable overview (separate from the editor) ---
    with st.expander("Zoomable overview (hover & zoom, every replicate)"):
        st.plotly_chart(make_plotly(x, Y, names, peaks),
                        use_container_width=True, key="overview")

    # --- exports ---
    st.divider()
    e1, e2 = st.columns(2)
    csv_bytes = build_combined_csv(x_col, x, Y, names, peaks, result,
                                   baseline_mode, clean_params)
    e1.download_button("⬇️ Cleaned data + integration (CSV)", data=csv_bytes,
                       file_name="spectra_processed.csv", mime="text/csv",
                       use_container_width=True)

    png_dir = tempfile.mkdtemp()
    png_path = os.path.join(png_dir, "integration.png")
    fd.plot_integration(x, Y, names, peaks, baseline=baseline_mode,
                        title="Integrated peaks", save_path=png_path,
                        show=False)
    with open(png_path, "rb") as fh:
        e2.download_button("⬇️ Annotated figure (PNG)", data=fh.read(),
                           file_name="spectra_integration.png",
                           mime="image/png", use_container_width=True)
    st.image(png_path, use_container_width=True)


# --------------------------------------------------------------------------- #
#  Sidebar controls
# --------------------------------------------------------------------------- #
def sidebar_controls():
    sb = st.sidebar
    sb.header("Data")
    up = sb.file_uploader("Spectra CSV", type=["csv", "txt"])
    sb.caption("First column = x-axis (wavenumber); every other column = a "
               "spectrum / replicate.")
    sb.header("Cleaning steps")

    with sb.expander("1 · Despike (cosmic rays)", expanded=True):
        despike = st.checkbox("Enable despike", value=True)
        d_threshold = st.slider("Sensitivity (lower removes more)",
                                1.0, 30.0, 7.0, 0.5)
        d_maxwidth = st.slider("Max spike width (points)", 1, 7, 3)
        d_prominence = st.slider("Noise floor (σ)", 0.0, 30.0, 6.0, 0.5)
    with sb.expander("2 · Resample (uniform grid)", expanded=True):
        resample = st.checkbox("Enable resample", value=True)
        r_method = st.selectbox("Interpolation", ["pchip", "linear"], 0)
        r_step = st.number_input("Spacing (0 = auto)", min_value=0.0,
                                 value=0.0, step=0.1)
    with sb.expander("3 · Baseline correction", expanded=True):
        baseline = st.checkbox("Enable baseline", value=True)
        b_method = st.selectbox("Method", ["arpls", "airpls", "imodpoly"], 0)
        b_lam = st.number_input("Smoothness λ", min_value=1.0, value=100000.0,
                                step=10000.0, format="%.0f")
        b_poly = st.slider("Polynomial order (imodpoly)", 1, 10, 5)
    with sb.expander("4 · FFT denoise", expanded=True):
        fft = st.checkbox("Enable FFT denoise", value=True)
        f_method = st.selectbox("Method",
                                ["lowpass", "psd_threshold", "notch"], 0)
        f_cutoff = st.number_input("Low-pass cutoff (cycles/point)",
                                   min_value=0.0, value=0.05, step=0.01,
                                   format="%.3f")
        f_nsigma = st.slider("n·σ (psd_threshold)", 1.0, 10.0, 4.0, 0.5)
        f_notch_raw = st.text_input("Notch frequencies (comma-separated)", "")
        f_notchwidth = st.number_input("Notch half-width", min_value=0.0,
                                       value=1.0, step=0.5)

    f_notch_freqs = []
    if f_notch_raw.strip():
        try:
            f_notch_freqs = [float(v) for v in f_notch_raw.split(",")
                             if v.strip()]
        except ValueError:
            sb.error("Notch frequencies must be comma-separated numbers.")

    params = dict(
        despike=despike, d_threshold=d_threshold, d_maxwidth=d_maxwidth,
        d_prominence=d_prominence, resample=resample, r_method=r_method,
        r_step=r_step, baseline=baseline, b_method=b_method, b_lam=b_lam,
        b_poly=b_poly, fft=fft, f_method=f_method, f_cutoff=f_cutoff,
        f_nsigma=f_nsigma, f_notch_freqs=f_notch_freqs,
        f_notchwidth=f_notchwidth)
    return up, params


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Spectra toolkit", page_icon="🔬",
                       layout="wide")
    st.title("🔬 SERS / Raman spectra toolkit")

    up, params = sidebar_controls()

    if up is None:
        st.info("⬅️ Upload a CSV in the sidebar to begin.")
        st.markdown("**Expected format**")
        st.code("Wavelength,S1,S2,S3,S4\n139.19,2368,1547,1492,1578\n"
                "141.24,2344,1549,1502,1569\n...", language="text")
        return

    # Reset peak state when a different file is loaded.
    if st.session_state.get("_file") != up.name:
        st.session_state._file = up.name
        for k in ("peaks", "canvas_ver"):
            st.session_state.pop(k, None)

    try:
        df = pd.read_csv(up)
    except Exception as e:                      # noqa: BLE001
        st.error(f"Could not read the CSV: {e}")
        return
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    if len(cols) < 2:
        st.error("The file needs at least two columns (x-axis + one spectrum).")
        return

    with st.expander("Column mapping (defaults are usually right)"):
        x_col = st.selectbox("X-axis column", cols, index=0)
        spec_cols = st.multiselect(
            "Spectrum columns", [c for c in cols if c != x_col],
            default=[c for c in cols if c != x_col])
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

    plot_dir = tempfile.mkdtemp()
    try:
        x_work, Y_clean, names, plots, messages = process(
            x, Y, names, params, plot_dir)
    except Exception as e:                      # noqa: BLE001
        st.error(f"Cleaning failed: {e}")
        return

    tab_clean, tab_integrate = st.tabs(
        ["Clean & preprocess", "Peak integration"])

    with tab_clean:
        for level, text in messages:
            getattr(st, level)(text)
        if not plots:
            st.warning("All cleaning steps are off — enable at least one, or "
                       "go straight to **Peak integration** on the raw data.")
        for stage, (path, caption) in plots.items():
            st.subheader(stage)
            st.image(path, use_container_width=True)
            st.caption(caption)
        st.subheader("Cleaned data")
        out = pd.DataFrame({x_col: x_work})
        for j, n in enumerate(names):
            out[n] = Y_clean[:, j]
        st.dataframe(out.head(15), use_container_width=True)

    with tab_integrate:
        integration_tab(x_work, Y_clean, names, x_col, params)


if __name__ == "__main__":
    main()
