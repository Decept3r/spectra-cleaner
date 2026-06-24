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

# This app uses a small custom Streamlit component (declared at the bottom of
# this block) that renders the spectrum as a real Plotly chart and lets the
# user edit integration windows as native, draggable Plotly rectangle shapes.
# That avoids streamlit-drawable-canvas's background-image mechanism entirely
# (it painted the spectrum onto the canvas via a one-shot drawImage that Fabric
# wiped on its next re-render, so the spectrum kept disappearing), and it needs
# no canvas-pixel<->wavenumber mapping: Plotly shapes live in data coords.
import shutil                                      # noqa: E402
import streamlit.components.v1 as components       # noqa: E402

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


# ---- Plotly editable-window component ------------------------------------ #
_EDITOR_HTML = '<!doctype html>\n<html>\n<head>\n<meta charset="utf-8"/>\n<script src="./plotly.min.js" onerror="(function(){var s=document.createElement(\'script\');s.src=\'https://cdn.plot.ly/plotly-2.35.2.min.js\';document.head.appendChild(s);})()"></script>\n<style>html,body{margin:0;padding:0;height:442px}#c{width:100%;height:430px}</style>\n</head>\n<body>\n<div id="c"></div>\n<script>\nfunction post(type, extra){ var m = Object.assign({isStreamlitMessage:true, type:type}, extra||{}); window.parent.postMessage(m, "*"); }\nvar Streamlit = {\n  ready:  function(){ post("streamlit:componentReady", {apiVersion:1}); },\n  height: function(h){ post("streamlit:setFrameHeight", {height:h}); },\n  value:  function(v){ post("streamlit:setComponentValue", {value:v, dataType:"json"}); }\n};\nvar GD=null, programmatic=false;\nvar PALETTE=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#17becf"];\n\nfunction currentShapes(){\n  var sh = (GD && GD.layout && GD.layout.shapes) || [];\n  return sh.map(function(s){ var a=Math.min(+s.x0,+s.x1), b=Math.max(+s.x0,+s.x1); return {x0:a, x1:b}; });\n}\nfunction wire(){\n  GD.on("plotly_relayout", function(ev){\n    if(programmatic) return;\n    var keys = Object.keys(ev||{});\n    var touched = keys.some(function(k){ return k.indexOf("shapes")>=0; });\n    if(!touched) return;                       // ignore zoom/autorange, react only to shape edits\n    Streamlit.value({shapes: currentShapes(), ts: Date.now()});\n  });\n}\nfunction build(args, theme){\n  args = args || {};\n  var x = args.x||[], ys = args.ys||[], names = args.names||[], wins = args.windows||[], mode = args.mode||"edit";\n  var dark = !!(theme && theme.base === "dark");\n  var bg = (theme && theme.backgroundColor) || (dark ? "#0e1117" : "#ffffff");\n  var fg = (theme && theme.textColor) || (dark ? "#fafafa" : "#262730");\n  var grid = dark ? "rgba(250,250,250,0.13)" : "rgba(0,0,0,0.09)";\n  var traces = ys.map(function(yy, j){\n    return {x:x, y:yy, type:"scatter", mode:"lines", name:(names[j]||("col "+j)),\n            line:{width:1.3, color:PALETTE[j % PALETTE.length]}, hoverinfo:"x+y+name"};\n  });\n  var shapes = wins.map(function(w){\n    return {type:"rect", xref:"x", yref:"paper", x0:w.x0, x1:w.x1, y0:0, y1:1,\n            fillcolor:"rgba(55,138,221,0.16)", line:{color:"rgba(55,138,221,0.7)", width:1},\n            layer:"above", editable:true};\n  });\n  var layout = {\n    margin:{l:58, r:14, t:10, b:42}, height:430, hovermode:"closest",\n    showlegend:true, legend:{orientation:"h", y:1.02, yanchor:"bottom", x:0, font:{color:fg}},\n    paper_bgcolor:bg, plot_bgcolor:bg, font:{color:fg},\n    xaxis:{title:{text:"Raman shift (cm⁻¹)"}, gridcolor:grid, zeroline:false, color:fg, fixedrange:true},\n    yaxis:{title:{text:"intensity (a.u.)"}, gridcolor:grid, zeroline:false, color:fg, fixedrange:true},\n    shapes:shapes,\n    dragmode: (mode === "add") ? "drawrect" : false,\n    newshape:{line:{color:"rgba(55,138,221,0.7)", width:1}, fillcolor:"rgba(55,138,221,0.16)", layer:"above"}\n  };\n  var config = {displaylogo:false, responsive:true, edits:{shapePosition:true},\n                modeBarButtonsToAdd:["drawrect","eraseshape"],\n                modeBarButtonsToRemove:["lasso2d","select2d","autoScale2d","zoom2d","zoomIn2d","zoomOut2d"]};\n  programmatic = true;\n  Plotly.react("c", traces, layout, config).then(function(){\n    if(!GD._wired){ wire(); GD._wired = true; }\n    setTimeout(function(){ programmatic = false; }, 80);\n    Streamlit.height(442);\n  });\n}\nwindow.addEventListener("message", function(e){\n  var d = e.data; if(!d || d.type !== "streamlit:render") return;\n  GD = document.getElementById("c");\n  build(d.args, d.theme);\n});\nStreamlit.ready();\n</script>\n</body>\n</html>\n'


@st.cache_resource(show_spinner=False)
def _editor_component():
    """Materialise (once) a tiny static frontend dir and declare the bidirectional
    Plotly editor component. plotly.min.js is copied from the installed plotly
    package so the component works offline; a CDN load is the fallback."""
    import plotly as _plotly
    base = os.path.join(tempfile.gettempdir(), "raman_window_editor_v1")
    os.makedirs(base, exist_ok=True)
    js_dst = os.path.join(base, "plotly.min.js")
    js_src = os.path.join(os.path.dirname(_plotly.__file__),
                          "package_data", "plotly.min.js")
    try:
        if os.path.isfile(js_src) and (
                not os.path.exists(js_dst)
                or os.path.getsize(js_dst) != os.path.getsize(js_src)):
            shutil.copyfile(js_src, js_dst)
    except OSError:
        pass
    with open(os.path.join(base, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(_EDITOR_HTML)
    return components.declare_component("raman_window_editor", path=base)


def _shapes_to_peaks(shapes, x, yref, xmin, xmax):
    """Turn the Plotly rectangle shapes returned by the editor (x0/x1 in cm^-1)
    back into integration windows; recompute each centre and ignore slivers --
    mirrors the old canvas_to_peaks contract so downstream code is unchanged."""
    peaks = []
    for s in shapes or []:
        try:
            x0, x1 = float(s.get("x0")), float(s.get("x1"))
        except (TypeError, ValueError):
            continue
        x0, x1 = min(x0, x1), max(x0, x1)
        x0, x1 = max(xmin, x0), min(xmax, x1)
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

    tool = st.radio("Tool", ["Edit windows (drag edge = resize, body = move)",
                             "Add a new window (draw on the plot)"],
                    horizontal=True, label_visibility="collapsed")
    mode = "edit" if tool.startswith("Edit") else "add"

    editor = _editor_component()
    ret = editor(
        x=x.astype(float).tolist(),
        ys=[Y[:, j].astype(float).tolist() for j in range(Y.shape[1])],
        names=list(names),
        windows=[{"x0": float(w["xl"]), "x1": float(w["xr"])} for w in peaks],
        mode=mode, seed=int(st.session_state.canvas_ver),
        key="raman_window_editor", default=None)
    st.caption("Drag a window\u2019s **edge** to resize or its **body** to move; "
               "edits update the table live. In **Add** mode, drag across the plot "
               "to draw a new window. Delete a window with the chart toolbar\u2019s "
               "**eraser** icon, or via the expander below. Only the left/right "
               "bounds set the integration range.")

    if isinstance(ret, dict) and ret.get("ts") and \
            ret.get("ts") != st.session_state.get("_editor_ts"):
        st.session_state._editor_ts = ret["ts"]
        prev_n = len(st.session_state.peaks)
        new_peaks = _shapes_to_peaks(ret.get("shapes", []), x, yref, xmin, xmax)
        st.session_state.peaks = new_peaks
        peaks = new_peaks
        # Re-seed the chart only when a window was added/removed, so it snaps to
        # clean full-height bands (a resize/move keeps the on-screen shape as-is).
        if len(new_peaks) != prev_n:
            st.session_state.canvas_ver += 1
            st.rerun()
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
            if st.button("\U0001F5D1 Delete this window", key=f"del_{sel}",
                         use_container_width=True):
                peaks.pop(sel)
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
