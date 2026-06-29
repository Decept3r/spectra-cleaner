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
import hashlib
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
    """One CSV: the full cleaned spectra first (a plain rectangular block that
    charts directly in Excel/Sheets), then processing metadata and the
    peak-integration table as a trailing, clearly-marked section."""
    # Cleaned spectra FIRST, header on row 1 -- identical layout to the Clean &
    # preprocess export, so spreadsheets can graph it straight away.
    spec = pd.DataFrame({x_col: x})
    for j, n in enumerate(names):
        spec[n] = Y[:, j]
    lines = [spec.to_csv(index=False).rstrip("\n"), ""]

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
    return ("\n".join(lines) + "\n").encode("utf-8")


# ---- Plotly editable-window component ------------------------------------ #
_EDITOR_HTML = "<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\"/>\n<script src=\"./plotly.min.js\" onerror=\"(function(){var s=document.createElement('script');s.src='https://cdn.plot.ly/plotly-2.35.2.min.js';document.head.appendChild(s);})()\"></script>\n<style>html,body{margin:0;padding:0;height:442px}#c{width:100%;height:430px}</style>\n</head>\n<body>\n<div id=\"c\"></div>\n<script>\nfunction post(t,e){var m=Object.assign({isStreamlitMessage:true,type:t},e||{});window.parent.postMessage(m,\"*\");}\nvar Streamlit={ready:function(){post(\"streamlit:componentReady\",{apiVersion:1});},height:function(h){post(\"streamlit:setFrameHeight\",{height:h});},value:function(v){post(\"streamlit:setComponentValue\",{value:v,dataType:\"json\"});}};\nvar GD=null,programmatic=false,lastSeed=null,lastKey=null,wired=false,clickWired=false;\nvar eraserOn=false,lastArgs=null,lastTheme=null;\nvar PALETTE=[\"#1f77b4\",\"#ff7f0e\",\"#2ca02c\",\"#d62728\",\"#9467bd\",\"#8c564b\",\"#e377c2\",\"#17becf\"];\nvar FILL=\"rgba(55,138,221,0.18)\",LINE=\"rgba(55,138,221,0.85)\";\nvar FILL_E=\"rgba(214,39,40,0.16)\",LINE_E=\"rgba(214,39,40,0.90)\";\nfunction band(x0,x1){return {type:\"rect\",xref:\"x\",yref:\"paper\",x0:x0,x1:x1,y0:0,y1:1,fillcolor:(eraserOn?FILL_E:FILL),line:{color:(eraserOn?LINE_E:LINE),width:1.5},layer:\"above\",editable:!eraserOn};}\nfunction winShapes(wins){return (wins||[]).map(function(w){return band(w.x0,w.x1);});}\nfunction rawShapes(){var sh=(GD&&GD.layout&&GD.layout.shapes)||[];return sh.map(function(s){return band(s.x0,s.x1);});}\nfunction shapeKey(){var sh=(GD&&GD.layout&&GD.layout.shapes)||[];return sh.map(function(s){return Math.round(Math.min(+s.x0,+s.x1))+\":\"+Math.round(Math.max(+s.x0,+s.x1));}).sort().join(\",\");}\nfunction sendShapes(){var sh=(GD&&GD.layout&&GD.layout.shapes)||[];var out=sh.map(function(s){var a=Math.min(+s.x0,+s.x1),b=Math.max(+s.x0,+s.x1);return {x0:a,x1:b};});lastKey=shapeKey();Streamlit.value({shapes:out,ts:Date.now()});}\nfunction wire(){GD.on(\"plotly_relayout\",function(){if(programmatic)return;if(shapeKey()!==lastKey){sendShapes();}});}\nfunction eraseAt(e){if(!eraserOn||!GD||!GD._fullLayout)return;if(GD._fullLayout.dragmode===\"drawrect\")return;var bb=GD.getBoundingClientRect(),xa=GD._fullLayout.xaxis,ya=GD._fullLayout.yaxis;if(!xa||!ya)return;var px=e.clientX-bb.left-xa._offset,py=e.clientY-bb.top-ya._offset;if(px<0||px>xa._length||py<0||py>ya._length)return;var xd=xa.p2d(px),sh=(GD.layout.shapes||[]);for(var i=0;i<sh.length;i++){var a=Math.min(+sh[i].x0,+sh[i].x1),b=Math.max(+sh[i].x0,+sh[i].x1);if(xd>=a&&xd<=b){var ns=sh.slice();ns.splice(i,1);programmatic=true;Plotly.relayout(GD,{shapes:ns}).then(function(){setTimeout(function(){programmatic=false;},80);sendShapes();});return;}}}\nfunction rerender(){if(lastArgs)build(lastArgs,lastTheme);}\nfunction build(args,theme){\n  if(typeof Plotly===\"undefined\"){setTimeout(function(){build(args,theme);},60);return;}\n  GD=document.getElementById(\"c\");args=args||{};lastArgs=args;lastTheme=theme;\n  var x=args.x||[],ys=args.ys||[],names=args.names||[],wins=args.windows||[],seed=args.seed;\n  var dark=!!(theme&&theme.base===\"dark\");\n  var bg=(theme&&theme.backgroundColor)||(dark?\"#0e1117\":\"#ffffff\");\n  var fg=(theme&&theme.textColor)||(dark?\"#fafafa\":\"#262730\");\n  var grid=dark?\"rgba(250,250,250,0.13)\":\"rgba(0,0,0,0.09)\";\n  var traces=ys.map(function(yy,j){return {x:x,y:yy,type:\"scatter\",mode:\"lines\",name:(names[j]||(\"col \"+j)),line:{width:1.3,color:PALETTE[j%PALETTE.length]},hoverinfo:\"x+y+name\"};});\n  var reseed=(seed!==lastSeed)||!wired;\n  var shapes=reseed?winShapes(wins):rawShapes();\n  var layout={margin:{l:58,r:14,t:10,b:42},height:430,hovermode:\"closest\",showlegend:true,legend:{orientation:\"h\",y:1.02,yanchor:\"bottom\",x:0,font:{color:fg}},paper_bgcolor:bg,plot_bgcolor:bg,font:{color:fg},xaxis:{title:{text:\"Raman shift (cm\u207b\u00b9)\"},gridcolor:grid,zeroline:false,color:fg,fixedrange:true},yaxis:{title:{text:\"intensity (a.u.)\"},gridcolor:grid,zeroline:false,color:fg,fixedrange:true},shapes:shapes,dragmode:false,newshape:{line:{color:LINE,width:1.5},fillcolor:FILL,layer:\"above\"}};\n  var eraserBtn={name:\"eraser\",title:\"Eraser - click a window to delete it\",icon:(Plotly.Icons&&Plotly.Icons.eraseshape)||Plotly.Icons.pencil,toggle:true,click:function(){eraserOn=!eraserOn;rerender();}};\n  var config={displaylogo:false,responsive:true,edits:{shapePosition:!eraserOn},modeBarButtonsToAdd:[\"drawrect\",eraserBtn],modeBarButtonsToRemove:[\"lasso2d\",\"select2d\",\"autoScale2d\",\"zoom2d\",\"zoomIn2d\",\"zoomOut2d\",\"pan2d\",\"eraseshape\"]};\n  programmatic=true;\n  Plotly.react(\"c\",traces,layout,config).then(function(){\n    if(!wired){wire();wired=true;}\n    if(!clickWired){GD.addEventListener(\"click\",eraseAt,true);clickWired=true;}\n    GD.style.cursor=eraserOn?\"crosshair\":\"\";\n    if(reseed){lastSeed=seed;lastKey=shapeKey();}\n    setTimeout(function(){programmatic=false;},120);\n    Streamlit.height(442);\n  });\n}\nwindow.addEventListener(\"message\",function(e){var d=e.data;if(!d||d.type!==\"streamlit:render\")return;build(d.args,d.theme);});\nStreamlit.ready();\n</script>\n</body>\n</html>\n"
_EDITOR_VER = hashlib.md5(_EDITOR_HTML.encode("utf-8")).hexdigest()[:8]


@st.cache_resource(show_spinner=False)
def _editor_component(ver):
    """Materialise (once) a tiny static frontend dir and declare the bidirectional
    Plotly editor component. plotly.min.js is copied from the installed plotly
    package so the component works offline; a CDN load is the fallback."""
    import plotly as _plotly
    base = os.path.join(tempfile.gettempdir(), "raman_window_editor_" + ver)
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
    return components.declare_component("raman_window_editor_" + ver, path=base)


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
def integration_tab(x, Y, names, x_col, clean_params, src_name):
    st.caption("Auto-detected windows appear as shaded bands. "
               "**Drag a band's edge to resize, or its body to move** — the "
               "table updates live. Use the chart toolbar to **draw** a new "
               "window (rectangle tool) or **erase** one (eraser tool, then "
               "click the window). Fine-tune exact bounds below.")

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

    mode = "edit"

    editor = _editor_component(_EDITOR_VER)
    ret = editor(
        x=x.astype(float).tolist(),
        ys=[Y[:, j].astype(float).tolist() for j in range(Y.shape[1])],
        names=list(names),
        windows=[{"x0": float(w["xl"]), "x1": float(w["xr"])} for w in peaks],
        mode=mode, seed=int(st.session_state.canvas_ver),
        key="raman_window_editor", default=None)

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
        st.info("No peaks yet. Click **Auto-detect**, or draw one on the "
                "spectrum with the rectangle tool in the chart toolbar.")


    # --- exports ---
    st.divider()
    e1, e2 = st.columns(2)
    stem = os.path.splitext(src_name)[0]
    csv_bytes = build_combined_csv(x_col, x, Y, names, peaks, result,
                                   baseline_mode, clean_params)
    e1.download_button("⬇️ Cleaned data + integration (CSV)", data=csv_bytes,
                       file_name=f"{stem}_processed_integrated.csv", mime="text/csv",
                       use_container_width=True)

    png_dir = tempfile.mkdtemp()
    png_path = os.path.join(png_dir, "integration.png")
    fd.plot_integration(x, Y, names, peaks, baseline=baseline_mode,
                        title="Integrated peaks", save_path=png_path,
                        show=False)
    with open(png_path, "rb") as fh:
        e2.download_button("⬇️ Annotated figure (PNG)", data=fh.read(),
                           file_name=f"{stem}_integration.png",
                           mime="image/png", use_container_width=True)
    st.image(png_path, use_container_width=True)


# --------------------------------------------------------------------------- #
#  UV-Vis analysis (real-time band around 850 nm)
# --------------------------------------------------------------------------- #
def _interp_x(x0, y0, x1, y1, level):
    """Linear-interpolate the x where y crosses `level` between two points."""
    if y1 == y0:
        return float(x0)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def _peak_stats(x, y):
    """For one trace over x return (peak_x, peak_y, fwhm, clipped). FWHM is the
    width at half-height above the in-region minimum; `clipped` is True when the
    trace never falls back to half-height inside the region."""
    y = np.asarray(y, dtype=float)
    if y.size == 0 or not np.isfinite(y).any():
        return float("nan"), float("nan"), float("nan"), False
    i = int(np.nanargmax(y))
    peak_x, peak_y = float(x[i]), float(y[i])
    floor = float(np.nanmin(y))
    if not np.isfinite(peak_y) or peak_y <= floor:
        return peak_x, peak_y, float("nan"), False
    half = floor + (peak_y - floor) / 2.0
    k = i
    while k > 0 and y[k] > half:
        k -= 1
    if y[k] > half:
        xl, cl = float(x[0]), True
    else:
        xl, cl = _interp_x(x[k], y[k], x[k + 1], y[k + 1], half), False
    k, n = i, y.size
    while k < n - 1 and y[k] > half:
        k += 1
    if y[k] > half:
        xr_, cr = float(x[-1]), True
    else:
        xr_, cr = _interp_x(x[k - 1], y[k - 1], x[k], y[k], half), False
    return peak_x, peak_y, abs(xr_ - xl), bool(cl or cr)


def _uvvis_figure(x, Y, names, ylabel):
    """Overlay every trace, coloured along a time gradient; Streamlit themes it."""
    cmap = plt.get_cmap("viridis")
    n = Y.shape[1]
    fig = go.Figure()
    for j, nm in enumerate(names):
        t = j / (n - 1) if n > 1 else 0.0
        r, g, b, _ = cmap(t)
        color = "#%02x%02x%02x" % (int(255 * r), int(255 * g), int(255 * b))
        fig.add_trace(go.Scatter(
            x=x, y=Y[:, j], mode="lines", name=str(nm),
            line=dict(width=1.4, color=color),
            hovertemplate="%{x:.0f} nm — %{y:.3f}<extra>" + str(nm) + "</extra>"))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=470,
        xaxis_title="wavelength (nm)", yaxis_title=ylabel,
        hovermode="closest",
        legend=dict(orientation="v", x=1.005, y=1, font=dict(size=10),
                    title_text=""))
    return fig


def uvvis_analysis(x, Y, names, x_col, src_name):
    """Focus on the broad band near 850 nm, then either normalise each trace's
    maximum to 1 (compare widths / shapes) or subtract a very broad background
    (preserving the broad peak)."""
    st.caption("Real-time UV-Vis — focus on the broad band near 850 nm, then "
               "either **normalize each trace's maximum to 1** (to compare peak "
               "widths / shapes) or **subtract a very broad background** "
               "(preserving the broad peak).")

    wmin, wmax = float(np.min(x)), float(np.max(x))
    if wmin <= 850.0 <= wmax:
        d_lo, d_hi = max(wmin, 700.0), min(wmax, 1000.0)
    else:
        d_lo, d_hi = wmin, wmax

    c1, c2 = st.columns([2.0, 1.3])
    lo, hi = c1.slider("Focus region (nm)", wmin, wmax, (d_lo, d_hi))
    op = c2.radio("Operation",
                  ["Normalize maxima to 1", "Broad background subtraction"])

    m = (x >= lo) & (x <= hi)
    if int(np.count_nonzero(m)) < 5:
        st.warning("The focus region is too narrow — widen it.")
        return
    xr = x[m]
    Yr = Y[m, :]

    if op == "Normalize maxima to 1":
        Yout = np.empty_like(Yr, dtype=float)
        for j in range(Yr.shape[1]):
            mx = float(np.nanmax(Yr[:, j]))
            Yout[:, j] = Yr[:, j] / mx if mx > 1e-9 else Yr[:, j]
        ylabel = "normalized absorbance (max = 1)"
        suffix = "uvvis_normalized"
        st.caption("Each trace divided by its own maximum inside the region — "
                   "every peak now reaches 1 so widths and shapes line up.")
    else:
        breadth = st.select_slider(
            "Background breadth", ["Broad", "Very broad", "Flattest"],
            value="Very broad",
            help="Larger = stiffer, flatter background that leaves the broad "
                 "peak intact.")
        lam = {"Broad": 1e6, "Very broad": 1e8, "Flattest": 1e10}[breadth]
        try:
            Yout, _bkg = fd.apply_baseline(xr, Yr, "arpls", lam=lam)
        except Exception as exc:                    # noqa: BLE001
            st.error(f"Background fit failed: {exc}")
            return
        ylabel = "background-subtracted absorbance"
        suffix = "uvvis_bgsub"
        st.caption(f"A very broad arpls background (λ = {lam:g}) subtracted "
                   "from every trace, leaving the broad peak.")

    st.plotly_chart(_uvvis_figure(xr, Yout, names, ylabel),
                    use_container_width=True, theme="streamlit")

    # peak summary: position, height and FWHM of each trace's band in the region
    stem = os.path.splitext(src_name)[0]
    rows = []
    for j, nm in enumerate(names):
        px, py, fw, clip = _peak_stats(xr, Yout[:, j])
        rows.append({
            "Series": str(nm),
            "Peak (nm)": round(px, 1) if np.isfinite(px) else None,
            "Peak value": round(py, 4) if np.isfinite(py) else None,
            "FWHM (nm)": (f"≥ {fw:.1f}" if clip else round(fw, 1))
                         if np.isfinite(fw) else None,
        })
    summary = pd.DataFrame(rows)
    st.markdown("**Peak summary**")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption("Peak = wavelength of each trace's maximum in the region; FWHM is "
               "measured at half-height above the in-region background. “≥” marks a "
               "peak that doesn't fall back to half-height inside the region — widen "
               "the focus region for its true width.")

    out = pd.DataFrame({x_col: xr})
    for j, n in enumerate(names):
        out[n] = Yout[:, j]
    d1, d2 = st.columns(2)
    d1.download_button(
        f"⬇️ {op} (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_{suffix}.csv",
        mime="text/csv", key="dl_uvvis", use_container_width=True)
    d2.download_button(
        "⬇️ Peak summary (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_uvvis_peaks.csv",
        mime="text/csv", key="dl_uvvis_peaks", use_container_width=True)


# --------------------------------------------------------------------------- #
#  Sidebar controls
# --------------------------------------------------------------------------- #
def sidebar_controls():
    sb = st.sidebar
    sb.header("Data")
    up = sb.file_uploader("Spectra CSV", type=["csv", "txt"])
    sb.caption("First column = x-axis; every other column = a spectrum, "
               "replicate, or time point.")
    analysis = sb.radio(
        "Analysis", ["SERS / Raman", "UV-Vis · real-time (~850 nm band)"],
        help="SERS / Raman: despike, baseline, FFT denoise, peak integration. "
             "UV-Vis: focus on the broad ~850 nm band, then normalize or "
             "broadly background-subtract.")
    if analysis.startswith("UV-Vis"):
        sb.caption("UV-Vis options appear on the main panel →")
        return up, analysis, None
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
    return up, analysis, params


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Spectra toolkit", page_icon="🔬",
                       layout="wide")
    st.title("🔬 SERS / Raman spectra toolkit")
    st.markdown("""<style>
    .block-container{padding-top:2rem;padding-bottom:3rem}
    h1{font-weight:700;letter-spacing:-0.01em}
    .stTabs [data-baseweb="tab-list"]{gap:1.4rem}
    .stTabs [data-baseweb="tab"]{font-weight:600}
    .stDownloadButton button,.stButton button{border-radius:8px;font-weight:600}
    [data-testid="stMetric"]{background:rgba(128,128,128,0.06);border:1px solid rgba(128,128,128,0.14);border-radius:12px;padding:0.5rem 0.9rem}
    [data-testid="stExpander"] details{border-radius:10px}
    </style>""", unsafe_allow_html=True)

    up, analysis, params = sidebar_controls()

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

    if analysis.startswith("UV-Vis"):
        uvvis_analysis(x, Y, names, x_col, up.name)
        return

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
        st.caption(
            f"Showing the first 15 of {len(out):,} rows — download the "
            "full cleaned dataset (every row, all spectra) below.")
        st.download_button(
            "⬇️ Cleaned data (CSV — all rows)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name=f"{os.path.splitext(up.name)[0]}_cleaned.csv",
            mime="text/csv", key="dl_cleaned_only")

    with tab_integrate:
        integration_tab(x_work, Y_clean, names, x_col, params, up.name)


if __name__ == "__main__":
    main()
