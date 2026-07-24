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
import re
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
import nanomeli_theme as theme                     # noqa: E402

theme.apply_matplotlib_theme()  # neon "instrument" look for the plots


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
_EDITOR_HTML = "<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\"/>\n<script src=\"./plotly.min.js\" onerror=\"(function(){var s=document.createElement('script');s.src='https://cdn.plot.ly/plotly-2.35.2.min.js';document.head.appendChild(s);})()\"></script>\n<style>html,body{margin:0;padding:0;height:442px}#c{width:100%;height:430px}</style>\n</head>\n<body>\n<div id=\"c\"></div>\n<script>\nfunction post(t,e){var m=Object.assign({isStreamlitMessage:true,type:t},e||{});window.parent.postMessage(m,\"*\");}\nvar Streamlit={ready:function(){post(\"streamlit:componentReady\",{apiVersion:1});},height:function(h){post(\"streamlit:setFrameHeight\",{height:h});},value:function(v){post(\"streamlit:setComponentValue\",{value:v,dataType:\"json\"});}};\nvar GD=null,programmatic=false,lastSeed=null,lastKey=null,wired=false,clickWired=false;\nvar eraserOn=false,lastArgs=null,lastTheme=null;\nvar MAGDIV=null,magWired=false,dragging=false,magPending=null,magRAF=false,lastXkey=null;\nvar curBg=\"#0e1117\",curFg=\"#fafafa\",curDark=true,curGrid=\"rgba(250,250,250,0.13)\";\nvar PALETTE=[\"#1f77b4\",\"#ff7f0e\",\"#2ca02c\",\"#d62728\",\"#9467bd\",\"#8c564b\",\"#e377c2\",\"#17becf\"];\nvar FILL=\"rgba(55,138,221,0.18)\",LINE=\"rgba(55,138,221,0.85)\";\nvar FILL_E=\"rgba(214,39,40,0.16)\",LINE_E=\"rgba(214,39,40,0.90)\";\nfunction band(x0,x1){return {type:\"rect\",xref:\"x\",yref:\"paper\",x0:x0,x1:x1,y0:0,y1:1,fillcolor:(eraserOn?FILL_E:FILL),line:{color:(eraserOn?LINE_E:LINE),width:1.5},layer:\"above\",editable:!eraserOn};}\nfunction winShapes(wins){return (wins||[]).map(function(w){return band(w.x0,w.x1);});}\nfunction rawShapes(){var sh=(GD&&GD.layout&&GD.layout.shapes)||[];return sh.map(function(s){return band(s.x0,s.x1);});}\nfunction shapeKey(){var sh=(GD&&GD.layout&&GD.layout.shapes)||[];return sh.map(function(s){return Math.round(Math.min(+s.x0,+s.x1))+\":\"+Math.round(Math.max(+s.x0,+s.x1));}).sort().join(\",\");}\nfunction sendShapes(){var sh=(GD&&GD.layout&&GD.layout.shapes)||[];var out=sh.map(function(s){var a=Math.min(+s.x0,+s.x1),b=Math.max(+s.x0,+s.x1);return {x0:a,x1:b};});lastKey=shapeKey();Streamlit.value({shapes:out,ts:Date.now()});}\nfunction wire(){GD.on(\"plotly_relayout\",function(){if(programmatic)return;if(shapeKey()!==lastKey){sendShapes();}});}\nfunction eraseAt(e){if(!eraserOn||!GD||!GD._fullLayout)return;if(GD._fullLayout.dragmode)return;var bb=GD.getBoundingClientRect(),xa=GD._fullLayout.xaxis,ya=GD._fullLayout.yaxis;if(!xa||!ya)return;var px=e.clientX-bb.left-xa._offset,py=e.clientY-bb.top-ya._offset;if(px<0||px>xa._length||py<0||py>ya._length)return;var xd=xa.p2d(px),sh=(GD.layout.shapes||[]);for(var i=0;i<sh.length;i++){var a=Math.min(+sh[i].x0,+sh[i].x1),b=Math.max(+sh[i].x0,+sh[i].x1);if(xd>=a&&xd<=b){var ns=sh.slice();ns.splice(i,1);programmatic=true;Plotly.relayout(GD,{shapes:ns}).then(function(){setTimeout(function(){programmatic=false;},80);sendShapes();});return;}}}\nfunction mouseDataX(e){var xa=GD._fullLayout.xaxis;var p=e.clientX-GD.getBoundingClientRect().left-xa._offset;var d=xa.p2d(p);if(d<xa.range[0])d=xa.range[0];if(d>xa.range[1])d=xa.range[1];return d;}\nfunction nearEdge(e){if(!GD||!GD._fullLayout)return null;var xa=GD._fullLayout.xaxis,ya=GD._fullLayout.yaxis,bb=GD.getBoundingClientRect();var px=e.clientX-bb.left-xa._offset,py=e.clientY-bb.top-ya._offset;if(px<0||px>xa._length||py<0||py>ya._length)return null;var md=xa.p2d(px),tol=Math.abs(xa.range[1]-xa.range[0])/xa._length*9;var sh=(GD.layout.shapes||[]),best=null,bd=tol;for(var i=0;i<sh.length;i++){var d0=Math.abs(md-(+sh[i].x0)),d1=Math.abs(md-(+sh[i].x1));if(d0<bd){bd=d0;best=+sh[i].x0;}if(d1<bd){bd=d1;best=+sh[i].x1;}}return best;}\nfunction ensureMag(){if(MAGDIV)return;MAGDIV=document.createElement(\"div\");MAGDIV.id=\"mag\";MAGDIV.style.cssText=\"position:absolute;top:8px;width:300px;height:178px;z-index:6;pointer-events:none;display:none;border-radius:8px;overflow:hidden;border:1px solid \"+(curDark?\"rgba(250,250,250,0.22)\":\"rgba(0,0,0,0.18)\")+\";box-shadow:0 6px 20px rgba(0,0,0,0.4)\";document.body.appendChild(MAGDIV);}\nfunction positionMag(cx){var xa=GD._fullLayout.xaxis,mid=(xa.range[0]+xa.range[1])/2;if(cx>mid){MAGDIV.style.left=\"10px\";MAGDIV.style.right=\"auto\";}else{MAGDIV.style.left=\"auto\";MAGDIV.style.right=\"10px\";}}\nfunction magRender(cx){if(typeof Plotly===\"undefined\"||!GD||!GD._fullLayout)return;var xa=GD._fullLayout.xaxis,hs=Math.abs(xa.range[1]-xa.range[0])/50;if(!(hs>0))hs=1;var lo=cx-hs,hi=cx+hs,mar=(hi-lo)*0.12,data=GD.data||[],ymin=Infinity,ymax=-Infinity,traces=[];for(var t=0;t<data.length;t++){var X=data[t].x||[],Y=data[t].y||[],xs=[],ysl=[];for(var k=0;k<X.length;k++){if(X[k]>=lo-mar&&X[k]<=hi+mar){xs.push(X[k]);ysl.push(Y[k]);if(X[k]>=lo&&X[k]<=hi){if(Y[k]<ymin)ymin=Y[k];if(Y[k]>ymax)ymax=Y[k];}}}traces.push({x:xs,y:ysl,type:\"scatter\",mode:\"lines\",line:{width:1.7,color:(data[t].line&&data[t].line.color)||\"#1f77b4\"},hoverinfo:\"skip\"});}if(!isFinite(ymin)){ymin=0;ymax=1;}var pad=(ymax-ymin)*0.1;if(!(pad>0))pad=1;ymin-=pad;ymax+=pad;ensureMag();positionMag(cx);MAGDIV.style.display=\"block\";var layout={margin:{l:6,r:6,t:20,b:18},width:300,height:178,paper_bgcolor:curBg,plot_bgcolor:curBg,font:{color:curFg,size:9},showlegend:false,xaxis:{range:[lo,hi],fixedrange:true,gridcolor:curGrid,zeroline:false,color:curFg,nticks:4,tickformat:\".0f\"},yaxis:{range:[ymin,ymax],fixedrange:true,gridcolor:curGrid,zeroline:false,showticklabels:false},shapes:[{type:\"line\",x0:cx,x1:cx,y0:0,y1:1,yref:\"paper\",line:{color:\"#d62728\",width:1.6}}],annotations:[{x:0.5,xref:\"paper\",y:1,yref:\"paper\",yanchor:\"bottom\",text:\"\u25be \"+cx.toFixed(1)+\" cm\u207b\u00b9\",showarrow:false,font:{color:curFg,size:11.5},bgcolor:curBg,borderpad:1}]};Plotly.react(MAGDIV,traces,layout,{staticPlot:true,displayModeBar:false});}\nfunction magUpdate(cx){magPending=cx;if(!magRAF){magRAF=true;requestAnimationFrame(function(){magRAF=false;if(magPending!==null)magRender(magPending);});}}\nfunction magHide(){if(MAGDIV)MAGDIV.style.display=\"none\";}\nfunction onDown(e){if(eraserOn||!GD||!GD._fullLayout)return;if(GD._fullLayout.dragmode)return;if(nearEdge(e)!==null){dragging=true;magUpdate(mouseDataX(e));}}\nfunction onMove(e){if(dragging)magUpdate(mouseDataX(e));}\nfunction onUp(){if(!dragging)return;dragging=false;setTimeout(magHide,1000);}\nfunction rerender(){if(lastArgs)build(lastArgs,lastTheme);}\nfunction build(args,theme){\n  if(typeof Plotly===\"undefined\"){setTimeout(function(){build(args,theme);},60);return;}\n  GD=document.getElementById(\"c\");args=args||{};lastArgs=args;lastTheme=theme;\n  var x=args.x||[],ys=args.ys||[],names=args.names||[],wins=args.windows||[],seed=args.seed;\n  var _xk=x.length?(x.length+\":\"+x[0]+\":\"+x[x.length-1]):\"\";var keepX=(_xk===lastXkey&&GD._fullLayout&&GD._fullLayout.xaxis&&GD._fullLayout.xaxis.range)?GD._fullLayout.xaxis.range.slice():null;lastXkey=_xk;\n  var dark=!!(theme&&theme.base===\"dark\");\n  var bg=(theme&&theme.backgroundColor)||(dark?\"#0e1117\":\"#ffffff\");\n  var fg=(theme&&theme.textColor)||(dark?\"#fafafa\":\"#262730\");\n  var grid=dark?\"rgba(250,250,250,0.13)\":\"rgba(0,0,0,0.09)\";\n  curDark=dark;curBg=bg;curFg=fg;curGrid=grid;\n  var traces=ys.map(function(yy,j){return {x:x,y:yy,type:\"scatter\",mode:\"lines\",name:(names[j]||(\"col \"+j)),line:{width:1.3,color:PALETTE[j%PALETTE.length]},hoverinfo:\"x+y+name\"};});\n  var reseed=(seed!==lastSeed)||!wired;\n  var shapes=reseed?winShapes(wins):rawShapes();\n  var layout={margin:{l:58,r:14,t:10,b:42},height:430,hovermode:\"closest\",showlegend:true,legend:{orientation:\"h\",y:1.02,yanchor:\"bottom\",x:0,font:{color:fg}},paper_bgcolor:bg,plot_bgcolor:bg,font:{color:fg},xaxis:{title:{text:\"Raman shift (cm\u207b\u00b9)\"},gridcolor:grid,zeroline:false,color:fg,fixedrange:false},yaxis:{title:{text:\"intensity (a.u.)\"},gridcolor:grid,zeroline:false,color:fg,fixedrange:true},shapes:shapes,dragmode:false,newshape:{line:{color:LINE,width:1.5},fillcolor:FILL,layer:\"above\"}};\n  if(keepX){layout.xaxis.range=keepX;}\n  var editBtn={name:\"editwin\",title:\"Edit windows (drag an edge to resize / move)\",icon:Plotly.Icons.pencil,click:function(){var w=eraserOn;eraserOn=false;if(w){rerender();}else{Plotly.relayout(GD,{dragmode:false});}}};\n  var eraserBtn={name:\"eraser\",title:\"Eraser - click a window to delete it\",icon:(Plotly.Icons&&Plotly.Icons.eraseshape)||Plotly.Icons.pencil,toggle:true,click:function(){eraserOn=!eraserOn;rerender();}};\n  var config={displaylogo:false,responsive:true,edits:{shapePosition:!eraserOn},modeBarButtonsToAdd:[editBtn,\"drawrect\",eraserBtn],modeBarButtonsToRemove:[\"lasso2d\",\"select2d\",\"eraseshape\",\"resetScale2d\"]};\n  programmatic=true;\n  Plotly.react(\"c\",traces,layout,config).then(function(){\n    if(!wired){wire();wired=true;}\n    if(!clickWired){GD.addEventListener(\"click\",eraseAt,true);clickWired=true;}\n    if(!magWired){GD.addEventListener(\"mousedown\",onDown,true);document.addEventListener(\"mousemove\",onMove,true);document.addEventListener(\"mouseup\",onUp,true);magWired=true;}\n    GD.style.cursor=eraserOn?\"crosshair\":\"\";\n    if(reseed){lastSeed=seed;lastKey=shapeKey();}\n    setTimeout(function(){programmatic=false;},120);\n    Streamlit.height(442);\n  });\n}\nwindow.addEventListener(\"message\",function(e){var d=e.data;if(!d||d.type!==\"streamlit:render\")return;build(d.args,d.theme);});\nStreamlit.ready();\n</script>\n</body>\n</html>\n"
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


_OCP_HTML = "<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\"/>\n<script src=\"./plotly.min.js\" onerror=\"(function(){var s=document.createElement('script');s.src='https://cdn.plot.ly/plotly-2.35.2.min.js';document.head.appendChild(s);})()\"></script>\n<style>\nhtml,body{margin:0;padding:0;height:588px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif}\n#bar{height:30px;display:flex;align-items:center;gap:12px;padding:2px 4px 4px 6px;box-sizing:border-box}\n#apply{font-size:13px;font-weight:600;padding:5px 15px;border-radius:7px;border:1px solid #34c9c0;background:#13b3aa;color:#04211f;cursor:pointer;transition:opacity .15s}\n#apply:disabled{cursor:default;opacity:.4}\n#status{font-size:12px;opacity:.85}\n#c{width:100%;height:548px}\n</style>\n</head>\n<body>\n<div id=\"bar\"><button id=\"apply\" disabled>&#10003; Apply moves</button><span id=\"status\"></span></div>\n<div id=\"c\"></div>\n<script>\nfunction post(t,e){var m=Object.assign({isStreamlitMessage:true,type:t},e||{});window.parent.postMessage(m,\"*\");}\nvar Streamlit={ready:function(){post(\"streamlit:componentReady\",{apiVersion:1});},height:function(h){post(\"streamlit:setFrameHeight\",{height:h});},value:function(v){post(\"streamlit:setComponentValue\",{value:v,dataType:\"json\"});}};\nvar GD=null,CX=[],CY=[],MX=[],COL=[],SYM=[],HOV=[],CLAB=[],CCOL=[],SLAB=[],SSYM=[],OFF=0,BAND=null,TITLE=\"\";\nvar lastSeed=null,wired=false,dragIdx=-1,dirty=0,applyWired=false;\nvar curBg=\"#0e1117\",curFg=\"#fafafa\",curGrid=\"rgba(250,250,250,0.13)\";\nfunction interp(x){var n=CX.length;if(n===0)return 0;if(x<=CX[0])return CY[0];if(x>=CX[n-1])return CY[n-1];var lo=0,hi=n-1;while(hi-lo>1){var mid=(lo+hi)>>1;if(CX[mid]<=x){lo=mid;}else{hi=mid;}}var t=(x-CX[lo])/((CX[hi]-CX[lo])||1);return CY[lo]+t*(CY[hi]-CY[lo]);}\nfunction mYs(){return MX.map(function(x){return interp(x)+OFF;});}\nfunction sendVal(){Streamlit.value({xs:MX.slice(),ts:Date.now()});}\nfunction setStatus(msg){var b=document.getElementById(\"apply\"),s=document.getElementById(\"status\");if(!b||!s)return;if(typeof msg===\"string\"){s.textContent=msg;s.style.color=curFg;return;}if(dirty>0){b.disabled=false;s.textContent=dirty+\" unsaved move\"+(dirty>1?\"s\":\"\");s.style.color=curFg;}else{b.disabled=true;s.textContent=\"\";}}\nfunction onApply(){if(dirty>0){sendVal();dirty=0;setStatus(\"saving\u2026\");}else{setStatus();}}\nfunction legendTraces(){var tr=[];var i;for(i=0;i<CLAB.length;i++){tr.push({x:[null],y:[null],type:\"scatter\",mode:\"markers\",name:CLAB[i],legendgroup:\"conc\",legendgrouptitle:{text:\"Concentration\"},marker:{size:11,color:CCOL[i],symbol:\"circle\",line:{width:1,color:\"#1A1620\"}}});}for(i=0;i<SLAB.length;i++){tr.push({x:[null],y:[null],type:\"scatter\",mode:\"markers\",name:SLAB[i],legendgroup:\"vol\",legendgrouptitle:{text:\"Volume\"},marker:{size:11,color:\"#6B6573\",symbol:SSYM[i],line:{width:1,color:\"#1A1620\"}}});}return tr;}\nfunction draw(reset){var keepx=null,keepy=null;if(!reset&&GD&&GD._fullLayout){var xap=GD._fullLayout.xaxis,yap=GD._fullLayout.yaxis;if(xap&&xap.autorange===false&&xap.range){keepx=xap.range.slice();}if(yap&&yap.autorange===false&&yap.range){keepy=yap.range.slice();}}var line={x:CX,y:CY,type:\"scatter\",mode:\"lines\",line:{width:1.6,color:\"#2E5A6B\"},hoverinfo:\"skip\",showlegend:false};var mk={x:MX.slice(),y:mYs(),type:\"scatter\",mode:\"markers\",showlegend:false,text:HOV,hovertemplate:\"%{x:.1f} min<br>%{text}<extra></extra>\",marker:{size:13,color:COL,symbol:SYM,line:{width:1,color:\"#1A1620\"}}};var traces=[line,mk].concat(legendTraces());var shapes=[];if(BAND){shapes.push({type:\"rect\",xref:\"paper\",x0:0,x1:1,yref:\"y\",y0:Math.min(BAND[0],BAND[1]),y1:Math.max(BAND[0],BAND[1]),fillcolor:\"rgba(240,215,55,0.28)\",line:{width:0},layer:\"below\"});}var layout={margin:{l:60,r:14,t:(TITLE?30:12),b:46},height:548,paper_bgcolor:curBg,plot_bgcolor:curBg,font:{color:curFg},hovermode:\"closest\",showlegend:true,legend:{orientation:\"v\",x:1.01,y:1,font:{color:curFg}},xaxis:{title:{text:\"time (min)\"},gridcolor:curGrid,zeroline:false,color:curFg},yaxis:{title:{text:\"potential (V)\"},gridcolor:curGrid,zeroline:false,color:curFg},shapes:shapes,dragmode:\"zoom\"};if(keepx){layout.xaxis.range=keepx;layout.xaxis.autorange=false;}if(keepy){layout.yaxis.range=keepy;layout.yaxis.autorange=false;}if(TITLE){layout.title={text:TITLE,font:{size:14,color:curFg},x:0.01};}var config={displaylogo:false,responsive:true,displayModeBar:true,scrollZoom:false,modeBarButtonsToRemove:[\"select2d\",\"lasso2d\",\"toImage\"]};Plotly.react(\"c\",traces,layout,config).then(function(){if(!wired){wireDrag();wired=true;}GD.style.cursor=\"\";Streamlit.height(588);});}\nfunction restyleMarkers(){Plotly.restyle(\"c\",{x:[MX.slice()],y:[mYs()]},[1]);}\nfunction nearestMarker(e){if(!GD||!GD._fullLayout)return -1;var xa=GD._fullLayout.xaxis,ya=GD._fullLayout.yaxis,bb=GD.getBoundingClientRect();var pxr=e.clientX-bb.left-xa._offset,pyr=e.clientY-bb.top-ya._offset;if(pxr<0||pxr>xa._length||pyr<0||pyr>ya._length)return -1;var mdx=xa.p2d(pxr),mdy=ya.p2d(pyr);var tolx=Math.abs(xa.range[1]-xa.range[0])/xa._length*16;var toly=Math.abs(ya.range[1]-ya.range[0])/ya._length*22;var best=-1,bestd=Infinity,i;for(i=0;i<MX.length;i++){var dx=Math.abs(mdx-MX[i])/tolx,dy=Math.abs(mdy-(interp(MX[i])+OFF))/toly;if(dx<1&&dy<1){var d=dx*dx+dy*dy;if(d<bestd){bestd=d;best=i;}}}return best;}\nfunction mouseDataX(e){var xa=GD._fullLayout.xaxis;var p=e.clientX-GD.getBoundingClientRect().left-xa._offset;var d=xa.p2d(p);if(d<xa.range[0])d=xa.range[0];if(d>xa.range[1])d=xa.range[1];return d;}\nfunction onDown(e){if(!GD||!GD._fullLayout)return;var i=nearestMarker(e);if(i>=0){dragIdx=i;GD.style.cursor=\"grabbing\";e.preventDefault();e.stopPropagation();}}\nfunction onHover(e){if(dragIdx>=0||!GD||!GD._fullLayout)return;GD.style.cursor=(nearestMarker(e)>=0)?\"grab\":\"\";}\nfunction onMove(e){if(dragIdx<0)return;MX[dragIdx]=mouseDataX(e);restyleMarkers();e.preventDefault();e.stopPropagation();}\nfunction onUp(e){if(dragIdx<0)return;dragIdx=-1;GD.style.cursor=\"\";dirty++;setStatus();}\nfunction wireDrag(){GD.addEventListener(\"mousedown\",onDown,true);GD.addEventListener(\"mousemove\",onHover);document.addEventListener(\"mousemove\",onMove,true);document.addEventListener(\"mouseup\",onUp,true);}\nfunction build(args,theme){if(typeof Plotly===\"undefined\"){setTimeout(function(){build(args,theme);},60);return;}GD=document.getElementById(\"c\");args=args||{};var dark=!!(theme&&theme.base===\"dark\");curBg=(theme&&theme.backgroundColor)||(dark?\"#0e1117\":\"#ffffff\");curFg=(theme&&theme.textColor)||(dark?\"#fafafa\":\"#262730\");curGrid=dark?\"rgba(250,250,250,0.13)\":\"rgba(0,0,0,0.09)\";CX=args.cx||[];CY=args.cy||[];COL=args.col||[];SYM=args.sym||[];HOV=args.hov||[];CLAB=args.clab||[];CCOL=args.ccol||[];SLAB=args.slab||[];SSYM=args.ssym||[];OFF=args.off||0;BAND=args.band||null;TITLE=args.title||\"\";var seed=args.seed;var reseed=(seed!==lastSeed)||!wired;if(reseed){MX=(args.mx||[]).slice();lastSeed=seed;dirty=0;}if(!applyWired){var ab=document.getElementById(\"apply\");if(ab){ab.addEventListener(\"click\",onApply);applyWired=true;}}draw(reseed);setStatus();}\nwindow.addEventListener(\"message\",function(e){var d=e.data;if(!d||d.type!==\"streamlit:render\")return;build(d.args,d.theme);});\nStreamlit.ready();\n</script>\n</body>\n</html>\n"
_OCP_VER = hashlib.md5(_OCP_HTML.encode("utf-8")).hexdigest()[:8]


@st.cache_resource(show_spinner=False)
def _ocp_component(ver):
    """Materialise the draggable OCP-marker frontend and declare the
    bidirectional component (same offline-plotly trick as the editor)."""
    import plotly as _plotly
    base = os.path.join(tempfile.gettempdir(), "ocp_editor_" + ver)
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
        fh.write(_OCP_HTML)
    return components.declare_component("ocp_editor_" + ver, path=base)


_LINK_HTML = "<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\"/>\n<script src=\"./plotly.min.js\" onerror=\"(function(){var s=document.createElement('script');s.src='https://cdn.plot.ly/plotly-2.35.2.min.js';document.head.appendChild(s);})()\"></script>\n<style>html,body{margin:0;padding:0;height:648px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif}#c{width:100%;height:648px}</style>\n</head>\n<body>\n<div id=\"c\"></div>\n<script>\nfunction post(t,e){var m=Object.assign({isStreamlitMessage:true,type:t},e||{});window.parent.postMessage(m,\"*\");}\nvar Streamlit={ready:function(){post(\"streamlit:componentReady\",{apiVersion:1});},height:function(h){post(\"streamlit:setFrameHeight\",{height:h});}};\nvar GD=null,OT=[],OV=[],WL=[],SP=[],SPT=[],SN=[],TITLE=\"\",wired=false,curK=-1;\nvar curBg=\"#ffffff\",curFg=\"#262730\",curGrid=\"rgba(0,0,0,0.09)\";\nfunction nearestSpec(t){var best=0,bd=Infinity,i;for(i=0;i<SPT.length;i++){var d=Math.abs(SPT[i]-t);if(d<bd){bd=d;best=i;}}return best;}\nfunction interpO(t){var n=OT.length;if(n===0)return 0;if(t<=OT[0])return OV[0];if(t>=OT[n-1])return OV[n-1];var lo=0,hi=n-1;while(hi-lo>1){var m=(lo+hi)>>1;if(OT[m]<=t){lo=m;}else{hi=m;}}var f=(t-OT[lo])/((OT[hi]-OT[lo])||1);return OV[lo]+f*(OV[hi]-OV[lo]);}\nfunction specTitle(k){if(k<0||k>=SPT.length)return \"UV-Vis spectrum\";return \"UV-Vis @ \"+SPT[k].toFixed(2)+\" min\"+(SN[k]?\"  \u00b7  \"+SN[k]:\"\");}\nfunction draw(){\n  var ocp={x:OT,y:OV,type:\"scatter\",mode:\"lines\",line:{width:1.7,color:\"#2E5A6B\"},xaxis:\"x\",yaxis:\"y\",hovertemplate:\"t = %{x:.2f} min<br>%{y:.3f} V<extra></extra>\"};\n  var rug={x:SPT,y:SPT.map(interpO),type:\"scatter\",mode:\"markers\",marker:{size:9,color:\"rgba(196,35,72,0.55)\",symbol:\"line-ns-open\"},xaxis:\"x\",yaxis:\"y\",hoverinfo:\"skip\"};\n  var cur={x:[SPT.length?SPT[curK]:0],y:[SPT.length?interpO(SPT[curK]):0],type:\"scatter\",mode:\"markers\",marker:{size:13,color:\"#C42348\",line:{width:1.6,color:\"#ffffff\"}},xaxis:\"x\",yaxis:\"y\",hoverinfo:\"skip\"};\n  var spec={x:WL,y:(SP.length?SP[curK]:[]),type:\"scatter\",mode:\"lines\",line:{width:1.7,color:\"#7B2FB0\"},fill:\"tozeroy\",fillcolor:\"rgba(123,47,176,0.09)\",xaxis:\"x2\",yaxis:\"y2\",hovertemplate:\"%{x:.0f} nm<br>%{y:.4f}<extra></extra>\"};\n  var layout={margin:{l:62,r:16,t:30,b:46},height:648,paper_bgcolor:curBg,plot_bgcolor:curBg,font:{color:curFg},showlegend:false,hovermode:\"x\",\n    xaxis:{domain:[0,1],anchor:\"y\",title:{text:\"time (min)\"},gridcolor:curGrid,zeroline:false,color:curFg,showspikes:true,spikemode:\"across\",spikethickness:1.2,spikecolor:\"#C42348\",spikedash:\"solid\",spikesnap:\"cursor\"},\n    yaxis:{domain:[0.575,1.0],anchor:\"x\",title:{text:\"potential (V)\"},gridcolor:curGrid,zeroline:false,color:curFg},\n    xaxis2:{domain:[0,1],anchor:\"y2\",title:{text:\"wavelength (nm)\"},gridcolor:curGrid,zeroline:false,color:curFg},\n    yaxis2:{domain:[0,0.43],anchor:\"x2\",title:{text:\"absorbance\"},gridcolor:curGrid,zeroline:false,color:curFg},\n    annotations:[{text:(TITLE||\"Potential vs time\"),x:0,xref:\"paper\",y:1.0,yref:\"paper\",yanchor:\"bottom\",showarrow:false,font:{size:13.5,color:curFg}},\n                 {text:specTitle(curK),x:0,xref:\"paper\",y:0.445,yref:\"paper\",yanchor:\"bottom\",showarrow:false,font:{size:13.5,color:curFg}}]};\n  var config={displaylogo:false,responsive:true,displayModeBar:true,modeBarButtonsToRemove:[\"select2d\",\"lasso2d\",\"autoScale2d\"]};\n  Plotly.react(\"c\",[ocp,rug,cur,spec],layout,config).then(function(){if(!wired){wire();wired=true;}Streamlit.height(658);});\n}\nfunction updateTo(t){if(!SP.length)return;var k=nearestSpec(t);curK=k;Plotly.restyle(\"c\",{x:[[t]],y:[[interpO(t)]]},[2]);Plotly.restyle(\"c\",{y:[SP[k]]},[3]);Plotly.relayout(\"c\",{\"annotations[1].text\":specTitle(k)});}\nfunction wire(){GD.on(\"plotly_hover\",function(d){if(!d||!d.points||!d.points.length)return;var i,p;for(i=0;i<d.points.length;i++){p=d.points[i];if(p.curveNumber===0&&typeof p.x===\"number\"){updateTo(p.x);return;}}});}\nfunction build(args,theme){if(typeof Plotly===\"undefined\"){setTimeout(function(){build(args,theme);},60);return;}GD=document.getElementById(\"c\");args=args||{};var dark=!!(theme&&theme.base===\"dark\");curBg=(theme&&theme.backgroundColor)||(dark?\"#0e1117\":\"#ffffff\");curFg=(theme&&theme.textColor)||(dark?\"#fafafa\":\"#262730\");curGrid=dark?\"rgba(250,250,250,0.13)\":\"rgba(0,0,0,0.09)\";OT=args.ot||[];OV=args.ov||[];WL=args.wl||[];SP=args.sp||[];SPT=args.spt||[];SN=args.sn||[];TITLE=args.title||\"\";if(curK<0||curK>=SP.length)curK=Math.floor(SP.length/2);draw();}\nwindow.addEventListener(\"message\",function(e){var d=e.data;if(!d||d.type!==\"streamlit:render\")return;build(d.args,d.theme);});\nStreamlit.ready();\n</script>\n</body>\n</html>\n"
_LINK_VER = hashlib.md5(_LINK_HTML.encode("utf-8")).hexdigest()[:8]


@st.cache_resource(show_spinner=False)
def _link_component(ver):
    """Materialise the linked OCP + UV-Vis viewer frontend (display-only; the
    hover interaction is entirely client-side, so nothing is sent back)."""
    import plotly as _plotly
    base = os.path.join(tempfile.gettempdir(), "link_view_" + ver)
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
        fh.write(_LINK_HTML)
    return components.declare_component("link_view_" + ver, path=base)


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
               "click the window). A zoomed inset pops up while you drag an edge for "
               "precise placement. Zoom the spectrum with the toolbar "
               "(magnifier / + / -); the pencil tool resumes window "
               "editing. Or fine-tune exact bounds below.")

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
#  Batch raw-file combiner (instrument exports -> one wavelength + many series)
# --------------------------------------------------------------------------- #
def _pick_col(cols, keys):
    low = [(str(c).lower(), c) for c in cols]
    for k in keys:
        for lc, c in low:
            if k in lc:
                return c
    return None


def _parse_sers_csv(f):
    """Raw SERS export: a CSV with Wavelength + Intensity columns (others ignored)."""
    try:
        f.seek(0)
    except Exception:
        pass
    df = pd.read_csv(f)
    df.columns = [str(c).strip() for c in df.columns]
    xcol = _pick_col(df.columns, ["wavelength", "wavenumber", "raman", "shift",
                                  "wave"]) or df.columns[0]
    ycol = (_pick_col(df.columns, ["intensity", "counts", "signal", "abs"])
            or (df.columns[1] if len(df.columns) > 1 else df.columns[0]))
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _parse_uvvis_txt(f):
    """Raw UV-Vis export: a header block ending at '...Begin Spectral Data...',
    then  wavelength <tab/space> intensity  rows."""
    try:
        raw = f.getvalue()
    except Exception:
        f.seek(0)
        raw = f.read()
    text = (raw.decode("utf-8", errors="replace")
            if isinstance(raw, bytes) else str(raw))
    lines = text.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if "begin spectral data" in ln.lower():
            start = i + 1
            break
    xs, ys = [], []
    for ln in lines[start:]:
        s = ln.strip()
        if not s:
            continue
        if s.startswith(">>>"):
            break
        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            xv, yv = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        xs.append(xv)
        ys.append(yv)
    return np.asarray(xs, float), np.asarray(ys, float)


def _combine_raw(files, is_sers):
    """Merge raw files onto the first file's wavelength axis; one series per file,
    each named after its source document. Returns (DataFrame, messages)."""
    parse = _parse_sers_csv if is_sers else _parse_uvvis_txt
    xname = "Wavelength" if is_sers else "Wavelength (nm)"
    x_ref, series, msgs = None, [], []
    for f in files:
        try:
            x, y = parse(f)
        except Exception as exc:                        # noqa: BLE001
            msgs.append(("warning", f"Skipped **{f.name}** — {exc}"))
            continue
        if x.size < 2 or y.size < 2:
            msgs.append(("warning", f"Skipped **{f.name}** — no spectral data found."))
            continue
        if x_ref is None:
            x_ref = x
        if y.size == x_ref.size and np.allclose(x, x_ref, atol=1e-6):
            yv = y
        else:
            yv = np.interp(x_ref, x, y)
            msgs.append(("info", f"**{f.name}** resampled onto the reference axis "
                                 f"({y.size}→{x_ref.size} pts)."))
        nm, taken = os.path.splitext(f.name)[0], [s[0] for s in series]
        col, k = nm, 1
        while col in taken:
            k += 1
            col = f"{nm} ({k})"
        series.append((col, yv))
    if x_ref is None or not series:
        return None, msgs
    out = pd.DataFrame({xname: x_ref})
    for col, yv in series:
        out[col] = yv
    return out, msgs


def combine_uploaded(files, is_sers):
    """Combine uploaded raw files, recomputing only when the file set changes."""
    sig = (bool(is_sers), tuple((f.name, f.size) for f in files))
    if st.session_state.get("_combine_sig") != sig:
        df, msgs = _combine_raw(files, is_sers)
        st.session_state["_combine_sig"] = sig
        st.session_state["_combine_df"] = df
        st.session_state["_combine_msgs"] = msgs
    return st.session_state["_combine_df"], st.session_state["_combine_msgs"]


# --------------------------------------------------------------------------- #
#  Kinetics / time-trend (OceanView strip-chart: Value vs Time)
# --------------------------------------------------------------------------- #
def _hms_to_sec(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _clock(sec):
    sec = sec % 86400.0
    return f"{int(sec // 3600):02d}:{int((sec % 3600) // 60):02d}:{sec % 60:05.2f}"


def _channel_of(name):
    m = re.match(r"^(.*?)__\d+__", name)
    pre = m.group(1) if m else os.path.splitext(name)[0]
    return pre.split("_")[-1] or pre


def _parse_ov_trend(f):
    """OceanView strip-chart export: Standard Time / Epoch Time / Value rows."""
    try:
        raw = f.getvalue()
    except Exception:
        f.seek(0)
        raw = f.read()
    text = (raw.decode("utf-8", errors="replace")
            if isinstance(raw, bytes) else str(raw))
    lines = text.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        low = ln.lower()
        if "begin spectral data" in low:
            start = i + 1
            break
        if low.startswith("hh:mm:ss"):
            start = i + 1
    t, v = [], []
    for ln in lines[start:]:
        p = ln.split("\t")
        if len(p) < 2:
            p = ln.split()
        if len(p) < 2:
            continue
        try:
            sec = _hms_to_sec(p[0].strip())
            val = float(p[-1])
        except (ValueError, IndexError):
            continue
        t.append(sec)
        v.append(val)
    return np.asarray(t, float), np.asarray(v, float)


def _stitch_trends(files):
    """Group OceanView files by channel and stitch the rolling windows into one
    continuous trace on a common 1-second grid. Returns (chans, grid_sec, t0)."""
    groups = {}
    for f in files:
        try:
            t, v = _parse_ov_trend(f)
        except Exception:                               # noqa: BLE001
            continue
        if t.size == 0:
            continue
        d = groups.setdefault(_channel_of(f.name), {})
        for ti, vi in zip(t, v):
            d[round(float(ti), 3)] = float(vi)
    groups = {k: d for k, d in groups.items() if d}
    if not groups:
        return {}, None, None
    t0 = min(min(d) for d in groups.values())
    t1 = max(max(d) for d in groups.values())
    grid = np.arange(t0, t1 + 1e-6, 1.0)
    chans = {}
    for ch, d in groups.items():
        ts = np.array(sorted(d))
        chans[ch] = np.interp(grid, ts, np.array([d[x] for x in ts]))
    return chans, grid, t0


def _detect_jumps(emin, s, min_rise, win_min=0.3, min_sep_min=1.5):
    """Flag sharp rises (reagent additions): absorbance rising by >= min_rise
    within ~win_min. Returns [(index, elapsed_min, rise), ...]."""
    if emin.size < 3:
        return []
    dt = float(np.median(np.diff(emin)))
    w = max(1, int(round(win_min / dt)))
    rise = np.zeros_like(s)
    rise[:-w] = s[w:] - s[:-w]
    cand = np.where(rise > min_rise)[0]
    out = []
    if cand.size:
        splits = np.where(np.diff(cand) > int(min_sep_min / dt))[0] + 1
        for g in np.split(cand, splits):
            i = int(g[int(np.argmax(rise[g]))])
            out.append((i, float(emin[i]), float(rise[i])))
    return out


def _kinetics_fig(emin, plot, labels, events, grid, ylab):
    pal = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#e87ba4"]
    fig = go.Figure()
    for j, ch in enumerate(plot):
        fig.add_trace(go.Scatter(
            x=emin, y=plot[ch], mode="lines", name=str(labels.get(ch, ch)),
            line=dict(width=1.8, color=pal[j % len(pal)]),
            hovertemplate="%{x:.1f} min — %{y:.3f}<extra>"
                          + str(labels.get(ch, ch)) + "</extra>"))
    for i, m, _r in events:
        fig.add_vline(x=m, line=dict(color="#9aa0a6", width=1, dash="dash"))
        fig.add_annotation(x=m, y=1.0, yref="paper", yanchor="bottom",
                           text=_clock(grid[i]).split(".")[0], showarrow=False,
                           font=dict(size=10))
    fig.update_layout(margin=dict(l=10, r=10, t=26, b=42), height=480,
                      xaxis_title="time (min)", yaxis_title=ylab,
                      hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    return fig


def kinetics_analysis(files):
    """OceanView strip-chart: stitch the rolling windows, plot absorbance vs time,
    and auto-detect + time-stamp the jumps (reagent additions)."""
    st.caption("OceanView strip-chart (absorbance vs time). Files are grouped by "
               "channel, stitched across the auto-saved rolling windows, and "
               "plotted. Sharp rises (e.g. reagent additions) are auto-detected "
               "and time-stamped below.")
    chans, grid, t0 = _stitch_trends(files)
    if not chans:
        st.warning("No OceanView trend data found — these should be the "
                   "strip-chart .txt files (Standard Time / Value rows).")
        return
    emin = (grid - t0) / 60.0

    st.markdown("**Channels** — rename to label each tracked wavelength")
    labels = {}
    lcols = st.columns(min(len(chans), 4))
    for i, ch in enumerate(chans):
        labels[ch] = lcols[i % len(lcols)].text_input(ch, value=ch, key=f"klab_{ch}")

    c1, c2 = st.columns([1.5, 1])
    base_lbl = c1.selectbox(
        "Baseline channel (optional — subtracted from the others)",
        ["(none)"] + [labels[c] for c in chans], index=0,
        help="A reference wavelength tracking common-mode drift / turbidity; "
             "subtracting it isolates the wavelength-specific signal.")
    sens = c2.slider("Jump sensitivity (min rise)", 0.01, 0.50, 0.05, 0.01,
                     help="Flag a jump when absorbance rises by at least this much "
                          "within ~0.3 min. Lower = more sensitive.")

    base = next((c for c in chans if labels[c] == base_lbl), None)
    if base is not None:
        plot = {c: chans[c] - chans[base] for c in chans if c != base}
        ylab = "absorbance (baseline-subtracted)"
    else:
        plot = dict(chans)
        ylab = "absorbance"
    if not plot:
        st.warning("Only the baseline channel is present — nothing left to plot.")
        return

    detect_ch = max(chans, key=lambda c: float(chans[c].max() - chans[c].min()))
    events = _detect_jumps(emin, chans[detect_ch], sens)  # additions from raw signal

    st.plotly_chart(_kinetics_fig(emin, plot, labels, events, grid, ylab),
                    use_container_width=True, theme="streamlit")

    st.markdown("**Detected jumps (additions)**")
    dt = float(np.median(np.diff(grid))) if grid.size > 1 else 1.0
    look = max(1, int(round(0.3 * 60 / dt)))
    rows = []
    for i, m, _r in events:
        j = min(i + look, len(grid) - 1)
        row = {"Time": _clock(grid[i]), "Elapsed (min)": round(m, 1)}
        for c in plot:
            row["Δ " + labels[c]] = round(float(plot[c][j] - plot[c][i]), 3)
        rows.append(row)
    ev_df = pd.DataFrame(rows)
    if rows:
        st.dataframe(ev_df, use_container_width=True, hide_index=True)
        st.caption(f"Detected on **{labels[detect_ch]}** (largest range). "
                   "Adjust sensitivity to add or drop events.")
    else:
        st.caption("No jumps above the current sensitivity — lower it to catch "
                   "smaller steps.")

    out = pd.DataFrame({"Time": [_clock(s) for s in grid],
                        "Elapsed_min": np.round(emin, 4)})
    for c in plot:
        out[labels[c]] = plot[c]
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Stitched trend (CSV)",
                       out.to_csv(index=False).encode("utf-8"),
                       "kinetics_trend.csv", "text/csv", key="dl_kin",
                       use_container_width=True)
    if rows:
        d2.download_button("⬇️ Detected jumps (CSV)",
                           ev_df.to_csv(index=False).encode("utf-8"),
                           "kinetics_jumps.csv", "text/csv", key="dl_kin_ev",
                           use_container_width=True)


# --------------------------------------------------------------------------- #
#  Open-circuit potential (potential vs time; reagent-addition annotations)
# --------------------------------------------------------------------------- #
OCP_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "triangle-down",
               "star", "cross", "x", "pentagon", "hexagon",
               "triangle-left", "triangle-right"]
OCP_MPL = {"circle": "o", "square": "s", "diamond": "D", "triangle-up": "^",
           "triangle-down": "v", "star": "*", "cross": "P", "x": "X",
           "pentagon": "p", "hexagon": "h", "triangle-left": "<",
           "triangle-right": ">"}
# validated standout categorical order (one colour per concentration)
def _distinct_colors(n):
    """n visually distinct colours: eight evenly-spaced rainbow hues first, then
    golden-ratio-spaced hues beyond eight so every concentration stays unique."""
    import colorsys

    def hexc(h, lig, sat):
        r, g, b = colorsys.hls_to_rgb(h % 1.0, lig, sat)
        return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))

    rainbow = [hexc(k / 8.0, 0.50, 0.90) for k in range(8)]
    if n <= 8:
        return rainbow[:n]
    out = list(rainbow)
    for k in range(n - 8):
        out.append(hexc(k * 0.6180339887498949 + 0.05, 0.44 + 0.12 * (k % 2), 0.82))
    return out


def _read_text(f):
    try:
        raw = f.getvalue()
    except Exception:
        f.seek(0)
        raw = f.read()
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)


def _parse_ocp_raw(text):
    """Raw OCP export: header, then '[Begin Data]', then Time (s), Voltage rows."""
    lines = text.splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if "[begin data]" in ln.lower():
            start = i + 1
            break
    if start < len(lines):
        h = lines[start].lower()
        if "time" in h and ("volt" in h or "(v)" in h):
            start += 1
    t, v = [], []
    for ln in lines[start:]:
        p = ln.replace(";", ",").split(",")
        if len(p) < 2:
            continue
        try:
            t.append(float(p[0]))
            v.append(float(p[1]))
        except ValueError:
            continue
    return np.asarray(t, float), np.asarray(v, float)


def _ocp_template_data(df):
    """From a filled template DataFrame return (tmin, volt, additions, error)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    vcol = _pick_col(cols, ["voltage", "potential", "(v)"])
    mcol = _pick_col(cols, ["time (min)", "min"])
    scol = _pick_col(cols, ["time (s)", "(s)", "second"])
    if vcol is None:
        return None, None, None, "no voltage / potential column found"
    volt = pd.to_numeric(df[vcol], errors="coerce").to_numpy(float)
    if mcol is not None:
        tmin = pd.to_numeric(df[mcol], errors="coerce").to_numpy(float)
    elif scol is not None:
        tmin = pd.to_numeric(df[scol], errors="coerce").to_numpy(float) / 60.0
    else:
        return None, None, None, "no time column (Time (min) or Time (s))"
    ann = [c for c in cols if c not in (vcol, mcol, scol)]
    additions = []
    # The row an entry sits in is only a placeholder -- the real minute is
    # chosen later by dragging.  Walk row-major so additions keep the order
    # they were typed in (top to bottom, left to right within a row).
    for r in range(len(df)):
        for c in ann:
            cell = df[c].iloc[r]
            if pd.isna(cell) or str(cell).strip() == "":
                continue
            raw = str(cell).strip()
            num = pd.to_numeric(raw, errors="coerce")
            vol = (f"{num:g} µL" if pd.notna(num) else raw)
            additions.append({"conc": c, "vol": vol})
    return tmin, volt, additions, None


def _ocp_fig(tmin, volt, additions, conc_color, vol_symbol, title="",
             band=None, offset=0.0, bounds=None):
    fig = go.Figure()
    if band:
        fig.add_hrect(y0=min(band), y1=max(band), line_width=0,
                      fillcolor="rgba(240,215,55,0.30)", layer="below")
    for bx in (bounds or []):
        fig.add_vline(x=float(bx), line_width=1, line_dash="dot",
                      line_color="rgba(150,150,160,0.6)")
    fig.add_trace(go.Scatter(x=tmin, y=volt, mode="lines",
                             line=dict(width=1.6, color="#2E5A6B"),
                             showlegend=False,
                             hovertemplate="%{x:.1f} min · %{y:.3f} V<extra></extra>"))
    if additions:
        fig.add_trace(go.Scatter(
            x=[a["t"] for a in additions],
            y=[a["v"] + offset for a in additions],
            mode="markers", showlegend=False,
            marker=dict(size=12,
                        color=[conc_color[a["conc"]] for a in additions],
                        symbol=[vol_symbol[a["vol"]] for a in additions],
                        line=dict(width=1, color="#1A1620")),
            customdata=[[a["conc"], a["vol"], a["v"]] for a in additions],
            hovertemplate="%{x:.1f} min · %{customdata[2]:.3f} V<br>"
                          "%{customdata[0]} · %{customdata[1]}<extra></extra>"))
        for c in conc_color:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=str(c),
                legendgroup="conc", legendgrouptitle_text="Concentration",
                marker=dict(size=11, color=conc_color[c], symbol="circle",
                            line=dict(width=1, color="#1A1620"))))
        for vol in vol_symbol:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=str(vol),
                legendgroup="vol", legendgrouptitle_text="Volume",
                marker=dict(size=11, color="#6B6573", symbol=vol_symbol[vol],
                            line=dict(width=1, color="#1A1620"))))
    fig.update_layout(margin=dict(l=10, r=10, t=34, b=42), height=520,
                      xaxis_title="time (min)", yaxis_title="potential (V)",
                      hovermode="closest", legend=dict(orientation="v", x=1.01, y=1))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14), x=0.01))
    return fig


def _ocp_png(tmin, volt, additions, conc_color, vol_symbol, title="",
             band=None, offset=0.0):
    """Render the annotated potential-vs-time figure to PNG bytes (white bg)."""
    import matplotlib.pyplot as _plt
    import matplotlib.patheffects as _pe
    from matplotlib.lines import Line2D
    light = {"figure.facecolor": "white", "axes.facecolor": "white",
             "savefig.facecolor": "white", "axes.edgecolor": "#888888",
             "axes.labelcolor": "#222222", "axes.titlecolor": "#222222",
             "xtick.color": "#444444", "ytick.color": "#444444",
             "text.color": "#222222", "axes.grid": True,
             "grid.color": "#DDDDDD", "font.size": 11}
    with _plt.rc_context(light):
        fig, ax = _plt.subplots(figsize=(11, 5.5))
        if band:
            ax.axhspan(min(band), max(band), color="#F0D737", alpha=0.30,
                       lw=0, zorder=0)
        for ln in ax.plot(tmin, volt, color="#2E5A6B", lw=1.4, zorder=1):
            ln.set_path_effects([_pe.Normal()])   # cancel the neon glow patch
        for a in additions:
            ax.scatter(a["t"], a["v"] + offset,
                       marker=OCP_MPL.get(vol_symbol[a["vol"]], "o"),
                       c=conc_color[a["conc"]], s=95, edgecolors="#1A1620",
                       linewidths=0.8, zorder=3)
        ax.set_xlabel("time (min)")
        ax.set_ylabel("potential (V)")
        if title:
            ax.set_title(title, loc="left", fontsize=12)
        ch = [Line2D([0], [0], marker="o", color="none", markersize=9,
                     markerfacecolor=conc_color[c], markeredgecolor="#1A1620",
                     label=str(c)) for c in conc_color]
        vh = [Line2D([0], [0], marker=OCP_MPL.get(vol_symbol[v], "o"),
                     color="none", markersize=9, markerfacecolor="#6B6573",
                     markeredgecolor="#1A1620", label=str(v)) for v in vol_symbol]
        leg1 = ax.legend(handles=ch, title="Concentration", loc="upper left",
                         bbox_to_anchor=(1.01, 1.0), fontsize=8,
                         title_fontsize=9)
        ax.add_artist(leg1)
        leg2 = ax.legend(handles=vh, title="Volume", loc="lower left",
                         bbox_to_anchor=(1.01, 0.0), fontsize=8,
                         title_fontsize=9)
        buf = io.BytesIO()
        # bbox_extra_artists makes the tight bbox include BOTH legends that
        # sit outside the axes, so neither gets clipped in the download.
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                    facecolor="white", pad_inches=0.18,
                    bbox_extra_artists=(leg1, leg2))
        _plt.close(fig)
    return buf.getvalue()


def _vol_num(v):
    """Leading numeric value of a volume label like '10 µL' -- used to sort
    volumes low->high. Tolerates thousands-separators and scientific
    notation (e.g. '1,000', '1e+03'); non-numeric labels sort last."""
    s = str(v).strip().replace(",", "")
    token = s.split()[0] if s.split() else s
    try:
        return float(token)
    except ValueError:
        pass
    num = ""
    for ch in token:
        if ch.isdigit() or ch in ".+-eE":
            num += ch
        else:
            break
    try:
        return float(num)
    except ValueError:
        return float("inf")


def _ocp_stitch(files):
    """Read several raw OCP runs, let the user order them, and join them into
    one continuous series (each run starts just after the previous one ends).
    Returns (t_s, volt, bounds_min, stem) or (None, None, None, None)."""
    parsed = []
    for f in files:
        txt = _read_text(f)
        if not ("[begin data]" in txt.lower()
                or "open circuit potential" in txt.lower()):
            st.error(f"**{f.name}** doesn't look like a raw OCP run. Uploading "
                     "several files stitches raw runs together — upload raw "
                     ".txt runs, or a single filled template for the graph.")
            return None, None, None, None
        t_s, v = _parse_ocp_raw(txt)
        if t_s.size < 2:
            st.error(f"Couldn't read Time / Voltage rows from **{f.name}**.")
            return None, None, None, None
        parsed.append((f.name, t_s, v))

    st.markdown(f"**{len(files)} raw runs uploaded — set the order.** They're "
                "joined end-to-end so each run's minute 0 lands right after the "
                "previous run's last minute. Edit the **Order** column (1 = "
                "first in time), then scroll down for the stitched template.")
    order_df = pd.DataFrame({
        "Order": list(range(1, len(parsed) + 1)),
        "File": [p[0] for p in parsed],
        "Points": [int(p[1].size) for p in parsed],
        "Length (min)": [round(float(p[1][-1] - p[1][0]) / 60.0, 2)
                         for p in parsed]})
    # Key the editor by the uploaded file set so a *different* upload starts
    # fresh instead of inheriting the previous run's stale Order edits.
    okey = "ocp_order_" + hashlib.md5(
        "|".join(sorted(p[0] for p in parsed)).encode("utf-8")).hexdigest()[:8]
    edited = st.data_editor(
        order_df, hide_index=True, use_container_width=True, key=okey,
        disabled=["File", "Points", "Length (min)"],
        column_config={"Order": st.column_config.NumberColumn(
            "Order", min_value=1, max_value=len(parsed), step=1,
            help="1 = first run in time, 2 = next, and so on.")})
    orders = pd.to_numeric(edited["Order"], errors="coerce").fillna(0).tolist()
    seq = sorted(range(len(parsed)), key=lambda i: (orders[i], i))

    all_t, all_v, bounds = [], [], []
    run_end, step = 0.0, 0.0
    for k, idx in enumerate(seq):
        _, t_s, v = parsed[idx]
        t0 = t_s - float(t_s[0])                       # zero-base this run
        d = np.diff(t_s)
        this_step = float(np.median(d)) if d.size else 1.0
        if k == 0:
            offset = 0.0
        else:
            offset = run_end + (step or this_step)
            bounds.append(offset / 60.0)
        tt = t0 + offset
        all_t.append(tt)
        all_v.append(np.asarray(v, float))
        run_end = float(tt[-1])
        step = this_step
    t_all = np.concatenate(all_t)
    v_all = np.concatenate(all_v)
    first = os.path.splitext(parsed[seq[0]][0])[0]
    stem = f"{first}_+{len(parsed) - 1}stitched"
    if bounds:
        st.caption("Runs join at (min): "
                   + ", ".join(f"{b:.2f}" for b in bounds)
                   + f"  ·  total {run_end / 60.0:.2f} min")
    return t_all, v_all, bounds, stem


def _ocp_raw_to_template(t_s, volt, stem, bounds=None):
    """Raw run -> curve preview + fill-in template CSV."""
    st.caption("Raw open-circuit-potential run loaded. Below is your **fill-in "
               "template** — rename each `Conc_n` header to a NaBH₄ "
               "concentration (delete the extra ones you don't need), enter the "
               "volume (µL) in any cell of that column (the row is just a "
               "placeholder — you'll drag it to the real minute later), save as "
               "CSV, and re-upload here for the annotated graph.")
    st.plotly_chart(
        _ocp_fig(t_s / 60.0, volt, [], {}, {}, title=stem, bounds=bounds),
        use_container_width=True, theme="streamlit")
    tmpl = pd.DataFrame({"Time (s)": np.round(t_s, 3),
                         "Time (min)": np.round(t_s / 60.0, 4),
                         "Voltage (V)": volt})
    for k in range(1, 16):
        tmpl[f"Conc_{k}"] = ""
    st.dataframe(tmpl.head(12), use_container_width=True)
    st.download_button(
        "⬇️ Fill-in template (CSV)",
        data=tmpl.to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_OCP_template.csv", mime="text/csv",
        key="dl_ocp_tmpl")


def _ocp_stem(stem):
    """Strip our own OCP filename suffixes so re-exports don't stack them."""
    s = str(stem)
    for suf in ("_OCP_project", "_OCP_annotated", "_OCP_template",
                "_OCP_additions"):
        if s.endswith(suf):
            return s[:-len(suf)]
    return s


def _ocp_project_csv(tmin, volt, real, conc_color, vol_symbol, title, lift, band):
    """Encode the whole annotated graph (curve + markers + settings) as one CSV
    that _ocp_load_project() can read back for zooming / further editing."""
    tmin = np.asarray(tmin, float)
    volt = np.asarray(volt, float)
    skeys = ["nanomeli_ocp_version", "title", "lift_pct", "band_on"]
    svals = ["1", str(title), str(int(lift)), "1" if band else "0"]
    if band:
        skeys += ["band_lo", "band_hi"]
        svals += ["%.6g" % float(band[0]), "%.6g" % float(band[1])]
    M = max(len(tmin), len(real), len(skeys), 1)

    def padf(arr):
        a = list(arr)
        return a + [np.nan] * (M - len(a))

    def pads(vals):
        v = list(vals)
        return v + [""] * (M - len(v))

    data = {
        "Time (s)": padf(np.round(tmin * 60.0, 3)),
        "Time (min)": padf(np.round(tmin, 5)),
        "Voltage (V)": padf(volt),
        "marker_min": pads([round(a["t"], 4) for a in real]),
        "marker_concentration": pads([a["conc"] for a in real]),
        "marker_volume": pads([a["vol"] for a in real]),
        "marker_symbol": pads([vol_symbol[a["vol"]] for a in real]),
        "marker_color": pads([conc_color[a["conc"]] for a in real]),
        "setting_key": pads(skeys),
        "setting_value": pads(svals),
    }
    return pd.DataFrame(data).to_csv(index=False).encode("utf-8")


def _ocp_load_project(df):
    """Parse a re-uploaded annotated-project CSV back into
    (tmin, volt, additions, preset, err). Markers keep their saved order."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    vcol = _pick_col(cols, ["voltage", "potential", "(v)"])
    mcol = _pick_col(cols, ["time (min)", "min"])
    scol = _pick_col(cols, ["time (s)", "(s)", "second"])
    if vcol is None:
        return None, None, None, None, "no voltage / potential column"
    volt = pd.to_numeric(df[vcol], errors="coerce").to_numpy(float)
    if mcol is not None:
        tmin = pd.to_numeric(df[mcol], errors="coerce").to_numpy(float)
    elif scol is not None:
        tmin = pd.to_numeric(df[scol], errors="coerce").to_numpy(float) / 60.0
    else:
        return None, None, None, None, "no time column"
    good = np.isfinite(tmin) & np.isfinite(volt)
    tmin, volt = tmin[good], volt[good]
    if tmin.size < 2:
        return None, None, None, None, "no curve rows"

    mm = pd.to_numeric(df.get("marker_min"), errors="coerce")
    additions, mins, syms = [], [], {}
    for r in range(len(df)):
        if r >= mm.size or pd.isna(mm.iloc[r]):
            continue
        conc = (str(df["marker_concentration"].iloc[r]).strip()
                if "marker_concentration" in cols else "")
        vol = (str(df["marker_volume"].iloc[r]).strip()
               if "marker_volume" in cols else "")
        additions.append({"conc": conc, "vol": vol})
        mins.append(float(mm.iloc[r]))
        if "marker_symbol" in cols:
            sym = str(df["marker_symbol"].iloc[r]).strip()
            if vol and sym and sym.lower() != "nan":
                syms[vol] = sym

    settings = {}
    if "setting_key" in cols:
        sv = df["setting_value"] if "setting_value" in cols else None
        for r in range(len(df)):
            k = str(df["setting_key"].iloc[r]).strip()
            if not k or k.lower() == "nan":
                continue
            settings[k] = "" if sv is None else str(sv.iloc[r]).strip()

    title = settings.get("title", "").strip()
    try:
        lift = int(round(float(settings.get("lift_pct", 4))))
    except (TypeError, ValueError):
        lift = 4
    lift = max(0, min(15, lift))
    band = None
    if str(settings.get("band_on", "0")).strip().lower() in ("1", "true", "yes"):
        try:
            band = (float(settings["band_lo"]), float(settings["band_hi"]))
        except (KeyError, TypeError, ValueError):
            band = None
    preset = {"mins": mins, "syms": syms, "title": title,
              "lift": lift, "band": band}
    return tmin, volt, additions, preset, None


def _ocp_filled(df, stem):
    """Filled template -> annotated drag-to-position potential-vs-time graph."""
    tmin, volt, additions, err = _ocp_template_data(df)
    if err:
        st.error(f"Template problem: {err}.")
        return
    _ocp_render(tmin, volt, additions, stem)


def _ocp_render(tmin, volt, additions, stem, preset=None):
    """Draw the annotated drag-to-position graph. When `preset` is given (a
    reloaded project) its marker minutes, symbols, title, lift and band are
    injected once, so the graph re-opens exactly as it was saved."""
    tmin = np.asarray(tmin, float)
    volt = np.asarray(volt, float)
    st.session_state.setdefault("ocp_title", stem)

    if not additions:
        title0 = st.text_input("Graph title", key="ocp_title")
        st.info("No additions marked yet — put the volume (µL) in a "
                "concentration column (any row), then re-upload. Showing the "
                "raw curve for now.")
        st.plotly_chart(_ocp_fig(tmin, volt, [], {}, {}, title=title0),
                        use_container_width=True, theme="streamlit")
        return

    concs = list(dict.fromkeys(a["conc"] for a in additions))
    vols = sorted(dict.fromkeys(a["vol"] for a in additions),
                  key=lambda v: (_vol_num(v), str(v)))
    palette = _distinct_colors(len(concs))
    conc_color = {c: palette[i] for i, c in enumerate(concs)}

    order = np.argsort(tmin)
    tx, vx = tmin[order], volt[order]
    tlo, thi = float(tx[0]), float(tx[-1])
    tspan = (thi - tlo) or 1.0
    n = len(additions)
    sig = (tuple((a["conc"], a["vol"]) for a in additions),
           round(tlo, 4), round(thi, 4))

    # Re-inject a freshly loaded project's saved state exactly once.
    fresh = False
    if preset is not None:
        psig = (sig, tuple(round(float(m), 4) for m in preset["mins"]),
                tuple(sorted(preset["syms"].items())), preset["title"],
                int(preset["lift"]),
                None if not preset["band"]
                else (round(preset["band"][0], 6), round(preset["band"][1], 6)))
        if st.session_state.get("_ocp_psig") != psig:
            fresh = True
            st.session_state["_ocp_psig"] = psig

    if fresh:
        st.session_state["ocp_title"] = preset["title"] or stem
    title = st.text_input("Graph title", key="ocp_title")

    st.markdown("**Symbol for each volume** (colour is auto-assigned per concentration)")
    scols = st.columns(min(len(vols), 4))
    vol_symbol = {}
    for i, vol in enumerate(vols):
        dk = f"ocpsym_{i}"
        st.session_state.setdefault(dk, OCP_SYMBOLS[i % len(OCP_SYMBOLS)])
        if fresh and preset["syms"].get(vol) in OCP_SYMBOLS:
            st.session_state[dk] = preset["syms"][vol]
        vol_symbol[vol] = scols[i % len(scols)].selectbox(
            str(vol), OCP_SYMBOLS, key=dk)

    vmin, vmax = float(np.nanmin(volt)), float(np.nanmax(volt))
    st.session_state.setdefault("ocp_band_on", False)
    st.session_state.setdefault("ocp_band_lo", round(vmin, 3))
    st.session_state.setdefault("ocp_band_hi", round(vmax, 3))
    st.session_state.setdefault("ocp_lift", 4)
    if fresh:
        st.session_state["ocp_lift"] = int(preset["lift"])
        st.session_state["ocp_band_on"] = bool(preset["band"])
        if preset["band"]:
            st.session_state["ocp_band_lo"] = float(preset["band"][0])
            st.session_state["ocp_band_hi"] = float(preset["band"][1])

    b1, b2, b3 = st.columns([1.1, 1, 1])
    hl = b1.checkbox("Highlight a potential band", key="ocp_band_on")
    ylo = b2.number_input("From (V)", step=0.01, format="%.3f",
                          disabled=not hl, key="ocp_band_lo")
    yhi = b3.number_input("To (V)", step=0.01, format="%.3f",
                          disabled=not hl, key="ocp_band_hi")
    lift = st.slider("Lift markers above the curve (% of span)", 0, 15,
                     key="ocp_lift",
                     help="Floats each symbol above the curve so the line stays "
                          "visible underneath; the marker keeps this height as "
                          "you drag it.")
    off = (lift / 100.0) * (vmax - vmin)

    st.caption("**Drag each marker left/right to the minute the addition was "
               "actually made** — move as many as you like, then click "
               "**✓ Apply moves** (top-left of the chart) to save them; the "
               "table and downloads update then. Use the chart toolbar (or drag "
               "a box over empty space) to **zoom into any region — horizontally "
               "in time and vertically in potential**, then double-click to zoom "
               "back out.")

    if fresh:
        st.session_state["_ocp_mx"] = [float(m) for m in preset["mins"]]
        st.session_state["_ocp_sig"] = sig
        st.session_state["_ocp_seed"] = st.session_state.get("_ocp_seed", 0) + 1
    if st.session_state.get("_ocp_sig") != sig:
        st.session_state["_ocp_sig"] = sig
        st.session_state["_ocp_mx"] = [tlo + tspan * (i + 0.5) / n
                                       for i in range(n)]
        st.session_state["_ocp_seed"] = st.session_state.get("_ocp_seed", 0) + 1
    mx = [float(v) for v in st.session_state["_ocp_mx"]]
    if len(mx) != n:
        mx = [tlo + tspan * (i + 0.5) / n for i in range(n)]
        st.session_state["_ocp_mx"] = mx

    comp = _ocp_component(_OCP_VER)
    ret = comp(
        cx=tx.astype(float).tolist(), cy=vx.astype(float).tolist(), mx=mx,
        col=[conc_color[a["conc"]] for a in additions],
        sym=[vol_symbol[a["vol"]] for a in additions],
        hov=[f"{a['conc']} · {a['vol']}" for a in additions],
        clab=[str(c) for c in concs], ccol=[conc_color[c] for c in concs],
        slab=[str(v) for v in vols], ssym=[vol_symbol[v] for v in vols],
        off=float(off), band=[float(ylo), float(yhi)] if hl else None,
        title=title, seed=int(st.session_state["_ocp_seed"]),
        key="ocp_editor", default=None)

    if isinstance(ret, dict) and ret.get("ts") and \
            ret.get("ts") != st.session_state.get("_ocp_ts"):
        st.session_state["_ocp_ts"] = ret["ts"]
        xs = ret.get("xs")
        if xs and len(xs) == n:
            st.session_state["_ocp_mx"] = [float(x) for x in xs]
            mx = [float(x) for x in xs]

    real = [{"t": mx[i], "v": float(np.interp(mx[i], tx, vx)),
             "conc": a["conc"], "vol": a["vol"]}
            for i, a in enumerate(additions)]
    real_sorted = sorted(real, key=lambda a: a["t"])

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Annotated figure (PNG)",
        data=_ocp_png(tmin, volt, real, conc_color, vol_symbol,
                      title=title, band=(ylo, yhi) if hl else None, offset=off),
        file_name=f"{_ocp_stem(stem)}_OCP_annotated.png", mime="image/png",
        key="dl_ocp_png", use_container_width=True)
    d2.download_button(
        "⬇️ Editable project (CSV)",
        data=_ocp_project_csv(tmin, volt, real, conc_color, vol_symbol,
                              title, lift, (ylo, yhi) if hl else None),
        file_name=f"{_ocp_stem(stem)}_OCP_project.csv", mime="text/csv",
        key="dl_ocp_proj", use_container_width=True,
        help="Re-upload this file any time to zoom around or keep editing this "
             "exact annotated graph.")

    ev = pd.DataFrame([{"Time (min)": round(a["t"], 2),
                        "Potential (V)": round(a["v"], 4),
                        "Concentration": a["conc"], "Volume": a["vol"]}
                       for a in real_sorted])
    st.markdown("**Additions** — minute is where you dragged each marker")
    st.dataframe(ev, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Additions (CSV)",
                       ev.to_csv(index=False).encode("utf-8"),
                       f"{_ocp_stem(stem)}_OCP_additions.csv", "text/csv",
                       key="dl_ocp_ev")


def ocp_analysis(files):
    """Open-circuit potential. One raw run -> a fill-in template; several raw
    runs -> stitched end-to-end into one template; a filled template -> the
    annotated drag-to-position graph."""
    if not isinstance(files, list):
        files = [files]
    files = [f for f in files if f is not None]
    if not files:
        return

    if len(files) > 1:
        t_s, volt, bounds, stem = _ocp_stitch(files)
        if t_s is None:
            return
        _ocp_raw_to_template(t_s, volt, stem, bounds=bounds)
        return

    f = files[0]
    text = _read_text(f)
    stem = os.path.splitext(f.name)[0]
    is_raw = ("[begin data]" in text.lower()
              or "open circuit potential" in text.lower())
    if is_raw:
        t_s, volt = _parse_ocp_raw(text)
        if t_s.size < 2:
            st.error("Couldn't read Time / Voltage rows from this file.")
            return
        _ocp_raw_to_template(t_s, volt, stem)
        return

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:                              # noqa: BLE001
        st.error(f"Couldn't read this as a template CSV: {e}")
        return
    dcols = [str(c).strip() for c in df.columns]
    if "marker_min" in dcols and "setting_key" in dcols:
        tmin, volt, additions, preset, err = _ocp_load_project(df)
        if err:
            st.error(f"Couldn't reload this annotated project: {err}.")
            return
        st.success("Annotated project reloaded — zoom, drag, restyle, or "
                   "re-export it below.")
        _ocp_render(tmin, volt, additions, _ocp_stem(stem), preset=preset)
        return
    _ocp_filled(df, stem)


def _natkey(s):
    """Natural sort key so file_2 sorts before file_10."""
    import re as _re
    return [int(t) if t.isdigit() else t.lower()
            for t in _re.split(r"(\d+)", str(s))]


def _uvvis_timeseries(files):
    """Uploaded UV-Vis files -> (wavelength, spectra[n_spec x n_wl], names, err).
    Several .txt/.csv files = one spectrum each (chronological by filename);
    a single CSV = its columns are the time-series (column 0 = wavelength)."""
    files = [f for f in files if f is not None]
    if not files:
        return None, None, None, "No UV-Vis files uploaded."
    if len(files) == 1 and files[0].name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(files[0])
        except Exception as e:                          # noqa: BLE001
            return None, None, None, f"Couldn't read the UV-Vis CSV: {e}"
        df.columns = [str(c).strip() for c in df.columns]
        if df.shape[1] < 2:
            return (None, None, None,
                    "The UV-Vis CSV needs a wavelength column plus at least one "
                    "spectrum column.")
        wl = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
        cols = list(df.columns[1:])
        mat = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        good = np.isfinite(wl)
        return wl[good], mat[good, :].T, cols, None
    files = sorted(files, key=lambda f: _natkey(f.name))
    dfc, msgs = _combine_raw(files, is_sers=False)
    for lvl, m in msgs:
        getattr(st, lvl)(m)
    if dfc is None or dfc.shape[1] < 2:
        return (None, None, None,
                "Couldn't read any UV-Vis spectra — each file should be one "
                "instrument spectrum (.txt with a 'Begin Spectral Data' block).")
    wl = pd.to_numeric(dfc.iloc[:, 0], errors="coerce").to_numpy(float)
    cols = list(dfc.columns[1:])
    mat = dfc[cols].to_numpy(float)
    good = np.isfinite(wl)
    return wl[good], mat[good, :].T, cols, None


def _downsample_spectra(wl, spectra, maxpts=1100):
    """Thin the wavelength axis for the live viewer (keeps the JSON small and
    the hover snappy). Interpolates every spectrum onto a uniform grid."""
    wl = np.asarray(wl, float)
    if wl.size <= maxpts:
        return wl, spectra
    order = np.argsort(wl)
    wls = wl[order]
    sp = spectra[:, order]
    grid = np.linspace(float(wls[0]), float(wls[-1]), maxpts)
    out = np.vstack([np.interp(grid, wls, sp[i]) for i in range(sp.shape[0])])
    return grid, out


def linked_ocp_uvvis(ocp_files, uvvis_files):
    """Stitch the OCP run(s), place the UV-Vis spectra on that same timeline,
    and show a linked view: hover the potential-vs-time curve to see the
    spectrum recorded at that moment."""
    st.caption("**Hover along the potential-vs-time curve to scrub through the "
               "UV-Vis spectra.** The OCP run(s) are stitched end-to-end and "
               "each spectrum is placed on that timeline; the bottom panel "
               "shows the spectrum recorded nearest the time you're hovering. "
               "The red ticks on the curve mark where spectra were taken.")

    # ---- OCP: one raw run, or several stitched end-to-end ----------------- #
    if len(ocp_files) > 1:
        t_s, volt, _bounds, stem = _ocp_stitch(ocp_files)
        if t_s is None:
            return
    else:
        txt = _read_text(ocp_files[0])
        if not ("[begin data]" in txt.lower()
                or "open circuit potential" in txt.lower()):
            st.error("The OCP file should be the instrument's raw .txt export "
                     "(with its [Begin Data] section).")
            return
        t_s, volt = _parse_ocp_raw(txt)
        if t_s.size < 2:
            st.error("Couldn't read Time / Voltage rows from the OCP file.")
            return
        stem = _ocp_stem(os.path.splitext(ocp_files[0].name)[0])
    tmin = np.asarray(t_s, float) / 60.0
    volt = np.asarray(volt, float)
    order = np.argsort(tmin)
    tmin, volt = tmin[order], volt[order]
    tlo, thi = float(tmin[0]), float(tmin[-1])

    # ---- UV-Vis spectra --------------------------------------------------- #
    wl, spectra, names, err = _uvvis_timeseries(uvvis_files)
    if err:
        st.error(err)
        return
    n_spec = int(spectra.shape[0])
    if n_spec < 1 or wl.size < 2:
        st.error("No usable UV-Vis spectra found.")
        return
    if n_spec == 1:
        st.info("Only one UV-Vis spectrum was found — the bottom panel will "
                "show it at every time. Upload the full set of spectra (one "
                "file per time point) to scrub through them.")

    # ---- place the spectra on the OCP timeline ---------------------------- #
    st.markdown("**Match the spectra to the OCP timeline**")
    c1, c2, c3 = st.columns([1.5, 1, 1])
    timing = c1.radio(
        "Spectrum timing", ["Evenly across the OCP run", "Fixed interval"],
        help="Evenly: the first spectrum sits at the start of the run and the "
             "last at the end. Fixed interval: set the seconds between spectra "
             "and the time of the first one (use this when you know the "
             "acquisition rate).")
    if timing.startswith("Fixed"):
        default_iv = round((thi - tlo) * 60.0 / max(1, n_spec - 1), 1) \
            if n_spec > 1 else 30.0
        start_min = c2.number_input("First spectrum at (min)",
                                    value=round(tlo, 2), step=0.1, format="%.2f")
        interval_s = c3.number_input("Interval between spectra (s)",
                                     min_value=0.1, value=float(default_iv),
                                     step=1.0)
        spec_t = start_min + np.arange(n_spec) * (interval_s / 60.0)
    else:
        spec_t = (np.linspace(tlo, thi, n_spec) if n_spec > 1
                  else np.array([(tlo + thi) / 2.0]))

    outside = int(np.count_nonzero((spec_t < tlo - 1e-9) | (spec_t > thi + 1e-9)))
    if outside:
        st.warning(f"{outside} spectrum time(s) fall outside the OCP run "
                   f"({tlo:.2f}–{thi:.2f} min); when hovering they snap to the "
                   "nearest available spectrum.")

    title = st.text_input("Title", value=stem, key="link_title")

    # thin the wavelength axis for a snappy live view
    wl_v, spectra_v = _downsample_spectra(wl, spectra, 1100)

    comp = _link_component(_LINK_VER)
    comp(ot=tmin.astype(float).tolist(), ov=volt.astype(float).tolist(),
         wl=np.asarray(wl_v, float).tolist(),
         sp=[np.asarray(r, float).tolist() for r in spectra_v],
         spt=[float(t) for t in spec_t], sn=[str(n) for n in names],
         title=title, key="link_view", default=None)

    # ---- time map + downloads -------------------------------------------- #
    tmap = pd.DataFrame({"Spectrum": names, "Time (min)": np.round(spec_t, 3)})
    with st.expander("Spectrum → time map"):
        st.dataframe(tmap, use_container_width=True, hide_index=True)
    d1, d2 = st.columns(2)
    ocp_csv = pd.DataFrame({"Time (s)": np.round(tmin * 60.0, 3),
                            "Time (min)": np.round(tmin, 5),
                            "Voltage (V)": volt})
    d1.download_button("⬇️ Stitched OCP (CSV)",
                       ocp_csv.to_csv(index=False).encode("utf-8"),
                       f"{stem}_OCP.csv", "text/csv", key="dl_link_ocp",
                       use_container_width=True)
    d2.download_button("⬇️ Spectrum time-map (CSV)",
                       tmap.to_csv(index=False).encode("utf-8"),
                       f"{stem}_spectrum_times.csv", "text/csv",
                       key="dl_link_map", use_container_width=True)


# --------------------------------------------------------------------------- #
#  Sidebar controls
# --------------------------------------------------------------------------- #
def sidebar_controls():
    sb = st.sidebar
    sb.header("Data")
    source = sb.radio(
        "Input", ["Single file", "Combine raw files", "Kinetics trend (OceanView)",
                  "Potential vs time (OCP)", "OCP + UV-Vis (linked)"],
        help="“Combine raw files” merges many raw instrument exports of the "
             "same format into one wavelength + many-series dataset, ready for "
             "the analyses below.")
    up = files = raw_type = None
    if source == "OCP + UV-Vis (linked)":
        up = sb.file_uploader(
            "OCP run(s) — raw .txt", type=["txt"],
            accept_multiple_files=True, key="link_ocp")
        files = sb.file_uploader(
            "UV-Vis spectra — .txt (one per time point) or a CSV",
            type=["txt", "csv"], accept_multiple_files=True, key="link_uv")
        sb.caption("Upload the OCP run(s) and the UV-Vis spectra taken "
                   "during the same experiment. The runs are stitched and "
                   "the spectra are dropped onto that timeline — hover the "
                   "curve to scrub the spectrum at each moment.")
        return up, files, raw_type, source, "Linked", None
    if source == "Potential vs time (OCP)":
        up = sb.file_uploader(
            "OCP file(s) — raw .txt run(s), or your filled-in template .csv",
            type=["txt", "csv"], accept_multiple_files=True)
        sb.caption("One raw run → a fill-in template. Several raw runs "
                   "→ order them and they're stitched end-to-end into "
                   "one template. A filled template → the annotated graph. "
                   "A saved **project** CSV → re-opens that graph to zoom "
                   "or keep editing.")
        return up, files, raw_type, source, "OCP", None
    if source == "Kinetics trend (OceanView)":
        files = sb.file_uploader(
            "OceanView trend files (.txt)", type=["txt"],
            accept_multiple_files=True)
        sb.caption("All the auto-saved strip-chart files; they're grouped "
                   "by channel and stitched into one continuous trace.")
        return up, files, raw_type, source, "Kinetics", None
    if source == "Combine raw files":
        raw_type = sb.radio(
            "Raw file format", ["SERS · .csv", "UV-Vis · .txt"],
            help="SERS: a CSV with Wavelength + Intensity columns. "
                 "UV-Vis: an instrument .txt (header, then wavelength/intensity).")
        files = sb.file_uploader("Raw files (up to ~25)", type=["csv", "txt"],
                                 accept_multiple_files=True)
        sb.caption("x-axis comes from the first file; each series is named after "
                   "its source file.")
        analysis = ("UV-Vis · real-time (~850 nm band)"
                    if raw_type.startswith("UV-Vis") else "SERS / Raman")
    else:
        up = sb.file_uploader("Spectra CSV", type=["csv", "txt"])
        sb.caption("First column = x-axis; every other column = a spectrum, "
                   "replicate, or time point.")
        analysis = sb.radio(
            "Analysis", ["SERS / Raman", "UV-Vis · real-time (~850 nm band)"],
            help="SERS / Raman: despike, baseline, FFT denoise, peak integration. "
                 "UV-Vis: focus on the broad ~850 nm band, then normalize or "
                 "broadly background-subtract.")

    if analysis.startswith("UV-Vis"):
        if source == "Single file":
            sb.caption("UV-Vis options appear on the main panel →")
        return up, files, raw_type, source, analysis, None

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
    return up, files, raw_type, source, analysis, params


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def _home_guide():
    """Themed 'how to use' landing page (shown before any data is loaded)."""
    _html = """<style>
    .nm-guide{font-family:'Space Grotesk',system-ui,sans-serif;color:#1A1620;margin-top:.3rem}
    .nm-guide .lead{font-size:1.06rem;color:#4A4453;max-width:780px;line-height:1.55;margin:.2rem 0 0}
    .nm-bar{height:3px;border-radius:3px;background:linear-gradient(90deg,#C42348,#C01C8E,#7B2FB0,#4A35C4,#2456C8);margin:1.1rem 0 1.3rem;opacity:.92}
    .nm-sec{font-weight:700;letter-spacing:.13em;text-transform:uppercase;font-size:.78rem;color:#7B2FB0;margin:1.5rem 0 .7rem}
    .nm-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(252px,1fr));gap:14px}
    .nm-card{background:#fff;border:1px solid #E9E2F0;border-left:4px solid #7B2FB0;border-radius:12px;padding:14px 16px;box-shadow:0 1px 2px rgba(26,22,40,.04)}
    .nm-card h4{margin:0 0 .35rem;font-size:1.03rem;font-weight:700;color:#1A1620}
    .nm-card p{margin:0;font-size:.9rem;color:#6B6573;line-height:1.5}
    .nm-card ul{margin:.45rem 0 0;padding-left:1.05rem}
    .nm-card li{font-size:.87rem;color:#4A4453;line-height:1.5;margin:.22rem 0}
    .nm-card b{color:#1A1620}
    .nm-step{display:flex;gap:12px;align-items:flex-start;margin:.45rem 0}
    .nm-num{flex:none;width:26px;height:26px;border-radius:50%;background:linear-gradient(135deg,#C42348,#7B2FB0);color:#fff;font-weight:700;font-size:.85rem;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(123,47,176,.3)}
    .nm-step div{font-size:.94rem;color:#4A4453;line-height:1.5}
    .nm-step b{color:#1A1620}
    .nm-fmt{font-family:'Space Mono',monospace;font-size:.82rem;background:#F7F4FA;border:1px solid #ECE6F1;border-radius:8px;padding:10px 13px;color:#5F5E5A;line-height:1.7;margin-top:.45rem}
    </style>
    <div class="nm-guide">
      <p class="lead">A toolkit for SERS / Raman and UV-Vis spectra — clean and integrate peaks,
      compare band shapes, batch-combine raw instrument files, and follow reactions in real time.</p>
      <div class="nm-bar"></div>

      <div class="nm-sec">How it flows</div>
      <div class="nm-step"><div class="nm-num">1</div><div><b>Choose your input</b> in the sidebar (Data &rarr; Input): a single CSV, a batch of raw files to combine, or OceanView kinetics files.</div></div>
      <div class="nm-step"><div class="nm-num">2</div><div><b>Pick an analysis</b> — SERS / Raman or UV-Vis — then tune it on the main panel.</div></div>
      <div class="nm-step"><div class="nm-num">3</div><div><b>Adjust &amp; export</b> — drag, zoom and fine-tune, then download CSVs and figures.</div></div>

      <div class="nm-sec">The three tracks</div>
      <div class="nm-grid">
        <div class="nm-card" style="border-left-color:#C42348">
          <h4>SERS / Raman</h4>
          <p>Clean noisy spectra, then integrate peaks with reproducibility stats.</p>
          <ul>
            <li><b>Clean &amp; preprocess</b> — despike cosmic rays, resample, baseline-correct, FFT denoise.</li>
            <li><b>Peak integration</b> — auto-detect bands, then drag boxes on a live plot to set windows.</li>
            <li>Per-peak areas, mean &plusmn; SD and <b>%RSD</b> across replicates; combined CSV + annotated PNG.</li>
          </ul>
        </div>
        <div class="nm-card" style="border-left-color:#7B2FB0">
          <h4>UV-Vis &middot; LSPR band</h4>
          <p>Focus the broad ~850&nbsp;nm band and compare shapes across a run.</p>
          <ul>
            <li><b>Normalize</b> each trace's maximum to 1 to compare widths / shapes, or</li>
            <li><b>Subtract a very broad background</b> while preserving the broad peak.</li>
            <li><b>Peak summary</b> — wavelength, height and <b>FWHM</b> for every series.</li>
          </ul>
        </div>
        <div class="nm-card" style="border-left-color:#2456C8">
          <h4>Real-time kinetics</h4>
          <p>Turn OceanView strip-charts into one continuous story.</p>
          <ul>
            <li>Drop in the auto-saved <b>.txt</b> files; they're <b>stitched</b> into one absorbance-vs-time trace.</li>
            <li>Sharp <b>additions are auto-detected and time-stamped</b>.</li>
            <li>Label channels, optional baseline subtraction; export the trend + the jump list.</li>
          </ul>
        </div>
      </div>

      <div class="nm-sec">Handy tools</div>
      <div class="nm-grid">
        <div class="nm-card" style="border-left-color:#C01C8E">
          <h4>Batch combine</h4>
          <p>Many raw files &rarr; one tidy dataset. Pick SERS (.csv) or UV-Vis (.txt) and drop in up to ~25 files; the wavelength axis comes from the first and each series is named after its file — download it, or pipe it straight into an analysis.</p>
        </div>
        <div class="nm-card" style="border-left-color:#4A35C4">
          <h4>The integration editor</h4>
          <p>Drag a band's <b>edge</b> to resize or its <b>body</b> to move — a <b>zoom loupe</b> pops up for precise placement. Use the toolbar to <b>draw</b> a new window or <b>zoom</b> the spectrum, the <b>eraser</b> to delete one, or type <b>exact bounds</b>. The areas table updates live.</p>
        </div>
      </div>

      <div class="nm-sec">Single-file format</div>
      <p class="lead" style="font-size:.92rem">First column = x-axis (wavelength / Raman shift); every other column is a spectrum, replicate, or time point.</p>
      <div class="nm-fmt">Wavelength,S1,S2,S3,S4<br>139.19,2368,1547,1492,1578<br>141.24,2344,1549,1502,1569<br>&hellip;</div>
    </div>"""
    st.markdown("\n".join(_l.lstrip() for _l in _html.splitlines()),
                unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Nanomeli · Spectra toolkit", page_icon="🧪",
                       layout="wide")
    theme.render_header()

    up, files, raw_type, source, analysis, params = sidebar_controls()

    if source == "OCP + UV-Vis (linked)":
        ocp_files = up or []
        uv_files = files or []
        if not ocp_files or not uv_files:
            st.info("⬅️ Upload BOTH an OCP run (.txt) and the UV-Vis "
                    "spectra (.txt files, one per time point, or a CSV) in "
                    "the sidebar to link them on one timeline.")
            return
        linked_ocp_uvvis(ocp_files, uv_files)
        return

    if source == "Potential vs time (OCP)":
        ocp_files = up if isinstance(up, list) else ([up] if up else [])
        if not ocp_files:
            st.info("⬅️ Upload your open-circuit-potential file(s) in the "
                    "sidebar — one raw run (.txt) for a template, several "
                    "raw runs to stitch, or your filled-in template (.csv) "
                    "for the annotated graph.")
            return
        ocp_analysis(ocp_files)
        return

    if source == "Kinetics trend (OceanView)":
        if not files:
            st.info("⬅️ Upload your OceanView strip-chart files (.txt) in "
                    "the sidebar to plot the time-trend and time-stamp the "
                    "jumps.")
            return
        kinetics_analysis(files)
        return

    if source == "Combine raw files":
        if not files:
            st.info("⬅️ Upload your raw instrument files in the sidebar "
                    "to combine them into one dataset.")
            st.markdown("**Raw formats**")
            st.markdown("- **SERS** — a CSV with `Wavelength` and `Intensity` "
                        "columns (extra columns are ignored).")
            st.markdown("- **UV-Vis** — an instrument `.txt`: a header block, "
                        "then `Begin Spectral Data`, then wavelength/intensity.")
            return
        is_sers = not raw_type.startswith("UV-Vis")
        df, _msgs = combine_uploaded(files, is_sers)
        for _lvl, _m in _msgs:
            getattr(st, _lvl)(_m)
        if df is None or df.shape[1] < 2:
            st.error("Couldn't extract any spectra — check the files match the "
                     "selected raw format.")
            return
        src_name = "combined_" + ("SERS" if is_sers else "UVVis")
        st.success(f"Combined **{df.shape[1] - 1}** file(s) onto a shared "
                   f"{df.columns[0]} axis · {len(df):,} points.")
        st.download_button(
            "⬇️ Combined raw data (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{src_name}.csv", mime="text/csv", key="dl_combined")
        st.dataframe(df.head(12), use_container_width=True)
        st.caption("This combined dataset now feeds the analysis below — "
                   "adjust the cleaning steps (SERS) or focus region (UV-Vis) "
                   "as usual.")
        data_id = "combine:" + str(tuple((f.name, f.size) for f in files))
    else:
        if up is None:
            _home_guide()
            return
        try:
            df = pd.read_csv(up)
        except Exception as e:                      # noqa: BLE001
            st.error(f"Could not read the CSV: {e}")
            return
        src_name = os.path.splitext(up.name)[0]
        data_id = up.name

    # Reset peak state when the active dataset changes.
    if st.session_state.get("_data_id") != data_id:
        st.session_state._data_id = data_id
        for k in ("peaks", "canvas_ver"):
            st.session_state.pop(k, None)

    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    if len(cols) < 2:
        st.error("The data needs at least two columns (x-axis + one spectrum).")
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
        uvvis_analysis(x, Y, names, x_col, src_name)
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
            file_name=f"{src_name}_cleaned.csv",
            mime="text/csv", key="dl_cleaned_only")

    with tab_integrate:
        integration_tab(x_work, Y_clean, names, x_col, params, src_name)


if __name__ == "__main__":
    main()
