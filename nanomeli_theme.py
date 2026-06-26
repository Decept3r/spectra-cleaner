"""
Nanomeli plasmonic theme for the SERS / Raman spectra toolkit.

Pure presentation -- no science here. Entry points:

  apply_matplotlib_theme()  dark "instrument" neon theme for the seaborn /
                            matplotlib diagnostic plots, incl. a soft glow on
                            every line. Call once, at import time.
  render_header()           injects the global CSS + the animated
                            plasmonic-coupling header banner. Call first inside
                            main().
  neon_spectrum_fig(...)    a glowing Plotly "live spectrum" for the stepwise
                            Clean pipeline.
  step_chip(...) / loading_bar(...)  small HTML widgets for the pipeline UI.

Design language borrows the lab's physics: a dispersed gold colloid glows ruby
and red-shifts through magenta / violet to blue as nanoparticles couple at an
interface. That red -> blue ramp is the whole UI's accent + data-viz language.
"""

import streamlit as st
import streamlit.components.v1 as components

# Plasmon ramp (light-UI) and its neon (dark-plot) cousins
RUBY, MAGENTA, VIOLET, INDIGO, BLUE = (
    "#C42348", "#C01C8E", "#7B2FB0", "#4A35C4", "#2456C8")
NEON = ["#FF4D6D", "#FF5FC4", "#B06CFF", "#6E8BFF", "#4D8BFF"]

_PATCHED = False


GLOBAL_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

@keyframes nmScan { 0%{left:-42%} 100%{left:104%} }
@keyframes nmShimmer { 0%{background-position:0% 50%} 100%{background-position:200% 50%} }

html, body, .stApp, [data-testid="stAppViewContainer"],
section[data-testid="stSidebar"], [data-baseweb] {
  font-family: 'Space Grotesk', system-ui, sans-serif;
}
.stApp {
  background:
    radial-gradient(1200px 460px at 88% -12%, rgba(123,47,176,0.09), transparent 60%),
    radial-gradient(960px 380px at -6% 2%, rgba(196,35,72,0.06), transparent 58%),
    #FBFAFC;
}
h1, h2, h3, h4 { font-family:'Space Grotesk', sans-serif; letter-spacing:-0.01em; color:#1A1620; }
button, input, select, textarea { font-family:'Space Grotesk', sans-serif; }
code, pre, kbd, samp, [data-testid="stMetricValue"] { font-family:'Space Mono', monospace; }
[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.1rem; padding-bottom:3rem; max-width:1500px; }

/* gradient dividers */
hr { height:2px !important; border:none !important; opacity:.85;
  background:linear-gradient(90deg,#C42348,#7B2FB0,#2456C8) !important; }

/* main-content subheaders get a plasmon marker */
.block-container h3::before {
  content:""; display:inline-block; width:10px; height:10px; border-radius:2px;
  margin-right:11px; vertical-align:middle;
  background:linear-gradient(135deg,#C42348,#7B2FB0);
  box-shadow:0 0 10px rgba(123,47,176,.5);
}

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:1.7rem; border-bottom:1px solid #ECE6F1; }
.stTabs [data-baseweb="tab"] { font-weight:600; color:#9A92A6; }
.stTabs [aria-selected="true"] { color:#1A1620; }
.stTabs [data-baseweb="tab-highlight"] {
  background:linear-gradient(90deg,#C42348,#C01C8E,#7B2FB0,#4A35C4,#2456C8);
  height:3px; border-radius:3px;
}

/* sidebar */
section[data-testid="stSidebar"] { background:#F3EFF8; border-right:1px solid #E9E2F0; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 {
  font-weight:700; text-transform:uppercase; letter-spacing:.14em;
  font-size:0.82rem; color:#7B2FB0;
}

/* the four cleaning-step expanders */
[data-testid="stExpander"] {
  border:1px solid #E9E2F0; border-left:3px solid #7B2FB0; border-radius:12px;
  background:#fff; box-shadow:0 1px 2px rgba(26,22,40,.04); overflow:hidden;
}
[data-testid="stExpander"] summary { font-weight:600; }
[data-testid="stExpander"] summary:hover { color:#7B2FB0; }

/* buttons -- bold. primary + download = full gradient; rest = violet tonal */
.stButton button {
  background:linear-gradient(180deg,#FBF4FD,#F2E7FB); color:#7B2FB0;
  border:1px solid #E2CDEE; border-radius:10px; font-weight:600;
  transition:all .15s ease;
}
.stButton button:hover {
  background:linear-gradient(120deg,#C42348,#7B2FB0); color:#fff;
  border-color:transparent; box-shadow:0 4px 14px rgba(123,47,176,.30);
}
.stButton button[kind="primary"],
[data-testid="baseButton-primary"], [data-testid="stBaseButton-primary"],
.stDownloadButton button {
  background:linear-gradient(120deg,#C42348,#9C2A9E 55%,#4A35C4) !important;
  color:#fff !important; border:none !important; border-radius:10px !important;
  font-weight:700 !important; letter-spacing:.01em;
  box-shadow:0 5px 16px rgba(123,47,176,.30) !important;
}
.stButton button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover, [data-testid="stBaseButton-primary"]:hover,
.stDownloadButton button:hover {
  filter:brightness(1.07); box-shadow:0 8px 22px rgba(123,47,176,.40) !important;
}
.stButton button:active, .stDownloadButton button:active { transform:translateY(1px); }

/* file uploader dropzone */
[data-testid="stFileUploaderDropzone"] {
  border:1.6px dashed #C49AD9; border-radius:12px;
  background:linear-gradient(180deg,#FBF4FD,#F3EAFA);
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color:#7B2FB0; background:linear-gradient(180deg,#F8EDFB,#EEE3F7);
}

/* metric cards */
[data-testid="stMetric"] {
  background:linear-gradient(180deg,#FFFFFF,#FBF5FC);
  border:1px solid #ECD9F0; border-left:4px solid #7B2FB0;
  border-radius:12px; padding:0.6rem 1rem;
  box-shadow:0 1px 3px rgba(123,47,176,.06);
}
[data-testid="stMetricValue"] { color:#1A1620; }
[data-testid="stMetricLabel"] { color:#6B6573; }

/* slider thumb halo (track fill comes from primaryColor) */
[data-testid="stSlider"] [role="slider"] { box-shadow:0 0 0 4px rgba(123,47,176,.18); }

/* neon plot panels (dark instrument look) framed nicely on the light page */
[data-testid="stPlotlyChart"] {
  border-radius:14px; overflow:hidden;
  box-shadow:0 10px 34px rgba(40,18,60,.22); border:1px solid #241B30;
}
[data-testid="stImage"] img {
  border-radius:12px; box-shadow:0 8px 26px rgba(40,18,60,.16);
}

/* dataframe + alerts + code */
[data-testid="stDataFrame"] { border:1px solid #ECE6F1; border-radius:12px; }
[data-testid="stNotification"], .stAlert { border-radius:12px; }
.stCode, pre { border-radius:10px !important; }
</style>"""


HERO_HTML = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box} html,body{margin:0;background:transparent}
.hero{position:relative;height:150px;border-radius:16px;overflow:hidden;
  background:linear-gradient(110deg,#FBEFF3,#F4EEF9 45%,#ECEFFB);
  border:1px solid #E9E2F0;font-family:'Space Grotesk',system-ui,sans-serif}
#f{position:absolute;inset:0;width:100%;height:100%}
.row{position:absolute;inset:0;display:flex;align-items:center;gap:20px;padding:0 30px}
.orb{position:relative;width:56px;height:56px;flex:none}
.orb b{position:absolute;inset:0;border-radius:50%;
  background:radial-gradient(circle at 36% 32%,#F0789A,#C42348 38%,#7B2FB0 78%,#2456C8);
  box-shadow:0 0 28px rgba(123,47,176,.5);animation:pp 3.6s ease-in-out infinite}
.orb i{position:absolute;left:14px;top:12px;width:15px;height:11px;border-radius:50%;
  background:rgba(255,255,255,.82);filter:blur(2px)}
.wm{font-weight:700;font-size:32px;letter-spacing:.17em;color:#1A1620;line-height:1}
.wm sup{color:#C8A24C;font-size:13px;letter-spacing:0;vertical-align:super}
.sub{font-size:12.5px;color:#6B6573;margin-top:7px;letter-spacing:.05em;font-weight:500}
.scan{position:absolute;left:0;bottom:0;height:3px;width:34%;
  background:linear-gradient(90deg,#C42348,#7B2FB0,#2456C8);
  box-shadow:0 0 12px rgba(123,47,176,.7);animation:sweep 2.8s ease-in-out infinite}
@keyframes pp{0%,100%{transform:scale(1)}50%{transform:scale(1.06)}}
@keyframes sweep{0%{left:-34%}100%{left:100%}}
</style></head><body>
<div class="hero">
  <canvas id="f"></canvas>
  <div class="row">
    <div class="orb"><b></b><i></i></div>
    <div>
      <div class="wm">NANOMELI<sup>Au</sup></div>
      <div class="sub">SERS / RAMAN SPECTRA TOOLKIT &middot; MOUNT ALLISON UNIVERSITY</div>
    </div>
  </div>
  <div class="scan"></div>
</div>
<script>
(function(){
  var cv=document.getElementById('f'),ctx=cv.getContext('2d');
  var ramp=['#C42348','#C01C8E','#7B2FB0','#4A35C4','#2456C8'];
  var dpr=Math.min(window.devicePixelRatio||1,2),P=[];
  function size(){var r=cv.getBoundingClientRect();cv.width=Math.max(4,r.width*dpr);cv.height=Math.max(4,r.height*dpr);}
  size();window.addEventListener('resize',size);
  var n=Math.max(16,Math.min(48,Math.round(cv.width*cv.height/24000)));
  for(var i=0;i<n;i++){P.push({x:Math.random(),y:Math.random(),vx:(Math.random()-0.5)*0.0009,vy:(Math.random()-0.5)*0.0009,r:5+Math.random()*9,h:Math.floor(Math.random()*ramp.length)});}
  function step(){
    var W=cv.width,H=cv.height;ctx.clearRect(0,0,W,H);
    for(var i=0;i<P.length;i++){var p=P[i];p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>1)p.vx*=-1;if(p.y<0||p.y>1)p.vy*=-1;p.x=Math.max(0,Math.min(1,p.x));p.y=Math.max(0,Math.min(1,p.y));}
    var thr=Math.min(W,H)*0.34,coup={};
    for(var i=0;i<P.length;i++)for(var j=i+1;j<P.length;j++){
      var a=P[i],b=P[j],dx=(a.x-b.x)*W,dy=(a.y-b.y)*H,d=Math.sqrt(dx*dx+dy*dy);
      if(d<thr){coup[i]=1;coup[j]=1;var g=ctx.createLinearGradient(a.x*W,a.y*H,b.x*W,b.y*H);g.addColorStop(0,ramp[a.h]);g.addColorStop(1,ramp[b.h]);ctx.strokeStyle=g;ctx.globalAlpha=(1-d/thr)*0.45;ctx.lineWidth=dpr;ctx.beginPath();ctx.moveTo(a.x*W,a.y*H);ctx.lineTo(b.x*W,b.y*H);ctx.stroke();}
    }
    ctx.globalAlpha=1;
    for(var i=0;i<P.length;i++){var p=P[i],hue=coup[i]?Math.min(ramp.length-1,p.h+2):p.h,c=ramp[hue],cx=p.x*W,cy=p.y*H,R=p.r*dpr;
      var rg=ctx.createRadialGradient(cx,cy,0,cx,cy,R*2.4);rg.addColorStop(0,c+'cc');rg.addColorStop(0.45,c+'4d');rg.addColorStop(1,c+'00');
      ctx.fillStyle=rg;ctx.beginPath();ctx.arc(cx,cy,R*2.4,0,6.2832);ctx.fill();
      ctx.fillStyle=c;ctx.beginPath();ctx.arc(cx,cy,Math.max(1,R*0.5),0,6.2832);ctx.fill();}
    requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
})();
</script>
</body></html>"""


def apply_matplotlib_theme():
    """Dark 'instrument' neon theme + a soft glow on every plotted line."""
    global _PATCHED
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        from matplotlib.axes import Axes
        import fft_denoise as fd

        # neon line roles
        fd.C_NOISY = "#6E6480"   # raw / muted
        fd.C_CLEAN = "#FF4D6D"   # cleaned signal -- neon ruby
        fd.C_SPEC = "#6E6480"    # full power spectrum -- muted
        fd.C_KEPT = "#B06CFF"    # kept points / spike marks / fills -- neon violet

        try:
            import seaborn as sns
            sns.set_style("dark")
        except Exception:
            pass

        ink = "#15101D"
        plt.rcParams.update({
            "figure.facecolor": ink,
            "axes.facecolor": ink,
            "savefig.facecolor": ink,
            "savefig.edgecolor": ink,
            "axes.edgecolor": "#3A2F4A",
            "axes.labelcolor": "#C9BFD9",
            "axes.titlecolor": "#F3EEF8",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.grid": True,
            "grid.color": "#271E33",
            "grid.linewidth": 0.9,
            "xtick.color": "#9A8FB0",
            "ytick.color": "#9A8FB0",
            "text.color": "#E8E2F0",
            "font.size": 11,
            "lines.linewidth": 1.6,
            "legend.frameon": True,
            "legend.facecolor": "#1E1726",
            "legend.edgecolor": "#3A2F4A",
            "legend.labelcolor": "#E8E2F0",
        })

        if not _PATCHED:
            _orig_plot = Axes.plot

            def _glow_plot(self, *args, **kwargs):
                lines = _orig_plot(self, *args, **kwargs)
                try:
                    for ln in lines:
                        lw = ln.get_linewidth() or 1.5
                        ln.set_path_effects([
                            pe.Stroke(linewidth=lw * 3.2,
                                      foreground=ln.get_color(), alpha=0.16),
                            pe.Stroke(linewidth=lw * 1.7,
                                      foreground=ln.get_color(), alpha=0.22),
                            pe.Normal()])
                except Exception:
                    pass
                return lines

            Axes.plot = _glow_plot
            _PATCHED = True
    except Exception:
        pass


def render_header():
    """Inject the plasmonic CSS and the animated coupling-field banner."""
    try:
        st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    except Exception:
        pass
    try:
        components.html(HERO_HTML, height=158)
    except Exception:
        pass


def neon_spectrum_fig(x, Y, names, stage=None):
    """A glowing Plotly 'live spectrum' on dark, for the stepwise pipeline."""
    import numpy as _np
    import plotly.graph_objects as go

    Y = _np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    xx = x if x is not None else _np.arange(Y.shape[0])
    fig = go.Figure()
    k = Y.shape[1]
    for j in range(k):
        col = NEON[j % len(NEON)]
        yj = Y[:, j]
        # glow underlay
        fig.add_trace(go.Scatter(
            x=xx, y=yj, mode="lines", line=dict(width=7, color=col),
            opacity=0.18, hoverinfo="skip", showlegend=False))
        # crisp line
        fig.add_trace(go.Scatter(
            x=xx, y=yj, mode="lines",
            name=(names[j] if j < len(names) else f"col {j}"),
            line=dict(width=1.9, color=col),
            hovertemplate="%{x:.0f} cm⁻¹ · %{y:.3g}<extra></extra>"))
    fig.update_layout(
        paper_bgcolor="#15101D", plot_bgcolor="#15101D",
        font=dict(color="#CFC6DD", family="Space Grotesk, system-ui, sans-serif"),
        margin=dict(l=58, r=16, t=14, b=42), height=360, hovermode="x unified",
        showlegend=(k > 1),
        legend=dict(orientation="h", y=1.04, yanchor="bottom", x=0,
                    font=dict(color="#CFC6DD")),
        xaxis=dict(title="Raman shift (cm⁻¹)", gridcolor="#271E33",
                   zeroline=False, color="#9A8FB0"),
        yaxis=dict(title="intensity (a.u.)", gridcolor="#271E33",
                   zeroline=False, color="#9A8FB0"))
    return fig


def step_chip(label, state, enabled=True):
    """Small status chip for the pipeline stepper."""
    if state == "active":
        css = ("color:#fff;border-color:transparent;"
               "background:linear-gradient(120deg,#C42348,#7B2FB0);"
               "box-shadow:0 4px 12px rgba(123,47,176,.32)")
        dot = "●"
    elif state == "applied":
        css = "color:#1A1620;border-color:#CDB6E0;background:#fff"
        dot = "✓"
    elif state == "skipped":
        css = "color:#B0A8BC;border-color:#E6E0ED;background:#F7F4FA"
        dot = "–"
    else:
        css = "color:#9A92A6;border-color:#E6E0ED;background:#F4F0F8"
        dot = "○"
    note = "" if enabled or state in ("applied", "skipped") else " · off"
    return (f'<span style="display:inline-flex;align-items:center;gap:7px;'
            f'font:600 12px \'Space Grotesk\';padding:7px 13px;border-radius:20px;'
            f'border:1px solid;{css}">{dot} {label}{note}</span>')


def loading_bar(label="Processing"):
    """An animated indeterminate plasmon scan bar (HTML for st.markdown)."""
    return (
        f'<div style="margin:8px 0 5px;font:600 11px \'Space Grotesk\';'
        f'letter-spacing:.16em;text-transform:uppercase;color:#7B2FB0">'
        f'{label}&hellip;</div>'
        f'<div style="position:relative;height:9px;border-radius:9px;'
        f'background:#EEE9F3;overflow:hidden">'
        f'<div style="position:absolute;top:0;bottom:0;width:38%;border-radius:9px;'
        f'background:linear-gradient(90deg,#C42348,#7B2FB0,#2456C8);'
        f'box-shadow:0 0 12px rgba(123,47,176,.6);'
        f'animation:nmScan 1.15s ease-in-out infinite"></div></div>')
