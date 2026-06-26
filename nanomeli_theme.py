"""
Nanomeli plasmonic theme for the SERS / Raman spectra toolkit.

Pure presentation -- no science here. Two entry points:

  apply_matplotlib_theme()  recolours the seaborn/matplotlib diagnostic plots
                            (despike, baseline, FFT, integration) to the
                            plasmon palette. Call once, at import time.
  render_header()           injects the global CSS and the animated
                            plasmonic-coupling header banner. Call once, first
                            thing inside main().

Design language borrows the lab's physics: a dispersed gold colloid glows ruby
and red-shifts through magenta / violet to blue as nanoparticles couple at an
interface. That red -> blue ramp is the whole UI's accent + data-viz language.
"""

import streamlit as st
import streamlit.components.v1 as components

# Plasmon ramp
RUBY, MAGENTA, VIOLET, INDIGO, BLUE = (
    "#C42348", "#C01C8E", "#7B2FB0", "#4A35C4", "#2456C8")


GLOBAL_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"],
section[data-testid="stSidebar"], [data-baseweb] {
  font-family: 'Space Grotesk', system-ui, sans-serif;
}
.stApp {
  background:
    radial-gradient(1100px 420px at 88% -12%, rgba(123,47,176,0.07), transparent 60%),
    radial-gradient(900px 360px at -6% 2%, rgba(196,35,72,0.05), transparent 58%),
    #FBFAFC;
}
h1, h2, h3, h4 { font-family:'Space Grotesk', sans-serif; letter-spacing:-0.01em; color:#1A1620; }
button, input, select, textarea { font-family:'Space Grotesk', sans-serif; }
code, pre, kbd, samp, [data-testid="stMetricValue"] { font-family:'Space Mono', monospace; }
[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.1rem; padding-bottom:3rem; max-width:1500px; }

/* main-content subheaders get a small plasmon marker */
.block-container h3::before {
  content:""; display:inline-block; width:9px; height:9px; border-radius:2px;
  margin-right:11px; vertical-align:middle;
  background:linear-gradient(135deg,#C42348,#7B2FB0);
}

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:1.6rem; border-bottom:1px solid #ECE6F1; }
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
  border:1px solid #E9E2F0; border-radius:12px; background:#fff;
  box-shadow:0 1px 2px rgba(26,22,40,.04); overflow:hidden;
}
[data-testid="stExpander"] summary { font-weight:600; }
[data-testid="stExpander"] summary:hover { color:#7B2FB0; }

/* download buttons = gradient CTA */
.stDownloadButton > button {
  background:linear-gradient(120deg,#C42348,#9C2A9E 55%,#4A35C4) !important;
  color:#fff !important; border:none !important; border-radius:9px !important;
  font-weight:600 !important; box-shadow:0 4px 14px rgba(123,47,176,.28) !important;
}
.stDownloadButton > button:hover {
  filter:brightness(1.06); box-shadow:0 7px 20px rgba(123,47,176,.36) !important;
}
/* secondary buttons = plasmon outline that fills on hover */
.stButton > button {
  background:#fff; color:#7B2FB0; border:1.5px solid #D9C6E8;
  border-radius:9px; font-weight:600; transition:all .15s ease;
}
.stButton > button:hover {
  background:linear-gradient(120deg,#C42348,#7B2FB0); color:#fff;
  border-color:transparent; box-shadow:0 4px 14px rgba(123,47,176,.28);
}
.stButton > button:active { transform:translateY(1px); }

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
.wm{font-weight:700;font-size:31px;letter-spacing:.16em;color:#1A1620;line-height:1}
.wm sup{color:#C8A24C;font-size:13px;letter-spacing:0;vertical-align:super}
.sub{font-size:12.5px;color:#6B6573;margin-top:7px;letter-spacing:.04em;font-weight:500}
@keyframes pp{0%,100%{transform:scale(1)}50%{transform:scale(1.06)}}
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
</div>
<script>
(function(){
  var cv=document.getElementById('f'),ctx=cv.getContext('2d');
  var ramp=['#C42348','#C01C8E','#7B2FB0','#4A35C4','#2456C8'];
  var dpr=Math.min(window.devicePixelRatio||1,2),P=[];
  function size(){var r=cv.getBoundingClientRect();cv.width=Math.max(4,r.width*dpr);cv.height=Math.max(4,r.height*dpr);}
  size();window.addEventListener('resize',size);
  var n=Math.max(14,Math.min(46,Math.round(cv.width*cv.height/26000)));
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
    """Recolour the seaborn/matplotlib diagnostic plots to the plasmon palette."""
    try:
        import matplotlib.pyplot as plt
        import fft_denoise as fd
        fd.C_NOISY = "#B7B0C2"   # raw / muted lilac-grey
        fd.C_CLEAN = "#C42348"   # cleaned signal -- ruby
        fd.C_SPEC = "#B7B0C2"    # full power spectrum -- muted
        fd.C_KEPT = "#7B2FB0"    # kept points / spike marks / fills -- violet
        try:
            import seaborn as sns
            sns.set_theme(style="white", context="notebook")
        except Exception:
            pass
        plt.rcParams.update({
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#E6E0ED",
            "axes.labelcolor": "#41394C",
            "axes.titlecolor": "#1A1620",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.grid": True,
            "grid.color": "#EFEAF3",
            "grid.linewidth": 0.9,
            "xtick.color": "#6B6573",
            "ytick.color": "#6B6573",
            "text.color": "#1A1620",
            "font.size": 11,
            "legend.frameon": True,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#E6E0ED",
        })
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
