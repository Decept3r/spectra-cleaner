# SERS / Raman Spectra Toolkit

A web app for cleaning SERS / Raman spectra and integrating their peaks, with
live diagnostic plots and exports. The same code also runs as a command-line
tool (`fft_denoise.py`).

Two tabs:

1. **Clean & preprocess** — remove cosmic-ray (muon) spikes, resample onto a
   uniform grid, baseline-correct, and FFT denoise.
2. **Peak integration** — auto-detect peaks, then adjust any integration window
   by *dragging a box across it on the plot* or editing the table. Get per-peak
   areas, cross-replicate mean +/- SD and %RSD (a reproducibility readout), and
   area ratios between bands.

## Input format

A CSV whose **first column is the x-axis** (wavenumber / Raman shift) and whose
**remaining columns are spectra / replicates**:

```
Wavelength,S1,S2,S3,S4
139.19,2368,1547,1492,1578
141.24,2344,1549,1502,1569
...
```

## Exports

- **Cleaned data + integration (CSV)** — one file with processing metadata, the
  peak-integration table (areas, stats, heights), and the full cleaned spectra,
  in clearly marked sections.
- **Annotated figure (PNG)** — the cleaned spectra with shaded integration
  regions and labelled peaks.

## Run it locally

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Command-line version

```bash
python fft_denoise.py mydata.csv --despike --resample \
    --baseline arpls --baseline-lam 1e5 --method lowpass --cutoff 0.05 \
    --out-dir output
```

Run `python fft_denoise.py --help` for all options.

## Files

| File | Purpose |
|------|---------|
| streamlit_app.py | Web interface (cleaning + integration tabs) |
| fft_denoise.py   | Processing engine + command-line tool |
| requirements.txt | Dependencies for deployment |
