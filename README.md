# SERS / Raman Spectra Cleaner

A web app for cleaning SERS / Raman spectra: removes cosmic-ray (muon) spikes,
resamples onto a uniform grid, corrects the fluorescence baseline, and FFT
denoises — with live diagnostic plots and a one-click cleaned-CSV download.

The same code also runs as a command-line tool (`fft_denoise.py`).

## Input format

A CSV whose **first column is the x-axis** (wavenumber / Raman shift) and whose
**remaining columns are spectra / replicates**:

```
Wavelength,S1,S2,S3,S4
139.19,2368,1547,1492,1578
141.24,2344,1549,1502,1569
...
```

## Use it on the web

Open the deployed app (Streamlit Community Cloud), upload a CSV, adjust the
controls on the left, and download the cleaned data. No installation needed.

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
| `streamlit_app.py` | Web interface (the engine, wrapped in a UI) |
| `fft_denoise.py`   | Processing engine + command-line tool |
| `requirements.txt` | Dependencies for deployment |
