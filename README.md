# Tibial SSEP Simulation + Time–Frequency Analysis

Goal:This project simulates **tibial somatosensory evoked potentials (SSEPs)**, optionally preprocesses the simulated trials, and computes/visualizes **time–frequency representations** (e.g., power, ITPC) using a **family of Morlet wavelets**. :contentReference[oaicite:0]{index=0}


The main steps include:
1. Generate custom waveforms that mimic the characteristics of SSEPs, such as their frequency content, amplitude, and temporal structure.
2. Simulate the SSEP signals by adding noise and other artifacts to the generated waveforms
3. Apply various signal processing techniques to the simulated SSEP signals, such as filtering, averaging, and time-frequency analysis, to extract relevant features and analyze the results.
4. Generate visualizations and reports to summarize the evoked responses. 
5. Anonymize unlabel the responses and save them in JPEG format for classification purposes. 
6. Evaluate the performance of different algorithms for SSEP analysis, such as machine learning classifiers, and compare their results to the ground truth labels.
7. Document the code and results in a clear and organized manner, including explanations of the methods used and the assumptions made in the analysis.



---

## Project structure

- `01_main_simulation_script.ipynb`  
  Main notebook: defines SSEP peak “ground truth” (e.g., normal/delayed/reduced conditions), generates multi-trial datasets, runs preprocessing, and produces plots/reports.

- `02_anonimizing_simulated_data.ipynb`  
  Utility notebook: copies `.jpg` outputs into a new folder while renaming them to random `subject_####.jpg` IDs and writing a CSV mapping.

- `custom_classes.py`  
  Core implementation: peak generators, dataset simulator, preprocessing pipeline, wavelet-family builder, and TF metrics/plots. :contentReference[oaicite:1]{index=1}

---

## Requirements

Typical environment:
- Python 3.10+
- `numpy`, `scipy`, `matplotlib`, `pandas`

Install (example):
```bash
pip install numpy scipy matplotlib pandas
```

## Usage example
Here is a minimal code example that resembles the structure of the main_simulation_script.ipynb notebook, demonstrating how to generate simulated SSEP data, preprocess it, and visualize the results using a family of Morlet wavelets.

```python
from custom_classes import Peak, GaussianReboundPeak, Dataset, Preprocessor, wavelet_family
import numpy as np

# 1) Define peaks
peaks = [
    GaussianReboundPeak("P45", latency_ms=45, amp_uv=2, width_ms=5,
                        rebound_frac=1.5, rebound_delay_ms=7, rebound_width_ms=5),
    Peak("P75", latency_ms=75, amp_uv=2, width_ms=6),
]

# 2) Simulate dataset
ds = Dataset(
    peaks=peaks,
    fs=2000,
    tmin_ms=-50,
    tmax_ms=200,
    noise_rms_uv=0.5,
    rng_seed=0,
    generate=True,
    n_trials=300,
)

# 3) Preprocess (optional)
pp = Preprocessor(fs=ds.fs)
pp.add_bandpass(low_hz=1.0, high_hz=200.0, order=4)
ds.preprocess(pp)

# 4) Build wavelet family + compute TF
frex = np.linspace(4, 80, 30)
fwhm = np.linspace(0.4, 0.15, len(frex))  # example values (seconds)
wavtime = np.arange(-1, 1+1/ds.fs, 1/ds.fs)  # odd length recommended

wf = wavelet_family(frex=frex, fwhm=fwhm, srate=ds.fs, wavtime=wavtime, n_data=len(ds.trials[0][0]))
wf.build()

ds.compute_tf_complex_from_wavelet_family(wf)
metrics = ds.tf_metrics(baseline_ms=(-50, 0), kind="db")

# 5) Plot a combined TF+ERP report
ds.plot_report(tf_metric="itpc", tlim_ms=(-50, 200), flim_hz=(4, 80), show=True)
```
