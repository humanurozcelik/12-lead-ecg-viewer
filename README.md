# 12-Lead ECG Viewer and Analysis Prototype

A Python desktop application for **12-channel ECG visualization and signal-analysis experiments**. The project combines a Tkinter/Matplotlib interface with ECG filtering, QRST landmark estimation, Modified Moving Average (MMA)-style T-wave alternans analysis, and pacemaker-artifact candidate visualization.

This repository is a portfolio-oriented version of a project developed during a biomedical engineering internship. It is intended for **education, algorithm development, and R&D demonstration**, not for clinical diagnosis or patient management.

## Features

### 12-Channel ECG Viewer
- Loads CSV files containing at least 12 numeric channels
- Displays all 12 channels in a 6 × 2 layout
- Supports animated playback and manual review with a time slider
- Converts raw ADC/ADU values to mV when an acquisition gain is supplied

### Display Filters
The main viewer provides optional:
- **50 Hz notch filter**
- **0.67 Hz high-pass filter** for baseline drift
- **35 Hz low-pass filter** for higher-frequency EMG/noise components

These toolbar filters are intended for visualization. The analysis windows use the original loaded signal and apply their own internal preprocessing.

### QRST Analysis
The QRST pipeline includes:
- QRS candidate detection from a band-limited derivative-energy envelope
- morphology-based candidate validation
- QRS onset and offset estimation
- Q, R, and S landmark estimation
- T-peak estimation with positive/negative polarity handling
- channel-by-channel visualization of detected landmarks

### MMA-TWA Analysis
The T-wave alternans workflow:
- identifies usable beats from QRS timing
- extracts ST-T segments
- rejects atypical segments using robust amplitude and morphology checks
- separates alternating beats into A/B sequences
- updates A/B templates with an MMA-style bounded update rule
- reports the maximum A/B template difference in the analysis window
- visualizes valid ST-T segments, A/B templates, and the TWA trend

The result is an **experimental signal-processing estimate**, not a clinically validated TWA measurement.

### Pacemaker Artifact Analysis
The pacemaker-artifact pipeline uses:
- high-pass preprocessing
- Shannon-energy transformation
- PCA/SVD-based first-component scoring across channels
- peak detection
- per-channel slope support for candidate validation

The output represents **artifact candidates** and should not be interpreted as pacemaker detection or device classification.

## Input Format

The application expects a CSV file with at least 12 numeric columns. The first 12 columns are interpreted as ECG channels:

```text
CH1, CH2, CH3, ... CH12
```

The program does not currently infer sampling frequency or physical amplitude units from file metadata.

### Sampling Frequency

The default sampling frequency is **500 Hz** and can be changed at runtime:

```bash
python ecg_viewer.py --fs 500
```

### Amplitude Units

By default, the program assumes that CSV amplitudes are already expressed in **mV**:

```bash
python ecg_viewer.py
```

If the file contains raw ADC/ADU values, provide the acquisition system's conversion factor in **ADU per mV**:

```bash
python ecg_viewer.py --gain YOUR_ADU_PER_MV
```

For example, the program divides each raw value by the supplied gain before displaying amplitudes in mV. Device-specific calibration constants are intentionally not embedded in this public repository.

## Installation

Python 3.11 is recommended.

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Tkinter is included with standard Python installations on Windows. On some Linux distributions, it may need to be installed separately through the system package manager.

## Run

Default settings:

```bash
python ecg_viewer.py
```

Custom sampling frequency:

```bash
python ecg_viewer.py --fs 500
```

Raw ADC/ADU input:

```bash
python ecg_viewer.py --fs 500 --gain YOUR_ADU_PER_MV
```

After launching the interface, use **Open ECG File** for manual review or **Play ECG File** for animated playback. Once a file is loaded, the QRST, MMA-TWA, and pacemaker-artifact analysis windows become available.

## Repository Structure

```text
.
├── ecg_viewer.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Main Technologies

- Python
- NumPy
- Pandas
- SciPy Signal
- Matplotlib
- Tkinter

## Methodological Scope

The algorithms in this repository are heuristic/experimental implementations developed for learning and R&D exploration. Performance can vary substantially with signal quality, lead configuration, acquisition hardware, sampling rate, rhythm, noise, pacing artifacts, and morphology.

Important limitations:

- no automatic sampling-rate or gain extraction from file metadata
- CSV input only in the public version
- no raw acquisition hardware interface
- QRST landmarks are algorithmic estimates rather than manual clinical annotations
- MMA-TWA output has not been established as a clinical measurement
- pacemaker output represents signal-artifact candidates only
- no prospective or clinical validation is claimed

## Disclaimer

This software is **not a medical device** and is not intended for diagnosis, treatment, monitoring, or clinical decision-making.
