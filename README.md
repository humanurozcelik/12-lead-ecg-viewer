# 12-Lead ECG Viewer

This is a Python desktop application I developed during my biomedical engineering internship to work with 12-channel ECG recordings.

The project started as a simple ECG viewer and gradually grew into a small analysis tool while I was learning more about ECG processing and medical device software.

## What it can do

- load and display 12-channel ECG data from CSV files
- show the signals in a 6 × 2 layout
- apply optional 50 Hz notch, 0.67 Hz high-pass, and 35 Hz low-pass filters
- estimate QRS onset/offset and Q, R, S, and T landmarks
- run an experimental MMA-style T-wave alternans analysis
- visualize possible pacemaker-related signal artifacts
- play ECG recordings as an animation and review them with a time slider

## Input

The program expects a CSV file with at least 12 numeric columns.

The default sampling frequency is 500 Hz:

```bash
python ecg_viewer.py --fs 500
```

If the data are raw ADC/ADU values, a gain value can also be supplied:

```bash
python ecg_viewer.py --fs 500 --gain YOUR_ADU_PER_MV
```

If no gain is given, the program assumes the values are already in mV.

## Installation

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python ecg_viewer.py
```

## Main files

```text
.
├── ecg_viewer.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Notes

The analysis methods in this project are experimental implementations for learning and portfolio use. Their performance can change with sampling rate, signal quality, rhythm, lead configuration, and noise.

The project is not intended for clinical diagnosis or patient monitoring.
