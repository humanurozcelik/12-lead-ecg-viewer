"""12-Lead ECG Viewer and Analysis Prototype

Desktop application for educational and R&D-oriented ECG visualization and
signal-analysis experiments. The implementation includes display filtering,
QRST landmark estimation, MMA-style T-wave alternans estimation, and
pacemaker-artifact candidate visualization.

This software is not a medical device and is not intended for diagnosis or
clinical decision-making.
"""

import argparse
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, filtfilt, iirnotch


Fs = 500
Wl = 5
Update = 50

# Public default: assume amplitudes are already in mV.
# For raw ADC/ADU input, pass the system conversion using --gain.
ADC_GAIN_ADU_PER_MV = 1.0

Channel_names = [
    'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6',
    'CH7', 'CH8', 'CH9', 'CH10', 'CH11', 'CH12'
]

data = None
raw_data = None
current_sample = 0
animation = None
current_file_path = None

AMPLITUDE_LABEL = 'Amplitude (mV)'

UI_BG = '#f4f6f8'
PANEL_BG = '#ffffff'
FILE_BUTTON_BG = '#dbeafe'
FILTER_BUTTON_BG = '#e5e7eb'
FILTER_ACTIVE_BG = '#bbf7d0'
ANALYSIS_BUTTON_BG = '#c7d2fe'
EXIT_BUTTON_BG = '#fee2e2'

notch_active = False
baseline_active = False
emg_active = False

window_samples = Fs * Wl
step_samples = int(Fs * Update / 1000)


def adu_to_mv(values):
    return np.asarray(values, dtype=float) / ADC_GAIN_ADU_PER_MV


def adu_to_uv(values):
    return adu_to_mv(values) * 1000.0


def notch_filter(signal):
    b, a = iirnotch(50, 30, Fs)
    return filtfilt(b, a, signal)


def baseline_filter(signal):
    b, a = butter(4, 0.67 / (Fs / 2), btype='highpass')
    return filtfilt(b, a, signal)


def emg_filter(signal):
    b, a = butter(4, 35 / (Fs / 2), btype='lowpass')
    return filtfilt(b, a, signal)


def bandpass_filter(signal, low, high, order=3):
    nyquist = Fs / 2
    b, a = butter(order, [low / nyquist, high / nyquist], btype='bandpass')
    return filtfilt(b, a, signal)


def robust_sigma(values):
    values = np.asarray(values, dtype=float)
    median_value = np.median(values)
    mad = np.median(np.abs(values - median_value))
    sigma = 1.4826 * mad
    if sigma == 0:
        sigma = np.std(values)
    return float(sigma)


def robust_threshold(signal):
    return np.median(signal) + 3.0 * robust_sigma(signal)


def pacemaker_preprocess(multichannel_signal):
    signals = np.asarray(multichannel_signal, dtype=float)
    if signals.ndim == 1:
        signals = signals[:, np.newaxis]

    nyquist = Fs / 2
    b, a = butter(2, 120 / nyquist, btype='highpass')
    highpass = np.zeros_like(signals, dtype=float)

    for channel in range(signals.shape[1]):
        highpass[:, channel] = filtfilt(b, a, signals[:, channel])

    squared = highpass ** 2
    shannon_energy = squared * np.log(squared + np.finfo(float).eps)
    return highpass, shannon_energy


def pacemaker_pc1(shannon_energy):
    energy = np.asarray(shannon_energy, dtype=float)
    if energy.ndim == 1:
        energy = energy[:, np.newaxis]

    centered = energy - np.mean(energy, axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    pc1 = u[:, 0] * singular_values[0]

    if abs(np.min(pc1)) > abs(np.max(pc1)):
        pc1 = -pc1

    return pc1


def detect_pacemaker_artifacts(multichannel_signal):
    highpass, shannon_energy = pacemaker_preprocess(multichannel_signal)
    pc1 = pacemaker_pc1(shannon_energy)
    score = pc1 - np.median(pc1)
    score_sigma = np.std(score)

    if not np.isfinite(score_sigma) or score_sigma <= 0:
        return {
            'spikes': np.array([], dtype=int),
            'pc1': pc1,
            'score': score,
            'threshold': np.nan,
            'lead_support': np.array([], dtype=int),
            'channel_support': np.empty((0, highpass.shape[1]), dtype=bool),
        }

    threshold = 6.0 * score_sigma
    spikes, _ = scipy.signal.find_peaks(
        score,
        height=threshold,
        distance=max(1, int(0.02 * Fs)),
    )

    slopes = np.abs(np.gradient(highpass, axis=0))
    slope_thresholds = np.zeros(slopes.shape[1], dtype=float)

    for channel in range(slopes.shape[1]):
        channel_slope = slopes[:, channel]
        slope_thresholds[channel] = (
            np.median(channel_slope)
            + 6.0 * max(robust_sigma(channel_slope), 1e-9)
        )

    validation_radius = max(1, int(0.006 * Fs))
    lead_support = []
    channel_support = []

    for spike in spikes:
        start = max(0, int(spike) - validation_radius)
        end = min(len(highpass), int(spike) + validation_radius + 1)
        local_slopes = np.max(slopes[start:end, :], axis=0)
        supported_channels = local_slopes >= slope_thresholds
        lead_support.append(int(np.sum(supported_channels)))
        channel_support.append(supported_channels)

    return {
        'spikes': np.asarray(spikes, dtype=int),
        'pc1': pc1,
        'score': score,
        'threshold': float(threshold),
        'lead_support': np.asarray(lead_support, dtype=int),
        'channel_support': np.asarray(channel_support, dtype=bool),
    }


def qrs_detection_signal(signal):
    centered = signal - np.median(signal)
    qrs_signal = bandpass_filter(centered, 5, 20, order=3)
    derivative = np.gradient(qrs_signal)
    energy = derivative ** 2
    integration_window = max(1, int(0.12 * Fs))
    kernel = np.ones(integration_window) / integration_window
    envelope = np.convolve(energy, kernel, mode='same')
    return qrs_signal, envelope


def validate_qrs_candidates(qrs_signal, envelope, candidate_peaks):
    if len(candidate_peaks) == 0:
        return np.array([], dtype=int)

    widths = scipy.signal.peak_widths(envelope, candidate_peaks, rel_height=0.5)[0]
    search_radius = int(0.10 * Fs)
    slopes = []
    sharpness_values = []

    for peak in candidate_peaks:
        start = max(0, peak - search_radius)
        end = min(len(qrs_signal), peak + search_radius + 1)
        region = qrs_signal[start:end]

        if len(region) < 3:
            slopes.append(0.0)
            sharpness_values.append(0.0)
            continue

        gradient = np.abs(np.gradient(region))
        max_slope = float(np.max(gradient))
        local_range = float(np.ptp(region))
        sharpness = max_slope / max(local_range, 1e-9)

        slopes.append(max_slope)
        sharpness_values.append(sharpness)

    slopes = np.asarray(slopes, dtype=float)
    sharpness_values = np.asarray(sharpness_values, dtype=float)

    finite_slopes = slopes[np.isfinite(slopes)]
    finite_sharpness = sharpness_values[np.isfinite(sharpness_values)]

    if len(finite_slopes) == 0 or len(finite_sharpness) == 0:
        return np.array([], dtype=int)

    slope_reference = np.percentile(finite_slopes, 75)
    sharpness_reference = np.percentile(finite_sharpness, 75)
    slope_threshold = max(0.35 * slope_reference, 1e-9)
    sharpness_threshold = max(0.45 * sharpness_reference, 0.015)
    maximum_width = 0.22 * Fs

    accepted = []

    for peak, width, slope, sharpness in zip(
        candidate_peaks,
        widths,
        slopes,
        sharpness_values,
    ):
        if (
            width <= maximum_width
            and slope >= slope_threshold
            and sharpness >= sharpness_threshold
        ):
            accepted.append(int(peak))

    return np.asarray(accepted, dtype=int)


def detect_qrs_events(signal):
    qrs_signal, envelope = qrs_detection_signal(signal)
    threshold = robust_threshold(envelope)
    candidate_peaks, _ = scipy.signal.find_peaks(
        envelope,
        height=threshold,
        distance=int(0.25 * Fs),
    )
    return validate_qrs_candidates(qrs_signal, envelope, candidate_peaks)


def qrs_morphology_signal(signal):
    centered = signal - np.median(signal)
    return bandpass_filter(centered, 0.5, 40, order=3)


def detect_qrs_boundaries(signal, qrs_events):
    morphology = qrs_morphology_signal(signal)
    slope_energy = np.gradient(morphology) ** 2
    smooth_window = max(1, int(0.02 * Fs))
    kernel = np.ones(smooth_window) / smooth_window
    slope_envelope = np.convolve(slope_energy, kernel, mode='same')

    left_search = int(0.15 * Fs)
    right_search = int(0.18 * Fs)
    quiet_samples = max(1, int(0.02 * Fs))

    onsets = []
    offsets = []

    for event in qrs_events:
        search_start = max(0, event - left_search)
        search_end = min(len(signal), event + right_search)
        region = slope_envelope[search_start:search_end]

        if len(region) == 0:
            onsets.append(np.nan)
            offsets.append(np.nan)
            continue

        baseline = np.median(region)
        noise = robust_sigma(region)
        threshold = baseline + 2.5 * noise
        event_local = int(np.clip(event - search_start, 0, len(region) - 1))

        onset = np.nan
        for i in range(event_local, quiet_samples, -1):
            quiet_region = region[i - quiet_samples:i]
            if np.all(quiet_region < threshold):
                onset = search_start + i
                break

        offset = np.nan
        for i in range(event_local, len(region) - quiet_samples):
            quiet_region = region[i:i + quiet_samples]
            if np.all(quiet_region < threshold):
                offset = search_start + i
                break

        onsets.append(onset)
        offsets.append(offset)

    return np.asarray(onsets, dtype=float), np.asarray(offsets, dtype=float)


def detect_r_peaks(signal, qrs_onsets, qrs_offsets):
    morphology = qrs_morphology_signal(signal)
    baseline_window = max(3, int(0.06 * Fs))
    r_peaks = []

    for onset, offset in zip(qrs_onsets, qrs_offsets):
        if np.isnan(onset) or np.isnan(offset):
            r_peaks.append(np.nan)
            continue

        onset = int(onset)
        offset = int(offset)

        if offset <= onset:
            r_peaks.append(np.nan)
            continue

        segment = morphology[onset:offset + 1]
        baseline_start = max(0, onset - baseline_window)
        baseline_region = morphology[baseline_start:onset]

        if len(baseline_region) < 3:
            r_peaks.append(np.nan)
            continue

        baseline = np.median(baseline_region)
        noise_sigma = robust_sigma(baseline_region)
        centered_segment = segment - baseline
        qrs_range = np.ptp(centered_segment)

        if qrs_range <= 0:
            r_peaks.append(np.nan)
            continue

        local_r = int(np.argmax(centered_segment))
        r_amplitude = centered_segment[local_r]
        minimum_amplitude = max(
            3.0 * noise_sigma,
            0.08 * qrs_range,
            1e-6,
        )

        if r_amplitude < minimum_amplitude:
            r_peaks.append(np.nan)
        else:
            r_peaks.append(onset + local_r)

    return np.asarray(r_peaks, dtype=float)


def detect_qs_points(signal, qrs_onsets, qrs_offsets, r_peaks):
    morphology = qrs_morphology_signal(signal)
    baseline_window = max(3, int(0.06 * Fs))
    q_points = []
    s_points = []

    for onset, offset, r_peak in zip(qrs_onsets, qrs_offsets, r_peaks):
        if np.isnan(onset) or np.isnan(offset) or np.isnan(r_peak):
            q_points.append(np.nan)
            s_points.append(np.nan)
            continue

        onset = int(onset)
        offset = int(offset)
        r_peak = int(r_peak)

        if not (onset < r_peak < offset):
            q_points.append(np.nan)
            s_points.append(np.nan)
            continue

        baseline_start = max(0, onset - baseline_window)
        baseline_region = morphology[baseline_start:onset]

        if len(baseline_region) < 3:
            q_points.append(np.nan)
            s_points.append(np.nan)
            continue

        baseline = np.median(baseline_region)
        noise_sigma = robust_sigma(baseline_region)
        qrs_segment = morphology[onset:offset + 1] - baseline
        qrs_range = np.ptp(qrs_segment)

        if qrs_range <= 0:
            q_points.append(np.nan)
            s_points.append(np.nan)
            continue

        minimum_deflection = max(
            3.0 * noise_sigma,
            0.08 * qrs_range,
            1e-6,
        )

        q_region = morphology[onset:r_peak] - baseline
        q_point = np.nan
        if len(q_region) > 0:
            local_q = int(np.argmin(q_region))
            q_deflection = -q_region[local_q]
            if q_deflection >= minimum_deflection:
                q_point = onset + local_q

        s_region = morphology[r_peak + 1:offset + 1] - baseline
        s_point = np.nan
        if len(s_region) > 0:
            local_s = int(np.argmin(s_region))
            s_deflection = -s_region[local_s]
            if s_deflection >= minimum_deflection:
                s_point = r_peak + 1 + local_s

        q_points.append(q_point)
        s_points.append(s_point)

    return (
        np.asarray(q_points, dtype=float),
        np.asarray(s_points, dtype=float),
    )


def detect_t_peaks(signal, qrs_events, qrs_offsets):
    centered = signal - np.median(signal)
    t_signal = bandpass_filter(centered, 0.5, 15, order=3)
    morphology = qrs_morphology_signal(signal)

    t_peaks = []
    t_polarities = []

    rr_intervals = np.diff(qrs_events)
    median_rr = int(np.median(rr_intervals)) if len(rr_intervals) > 0 else int(0.8 * Fs)

    st_guard = int(0.06 * Fs)
    baseline_window = max(3, int(0.04 * Fs))
    refinement_radius = int(0.025 * Fs)

    for i, (qrs_event, qrs_offset) in enumerate(zip(qrs_events, qrs_offsets)):
        if np.isnan(qrs_offset):
            t_peaks.append(np.nan)
            t_polarities.append('unknown')
            continue

        qrs_event = int(qrs_event)
        qrs_offset = int(qrs_offset)

        if i + 1 < len(qrs_events):
            next_qrs = int(qrs_events[i + 1])
            rr = next_qrs - qrs_event
        else:
            rr = median_rr
            next_qrs = min(len(signal) - 1, qrs_event + rr)

        if rr <= 0:
            t_peaks.append(np.nan)
            t_polarities.append('unknown')
            continue

        search_start = qrs_offset + st_guard
        search_end = min(
            int(qrs_event + 0.60 * rr),
            next_qrs - int(0.20 * rr),
            len(signal) - 1,
        )

        if search_end <= search_start:
            t_peaks.append(np.nan)
            t_polarities.append('unknown')
            continue

        baseline_start = max(0, search_start - baseline_window)
        baseline_region = t_signal[baseline_start:search_start]

        if len(baseline_region) < 3:
            t_peaks.append(np.nan)
            t_polarities.append('unknown')
            continue

        baseline = np.median(baseline_region)
        noise_sigma = robust_sigma(baseline_region)
        region = t_signal[search_start:search_end] - baseline

        if len(region) == 0:
            t_peaks.append(np.nan)
            t_polarities.append('unknown')
            continue

        local_t = int(np.argmax(np.abs(region)))
        t_amplitude = region[local_t]
        region_range = np.ptp(region)
        minimum_amplitude = max(
            3.0 * noise_sigma,
            0.08 * region_range,
            1e-6,
        )

        if abs(t_amplitude) < minimum_amplitude:
            t_peaks.append(np.nan)
            t_polarities.append('unknown')
            continue

        candidate_index = search_start + local_t
        polarity = 'positive' if t_amplitude >= 0 else 'negative'

        refine_start = max(
            search_start,
            candidate_index - refinement_radius,
        )
        refine_end = min(
            search_end,
            candidate_index + refinement_radius + 1,
        )
        refine_region = morphology[refine_start:refine_end]

        if len(refine_region) == 0:
            t_peaks.append(np.nan)
            t_polarities.append('unknown')
            continue

        if polarity == 'positive':
            refined_peak = refine_start + int(np.argmax(refine_region))
        else:
            refined_peak = refine_start + int(np.argmin(refine_region))

        t_peaks.append(refined_peak)
        t_polarities.append(polarity)

    return (
        np.asarray(t_peaks, dtype=float),
        np.asarray(t_polarities, dtype=object),
    )


def analyze_qrst(signal):
    qrs_events = detect_qrs_events(signal)
    qrs_onsets, qrs_offsets = detect_qrs_boundaries(signal, qrs_events)
    r_peaks = detect_r_peaks(signal, qrs_onsets, qrs_offsets)
    q_points, s_points = detect_qs_points(
        signal,
        qrs_onsets,
        qrs_offsets,
        r_peaks,
    )
    t_peaks, t_polarities = detect_t_peaks(
        signal,
        qrs_events,
        qrs_offsets,
    )

    return {
        'qrs_events': qrs_events,
        'qrs_onsets': qrs_onsets,
        'qrs_offsets': qrs_offsets,
        'q_points': q_points,
        'r_peaks': r_peaks,
        's_points': s_points,
        't_peaks': t_peaks,
        't_polarities': t_polarities,
    }


def get_twa_beats(qrs_events, qrs_offsets):
    beats = []
    usable_count = min(len(qrs_events), len(qrs_offsets))

    for i in range(max(0, usable_count - 1)):
        qrs_event = int(qrs_events[i])
        next_qrs = int(qrs_events[i + 1])
        qrs_offset = qrs_offsets[i]

        if np.isnan(qrs_offset):
            continue

        qrs_offset = int(qrs_offset)
        rr = next_qrs - qrs_event

        if rr <= 0:
            continue

        if not (qrs_event < qrs_offset < next_qrs):
            continue

        beats.append(
            {
                'beat_index': i,
                'qrs_event': qrs_event,
                'qrs_offset': qrs_offset,
                'next_qrs': next_qrs,
                'rr': rr,
            }
        )

    return beats


def extract_stt_segments(signal, beats):
    if len(beats) == 0:
        return np.empty((0, 0)), np.array([]), []

    centered = signal - np.median(signal)
    stt_signal = bandpass_filter(centered, 0.5, 15, order=3)

    rr_values = np.asarray([beat['rr'] for beat in beats], dtype=float)
    median_rr = int(np.median(rr_values))
    stt_length = max(1, int(0.45 * median_rr))
    next_qrs_guard = max(1, int(0.15 * median_rr))

    segments = []
    segment_times = []
    accepted_beats = []

    for beat in beats:
        start = beat['qrs_offset']
        end = start + stt_length
        safe_end = beat['next_qrs'] - next_qrs_guard

        if end > safe_end or end > len(stt_signal):
            continue

        segment = stt_signal[start:end]

        if len(segment) != stt_length or np.any(~np.isfinite(segment)):
            continue

        segments.append(segment)
        segment_times.append(beat['qrs_event'] / Fs)
        accepted_beats.append(beat)

    if len(segments) == 0:
        return (
            np.empty((0, stt_length)),
            np.array([]),
            [],
        )

    return (
        np.asarray(segments, dtype=float),
        np.asarray(segment_times, dtype=float),
        accepted_beats,
    )


def validate_stt_segments(segments, segment_times, accepted_beats):
    if len(segments) == 0:
        return np.empty((0, 0)), np.array([]), []

    reference = np.median(segments, axis=0)
    segment_ranges = np.ptp(segments, axis=1)
    range_median = np.median(segment_ranges)
    range_sigma = robust_sigma(segment_ranges)
    range_threshold = (
        np.inf
        if range_sigma == 0
        else range_median + 3.0 * range_sigma
    )

    morphology_errors = np.mean(np.abs(segments - reference), axis=1)
    error_median = np.median(morphology_errors)
    error_sigma = robust_sigma(morphology_errors)
    error_threshold = (
        np.inf
        if error_sigma == 0
        else error_median + 3.0 * error_sigma
    )

    valid_segments = []
    valid_times = []
    valid_beats = []

    for segment, time_value, beat, segment_range, error in zip(
        segments,
        segment_times,
        accepted_beats,
        segment_ranges,
        morphology_errors,
    ):
        if segment_range <= range_threshold and error <= error_threshold:
            valid_segments.append(segment)
            valid_times.append(time_value)
            valid_beats.append(beat)

    if len(valid_segments) == 0:
        return (
            np.empty((0, segments.shape[1])),
            np.array([]),
            [],
        )

    return (
        np.asarray(valid_segments, dtype=float),
        np.asarray(valid_times, dtype=float),
        valid_beats,
    )


def mma_update(template, segment, update_limit):
    difference = segment - template
    update = difference / 8.0
    update = np.clip(update, -update_limit, update_limit)
    return template + update


def calculate_mma_twa(segments, segment_times, beats):
    measurement_start = int(0.06 * Fs)
    init_count = 3

    a_segments = [
        segment
        for segment, beat in zip(segments, beats)
        if beat['beat_index'] % 2 == 0
    ]
    b_segments = [
        segment
        for segment, beat in zip(segments, beats)
        if beat['beat_index'] % 2 != 0
    ]

    a_count = len(a_segments)
    b_count = len(b_segments)
    enough_for_initialization = (
        a_count >= init_count and b_count >= init_count
    )

    template_a = None
    template_b = None

    if enough_for_initialization:
        template_a = np.median(
            np.asarray(a_segments[:init_count], dtype=float),
            axis=0,
        )
        template_b = np.median(
            np.asarray(b_segments[:init_count], dtype=float),
            axis=0,
        )

    if not enough_for_initialization:
        return {
            'template_a': template_a,
            'template_b': template_b,
            'twa_times': np.array([]),
            'twa_values': np.array([]),
            'twa_index': np.nan,
            'final_twa': np.nan,
            'status': 'Insufficient valid beats',
            'a_count': a_count,
            'b_count': b_count,
            'init_count': init_count,
            'measurement_start': measurement_start,
        }

    if segments.shape[1] <= measurement_start:
        return {
            'template_a': template_a,
            'template_b': template_b,
            'twa_times': np.array([]),
            'twa_values': np.array([]),
            'twa_index': np.nan,
            'final_twa': np.nan,
            'status': 'Invalid ST-T window',
            'a_count': a_count,
            'b_count': b_count,
            'init_count': init_count,
            'measurement_start': measurement_start,
        }

    segment_ranges = np.ptp(segments, axis=1)
    typical_range = np.median(segment_ranges)
    update_limit = max(0.10 * typical_range, 1e-9)

    twa_times = []
    twa_values = []
    seen_a = 0
    seen_b = 0

    for segment, time_value, beat in zip(segments, segment_times, beats):
        beat_index = beat['beat_index']

        if beat_index % 2 == 0:
            seen_a += 1
            if seen_a <= init_count:
                continue
            template_a = mma_update(template_a, segment, update_limit)
        else:
            seen_b += 1
            if seen_b <= init_count:
                continue
            template_b = mma_update(template_b, segment, update_limit)

        difference = np.abs(template_a - template_b)
        measurement_difference = difference[measurement_start:]
        current_twa = float(np.max(measurement_difference))

        twa_times.append(time_value)
        twa_values.append(current_twa)

    final_difference = np.abs(template_a - template_b)
    measurement_difference = final_difference[measurement_start:]
    twa_index = measurement_start + int(np.argmax(measurement_difference))
    final_twa = float(final_difference[twa_index])

    return {
        'template_a': template_a,
        'template_b': template_b,
        'twa_times': np.asarray(twa_times, dtype=float),
        'twa_values': np.asarray(twa_values, dtype=float),
        'twa_index': twa_index,
        'final_twa': final_twa,
        'status': 'MMA-TWA estimate',
        'a_count': a_count,
        'b_count': b_count,
        'init_count': init_count,
        'measurement_start': measurement_start,
    }


def prepare_twa_analysis(signal):
    qrst = analyze_qrst(signal)
    qrs_events = qrst['qrs_events']
    qrs_offsets = qrst['qrs_offsets']

    beats = get_twa_beats(qrs_events, qrs_offsets)
    segments, segment_times, extracted_beats = extract_stt_segments(
        signal,
        beats,
    )
    valid_segments, valid_times, valid_beats = validate_stt_segments(
        segments,
        segment_times,
        extracted_beats,
    )
    mma = calculate_mma_twa(
        valid_segments,
        valid_times,
        valid_beats,
    )

    return {
        'qrst': qrst,
        'beats': beats,
        'segments': segments,
        'segment_times': segment_times,
        'extracted_beats': extracted_beats,
        'valid_segments': valid_segments,
        'valid_times': valid_times,
        'valid_beats': valid_beats,
        'template_a': mma['template_a'],
        'template_b': mma['template_b'],
        'twa_times': mma['twa_times'],
        'twa_values': mma['twa_values'],
        'twa_index': mma['twa_index'],
        'final_twa': mma['final_twa'],
        'status': mma['status'],
        'a_count': mma['a_count'],
        'b_count': mma['b_count'],
        'init_count': mma['init_count'],
        'measurement_start': mma['measurement_start'],
    }


def update_y_limits():
    if data is None:
        return

    for i in range(12):
        signal = adu_to_mv(data[:, i])
        low = np.percentile(signal, 1)
        high = np.percentile(signal, 99)
        margin = (high - low) * 0.15

        if margin == 0:
            margin = 1

        axes[i].set_ylim(low - margin, high + margin)


def apply_filters():
    global data

    if raw_data is None:
        return

    data = raw_data.copy()

    for i in range(12):
        signal = data[:, i]

        if notch_active:
            signal = notch_filter(signal)

        if baseline_active:
            signal = baseline_filter(signal)

        if emg_active:
            signal = emg_filter(signal)

        data[:, i] = signal

    update_y_limits()


def refresh_display():
    if data is None:
        return

    if time_slider['state'] == 'normal':
        show_window(time_slider.get())
    else:
        canvas.draw_idle()


def load_csv(file_path):
    global raw_data
    global data

    dataframe = pd.read_csv(file_path, index_col=False)

    if dataframe.shape[1] < 12:
        raise ValueError('File must contain at least 12 channels')

    dataframe = dataframe.iloc[:, :12]

    if dataframe.isna().any().any():
        raise ValueError('ECG data contains missing or invalid values')

    raw_data = dataframe.to_numpy(dtype=float)
    data = raw_data.copy()
    apply_filters()


def current_metadata():
    if raw_data is None:
        return 'No ECG file loaded'

    if current_file_path:
        file_name = current_file_path.replace('\\', '/').split('/')[-1]
    else:
        file_name = 'ECG'

    duration = len(raw_data) / Fs

    return (
        f'{file_name} | '
        f'12 channels | '
        f'Fs: {Fs} Hz | '
        f'Gain: {ADC_GAIN_ADU_PER_MV:g} ADU/mV | '
        f'{duration:.1f} s'
    )


def show_loaded_view(file_path, mode):
    global current_file_path

    current_file_path = file_path
    welcome_frame.pack_forget()

    if not viewer_frame.winfo_ismapped():
        viewer_frame.pack(fill=tk.BOTH, expand=True)

    for button in (
        notch_button,
        baseline_button,
        emg_button,
        qrst_button,
        twa_button,
        pacemaker_button,
    ):
        button.config(state='normal')

    status_label.config(
        text=f'{current_metadata()} | Mode: {mode}'
    )


def stop_playback_animation():
    global animation

    if animation is not None:
        if animation.event_source is not None:
            animation.event_source.stop()
        animation = None

    for line in lines:
        line.set_animated(False)


def select_file():
    global current_sample
    global animation

    file_path = filedialog.askopenfilename(
        title='Select ECG CSV file for playback',
        filetypes=[('CSV files', '*.csv')],
    )

    if not file_path:
        return

    try:
        load_csv(file_path)
        current_sample = 0
        time_slider.config(from_=0, to=1, state='disabled')
        time_slider.set(0)
        show_loaded_view(file_path, 'Playback')
        stop_playback_animation()

        animation = FuncAnimation(
            figure,
            update_plot,
            interval=Update,
            blit=True,
            cache_frame_data=False,
        )

        canvas.draw()

    except Exception as error:
        status_label.config(text=f'Error: {error}')


def open_file():
    global current_sample
    global animation

    file_path = filedialog.askopenfilename(
        title='Open ECG CSV file for review',
        filetypes=[('CSV files', '*.csv')],
    )

    if not file_path:
        return

    try:
        stop_playback_animation()
        load_csv(file_path)
        current_sample = 0

        total_duration = len(data) / Fs
        max_start_time = max(0, total_duration - Wl)

        time_slider.config(
            from_=0,
            to=max_start_time,
            state='normal',
        )
        time_slider.set(0)
        show_loaded_view(file_path, 'Review')
        show_window(0)
        canvas.draw()

    except Exception as error:
        status_label.config(text=f'Error: {error}')


def show_window(value):
    if data is None:
        return

    start_time = float(value)
    start = int(start_time * Fs)
    end = min(start + window_samples, len(data))
    time_axis = np.arange(start, end) / Fs

    for i in range(12):
        signal = adu_to_mv(data[start:end, i])
        lines[i].set_data(time_axis, signal)
        axes[i].set_xlim(start_time, start_time + Wl)

    canvas.draw_idle()


def update_plot(frame):
    global current_sample

    if data is None:
        return lines

    if current_sample >= len(data):
        if animation is not None:
            animation.event_source.stop()
        return lines

    start = max(0, current_sample - window_samples)
    end = current_sample

    if end <= start:
        current_sample += step_samples
        return lines

    time_axis = np.arange(end - start) / Fs

    for i in range(12):
        signal = adu_to_mv(data[start:end, i])
        lines[i].set_data(time_axis, signal)
        axes[i].set_xlim(0, Wl)

    current_sample += step_samples
    return lines


def notch():
    global notch_active
    notch_active = not notch_active
    apply_filters()
    refresh_display()
    notch_button.config(
        bg=FILTER_ACTIVE_BG if notch_active else FILTER_BUTTON_BG
    )


def baseline():
    global baseline_active
    baseline_active = not baseline_active
    apply_filters()
    refresh_display()
    baseline_button.config(
        bg=FILTER_ACTIVE_BG if baseline_active else FILTER_BUTTON_BG
    )


def emg():
    global emg_active
    emg_active = not emg_active
    apply_filters()
    refresh_display()
    emg_button.config(
        bg=FILTER_ACTIVE_BG if emg_active else FILTER_BUTTON_BG
    )


def points_in_window(points, start, end):
    valid = points[~np.isnan(points)].astype(int)
    return valid[(valid >= start) & (valid < end)]


def vertical_marker_data(points, y_min, y_max):
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return np.array([]), np.array([])

    x_values = np.repeat(points / Fs, 3)
    y_values = np.tile([y_min, y_max, np.nan], len(points))
    x_values[2::3] = np.nan
    return x_values, y_values


def qrst_analysis_window():
    if raw_data is None:
        status_label.config(text='Load an ECG file first')
        return

    analysis_window = tk.Toplevel(root)
    analysis_window.title('Q-R-S-T Analysis')
    analysis_window.geometry('1200x700')

    control_frame = tk.Frame(analysis_window, bg=UI_BG)
    control_frame.pack(fill=tk.X, padx=20, pady=(8, 6))

    tk.Label(
        control_frame,
        text='Channel:',
        font=('Arial', 11),
    ).pack(side=tk.LEFT, padx=5)

    channel_var = tk.StringVar()
    channel_box = ttk.Combobox(
        control_frame,
        textvariable=channel_var,
        values=Channel_names,
        state='readonly',
        width=10,
    )
    channel_box.current(0)
    channel_box.pack(side=tk.LEFT, padx=5)

    metadata_label = tk.Label(
        control_frame,
        text=current_metadata(),
        font=('Arial', 10),
        bg=UI_BG,
        fg='#374151',
        anchor='e',
        width=70,
    )
    metadata_label.pack(side=tk.RIGHT, padx=10)

    analysis_figure, analysis_ax = plt.subplots(figsize=(12, 6))
    analysis_figure.subplots_adjust(right=0.80)

    analysis_canvas = FigureCanvasTkAgg(
        analysis_figure,
        master=analysis_window,
    )
    analysis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    signal_line, = analysis_ax.plot(
        [],
        [],
        color='black',
        linewidth=1.0,
        label='ECG',
    )
    qrs_line, = analysis_ax.plot(
        [],
        [],
        color='gray',
        alpha=0.25,
        linewidth=1,
        linestyle=':',
        label='QRS detector event',
    )
    onset_line, = analysis_ax.plot(
        [],
        [],
        color='blue',
        alpha=0.6,
        linewidth=1.1,
        linestyle='--',
        label='QRS onset',
    )
    offset_line, = analysis_ax.plot(
        [],
        [],
        color='red',
        alpha=0.6,
        linewidth=1.1,
        linestyle='--',
        label='QRS offset',
    )
    q_scatter = analysis_ax.scatter(
        [],
        [],
        color='orange',
        label='Q point',
        s=40,
        zorder=5,
    )
    r_scatter = analysis_ax.scatter(
        [],
        [],
        color='green',
        label='R peak',
        s=40,
        zorder=5,
    )
    s_scatter = analysis_ax.scatter(
        [],
        [],
        color='purple',
        label='S point',
        s=40,
        zorder=5,
    )
    t_scatter = analysis_ax.scatter(
        [],
        [],
        color='red',
        label='T peak',
        s=40,
        zorder=5,
    )

    analysis_ax.set_xlabel('Time (s)')
    analysis_ax.set_ylabel(AMPLITUDE_LABEL)
    analysis_ax.grid(True, alpha=0.3)
    analysis_ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )

    analysis_cache = {}
    total_duration = len(raw_data) / Fs
    max_analysis_time = max(0, total_duration - Wl)

    analysis_slider = tk.Scale(
        analysis_window,
        from_=0,
        to=max_analysis_time,
        orient=tk.HORIZONTAL,
        resolution=0.1,
        showvalue=0,
    )
    analysis_slider.pack(fill=tk.X, padx=40, pady=10)

    def scatter_offsets(points, display_signal):
        if len(points) == 0:
            return np.empty((0, 2))

        return np.column_stack(
            (
                points / Fs,
                adu_to_mv(display_signal[points]),
            )
        )

    def plot_selected_channel(event=None):
        channel_name = channel_var.get()
        channel_index = Channel_names.index(channel_name)

        if channel_index not in analysis_cache:
            analysis_signal = raw_data[:, channel_index]
            display_signal = qrs_morphology_signal(analysis_signal)
            qrst = analyze_qrst(analysis_signal)
            analysis_cache[channel_index] = (display_signal, qrst)

        display_signal, qrst = analysis_cache[channel_index]

        qrs_all = qrst['qrs_events']
        onset_all = qrst['qrs_onsets']
        offset_all = qrst['qrs_offsets']
        q_all = qrst['q_points']
        r_all = qrst['r_peaks']
        s_all = qrst['s_points']
        t_all = qrst['t_peaks']

        start_time = float(analysis_slider.get())
        start = int(start_time * Fs)
        end = min(start + window_samples, len(display_signal))
        time_axis = np.arange(start, end) / Fs
        visible_signal = adu_to_mv(display_signal[start:end])

        qrs = qrs_all[(qrs_all >= start) & (qrs_all < end)]
        onsets = points_in_window(onset_all, start, end)
        offsets = points_in_window(offset_all, start, end)
        q_points = points_in_window(q_all, start, end)
        r_peaks = points_in_window(r_all, start, end)
        s_points = points_in_window(s_all, start, end)
        t_peaks = points_in_window(t_all, start, end)

        signal_line.set_data(time_axis, visible_signal)
        analysis_ax.set_xlim(start_time, start_time + Wl)

        if len(visible_signal) > 0:
            y_min = float(np.min(visible_signal))
            y_max = float(np.max(visible_signal))
            margin = (y_max - y_min) * 0.08

            if margin == 0:
                margin = 1.0

            analysis_ax.set_ylim(y_min - margin, y_max + margin)

        y_min, y_max = analysis_ax.get_ylim()

        qrs_line.set_data(*vertical_marker_data(qrs, y_min, y_max))
        onset_line.set_data(*vertical_marker_data(onsets, y_min, y_max))
        offset_line.set_data(*vertical_marker_data(offsets, y_min, y_max))
        q_scatter.set_offsets(scatter_offsets(q_points, display_signal))
        r_scatter.set_offsets(scatter_offsets(r_peaks, display_signal))
        s_scatter.set_offsets(scatter_offsets(s_points, display_signal))
        t_scatter.set_offsets(scatter_offsets(t_peaks, display_signal))

        analysis_ax.set_title(f'{channel_name} - Q-R-S-T Analysis')
        analysis_canvas.draw_idle()

    analysis_slider.config(command=plot_selected_channel)
    channel_box.bind('<<ComboboxSelected>>', plot_selected_channel)
    plot_selected_channel()


def twa_analysis_window():
    if raw_data is None:
        status_label.config(text='Load an ECG file first')
        return

    twa_window = tk.Toplevel(root)
    twa_window.title('MMA-TWA Analysis')
    twa_window.geometry('1300x900')

    control_frame = tk.Frame(twa_window, bg=UI_BG)
    control_frame.pack(fill=tk.X, padx=20, pady=(8, 6))

    tk.Label(
        control_frame,
        text='Channel:',
        font=('Arial', 11),
    ).pack(side=tk.LEFT, padx=5)

    channel_var = tk.StringVar()
    channel_box = ttk.Combobox(
        control_frame,
        textvariable=channel_var,
        values=Channel_names,
        state='readonly',
        width=10,
    )
    channel_box.current(0)
    channel_box.pack(side=tk.LEFT, padx=5)

    info_frame = tk.Frame(control_frame, bg=UI_BG)
    info_frame.pack(side=tk.RIGHT, padx=10)

    primary_info_label = tk.Label(
        info_frame,
        text='',
        font=('Arial', 10, 'bold'),
        bg=UI_BG,
        fg='#1f2937',
        anchor='e',
        width=88,
    )
    primary_info_label.pack(anchor='e')

    secondary_info_label = tk.Label(
        info_frame,
        text='',
        font=('Arial', 9),
        bg=UI_BG,
        fg='#6b7280',
        anchor='e',
        width=88,
    )
    secondary_info_label.pack(anchor='e')

    twa_figure = plt.figure(figsize=(13, 9))
    grid = twa_figure.add_gridspec(
        3,
        2,
        height_ratios=[1.2, 1, 1],
    )

    ecg_ax = twa_figure.add_subplot(grid[0, :])
    segments_ax = twa_figure.add_subplot(grid[1, 0])
    mma_ax = twa_figure.add_subplot(grid[1, 1])
    trend_ax = twa_figure.add_subplot(grid[2, :])
    twa_figure.subplots_adjust(hspace=0.45, wspace=0.30)

    twa_canvas = FigureCanvasTkAgg(
        twa_figure,
        master=twa_window,
    )
    twa_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    twa_cache = {}

    def plot_twa_channel(event=None):
        ecg_ax.clear()
        segments_ax.clear()
        mma_ax.clear()
        trend_ax.clear()

        channel_name = channel_var.get()
        channel_index = Channel_names.index(channel_name)

        if channel_index not in twa_cache:
            signal = raw_data[:, channel_index]
            results = prepare_twa_analysis(signal)
            display_signal = qrs_morphology_signal(signal)
            twa_cache[channel_index] = (display_signal, results)

        display_signal, results = twa_cache[channel_index]
        qrst = results['qrst']
        r_peaks = qrst['r_peaks']
        t_peaks = qrst['t_peaks']
        segments = results['segments']
        valid_segments = results['valid_segments']
        template_a = results['template_a']
        template_b = results['template_b']
        twa_times = results['twa_times']
        twa_values = results['twa_values']
        twa_index = results['twa_index']
        final_twa = results['final_twa']
        a_count = results['a_count']
        b_count = results['b_count']

        time_axis = np.arange(len(display_signal)) / Fs
        display_signal_mv = adu_to_mv(display_signal)

        ecg_ax.plot(
            time_axis,
            display_signal_mv,
            color='black',
            linewidth=0.8,
            label=channel_name,
        )

        valid_r = r_peaks[~np.isnan(r_peaks)].astype(int)
        if len(valid_r) > 0:
            ecg_ax.scatter(
                valid_r / Fs,
                display_signal_mv[valid_r],
                color='green',
                s=28,
                label='R peak',
                zorder=5,
            )

        valid_t = t_peaks[~np.isnan(t_peaks)].astype(int)
        if len(valid_t) > 0:
            ecg_ax.scatter(
                valid_t / Fs,
                display_signal_mv[valid_t],
                color='red',
                s=28,
                label='T peak',
                zorder=5,
            )

        ecg_ax.set_title(f'{channel_name} - ECG with R and T Peaks')
        ecg_ax.set_xlabel('Time (s)')
        ecg_ax.set_ylabel(AMPLITUDE_LABEL)
        ecg_ax.grid(True, alpha=0.3)
        ecg_ax.legend()

        if len(valid_segments) > 0:
            segment_time = np.arange(valid_segments.shape[1]) / Fs

            for segment in valid_segments:
                segments_ax.plot(
                    segment_time,
                    adu_to_mv(segment),
                    alpha=0.25,
                    linewidth=0.8,
                )

            reference = np.median(valid_segments, axis=0)
            segments_ax.plot(
                segment_time,
                adu_to_mv(reference),
                color='black',
                linewidth=1.8,
                label='Median reference',
            )
            segments_ax.legend()

        segments_ax.set_title('Valid ST-T Segments')
        segments_ax.set_xlabel('Time after QRS offset (s)')
        segments_ax.set_ylabel(AMPLITUDE_LABEL)
        segments_ax.grid(True, alpha=0.3)

        extracted_count = len(segments)
        valid_count = len(valid_segments)
        acceptance_rate = 100.0 * valid_count / max(extracted_count, 1)

        primary_info_label.config(
            text=(
                f'Segments: {extracted_count}   |   '
                f'Valid: {valid_count}   |   '
                f'Accepted: {acceptance_rate:.1f}%'
            )
        )
        secondary_info_label.config(
            text=f'A beats: {a_count}   |   B beats: {b_count}'
        )

        if template_a is not None and template_b is not None:
            template_time = np.arange(len(template_a)) / Fs
            mma_ax.plot(
                template_time,
                adu_to_mv(template_a),
                label='MMA A',
            )
            mma_ax.plot(
                template_time,
                adu_to_mv(template_b),
                label='MMA B',
            )

            if np.isfinite(final_twa):
                twa_index_int = int(twa_index)
                twa_time_point = twa_index_int / Fs
                mma_ax.axvline(
                    twa_time_point,
                    linestyle='--',
                    linewidth=1.2,
                    label='Max |A-B|',
                )

                final_twa_uv = float(adu_to_uv(final_twa))
                mma_ax.text(
                    0.02,
                    0.95,
                    f'TWA = {final_twa_uv:.2f} µV\n'
                    f'A = {a_count}, B = {b_count}',
                    transform=mma_ax.transAxes,
                    verticalalignment='top',
                )
            else:
                mma_ax.text(
                    0.02,
                    0.95,
                    'Insufficient valid beats for MMA-TWA\n'
                    f'A = {a_count}, B = {b_count}',
                    transform=mma_ax.transAxes,
                    verticalalignment='top',
                )

            mma_ax.legend()
        else:
            mma_ax.text(
                0.5,
                0.5,
                'Insufficient valid beats for MMA-TWA',
                transform=mma_ax.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
            )

        mma_ax.set_title('MMA A / B Templates')
        mma_ax.set_xlabel('Time after QRS offset (s)')
        mma_ax.set_ylabel(AMPLITUDE_LABEL)
        mma_ax.grid(True, alpha=0.3)

        if np.isfinite(final_twa) and len(twa_values) > 0:
            trend_ax.plot(
                twa_times,
                adu_to_uv(twa_values),
                marker='o',
                linewidth=1,
            )
        elif np.isfinite(final_twa):
            trend_ax.text(
                0.5,
                0.5,
                f'TWA = {float(adu_to_uv(final_twa)):.2f} µV\n'
                'No post-initialization trend points',
                transform=trend_ax.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
            )
        else:
            trend_ax.text(
                0.5,
                0.5,
                'Insufficient valid beats for MMA-TWA',
                transform=trend_ax.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
            )

        trend_ax.set_title('TWA Trend')
        trend_ax.set_xlabel('Time (s)')
        trend_ax.set_ylabel('TWA (µV)')
        trend_ax.grid(True, alpha=0.3)
        twa_canvas.draw_idle()

    channel_box.bind('<<ComboboxSelected>>', plot_twa_channel)
    plot_twa_channel()


def pacemaker_analysis_window():
    if raw_data is None:
        status_label.config(text='Load an ECG file first')
        return

    pacemaker_window = tk.Toplevel(root)
    pacemaker_window.title('Pacemaker Artifact Analysis')
    pacemaker_window.geometry('1300x700')

    control_frame = tk.Frame(pacemaker_window, bg=UI_BG)
    control_frame.pack(fill=tk.X, padx=20, pady=(8, 6))

    tk.Label(
        control_frame,
        text='Channel:',
        font=('Arial', 11),
    ).pack(side=tk.LEFT, padx=5)

    channel_var = tk.StringVar()
    channel_box = ttk.Combobox(
        control_frame,
        textvariable=channel_var,
        values=Channel_names,
        state='readonly',
        width=10,
    )
    channel_box.current(0)
    channel_box.pack(side=tk.LEFT, padx=5)

    results = detect_pacemaker_artifacts(raw_data)
    spike_indices = results['spikes']
    lead_support = results['lead_support']
    channel_support = results['channel_support']

    info_frame = tk.Frame(control_frame, bg=UI_BG)
    info_frame.pack(side=tk.RIGHT, padx=10)

    metadata_label = tk.Label(
        info_frame,
        text=current_metadata(),
        font=('Arial', 9),
        bg=UI_BG,
        fg='#6b7280',
        anchor='e',
        width=82,
    )
    metadata_label.pack(anchor='e')

    info_label = tk.Label(
        info_frame,
        text='',
        font=('Arial', 10, 'bold'),
        bg=UI_BG,
        fg='#1f2937',
        anchor='e',
        width=82,
    )
    info_label.pack(anchor='e')

    pacemaker_figure, pacemaker_ax = plt.subplots(figsize=(13, 6))
    pacemaker_canvas = FigureCanvasTkAgg(
        pacemaker_figure,
        master=pacemaker_window,
    )
    pacemaker_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    pacemaker_signal_line, = pacemaker_ax.plot(
        [],
        [],
        color='black',
        linewidth=0.8,
        label='ECG',
    )
    pacemaker_artifact_line, = pacemaker_ax.plot(
        [],
        [],
        color='red',
        linestyle='--',
        linewidth=1.6,
        alpha=0.95,
        label='Channel-supported artifact',
    )

    pacemaker_ax.set_xlabel('Time (s)')
    pacemaker_ax.set_ylabel(AMPLITUDE_LABEL)
    pacemaker_ax.grid(True, alpha=0.3)
    pacemaker_ax.legend(
        loc='upper right',
        frameon=True,
        borderaxespad=0.6,
    )

    total_duration = len(raw_data) / Fs
    pacemaker_window_length = 10
    max_start_time = max(
        0,
        total_duration - pacemaker_window_length,
    )

    pacemaker_slider = tk.Scale(
        pacemaker_window,
        from_=0,
        to=max_start_time,
        orient=tk.HORIZONTAL,
        resolution=0.1,
        showvalue=0,
    )
    pacemaker_slider.pack(fill=tk.X, padx=40, pady=10)

    def plot_pacemaker(event=None):
        channel_name = channel_var.get()
        channel_index = Channel_names.index(channel_name)
        start_time = float(pacemaker_slider.get())
        start = int(start_time * Fs)
        end = min(
            start + pacemaker_window_length * Fs,
            len(raw_data),
        )

        time_axis = np.arange(start, end) / Fs
        signal = adu_to_mv(raw_data[start:end, channel_index])

        selected_channel_support = (
            channel_support[:, channel_index]
            if len(channel_support) > 0
            else np.array([], dtype=bool)
        )

        channel_spike_indices = spike_indices[selected_channel_support]
        channel_lead_support = lead_support[selected_channel_support]

        visible_mask = (
            (channel_spike_indices >= start)
            & (channel_spike_indices < end)
        )
        visible_spikes = channel_spike_indices[visible_mask]
        visible_support = channel_lead_support[visible_mask]

        pacemaker_signal_line.set_data(time_axis, signal)
        pacemaker_ax.set_xlim(
            start_time,
            start_time + pacemaker_window_length,
        )

        if len(signal) > 0:
            y_min = float(np.min(signal))
            y_max = float(np.max(signal))
            margin = (y_max - y_min) * 0.08

            if margin == 0:
                margin = 1.0

            pacemaker_ax.set_ylim(y_min - margin, y_max + margin)

        y_min, y_max = pacemaker_ax.get_ylim()
        pacemaker_artifact_line.set_data(
            *vertical_marker_data(
                visible_spikes,
                y_min,
                y_max,
            )
        )

        pacemaker_ax.set_title(
            f'{channel_name} - Pacemaker Artifact Analysis'
        )

        median_support = (
            int(np.median(visible_support))
            if len(visible_support) > 0
            else 0
        )

        info_label.config(
            text=(
                f'Global candidates: {len(spike_indices)}   |   '
                f'{channel_name} supported: {len(channel_spike_indices)}   |   '
                f'Visible: {len(visible_spikes)}   |   '
                f'Median lead support: {median_support}/12   |   '
                f'12-channel PCA-PC1 + channel slope support'
            )
        )

        pacemaker_canvas.draw_idle()

    pacemaker_slider.config(command=plot_pacemaker)
    channel_box.bind('<<ComboboxSelected>>', plot_pacemaker)
    plot_pacemaker()


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description='12-lead ECG viewer and analysis prototype'
    )
    parser.add_argument(
        '--fs',
        type=int,
        default=500,
        help='Sampling frequency in Hz (default: 500)',
    )
    parser.add_argument(
        '--gain',
        type=float,
        default=1.0,
        help=(
            'Input conversion factor in ADU per mV. '
            'Use 1.0 when the CSV amplitudes are already in mV.'
        ),
    )
    return parser


def main():
    global Fs
    global ADC_GAIN_ADU_PER_MV
    global window_samples
    global step_samples
    global root
    global welcome_frame
    global viewer_frame
    global figure
    global axes
    global lines
    global canvas
    global time_slider
    global status_label
    global notch_button
    global baseline_button
    global emg_button
    global qrst_button
    global twa_button
    global pacemaker_button

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.fs <= 0:
        parser.error('--fs must be greater than 0')

    if args.gain <= 0:
        parser.error('--gain must be greater than 0')

    Fs = args.fs
    ADC_GAIN_ADU_PER_MV = args.gain
    window_samples = int(Fs * Wl)
    step_samples = max(1, int(Fs * Update / 1000))

    root = tk.Tk()
    root.title('ECG Viewer 1.0')
    root.geometry('1400x900')
    root.minsize(1100, 700)
    root.configure(bg=UI_BG)

    title_frame = tk.Frame(root, bg=UI_BG)
    title_frame.pack(fill=tk.X, padx=18, pady=(10, 4))

    tk.Label(
        title_frame,
        text='ECG Viewer 1.0',
        font=('Arial', 18, 'bold'),
        bg=UI_BG,
    ).pack(side=tk.LEFT)

    tk.Label(
        title_frame,
        text='12-channel ECG visualization and analysis',
        font=('Arial', 10),
        fg='#5f6368',
        bg=UI_BG,
    ).pack(side=tk.LEFT, padx=12, pady=6)

    toolbar = tk.Frame(root, bg=UI_BG)
    toolbar.pack(fill=tk.X, padx=14, pady=(5, 6))

    file_group = tk.LabelFrame(
        toolbar,
        text='File',
        font=('Arial', 10, 'bold'),
        bg=PANEL_BG,
        padx=6,
        pady=7,
    )
    file_group.pack(side=tk.LEFT, padx=6)

    play_button = tk.Button(
        file_group,
        text='Play ECG File',
        command=select_file,
        font=('Arial', 10),
        bg=FILE_BUTTON_BG,
        relief=tk.FLAT,
        padx=12,
    )
    play_button.pack(side=tk.LEFT, padx=3)

    open_button = tk.Button(
        file_group,
        text='Open ECG File',
        command=open_file,
        font=('Arial', 10),
        bg=FILE_BUTTON_BG,
        relief=tk.FLAT,
        padx=12,
    )
    open_button.pack(side=tk.LEFT, padx=3)

    filter_group = tk.LabelFrame(
        toolbar,
        text='Display Filters',
        font=('Arial', 10, 'bold'),
        bg=PANEL_BG,
        padx=6,
        pady=7,
    )
    filter_group.pack(side=tk.LEFT, padx=6)

    notch_button = tk.Button(
        filter_group,
        text='Notch 50 Hz',
        command=notch,
        font=('Arial', 10),
        bg=FILTER_BUTTON_BG,
        relief=tk.FLAT,
        state='disabled',
        padx=10,
    )
    notch_button.pack(side=tk.LEFT, padx=3)

    baseline_button = tk.Button(
        filter_group,
        text='Baseline 0.67 Hz',
        command=baseline,
        font=('Arial', 10),
        bg=FILTER_BUTTON_BG,
        relief=tk.FLAT,
        state='disabled',
        padx=10,
    )
    baseline_button.pack(side=tk.LEFT, padx=3)

    emg_button = tk.Button(
        filter_group,
        text='EMG 35 Hz',
        command=emg,
        font=('Arial', 10),
        bg=FILTER_BUTTON_BG,
        relief=tk.FLAT,
        state='disabled',
        padx=10,
    )
    emg_button.pack(side=tk.LEFT, padx=3)

    analysis_group = tk.LabelFrame(
        toolbar,
        text='Analysis',
        font=('Arial', 10, 'bold'),
        bg=PANEL_BG,
        padx=6,
        pady=7,
    )
    analysis_group.pack(side=tk.LEFT, padx=6)

    qrst_button = tk.Button(
        analysis_group,
        text='Q-R-S-T',
        command=qrst_analysis_window,
        font=('Arial', 10),
        bg=ANALYSIS_BUTTON_BG,
        relief=tk.FLAT,
        state='disabled',
        padx=12,
    )
    qrst_button.pack(side=tk.LEFT, padx=3)

    twa_button = tk.Button(
        analysis_group,
        text='MMA-TWA',
        command=twa_analysis_window,
        font=('Arial', 10),
        bg=ANALYSIS_BUTTON_BG,
        relief=tk.FLAT,
        state='disabled',
        padx=12,
    )
    twa_button.pack(side=tk.LEFT, padx=3)

    pacemaker_button = tk.Button(
        analysis_group,
        text='Pacemaker',
        command=pacemaker_analysis_window,
        font=('Arial', 10),
        bg=ANALYSIS_BUTTON_BG,
        relief=tk.FLAT,
        state='disabled',
        padx=12,
    )
    pacemaker_button.pack(side=tk.LEFT, padx=3)

    exit_button = tk.Button(
        toolbar,
        text='Exit',
        command=root.destroy,
        font=('Arial', 10),
        bg=EXIT_BUTTON_BG,
        relief=tk.FLAT,
        padx=16,
    )
    exit_button.pack(side=tk.RIGHT, padx=5, pady=9)

    status_label = tk.Label(
        root,
        text='No ECG file loaded',
        font=('Arial', 10),
        bg=PANEL_BG,
        fg='#374151',
        anchor='w',
        relief=tk.GROOVE,
        padx=10,
        pady=5,
    )
    status_label.pack(fill=tk.X, padx=18, pady=(2, 6))

    welcome_frame = tk.Frame(root, bg=PANEL_BG)
    welcome_frame.pack(
        fill=tk.BOTH,
        expand=True,
        padx=18,
        pady=8,
    )

    welcome_content = tk.Frame(welcome_frame, bg=PANEL_BG)
    welcome_content.place(relx=0.5, rely=0.46, anchor='center')

    tk.Label(
        welcome_content,
        text='No ECG file loaded',
        font=('Arial', 24, 'bold'),
        bg=PANEL_BG,
        fg='#1f2937',
    ).pack(pady=(0, 8))

    tk.Label(
        welcome_content,
        text='Load a 12-channel CSV file to start visualization and analysis.',
        font=('Arial', 12),
        bg=PANEL_BG,
        fg='#6b7280',
    ).pack(pady=(0, 20))

    welcome_buttons = tk.Frame(welcome_content, bg=PANEL_BG)
    welcome_buttons.pack()

    tk.Button(
        welcome_buttons,
        text='Open ECG File',
        command=open_file,
        font=('Arial', 11),
        bg=FILE_BUTTON_BG,
        relief=tk.FLAT,
        padx=18,
        pady=6,
    ).pack(side=tk.LEFT, padx=5)

    tk.Button(
        welcome_buttons,
        text='Play ECG File',
        command=select_file,
        font=('Arial', 11),
        bg=FILE_BUTTON_BG,
        relief=tk.FLAT,
        padx=18,
        pady=6,
    ).pack(side=tk.LEFT, padx=5)

    viewer_frame = tk.Frame(root, bg=PANEL_BG)

    figure, ax_matrix = plt.subplots(
        6,
        2,
        figsize=(14, 8),
    )
    figure.subplots_adjust(
        left=0.05,
        right=0.985,
        top=0.97,
        bottom=0.05,
        hspace=0.32,
        wspace=0.14,
    )

    axes = []
    for row in range(6):
        axes.append(ax_matrix[row, 0])
    for row in range(6):
        axes.append(ax_matrix[row, 1])

    lines = []
    for i in range(12):
        ax = axes[i]
        ax.set_title(
            Channel_names[i],
            loc='left',
            fontsize=10,
            fontweight='bold',
        )
        ax.set_ylabel('mV', fontsize=8)
        ax.set_xlim(0, Wl)
        ax.minorticks_on()
        ax.grid(
            True,
            which='major',
            linewidth=0.5,
            alpha=0.18,
        )
        ax.grid(
            True,
            which='minor',
            linewidth=0.25,
            alpha=0.10,
        )
        line, = ax.plot(
            [],
            [],
            linewidth=0.8,
            color='black',
        )
        lines.append(line)

    canvas = FigureCanvasTkAgg(figure, master=viewer_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    time_slider = tk.Scale(
        viewer_frame,
        from_=0,
        to=1,
        orient=tk.HORIZONTAL,
        command=show_window,
        resolution=0.1,
        showvalue=0,
        state='disabled',
        highlightthickness=0,
    )
    time_slider.pack(fill=tk.X, padx=40, pady=(4, 12))

    root.mainloop()


if __name__ == '__main__':
    main()
