import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# Placeholder for actual breath data if needed for some features
# from .breath_detection import detect_breaths_from_flow # Assuming in same package

def get_signal_segment(signal: pd.Series, event_start_time: pd.Timestamp,
                       event_end_time: pd.Timestamp,
                       pre_event_window_s: int = 30,
                       post_event_window_s: int = 30) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """
    Extracts signal segments for pre-event, during-event, and post-event periods.

    Args:
        signal (pd.Series): The signal to segment (e.g., flow_rate, pressure).
        event_start_time (pd.Timestamp): Start time of the event.
        event_end_time (pd.Timestamp): End time of the event.
        pre_event_window_s (int): Duration of the pre-event window in seconds.
        post_event_window_s (int): Duration of the post-event window in seconds.

    Returns:
        Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        Pre-event segment, during-event segment, post-event segment.
        Returns None for a segment if data is not available.
    """
    if signal.empty:
        return None, None, None

    # Ensure timestamps are compatible (naive vs. aware)
    if event_start_time.tzinfo != signal.index.tzinfo:
        if signal.index.tzinfo is None: # Make event times naive
            event_start_time = event_start_time.tz_localize(None)
            event_end_time = event_end_time.tz_localize(None)
        else: # Make event times aware (match signal)
            event_start_time = event_start_time.tz_localize(signal.index.tzinfo)
            event_end_time = event_end_time.tz_localize(signal.index.tzinfo)


    pre_start = event_start_time - pd.Timedelta(seconds=pre_event_window_s)
    post_end = event_end_time + pd.Timedelta(seconds=post_event_window_s)

    try:
        pre_segment = signal.loc[pre_start : event_start_time - pd.Timedelta(nanoseconds=1)] # up to, but not including event_start
        if pre_segment.empty: pre_segment = None
    except Exception: # KeyError if out of bounds
        pre_segment = None

    try:
        during_segment = signal.loc[event_start_time : event_end_time - pd.Timedelta(nanoseconds=1)] # up to, but not including event_end
        if during_segment.empty: during_segment = None
    except Exception:
        during_segment = None

    try:
        post_segment = signal.loc[event_end_time : post_end - pd.Timedelta(nanoseconds=1)]
        if post_segment.empty: post_segment = None
    except Exception:
        post_segment = None

    return pre_segment, during_segment, post_segment


def calculate_segment_stats(segment: Optional[pd.Series], prefix: str) -> dict:
    """Calculates basic statistics for a signal segment."""
    stats = {}
    if segment is not None and not segment.empty:
        stats[f'{prefix}_mean'] = segment.mean()
        stats[f'{prefix}_median'] = segment.median()
        stats[f'{prefix}_std'] = segment.std()
        stats[f'{prefix}_min'] = segment.min()
        stats[f'{prefix}_max'] = segment.max()
        stats[f'{prefix}_skew'] = skew(segment.dropna()) if not segment.dropna().empty else 0
        stats[f'{prefix}_kurtosis'] = kurtosis(segment.dropna()) if not segment.dropna().empty else 0
        stats[f'{prefix}_abs_mean'] = segment.abs().mean() # Mean of absolute values
        stats[f'{prefix}_duration_s'] = (segment.index[-1] - segment.index[0]).total_seconds() if len(segment.index) > 1 else 0
    else: # Fill with NaNs or Zeros if segment is None or empty
        for stat_name in ['mean', 'median', 'std', 'min', 'max', 'skew', 'kurtosis', 'abs_mean', 'duration_s']:
            stats[f'{prefix}_{stat_name}'] = np.nan
    return stats

def inspiratory_flattening_index(insp_flow_segment: pd.Series, sampling_freq_hz: float) -> float:
    """
    Calculates an inspiratory flattening index.
    A simple version: ratio of actual volume in first 50% of insp. time vs. ideal triangular breath.
    Assumes insp_flow_segment is positive flow during inspiration.
    """
    if insp_flow_segment is None or insp_flow_segment.empty or insp_flow_segment.max() <= 1e-3:
        return np.nan

    insp_duration_s = (insp_flow_segment.index[-1] - insp_flow_segment.index[0]).total_seconds()
    if insp_duration_s == 0: return np.nan

    mid_insp_time_s = insp_duration_s / 2
    peak_flow = insp_flow_segment.max()

    # Actual volume in first 50% of time
    first_half_segment = insp_flow_segment[insp_flow_segment.index <= insp_flow_segment.index[0] + pd.Timedelta(seconds=mid_insp_time_s)]
    if first_half_segment.empty: return np.nan

    actual_vol_first_half = np.trapz(first_half_segment.values, dx=1.0/sampling_freq_hz)

    # Ideal volume in first 50% for a triangular breath peaking at peak_flow at mid_insp_time_s
    # Area of triangle = 0.5 * base * height. Here, base = mid_insp_time_s, height = peak_flow.
    # This assumes peak occurs at mid-inspiration for the "ideal" triangle.
    # A more robust "ideal" might assume peak occurs at actual peak time.
    # For simplicity, let's use peak_flow as the height of this hypothetical triangle.
    ideal_vol_first_half = 0.5 * mid_insp_time_s * peak_flow * 0.5 # (0.5 * base * height for the half-triangle)
                                                                # Area of first half of a symmetric triangle is 1/4 of (total_duration * peak_flow)
                                                                # Or, 0.5 * (duration of first half) * (flow at end of first half, if linear ramp to peak)

    # Let's use a simpler definition: ratio of mean flow in first half to peak flow.
    # For a perfect square wave (max flattening), this would be 1.
    # For a perfect triangle (peaking at mid insp), mean flow in first half is peak_flow/2. Ratio = 0.5.
    # So, higher values mean more flattening.
    mean_flow_first_half = first_half_segment.mean()
    if peak_flow < 1e-3: return np.nan # Avoid division by zero if no flow

    flattening_index = mean_flow_first_half / peak_flow
    return flattening_index


def flow_shape_features(flow_segment: pd.Series, sampling_freq_hz: float, prefix: str) -> dict:
    """ Calculates features related to flow shape from a segment (e.g., an inspiration)."""
    features = {}
    if flow_segment is None or len(flow_segment) < 2: # Need at least 2 points for duration/integration
        features[f'{prefix}_flattening_idx'] = np.nan
        features[f'{prefix}_integral'] = np.nan
        features[f'{prefix}_slope_first_half'] = np.nan
        features[f'{prefix}_slope_second_half'] = np.nan
        return features

    features[f'{prefix}_flattening_idx'] = inspiratory_flattening_index(flow_segment, sampling_freq_hz)
    features[f'{prefix}_integral'] = np.trapz(flow_segment.values, dx=1.0/sampling_freq_hz)

    # Slopes
    duration_s = (flow_segment.index[-1] - flow_segment.index[0]).total_seconds()
    if duration_s > 0:
        mid_point_idx = len(flow_segment) // 2
        if mid_point_idx > 0:
            first_half = flow_segment.iloc[:mid_point_idx]
            second_half = flow_segment.iloc[mid_point_idx:]

            if len(first_half) >= 2:
                dt_first = (first_half.index[-1] - first_half.index[0]).total_seconds()
                features[f'{prefix}_slope_first_half'] = (first_half.iloc[-1] - first_half.iloc[0]) / dt_first if dt_first > 0 else np.nan
            else:
                features[f'{prefix}_slope_first_half'] = np.nan

            if len(second_half) >= 2:
                dt_second = (second_half.index[-1] - second_half.index[0]).total_seconds()
                features[f'{prefix}_slope_second_half'] = (second_half.iloc[-1] - second_half.iloc[0]) / dt_second if dt_second > 0 else np.nan
            else:
                features[f'{prefix}_slope_second_half'] = np.nan
        else:
            features[f'{prefix}_slope_first_half'] = np.nan
            features[f'{prefix}_slope_second_half'] = np.nan
    else:
        features[f'{prefix}_slope_first_half'] = np.nan
        features[f'{prefix}_slope_second_half'] = np.nan

    return features


def periodicity_features(segment: Optional[pd.Series], sampling_freq_hz: float, prefix: str) -> dict:
    """Calculates features related to signal periodicity (e.g., for Cheyne-Stokes)."""
    features = {}
    if segment is None or len(segment) < sampling_freq_hz * 5: # Need at least a few seconds of data, preferably more
        features[f'{prefix}_power_lf'] = np.nan # Low frequency power (e.g. 0.01-0.05 Hz for CS-like)
        features[f'{prefix}_power_hf'] = np.nan # Higher frequency power (e.g. respiratory band 0.1-0.5 Hz)
        features[f'{prefix}_lf_hf_ratio'] = np.nan
        return features

    # Welch's method for power spectral density
    # Define frequency bands:
    # LF for Cheyne-Stokes like patterns (e.g., cycles of 20-100s, so ~0.01-0.05 Hz)
    # HF for normal respiratory band (e.g., cycles of 2-10s, so ~0.1-0.5 Hz)
    freqs, psd = welch(segment.dropna(), fs=sampling_freq_hz, nperseg=min(len(segment.dropna()), 256)) # nperseg can be tuned

    lf_band = (0.01, 0.05)
    hf_band = (0.1, 0.5)

    lf_power = np.trapz(psd[(freqs >= lf_band[0]) & (freqs <= lf_band[1])],
                        freqs[(freqs >= lf_band[0]) & (freqs <= lf_band[1])])
    hf_power = np.trapz(psd[(freqs >= hf_band[0]) & (freqs <= hf_band[1])],
                        freqs[(freqs >= hf_band[0]) & (freqs <= hf_band[1])])

    features[f'{prefix}_power_lf'] = lf_power if lf_power is not None else np.nan
    features[f'{prefix}_power_hf'] = hf_power if hf_power is not None else np.nan
    if hf_power is not None and hf_power > 1e-6 and lf_power is not None: # Avoid division by zero
        features[f'{prefix}_lf_hf_ratio'] = lf_power / hf_power
    else:
        features[f'{prefix}_lf_hf_ratio'] = np.nan

    return features


def extract_features_for_event(
    event_row: pd.Series,
    flow_series: pd.Series,
    pressure_series: pd.Series,
    # breath_df: Optional[pd.DataFrame], # If using breath-by-breath features
    sampling_freq_hz: float,
    pre_event_window_s: int = 30,
    post_event_window_s: int = 30
) -> dict:
    """
    Extracts a feature vector for a single respiratory event.

    Args:
        event_row (pd.Series): A row from the event DataFrame (from event_detection.py).
                               Must contain 'event_start_time', 'event_end_time', 'event_type'.
        flow_series (pd.Series): Full flow rate signal.
        pressure_series (pd.Series): Full pressure signal.
        # breath_df (Optional[pd.DataFrame]): DataFrame of detected breaths.
        sampling_freq_hz (float): Sampling frequency.
        pre_event_window_s (int): Window size for pre-event features.
        post_event_window_s (int): Window size for post-event features.

    Returns:
        dict: A dictionary of extracted features for the event.
    """
    features = {
        'event_id': event_row.name, # Assuming event_row is a row from a DataFrame with an index
        'event_type_original': event_row['event_type'],
        'event_duration_s': event_row['event_duration_s'],
        'avg_flow_reduction_percent_original': event_row.get('avg_flow_reduction_percent', np.nan),
        'min_flow_during_event_ratio_original': event_row.get('min_flow_during_event_ratio', np.nan),
        'excluded_due_to_leak_original': event_row.get('excluded_due_to_leak', False)
    }

    event_start = event_row['event_start_time']
    event_end = event_row['event_end_time']

    # Get signal segments for flow and pressure
    flow_pre, flow_during, flow_post = get_signal_segment(flow_series, event_start, event_end, pre_event_window_s, post_event_window_s)
    pressure_pre, pressure_during, pressure_post = get_signal_segment(pressure_series, event_start, event_end, pre_event_window_s, post_event_window_s)

    # Basic stats for each segment
    features.update(calculate_segment_stats(flow_pre, 'flow_pre'))
    features.update(calculate_segment_stats(flow_during, 'flow_during'))
    features.update(calculate_segment_stats(flow_post, 'flow_post'))

    features.update(calculate_segment_stats(pressure_pre, 'pressure_pre'))
    features.update(calculate_segment_stats(pressure_during, 'pressure_during'))
    features.update(calculate_segment_stats(pressure_post, 'pressure_post'))

    # Flow Shape features (example for inspiratory part of breaths if available, or just whole segments)
    # This requires identifying inspirations within segments, or applying to segments directly.
    # For simplicity, apply to overall flow_during segment if it's a hypopnea (where flow exists)
    if event_row['event_type'] == 'hypopnea_candidate' and flow_during is not None:
        # A more advanced version would segment breaths within flow_during and analyze each.
        # For now, simple features on the whole hypopnea flow:
        features.update(flow_shape_features(flow_during, sampling_freq_hz, 'hypopnea_flow_shape'))
    else: # For apneas, flow is minimal, so shape features less relevant on 'during' segment.
        features.update(flow_shape_features(None, sampling_freq_hz, 'hypopnea_flow_shape')) # Add NaNs


    # Recovery breath features from flow_post (e.g. first breath after event)
    # This would ideally use the breath_df to find the first 1-2 breaths in flow_post
    if flow_post is not None and len(flow_post) > sampling_freq_hz * 1 : # Need at least 1s of post data
        # Simplified: max flow in first 5s post event vs mean flow pre event
        first_5s_post_flow = flow_post.first('5s')
        if not first_5s_post_flow.empty:
            features['recovery_peak_flow_post5s'] = first_5s_post_flow.max()
            if flow_pre is not None and not flow_pre.empty and flow_pre.abs().mean() > 1e-3:
                 features['recovery_peak_ratio_vs_pre_mean'] = first_5s_post_flow.max() / flow_pre.abs().mean()
            else:
                 features['recovery_peak_ratio_vs_pre_mean'] = np.nan
        else:
            features['recovery_peak_flow_post5s'] = np.nan
            features['recovery_peak_ratio_vs_pre_mean'] = np.nan
    else:
        features['recovery_peak_flow_post5s'] = np.nan
        features['recovery_peak_ratio_vs_pre_mean'] = np.nan


    # Pre-event stability / Periodicity
    # Using flow_pre segment for this
    features.update(periodicity_features(flow_pre, sampling_freq_hz, 'flow_pre_periodicity'))
    if flow_pre is not None and not flow_pre.empty:
        # Coefficient of variation of flow in pre-event window
        if flow_pre.mean() != 0:
             features['flow_pre_coeff_var'] = flow_pre.std() / np.abs(flow_pre.mean())
        else:
             features['flow_pre_coeff_var'] = np.nan
    else:
        features['flow_pre_coeff_var'] = np.nan


    # Pressure change features
    if pressure_during is not None and not pressure_during.empty:
        features['pressure_change_during_event'] = pressure_during.iloc[-1] - pressure_during.iloc[0]
        features['pressure_max_during_event'] = pressure_during.max()
        if pressure_pre is not None and not pressure_pre.empty:
            features['pressure_rise_from_pre_to_during'] = pressure_during.mean() - pressure_pre.mean()
        else:
            features['pressure_rise_from_pre_to_during'] = np.nan
    else:
        features['pressure_change_during_event'] = np.nan
        features['pressure_max_during_event'] = np.nan
        features['pressure_rise_from_pre_to_during'] = np.nan

    # Ensure all expected feature keys are present, filling with NaN if calculation failed
    # This list should be maintained based on features defined.
    expected_feature_prefixes = [
        'flow_pre', 'flow_during', 'flow_post',
        'pressure_pre', 'pressure_during', 'pressure_post'
    ]
    stat_suffixes = ['mean', 'median', 'std', 'min', 'max', 'skew', 'kurtosis', 'abs_mean', 'duration_s']

    for pfx in expected_feature_prefixes:
        for sfx in stat_suffixes:
            key = f'{pfx}_{sfx}'
            if key not in features:
                features[key] = np.nan

    other_expected_keys = [
        'hypopnea_flow_shape_flattening_idx', 'hypopnea_flow_shape_integral',
        'hypopnea_flow_shape_slope_first_half', 'hypopnea_flow_shape_slope_second_half',
        'recovery_peak_flow_post5s', 'recovery_peak_ratio_vs_pre_mean',
        'flow_pre_periodicity_power_lf', 'flow_pre_periodicity_power_hf', 'flow_pre_periodicity_lf_hf_ratio',
        'flow_pre_coeff_var', 'pressure_change_during_event', 'pressure_max_during_event',
        'pressure_rise_from_pre_to_during'
    ]
    for key in other_expected_keys:
        if key not in features:
            features[key] = np.nan

    return features

if __name__ == '__main__':
    # Setup for testing - requires data similar to event_detection output
    sampling_freq_hz = 25
    duration_sec = 200 # Longer duration for more context
    num_samples = duration_sec * sampling_freq_hz
    time_stamps = np.arange(num_samples) / sampling_freq_hz
    time_index = pd.to_datetime(time_stamps, unit='s')

    # Create dummy flow and pressure series
    flow_s = pd.Series(np.random.randn(num_samples) * 0.3, index=time_index, name='flow_rate')
    pressure_s = pd.Series(np.ones(num_samples) * 10 + np.random.randn(num_samples) * 0.1, index=time_index, name='pressure')

    # Simulate some flow characteristics for events
    # Event 1: "Obstructive-like" hypopnea
    event1_start_s, event1_end_s = 60, 75
    event1_start_idx, event1_end_idx = int(event1_start_s*sampling_freq_hz), int(event1_end_s*sampling_freq_hz)
    flow_s.iloc[event1_start_idx - 10*sampling_freq_hz : event1_start_idx] *= 1.5 # Bit unstable before
    flow_s.iloc[event1_start_idx : event1_end_idx] = 0.1 + 0.05 * np.sin(np.linspace(0, 3*np.pi, event1_end_idx - event1_start_idx)) # Flattened flow
    flow_s.iloc[event1_end_idx : event1_end_idx + 5*sampling_freq_hz] *= 2.5 # Recovery breaths
    pressure_s.iloc[event1_start_idx : event1_end_idx + 5*sampling_freq_hz] += 2 # Pressure increase

    # Event 2: "Central-like" apnea
    event2_start_s, event2_end_s = 120, 135
    event2_start_idx, event2_end_idx = int(event2_start_s*sampling_freq_hz), int(event2_end_s*sampling_freq_hz)
    # Smooth decrement into apnea
    for i in range(10 * sampling_freq_hz):
        flow_s.iloc[event2_start_idx - i] *= ( (10*sampling_freq_hz - i) / (10*sampling_freq_hz) )**2
    flow_s.iloc[event2_start_idx : event2_end_idx] = 0.01 * np.random.randn(event2_end_idx - event2_start_idx) # Apnea
    # Smooth recovery
    for i in range(10 * sampling_freq_hz):
         flow_s.iloc[event2_end_idx + i] *= (i / (10*sampling_freq_hz))**2


    # Create a dummy events DataFrame
    events_data = [
        {'event_start_time': pd.Timestamp(f'{event1_start_s}s'), 'event_end_time': pd.Timestamp(f'{event1_end_s}s'),
         'event_type': 'hypopnea_candidate', 'event_duration_s': event1_end_s - event1_start_s,
         'avg_flow_reduction_percent': 60.0, 'min_flow_during_event_ratio': 0.3, 'excluded_due_to_leak': False},
        {'event_start_time': pd.Timestamp(f'{event2_start_s}s'), 'event_end_time': pd.Timestamp(f'{event2_end_s}s'),
         'event_type': 'apnea_candidate', 'event_duration_s': event2_end_s - event2_start_s,
         'avg_flow_reduction_percent': 95.0, 'min_flow_during_event_ratio': 0.05, 'excluded_due_to_leak': False},
    ]
    # Ensure event timestamps are compatible with signal index tz
    if flow_s.index.tz is not None:
        for ev in events_data:
            ev['event_start_time'] = ev['event_start_time'].tz_localize(flow_s.index.tz)
            ev['event_end_time'] = ev['event_end_time'].tz_localize(flow_s.index.tz)

    events_df = pd.DataFrame(events_data)

    print("--- Testing Feature Extraction for Events ---")
    all_event_features = []
    for idx, event_row in events_df.iterrows():
        # print(f"\nProcessing event {idx} ({event_row['event_type']})")
        # Manually set name for event_row if it's not set (when iterating from df.iterrows, idx is name)
        event_row.name = idx
        features = extract_features_for_event(event_row, flow_s, pressure_s, sampling_freq_hz)
        all_event_features.append(features)

    features_df = pd.DataFrame(all_event_features)

    if not features_df.empty:
        print(f"Extracted features for {len(features_df)} events.")
        pd.set_option('display.max_columns', None)
        print("Features for first event:")
        print(features_df.head(1).T) # Transpose for better readability of many features

        # Basic checks
        assert not features_df.isnull().all().all(), "Feature DataFrame should not be all NaNs."
        assert 'flow_pre_mean' in features_df.columns, "flow_pre_mean missing."
        assert 'pressure_during_std' in features_df.columns, "pressure_during_std missing."
        assert 'hypopnea_flow_shape_flattening_idx' in features_df.columns
        assert 'recovery_peak_flow_post5s' in features_df.columns
        assert 'flow_pre_periodicity_lf_hf_ratio' in features_df.columns

        # Check if hypopnea features are NaN for apnea and present for hypopnea
        hypopnea_event_features = features_df[features_df['event_type_original'] == 'hypopnea_candidate']
        apnea_event_features = features_df[features_df['event_type_original'] == 'apnea_candidate']

        if not hypopnea_event_features.empty:
            assert not hypopnea_event_features['hypopnea_flow_shape_flattening_idx'].isnull().all(), \
                "Flattening index should be calculated for hypopneas."
        if not apnea_event_features.empty:
            assert apnea_event_features['hypopnea_flow_shape_flattening_idx'].isnull().all(), \
                "Flattening index should be NaN for apneas."

        print("\nFeature extraction tests passed basic assertions.")
    else:
        print("No features extracted. Check input data or feature extraction logic.")

    # Test with an event at the very beginning or end of the signal
    print("\n--- Testing Edge Case Event (start of signal) ---")
    edge_event_data = {
        'event_start_time': flow_s.index[0],
        'event_end_time': flow_s.index[int(10*sampling_freq_hz)],
        'event_type': 'hypopnea_candidate', 'event_duration_s': 10.0,
        'avg_flow_reduction_percent': 50.0, 'min_flow_during_event_ratio': 0.4,
        'excluded_due_to_leak': False
    }
    edge_event_row = pd.Series(edge_event_data, name='edge_event')
    edge_features = extract_features_for_event(edge_event_row, flow_s, pressure_s, sampling_freq_hz)
    assert edge_features['flow_pre_mean'] is np.nan or edge_features['flow_pre_duration_s'] == 0, "Pre-event features should be NaN or duration 0 for event at start."
    assert edge_features['flow_during_mean'] is not np.nan, "During-event features should be present."
    assert edge_features['flow_post_mean'] is not np.nan, "Post-event features should be present."
    print("Correctly handled event at the start of the signal (pre-event features are NaN/zero duration).")

    print("\nFeature engineering module tests complete.")
from typing import Optional, Tuple # Add this import for Python 3.8+ if needed for type hints
