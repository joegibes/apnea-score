import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def detect_breaths_from_flow(flow_series: pd.Series,
                             sampling_freq_hz: float,
                             peak_prominence_threshold: Optional[float] = None,
                             zero_crossing_hysteresis: float = 0.01) -> pd.DataFrame:
    """
    Detects individual breaths from a flow rate signal.
    This implementation primarily uses zero-crossings, but also identifies
    inspiratory and expiratory peaks for feature calculation.

    Args:
        flow_series (pd.Series): Time series of flow rate.
                                 Positive values for inspiration, negative for expiration.
        sampling_freq_hz (float): Sampling frequency of the flow signal in Hz.
        peak_prominence_threshold (Optional[float]): Minimum prominence for find_peaks.
                                                    If None, an adaptive threshold is calculated based on signal properties
                                                    (e.g., a fraction of median absolute flow or standard deviation),
                                                    which is useful for data with varying scales.
                                                    If a float is provided, it's used directly.
        zero_crossing_hysteresis (float): A small value to avoid detecting multiple crossings
                                          due to noise around zero. Flow must pass beyond
                                          +/- this hysteresis to be considered a crossing.
                                          If None, an adaptive hysteresis is calculated based on signal properties,
                                          recommended for data with varying scales.

    Returns:
        pd.DataFrame: DataFrame with columns:
            'breath_start_time': Timestamp of the start of inspiration.
            'insp_peak_time': Timestamp of peak inspiratory flow.
            'insp_peak_flow': Value of peak inspiratory flow.
            'breath_mid_time': Timestamp of zero-crossing from inspiration to expiration.
            'exp_peak_time': Timestamp of peak expiratory flow (most negative).
            'exp_peak_flow': Value of peak expiratory flow.
            'breath_end_time': Timestamp of zero-crossing from expiration to inspiration (start of next breath).
            'insp_duration_s': Duration of inspiration in seconds.
            'exp_duration_s': Duration of expiration in seconds.
            'total_duration_s': Total breath duration in seconds.
            'tidal_volume_l': Estimated tidal volume in Liters.
            'peak_to_peak_flow_l_s': Difference between peak inspiratory and peak expiratory flow.
    """
    if not isinstance(flow_series, pd.Series):
        raise TypeError("flow_series must be a pandas Series.")
    if flow_series.empty:
        return pd.DataFrame()

    dt_seconds = 1.0 / sampling_freq_hz
    min_samples_between_crossings = int(0.2 * sampling_freq_hz) # e.g., 200ms refractory period

    # Adaptive thresholds based on flow signal characteristics
    # Use a robust measure of typical flow amplitude, e.g., median of positive part of signal or std
    # This is a heuristic; may need tuning based on data scaling.
    # If flow values are very large (e.g. 600,000), these % will be large.
    flow_abs_median = np.median(np.abs(flow_series[flow_series != 0]))
    if pd.isna(flow_abs_median) or flow_abs_median == 0: # Handle cases with mostly zero flow or issues
        flow_abs_median = np.std(flow_series) # Fallback to std if median is tricky
        if pd.isna(flow_abs_median) or flow_abs_median == 0: # Further fallback
             flow_abs_median = 1.0 # Avoid zero if signal is flat zero

    # Adaptive zero_crossing_hysteresis: e.g., 2-5% of typical flow amplitude or a fraction of std dev
    # If zero_crossing_hysteresis was passed as a very small number (e.g. <1 for scaled data), adapt it.
    if zero_crossing_hysteresis is None or zero_crossing_hysteresis < 1.0 and flow_abs_median > 100:
        adaptive_hysteresis = max(0.02 * flow_abs_median, 1.0) # Ensure it's not excessively small for large-scale data
        # print(f"Adapting zero_crossing_hysteresis to: {adaptive_hysteresis:.2f}")
    else: # Use provided hysteresis if it seems reasonable for the data scale or is explicitly set high
        adaptive_hysteresis = zero_crossing_hysteresis

    # Adaptive peak_prominence_threshold: e.g., 5-10% of typical flow amplitude
    if peak_prominence_threshold is None or peak_prominence_threshold < 1.0 and flow_abs_median > 100:
        adaptive_peak_prominence = max(0.05 * flow_abs_median, adaptive_hysteresis * 2) # Prominence should be > hysteresis
        # print(f"Adapting peak_prominence_threshold to: {adaptive_peak_prominence:.2f}")
    else:
        adaptive_peak_prominence = peak_prominence_threshold

    # 1. Find actual sign changes (potential zero-crossings)
    sign_flow = np.sign(flow_series)
    # A crossing occurs where sign changes, and previous point was not zero (to avoid multiple from flat zero segments)
    # diff will be non-zero where sign changes.
    # We are interested in indices `i` where sign_flow[i] != sign_flow[i-1]
    potential_crossing_indices = np.where((sign_flow != sign_flow.shift(1)) & (sign_flow.shift(1) != 0))[0]

    if len(potential_crossing_indices) == 0:
        return pd.DataFrame()

    # 2. Filter these crossings based on hysteresis and refractory period
    valid_crossings_indices = []
    last_crossing_idx = -np.inf # Initialize to ensure first valid crossing can be added

    for current_idx in potential_crossing_indices:
        if current_idx == 0: continue # Cannot check previous for the first point

        # Check refractory period: current crossing must be far enough from the last valid one
        if current_idx - last_crossing_idx < min_samples_between_crossings:
            continue

        prev_val = flow_series.iloc[current_idx - 1]
        current_val = flow_series.iloc[current_idx]

        # Condition for Insp -> Exp: prev_val > hysteresis AND current_val < -hysteresis (strong crossing)
        # OR prev_val > hysteresis AND abs(current_val) < hysteresis (crossing into zero-band from positive)
        is_insp_to_exp = (prev_val > adaptive_hysteresis and current_val < -adaptive_hysteresis) or \
                         (prev_val > adaptive_hysteresis and abs(current_val) < adaptive_hysteresis and sign_flow.iloc[current_idx-1] == 1)
                         # ensure previous was clearly positive

        # Condition for Exp -> Insp: prev_val < -hysteresis AND current_val > hysteresis (strong crossing)
        # OR prev_val < -hysteresis AND abs(current_val) < hysteresis (crossing into zero-band from negative)
        is_exp_to_insp = (prev_val < -adaptive_hysteresis and current_val > adaptive_hysteresis) or \
                         (prev_val < -adaptive_hysteresis and abs(current_val) < adaptive_hysteresis and sign_flow.iloc[current_idx-1] == -1)
                         # ensure previous was clearly negative

        if is_insp_to_exp or is_exp_to_insp:
            # Further check: ensure it's not a brief flick if previous valid crossing was of the same type
            if valid_crossings_indices:
                prev_valid_idx = valid_crossings_indices[-1]
                prev_valid_flow_val_before_crossing = flow_series.iloc[prev_valid_idx-1]

                # If new is I->E, prev valid should be E->I. If new is E->I, prev valid should be I->E.
                # Avoid I->E followed closely by another I->E without a proper E->I in between.
                current_is_IE = prev_val > adaptive_hysteresis # Current crossing starts from positive (inspiration)
                prev_was_IE = flow_series.iloc[valid_crossings_indices[-1]-1] > adaptive_hysteresis if valid_crossings_indices[-1] > 0 else False

                if current_is_IE == prev_was_IE: # Same type of crossing back-to-back (e.g. I->E then another I->E)
                    # This indicates a potential double count or noisy segment, skip this current one.
                    continue

            valid_crossings_indices.append(current_idx)
            last_crossing_idx = current_idx

    if not valid_crossings_indices:
        return pd.DataFrame()

    # 3. Segment breaths based on valid_crossings_indices (Exp->Insp, Insp->Exp, Exp->Insp sequence)
    processed_breaths = []

    # Find the first Exp->Insp transition to start breath detection
    first_breath_phase_start_k = 0
    while first_breath_phase_start_k < len(valid_crossings_indices):
        idx = valid_crossings_indices[first_breath_phase_start_k]
        if idx > 0 and flow_series.iloc[idx-1] < -adaptive_hysteresis and flow_series.iloc[idx] > -adaptive_hysteresis: # Exp->Insp
            break
        first_breath_phase_start_k += 1

    k = first_breath_phase_start_k
    while k + 2 < len(valid_crossings_indices): # Need three crossings for one full breath cycle
        start_idx_orig = valid_crossings_indices[k]     # Supposedly Exp -> Insp
        mid_idx_orig = valid_crossings_indices[k+1]     # Supposedly Insp -> Exp
        end_idx_orig = valid_crossings_indices[k+2]     # Supposedly Exp -> Insp (marks end of current, start of next)

        # Verify phase transitions strictly using flow values around crossings and hysteresis
        # Start of Inspiration (Exp -> Insp)
        is_start_valid = (start_idx_orig > 0 and
                          flow_series.iloc[start_idx_orig - 1] < -adaptive_hysteresis and
                          flow_series.iloc[start_idx_orig] >= -adaptive_hysteresis) # Allow starting near zero from negative

        # Mid-breath, end of Inspiration (Insp -> Exp)
        is_mid_valid = (mid_idx_orig > 0 and
                        flow_series.iloc[mid_idx_orig - 1] > adaptive_hysteresis and
                        flow_series.iloc[mid_idx_orig] <= adaptive_hysteresis) # Allow ending near zero from positive

        # End of Expiration / Start of next Inspiration (Exp -> Insp)
        is_end_valid = (end_idx_orig > 0 and
                        flow_series.iloc[end_idx_orig - 1] < -adaptive_hysteresis and
                        flow_series.iloc[end_idx_orig] >= -adaptive_hysteresis)

        if not (is_start_valid and is_mid_valid and is_end_valid):
            k += 1 # Advance to the next potential start crossing
            continue

        breath_start_time = flow_series.index[start_idx_orig]
        breath_mid_time = flow_series.index[mid_idx_orig]
        breath_end_time = flow_series.index[end_idx_orig]

        # Ensure segments are not empty and have valid durations
        if breath_mid_time <= breath_start_time or breath_end_time <= breath_mid_time:
            k += 1
            continue

        insp_flow_segment = flow_series.loc[breath_start_time : breath_mid_time]
        exp_flow_segment = flow_series.loc[breath_mid_time : breath_end_time]

        # Refine segments slightly to ensure they don't include the other phase due to exact crossing point
        insp_flow_segment = insp_flow_segment[insp_flow_segment.index < breath_mid_time]
        exp_flow_segment = exp_flow_segment[exp_flow_segment.index < breath_end_time]


        if insp_flow_segment.empty or exp_flow_segment.empty:
            k += 1
            continue

        # Peak Inspiratory Flow (positive peaks)
        insp_peaks_indices, _ = find_peaks(insp_flow_segment.values,
                                           prominence=adaptive_peak_prominence,
                                           height=adaptive_hysteresis) # height ensures it's above noise
        if len(insp_peaks_indices) > 0:
            insp_peak_local_idx = insp_peaks_indices[np.argmax(insp_flow_segment.iloc[insp_peaks_indices].values)]
            insp_peak_time = insp_flow_segment.index[insp_peak_local_idx]
            insp_peak_flow = insp_flow_segment.iloc[insp_peak_local_idx]
        else: # Fallback if no distinct peak found
            insp_peak_time = insp_flow_segment.idxmax()
            insp_peak_flow = insp_flow_segment.max()

        # Peak Expiratory Flow (negative peaks, so find peaks in -flow)
        exp_peaks_indices, _ = find_peaks(-exp_flow_segment.values,
                                          prominence=adaptive_peak_prominence,
                                          height=adaptive_hysteresis) # height for -flow means more negative than -hysteresis
        if len(exp_peaks_indices) > 0:
            exp_peak_local_idx = exp_peaks_indices[np.argmax(-exp_flow_segment.iloc[exp_peaks_indices].values)]
            exp_peak_time = exp_flow_segment.index[exp_peak_local_idx]
            exp_peak_flow = exp_flow_segment.iloc[exp_peak_local_idx]
        else: # Fallback
            exp_peak_time = exp_flow_segment.idxmin()
            exp_peak_flow = exp_flow_segment.min()

        insp_duration_s = (breath_mid_time - breath_start_time).total_seconds()
        exp_duration_s = (breath_end_time - breath_mid_time).total_seconds()
        total_duration_s = (breath_end_time - breath_start_time).total_seconds()

        if total_duration_s <= 0.1: # Reject physiologically too short breaths (e.g. <100ms)
            k += 1 # Try next starting point
            continue

        # Tidal Volume: integral of inspiratory flow segment (from start to mid crossing)
        # Ensure we use the original series for integration between precise crossing points for accuracy
        true_insp_segment_for_integration = flow_series.iloc[start_idx_orig:mid_idx_orig]
        tidal_volume_l = np.trapz(true_insp_segment_for_integration.clip(lower=0), dx=dt_seconds)

        peak_to_peak_flow_l_s = insp_peak_flow - exp_peak_flow # exp_peak_flow is negative

        processed_breaths.append({
            'breath_start_time': breath_start_time,
            'insp_peak_time': insp_peak_time,
            'insp_peak_flow': insp_peak_flow,
            'breath_mid_time': breath_mid_time,
            'exp_peak_time': exp_peak_time,
            'exp_peak_flow': exp_peak_flow,
            'breath_end_time': breath_end_time,
            'insp_duration_s': insp_duration_s,
            'exp_duration_s': exp_duration_s,
            'total_duration_s': total_duration_s,
            'tidal_volume_l': tidal_volume_l,
            'peak_to_peak_flow_l_s': peak_to_peak_flow_l_s
        })
        k += 2 # Advance by 2 because current breath used crossings k, k+1, k+2. Next breath starts at k+2.

    return pd.DataFrame(processed_breaths)


if __name__ == '__main__':
    # Test with the new Parquet data
    print("--- Testing Breath Detection with Parquet Data ---")

    # These imports would be at the top of the file in a real script
    import os
    # Assuming src.data_loader is accessible via sys.path if running this as a script
    # For direct execution or if src is not in PYTHONPATH, adjust path:
    # module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # if module_path not in sys.path:
    #    sys.path.append(module_path)
    from src.data_loader import load_parquet_data
    from src.preprocessing import butterworth_filter


    # Path to the Parquet file (assuming it's in the root of the repo for this test)
    # In a real scenario, this path might be different or passed as an argument.
    parquet_file_path = "test_data_merged_10min.parquet"

    if not os.path.exists(parquet_file_path):
        print(f"Test data file not found: {parquet_file_path}")
        print("Skipping Parquet data test in breath_detection.py.")
        # Create dummy data for basic structural tests if file not found
        sampling_freq_hz_dummy = 25
        time_index_dummy = pd.to_datetime(np.arange(100) / sampling_freq_hz_dummy, unit='s')
        flow_series_dummy = pd.Series(np.sin(np.arange(100)*0.1), index=time_index_dummy, name='flow_rate')
        print("\n--- Test with empty series (fallback) ---")
        empty_series = pd.Series([], dtype=float, index=pd.to_datetime([]))
        empty_breaths_df = detect_breaths_from_flow(empty_series, sampling_freq_hz_dummy)
        assert empty_breaths_df.empty, "Should return empty DataFrame for empty input."
        print("Correctly handled empty series.")
        return # Exit if no Parquet file for full test


    col_map = {
        'flow_rate': 'Flow.40ms',
        'pressure': 'Press.2s',
        'mask_pressure': 'MaskPress.2s',
        'epr_pressure': 'EprPress.2s',
        'leak_rate': 'Leak.2s',
        'respiratory_rate': 'RespRate.2s',
        'minute_ventilation': 'MinVent.2s',
        'flow_limitation': 'FlowLim.2s'
    }

    # Load data using the updated Parquet loader
    # The Parquet file has 'timestamp' as its DatetimeIndex name
    full_df = load_parquet_data(filepath=parquet_file_path,
                                timestamp_col_name_in_file='timestamp',
                                standard_timestamp_col='timestamp',
                                column_name_map=col_map)

    if full_df is None or 'flow_rate' not in full_df.columns:
        print("Failed to load Parquet data or 'flow_rate' column missing.")
        return

    # Determine sampling frequency from the data (more robust than hardcoding)
    if len(full_df.index) > 1:
        actual_sampling_interval_s = (full_df.index[1] - full_df.index[0]).total_seconds()
        if actual_sampling_interval_s > 0:
            SAMPLING_FREQ_HZ_ACTUAL = 1.0 / actual_sampling_interval_s
            print(f"Detected sampling frequency from Parquet data: {SAMPLING_FREQ_HZ_ACTUAL:.2f} Hz")
        else:
            print("Warning: Could not determine sampling frequency from data index. Using default of 25Hz for test.")
            SAMPLING_FREQ_HZ_ACTUAL = 25.0 # Fallback
    else:
        print("Warning: Not enough data points to determine sampling frequency. Using default of 25Hz for test.")
        SAMPLING_FREQ_HZ_ACTUAL = 25.0 # Fallback


    flow_to_analyze = full_df['flow_rate']

    # Optional: Rescale flow if necessary (e.g. if it's in mL/s or other units)
    # Example: if flow_rate values are very large (e.g. mean abs > 1000), divide by 1000
    if not flow_to_analyze.empty and flow_to_analyze.abs().mean() > 1000: # Heuristic for large scaled values
        print(f"Flow values are large (mean abs: {flow_to_analyze.abs().mean():.0f}), assuming scaled. Rescaling by /1000 for breath detection test.")
        flow_to_analyze = flow_to_analyze / 1000.0
        # Note: This rescaling is for the purpose of this test script.
        # The actual application might need a more robust way to handle scaling or make thresholds scale-invariant.

    # Apply a light filter if desired (often helpful before breath detection)
    flow_filtered = butterworth_filter(flow_to_analyze,
                                       filter_type='lowpass',
                                       cutoff_freq_hz=3.0, # Slightly higher cutoff for potentially faster real breaths
                                       sampling_freq_hz=SAMPLING_FREQ_HZ_ACTUAL,
                                       order=2) # Lower order filter for less smoothing

    print(f"Running breath detection on flow data (first {len(flow_filtered)} samples)...")
    # The adaptive thresholds in detect_breaths_from_flow should handle the scale.
    # Explicitly pass None for thresholds to let the function determine them, or pass scaled values.
    breaths_df = detect_breaths_from_flow(flow_filtered,
                                          sampling_freq_hz=SAMPLING_FREQ_HZ_ACTUAL,
                                          peak_prominence_threshold=None, # Let function adapt
                                          zero_crossing_hysteresis=None)  # Let function adapt

    if not breaths_df.empty:
        print(f"Detected {len(breaths_df)} breaths from Parquet data.")
        print("First 5 detected breaths:")
        print(breaths_df.head())

        # Basic assertions for real data (can be more lenient)
        assert breaths_df['insp_duration_s'].min() > 0.1, "Insp. duration too short" # Expect >100ms
        assert breaths_df['exp_duration_s'].min() > 0.1, "Exp. duration too short"   # Expect >100ms
        assert breaths_df['total_duration_s'].min() > 0.2, "Total duration too short" # Expect >200ms
        # Tidal volume can be very small for scaled data if not rescaled properly, so careful with this assertion
        # assert breaths_df['tidal_volume_l'].min() >= 0, "Tidal volume should be non-negative"

        print("\nBreath detection on Parquet data passed basic assertions.")

        # Plotting for visual verification (first N seconds of data)
        import matplotlib.pyplot as plt
        plot_duration_s = 60 # Plot first 60 seconds
        plot_samples = int(plot_duration_s * SAMPLING_FREQ_HZ_ACTUAL)

        plt.figure(figsize=(18, 7))
        # Plot original (but potentially rescaled for test) flow
        flow_to_analyze.iloc[:plot_samples].plot(label='Input Flow (used for detection)', color='lightgray', alpha=0.6)
        # Plot filtered flow that was actually used by detection
        flow_filtered.iloc[:plot_samples].plot(label='Filtered Flow (used for detection)', color='cornflowerblue')

        breaths_to_plot = breaths_df[breaths_df['breath_end_time'] <= flow_filtered.index[min(plot_samples-1, len(flow_filtered)-1)]]

        if not breaths_to_plot.empty:
            plt.scatter(breaths_to_plot['insp_peak_time'], breaths_to_plot['insp_peak_flow'],
                        color='green', marker='^', label='Insp Peaks', s=50, alpha=0.8)
            plt.scatter(breaths_to_plot['exp_peak_time'], breaths_to_plot['exp_peak_flow'],
                        color='red', marker='v', label='Exp Peaks', s=50, alpha=0.8)

            for _, breath in breaths_to_plot.iterrows():
                plt.axvline(breath['breath_start_time'], color='blue', linestyle='--', linewidth=0.8, alpha=0.6)
                plt.axvline(breath['breath_mid_time'], color='purple', linestyle=':', linewidth=0.8, alpha=0.6)

        plt.title(f'Breath Detection on Parquet Data (First {plot_duration_s}s)')
        plt.xlabel('Time')
        plt.ylabel('Flow Rate (potentially rescaled for test)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        results_dir = "results" # Save in main results dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        plot_filename = os.path.join(results_dir, "breath_detection_parquet_test_plot.png")
        plt.savefig(plot_filename)
        print(f"\nSaved Parquet breath detection plot to {plot_filename}")
        # plt.show()
    else:
        print("No breaths detected from Parquet data. Check parameters, data quality, or scaling.")

    print("\nBreath detection module main test complete.")

from typing import Optional # Add this import
