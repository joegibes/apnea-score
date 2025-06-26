import pandas as pd
import numpy as np
from typing import Optional
from scipy.signal import find_peaks, butter, filtfilt

def detect_breaths_from_flow(flow_series: pd.Series,
                             sampling_freq_hz: float,
                             peak_prominence_threshold: Optional[float] = None,
                             zero_crossing_hysteresis: float = 0.005) -> pd.DataFrame:
    """
    Detects individual breaths from a flow rate signal.
    This implementation primarily uses zero-crossings, but also identifies
    inspiratory and expiratory peaks for feature calculation.

    Args:
        flow_series (pd.Series): Time series of flow rate.
                                 Positive values for inspiration, negative for expiration.
        sampling_freq_hz (float): Sampling frequency of the flow signal in Hz.
        peak_prominence_threshold (Optional[float]): Minimum prominence for find_peaks.
                                                    If None, a default based on signal std might be used,
                                                    or a simpler threshold if find_peaks is too sensitive.
                                                    A sensible default might be 0.05 to 0.1 L/s for human data.
        zero_crossing_hysteresis (float): A small value to avoid detecting multiple crossings
                                          due to noise around zero. Flow must pass beyond
                                          +/- this hysteresis to be considered a crossing.

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
        return pd.DataFrame() # Return empty if no data

    # --- Step 1: Low-pass filter the flow signal to suppress noise ---
    def butter_lowpass_filter(data, cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    filtered_flow = pd.Series(
        butter_lowpass_filter(flow_series.values, cutoff=4.0, fs=sampling_freq_hz, order=4),
        index=flow_series.index
    )

    # Default prominence if not provided: 10% of std of flow, or a minimum of 0.05
    if peak_prominence_threshold is None:
        flow_std = filtered_flow.std()
        peak_prominence_threshold = max(0.05, flow_std * 0.1) if flow_std > 0 else 0.05

    dt_seconds = 1.0 / sampling_freq_hz

    # --- Step 2: Robust zero-crossing detection with larger hysteresis and sign change ---
    sign = np.sign(filtered_flow)
    valid_crossings_indices = []
    last_crossing_idx = -10
    min_gap_samples = int(0.3 * sampling_freq_hz)  # 300ms minimum between crossings
    for idx in range(1, len(filtered_flow)):
        # Prevent double-counting: skip if too close to previous crossing
        if idx - last_crossing_idx < min_gap_samples:
            continue
        # Exp to Insp: crosses from below -hysteresis to above +hysteresis
        if filtered_flow.iloc[idx-1] < -zero_crossing_hysteresis and filtered_flow.iloc[idx] > zero_crossing_hysteresis:
            valid_crossings_indices.append(idx)
            last_crossing_idx = idx
        # Insp to Exp: crosses from above +hysteresis to below -hysteresis
        elif filtered_flow.iloc[idx-1] > zero_crossing_hysteresis and filtered_flow.iloc[idx] < -zero_crossing_hysteresis:
            valid_crossings_indices.append(idx)
            last_crossing_idx = idx

    # --- Step 3: Remove implausible breaths based on duration ---
    if not valid_crossings_indices:
        return pd.DataFrame()

    min_breath_samples = int(0.5 * sampling_freq_hz)
    max_breath_samples = int(10.0 * sampling_freq_hz)
    filtered_crossings = [valid_crossings_indices[0]]
    for idx in valid_crossings_indices[1:]:
        gap = idx - filtered_crossings[-1]
        if min_breath_samples <= gap <= max_breath_samples:
            filtered_crossings.append(idx)
    valid_crossings_indices = filtered_crossings


    if not valid_crossings_indices:
        # print("Warning: No valid zero crossings found with current hysteresis.")
        return pd.DataFrame()

    crossing_times = flow_series.index[valid_crossings_indices]
    crossing_signs = sign.iloc[valid_crossings_indices] # Sign of flow *at* the crossing point (new phase)

    breaths = []
    # Iterate through crossings to define breath segments
    # A breath starts at an exp_to_insp crossing and ends at the next exp_to_insp crossing.
    # The mid-point is an insp_to_exp crossing.

    # Find start of first full breath (must be exp_to_insp)
    first_breath_start_idx = 0
    while first_breath_start_idx < len(crossing_times):
        # A breath starts when flow transitions from negative (expiration) to positive (inspiration)
        # So, sign.iloc[valid_crossings_indices[first_breath_start_idx]-1] should be -1
        # and sign.iloc[valid_crossings_indices[first_breath_start_idx]] should be 1
        idx_at_crossing = valid_crossings_indices[first_breath_start_idx]
        if idx_at_crossing > 0 and \
           sign.iloc[idx_at_crossing-1] < 0 and sign.iloc[idx_at_crossing] >= 0: # Exp to Insp
            break
        first_breath_start_idx += 1

    if first_breath_start_idx >= len(crossing_times) -1: # Need at least two crossings for one full breath phase
        # print("Warning: Not enough crossings to define a full breath cycle.")
        return pd.DataFrame()

    for i in range(first_breath_start_idx, len(crossing_times) - 1):
        start_crossing_original_idx = valid_crossings_indices[i]
        mid_crossing_original_idx = -1
        end_crossing_original_idx = -1

        # Current crossing is Exp -> Insp (start of breath)
        if not (start_crossing_original_idx > 0 and sign.iloc[start_crossing_original_idx-1] < 0 and sign.iloc[start_crossing_original_idx] >=0) :
            continue

        breath_start_time = flow_series.index[start_crossing_original_idx]

        # Next crossing should be Insp -> Exp (mid-breath)
        if i + 1 < len(crossing_times):
            # Check if next crossing is Insp -> Exp
            idx_at_next_crossing = valid_crossings_indices[i+1]
            if not (idx_at_next_crossing > 0 and sign.iloc[idx_at_next_crossing-1] > 0 and sign.iloc[idx_at_next_crossing] <= 0):
                continue # Not a valid mid-breath point, skip this "breath"
            mid_crossing_original_idx = idx_at_next_crossing
            breath_mid_time = flow_series.index[mid_crossing_original_idx]
        else:
            continue # Incomplete breath

        # The crossing after that should be Exp -> Insp (end of current breath, start of next)
        if i + 2 < len(crossing_times):
            idx_at_following_crossing = valid_crossings_indices[i+2]
            if not (idx_at_following_crossing > 0 and sign.iloc[idx_at_following_crossing-1] < 0 and sign.iloc[idx_at_following_crossing] >=0):
                 # This means the expiratory phase didn't complete as expected before another insp.
                 # Or it's the end of the data. For now, we'll take the exp phase up to this point
                 # or if this is not an Exp->Insp, we might not have a clean end.
                 # Let's assume for now a full breath cycle requires three such points.
                 continue
            end_crossing_original_idx = idx_at_following_crossing
            breath_end_time = flow_series.index[end_crossing_original_idx]
        else:
            # If we don't have i+2, it means this is the last breath segment, use end of series for exp_end
            # This will make the last expiration potentially incomplete.
            # For simplicity, we only process full breaths that have a defined start, mid, and next_start crossing
            continue


        # Segment for inspiration: from breath_start_time to breath_mid_time
        insp_flow_segment = flow_series.loc[breath_start_time:breath_mid_time]
        # Segment for expiration: from breath_mid_time to breath_end_time
        exp_flow_segment = flow_series.loc[breath_mid_time:breath_end_time]

        if insp_flow_segment.empty or exp_flow_segment.empty:
            continue

        # Find peak inspiratory flow
        insp_peaks_indices, _ = find_peaks(insp_flow_segment, prominence=peak_prominence_threshold, height=zero_crossing_hysteresis)
        if len(insp_peaks_indices) > 0:
            # insp_peak_idx = insp_peaks_indices[np.argmax(insp_flow_segment.iloc[insp_peaks_indices])] # Global index
            insp_peak_local_idx = insp_peaks_indices[np.argmax(insp_flow_segment.iloc[insp_peaks_indices].values)]
            insp_peak_time = insp_flow_segment.index[insp_peak_local_idx]
            insp_peak_flow = insp_flow_segment.iloc[insp_peak_local_idx]
        else: # Fallback if no peak found (e.g. very flat insp)
            insp_peak_time = insp_flow_segment.idxmax() if not insp_flow_segment.empty else breath_start_time
            insp_peak_flow = insp_flow_segment.max() if not insp_flow_segment.empty else 0


        # Find peak expiratory flow (most negative)
        # For expiratory flow, peaks are negative, so we look for "valleys" in the positive flow or peaks in -flow
        exp_peaks_indices, _ = find_peaks(-exp_flow_segment, prominence=peak_prominence_threshold, height=zero_crossing_hysteresis)
        if len(exp_peaks_indices) > 0:
            # exp_peak_idx = exp_peaks_indices[np.argmax(-exp_flow_segment.iloc[exp_peaks_indices])] # Global index
            exp_peak_local_idx = exp_peaks_indices[np.argmax(-exp_flow_segment.iloc[exp_peaks_indices].values)]
            exp_peak_time = exp_flow_segment.index[exp_peak_local_idx]
            exp_peak_flow = exp_flow_segment.iloc[exp_peak_local_idx]
        else: # Fallback
            exp_peak_time = exp_flow_segment.idxmin() if not exp_flow_segment.empty else breath_mid_time
            exp_peak_flow = exp_flow_segment.min() if not exp_flow_segment.empty else 0

        # Calculate durations
        insp_duration_s = (breath_mid_time - breath_start_time).total_seconds()
        exp_duration_s = (breath_end_time - breath_mid_time).total_seconds() # End of exp is start of next insp
        total_duration_s = (breath_end_time - breath_start_time).total_seconds()

        # Estimate tidal volume by integrating inspiratory flow
        # Trapezoidal integration: sum of (value_i + value_{i+1})/2 * dt
        # Ensure insp_flow_segment contains only positive flow for this.
        # For more accuracy, one might re-integrate from precise zero-crossing to zero-crossing.
        # The current insp_flow_segment is already defined by zero crossings.
        tidal_volume_l = np.trapz(insp_flow_segment.clip(lower=0), dx=dt_seconds) # Clip to ensure only positive flow contributes

        peak_to_peak_flow_l_s = insp_peak_flow - exp_peak_flow # exp_peak_flow is negative

        breaths.append({
            'breath_start_time': breath_start_time,
            'insp_peak_time': insp_peak_time,
            'insp_peak_flow': insp_peak_flow,
            'breath_mid_time': breath_mid_time, # Insp-Exp crossing
            'exp_peak_time': exp_peak_time,
            'exp_peak_flow': exp_peak_flow,
            'breath_end_time': breath_end_time, # Exp-Insp crossing (effectively start of next breath)
            'insp_duration_s': insp_duration_s,
            'exp_duration_s': exp_duration_s,
            'total_duration_s': total_duration_s,
            'tidal_volume_l': tidal_volume_l,
            'peak_to_peak_flow_l_s': peak_to_peak_flow_l_s
        })

        # Important: Advance 'i' because the end of the current breath (i+2) is the start of the next one.
        # The loop structure should handle this correctly if we iterate i and define a breath from i, i+1, i+2.
        # To avoid re-processing, we should advance `i` by 2 if a full breath is found.
        # However, the current loop structure `for i in range(first_breath_start_idx, len(crossing_times) - 2)`
        # and using `crossing_times[i], crossing_times[i+1], crossing_times[i+2]` might be cleaner.
        # Let's adjust loop to make this more explicit.

    # Revised loop for clarity:
    # Each iteration `k` tries to define a breath using crossings at `k` (Exp->Insp), `k+1` (Insp->Exp), `k+2` (Exp->Insp)
    processed_breaths = []
    k = first_breath_start_idx
    while k + 2 < len(valid_crossings_indices): # Need three crossings for one full breath cycle
        start_idx_orig = valid_crossings_indices[k]
        mid_idx_orig = valid_crossings_indices[k+1]
        end_idx_orig = valid_crossings_indices[k+2]

        # Check phase transitions:
        # Start: Exp -> Insp (flow[start-1] < 0, flow[start] >= 0)
        is_start_valid = (start_idx_orig > 0 and
                          flow_series.iloc[start_idx_orig-1] < -zero_crossing_hysteresis and
                          flow_series.iloc[start_idx_orig] > -zero_crossing_hysteresis) # allow near zero start
        # Mid: Insp -> Exp (flow[mid-1] > 0, flow[mid] <= 0)
        is_mid_valid = (mid_idx_orig > 0 and
                        flow_series.iloc[mid_idx_orig-1] > zero_crossing_hysteresis and
                        flow_series.iloc[mid_idx_orig] < zero_crossing_hysteresis) # allow near zero end
        # End: Exp -> Insp (flow[end-1] < 0, flow[end] >= 0) - this is start of NEXT breath
        is_end_valid = (end_idx_orig > 0 and
                        flow_series.iloc[end_idx_orig-1] < -zero_crossing_hysteresis and
                        flow_series.iloc[end_idx_orig] > -zero_crossing_hysteresis)

        if not (is_start_valid and is_mid_valid and is_end_valid):
            k += 1 # Move to the next potential start
            continue

        breath_start_time = flow_series.index[start_idx_orig]
        breath_mid_time = flow_series.index[mid_idx_orig]
        breath_end_time = flow_series.index[end_idx_orig] # This is effectively the start of the next breath

        insp_flow_segment = flow_series.loc[breath_start_time:breath_mid_time]
        # Exclude the mid_time itself from insp_flow if it's already negative, to avoid issues with idxmax
        if not insp_flow_segment.empty and insp_flow_segment.iloc[-1] < 0 :
             insp_flow_segment = insp_flow_segment.iloc[:-1]

        exp_flow_segment = flow_series.loc[breath_mid_time:breath_end_time]
        if not exp_flow_segment.empty and exp_flow_segment.iloc[0] > 0 : # Ensure exp starts negative or zero
            exp_flow_segment = exp_flow_segment.iloc[1:]


        if insp_flow_segment.empty or exp_flow_segment.empty:
            k += 1
            continue

        # Peak Inspiratory Flow
        insp_peaks_indices, _ = find_peaks(insp_flow_segment.values, prominence=peak_prominence_threshold, height=zero_crossing_hysteresis)
        if len(insp_peaks_indices) > 0:
            insp_peak_local_idx = insp_peaks_indices[np.argmax(insp_flow_segment.iloc[insp_peaks_indices].values)]
            insp_peak_time = insp_flow_segment.index[insp_peak_local_idx]
            insp_peak_flow = insp_flow_segment.iloc[insp_peak_local_idx]
        else:
            insp_peak_time = insp_flow_segment.idxmax() if not insp_flow_segment.empty else breath_start_time
            insp_peak_flow = insp_flow_segment.max() if not insp_flow_segment.empty else 0

        # Peak Expiratory Flow
        exp_peaks_indices, _ = find_peaks(-exp_flow_segment.values, prominence=peak_prominence_threshold, height=zero_crossing_hysteresis)
        if len(exp_peaks_indices) > 0:
            exp_peak_local_idx = exp_peaks_indices[np.argmax(-exp_flow_segment.iloc[exp_peaks_indices].values)]
            exp_peak_time = exp_flow_segment.index[exp_peak_local_idx]
            exp_peak_flow = exp_flow_segment.iloc[exp_peak_local_idx]
        else:
            exp_peak_time = exp_flow_segment.idxmin() if not exp_flow_segment.empty else breath_mid_time
            exp_peak_flow = exp_flow_segment.min() if not exp_flow_segment.empty else 0

        insp_duration_s = (breath_mid_time - breath_start_time).total_seconds()
        exp_duration_s = (breath_end_time - breath_mid_time).total_seconds()
        total_duration_s = (breath_end_time - breath_start_time).total_seconds()

        # Tidal Volume from inspiratory flow (integral of positive flow during inspiration)
        # Ensure the segment for integration is purely inspiratory phase based on detected crossings
        # The insp_flow_segment from start_idx_orig to mid_idx_orig should be okay
        # We integrate flow_series from start_time to mid_time
        true_insp_segment = flow_series.iloc[start_idx_orig:mid_idx_orig+1]
        tidal_volume_l = np.trapz(true_insp_segment.clip(lower=0), dx=dt_seconds)

        peak_to_peak_flow_l_s = insp_peak_flow - exp_peak_flow

        if total_duration_s <= 0: # Sanity check for valid breath
            k+=1
            continue

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
    # Setup for testing
    sampling_freq_hz = 50  # Hz
    duration_sec = 30    # seconds
    num_samples = duration_sec * sampling_freq_hz
    time_stamps = np.arange(num_samples) / sampling_freq_hz
    time_index = pd.to_datetime(time_stamps, unit='s')

    # Create dummy flow rate data (sine wave for breathing + some noise)
    # Respiratory rate: 0.25 Hz = 15 breaths/min. Period = 4s. Insp = 1.6s, Exp = 2.4s (approx I:E 1:1.5)
    t_insp = 1.6
    t_exp = 2.4
    t_total = t_insp + t_exp

    flow_rate = []
    for t_current in time_stamps:
        t_in_cycle = t_current % t_total
        if t_in_cycle < t_insp: # Inspiration (positive half-sine)
            flow = 0.5 * np.sin((np.pi / t_insp) * t_in_cycle)
        else: # Expiration (negative half-sine, scaled)
            flow = -0.35 * np.sin((np.pi / t_exp) * (t_in_cycle - t_insp))
        flow_rate.append(flow)

    flow_rate = np.array(flow_rate)
    flow_rate += 0.02 * np.random.randn(num_samples) # Add some noise

    # Introduce an apnea: 5 seconds of zero flow
    apnea_start_sec = 10
    apnea_duration_sec = 5
    apnea_start_idx = int(apnea_start_sec * sampling_freq_hz)
    apnea_end_idx = int((apnea_start_sec + apnea_duration_sec) * sampling_freq_hz)
    flow_rate[apnea_start_idx:apnea_end_idx] = 0.001 * np.random.randn(apnea_end_idx - apnea_start_idx)

    # Introduce a period of very shallow breathing (hypopnea-like)
    hyp_start_sec = 20
    hyp_duration_sec = 5
    hyp_start_idx = int(hyp_start_sec * sampling_freq_hz)
    hyp_end_idx = int((hyp_start_sec + hyp_duration_sec) * sampling_freq_hz)
    flow_rate[hyp_start_idx:hyp_end_idx] *= 0.3


    flow_series = pd.Series(flow_rate, index=time_index, name='flow_rate')

    print("--- Testing Breath Detection ---")
    # Use a filter from preprocessing if available and if it's meant to be used before breath detection
    # For now, use raw (but noisy) flow.
    # from preprocessing import butterworth_filter # Assuming preprocessing.py is in PYTHONPATH
    # flow_filtered = butterworth_filter(flow_series, 'lowpass', cutoff_freq_hz=2.0, sampling_freq_hz=sampling_freq_hz)

    breaths_df = detect_breaths_from_flow(flow_series,
                                          sampling_freq_hz,
                                          peak_prominence_threshold=0.05, # L/s
                                          zero_crossing_hysteresis=0.02) # L/s

    if not breaths_df.empty:
        print(f"Detected {len(breaths_df)} breaths.")
        print("First 5 detected breaths:")
        print(breaths_df.head())
        print("\nLast 5 detected breaths:")
        print(breaths_df.tail())

        # Sanity checks
        assert breaths_df['insp_duration_s'].min() > 0, "Inspiratory duration should be positive"
        assert breaths_df['exp_duration_s'].min() > 0, "Expiratory duration should be positive"
        assert breaths_df['total_duration_s'].min() > 0, "Total duration should be positive"
        assert breaths_df['tidal_volume_l'].min() >= 0, "Tidal volume should be non-negative"
        assert (breaths_df['insp_peak_flow'] > -0.01).all() , "Insp peak flow should be positive (allowing for small noise)" # Check it's mostly positive
        assert (breaths_df['exp_peak_flow'] < 0.01).all() , "Exp peak flow should be negative (allowing for small noise)" # Check it's mostly negative

        print("\nBreath detection tests passed basic assertions.")

        # Optional: Plotting for visual verification
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 6))
        plt.plot(flow_series.index, flow_series.values, label='Flow Rate', color='lightgray', alpha=0.7)
        # plt.plot(flow_filtered.index, flow_filtered.values, label='Filtered Flow Rate', color='cornflowerblue')

        if not breaths_df.empty:
            plt.scatter(breaths_df['insp_peak_time'], breaths_df['insp_peak_flow'], color='green', marker='^', label='Insp Peaks')
            plt.scatter(breaths_df['exp_peak_time'], breaths_df['exp_peak_flow'], color='red', marker='v', label='Exp Peaks')

            for _, breath in breaths_df.iterrows():
                plt.axvline(breath['breath_start_time'], color='blue', linestyle='--', linewidth=0.7, alpha=0.5) # Start of Insp
                plt.axvline(breath['breath_mid_time'], color='purple', linestyle=':', linewidth=0.7, alpha=0.5) # Insp to Exp
                # plt.axvline(breath['breath_end_time'], color='blue', linestyle='--', linewidth=0.7, alpha=0.5) # End of Exp (start of next insp)


        plt.title('Breath Detection Test')
        plt.xlabel('Time')
        plt.ylabel('Flow Rate (L/s)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # Save the plot to a file instead of showing interactively
        plot_filename = "results/breath_detection_test_plot.png"
        plt.savefig(plot_filename)
        print(f"\nSaved breath detection plot to {plot_filename}")
        # plt.show() # Comment out for non-interactive environments

    else:
        print("No breaths detected. Check parameters or input signal.")

    print("\n--- Test with empty series ---")
    empty_series = pd.Series([], dtype=float, index=pd.to_datetime([]))
    empty_breaths_df = detect_breaths_from_flow(empty_series, sampling_freq_hz)
    assert empty_breaths_df.empty, "Should return empty DataFrame for empty input."
    print("Correctly handled empty series.")

    print("\n--- Test with flat line (zero flow) ---")
    flat_series = pd.Series(np.zeros(100), index=pd.to_datetime(np.arange(100)/sampling_freq_hz, unit='s'))
    flat_breaths_df = detect_breaths_from_flow(flat_series, sampling_freq_hz)
    assert flat_breaths_df.empty, "Should return empty DataFrame for flat zero flow."
    print("Correctly handled flat zero flow series.")

    print("\n--- Test with positive-only flow (no expiration) ---")
    positive_flow = pd.Series(np.linspace(0.1, 0.5, 100), index=pd.to_datetime(np.arange(100)/sampling_freq_hz, unit='s'))
    positive_breaths_df = detect_breaths_from_flow(positive_flow, sampling_freq_hz)
    assert positive_breaths_df.empty, "Should return empty DataFrame for positive-only flow."
    print("Correctly handled positive-only flow series.")

    print("\nBreath detection module tests complete.")
