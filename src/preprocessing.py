import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt, filtfilt

def butterworth_filter(data: pd.Series,
                       filter_type: str,
                       cutoff_freq_hz: float,
                       sampling_freq_hz: float,
                       order: int = 4) -> pd.Series:
    """
    Applies a Butterworth filter (low-pass, high-pass, or band-pass) to the data.

    Args:
        data (pd.Series): Input data series.
        filter_type (str): Type of filter: 'lowpass', 'highpass', 'bandpass'.
        cutoff_freq_hz (float or tuple): Cutoff frequency or (low_cut, high_cut) for bandpass.
        sampling_freq_hz (float): Sampling frequency of the data.
        order (int): Order of the Butterworth filter.

    Returns:
        pd.Series: Filtered data.
    """
    nyquist_freq_hz = 0.5 * sampling_freq_hz

    if filter_type == 'lowpass':
        if cutoff_freq_hz >= nyquist_freq_hz:
            # print(f"Warning: Lowpass cutoff frequency ({cutoff_freq_hz} Hz) is at or above Nyquist frequency ({nyquist_freq_hz} Hz). Skipping filter.")
            return data
        normalized_cutoff = cutoff_freq_hz / nyquist_freq_hz
        sos = butter(order, normalized_cutoff, btype='low', analog=False, output='sos')
    elif filter_type == 'highpass':
        if cutoff_freq_hz <= 0:
             # print(f"Warning: Highpass cutoff frequency ({cutoff_freq_hz} Hz) is at or below 0 Hz. Skipping filter.")
            return data
        if cutoff_freq_hz >= nyquist_freq_hz: # cannot be higher than nyquist
            # print(f"Warning: Highpass cutoff frequency ({cutoff_freq_hz} Hz) is at or above Nyquist ({nyquist_freq_hz} Hz), effectively removing all. Returning original.")
            return data # Or raise error, or return zeros. For now, return original.
        normalized_cutoff = cutoff_freq_hz / nyquist_freq_hz
        sos = butter(order, normalized_cutoff, btype='high', analog=False, output='sos')
    elif filter_type == 'bandpass':
        if not isinstance(cutoff_freq_hz, (list, tuple)) or len(cutoff_freq_hz) != 2:
            raise ValueError("cutoff_freq_hz must be a list or tuple of two frequencies for bandpass.")
        low_cut, high_cut = cutoff_freq_hz
        if low_cut <= 0 or high_cut >= nyquist_freq_hz or low_cut >= high_cut:
            # print(f"Warning: Invalid bandpass frequencies ({low_cut}, {high_cut} Hz) relative to Nyquist ({nyquist_freq_hz} Hz). Skipping filter.")
            return data
        normalized_low = low_cut / nyquist_freq_hz
        normalized_high = high_cut / nyquist_freq_hz
        sos = butter(order, [normalized_low, normalized_high], btype='band', analog=False, output='sos')
    else:
        raise ValueError("filter_type must be 'lowpass', 'highpass', or 'bandpass'")

    # Use sosfiltfilt for zero-phase filtering, good for offline processing
    # fill_method='pad' and padlen can help with edge effects for short series
    padlen = min(3 * order, len(data) -1) if len(data) > 3 * order else 0

    if padlen > 0 : # sosfiltfilt needs len(x) > padlen
      filtered_data = sosfiltfilt(sos, data, padlen=padlen)
    else: # For very short series, basic filtfilt might be more stable or just skip
      # print("Warning: Data series too short for robust sosfiltfilt padding. Using filtfilt or returning original if too short.")
      if len(data) > order * 2: # filtfilt needs len(data) > N (order) * 2 typically
          # For filtfilt, we need b, a coefficients
          b, a = butter(order, normalized_cutoff if filter_type != 'bandpass' else [normalized_low, normalized_high],
                        btype=filter_type, analog=False, output='ba')
          filtered_data = filtfilt(b, a, data)
      else:
          # print("Warning: Series too short for filtering. Returning original data.")
          return data.copy()


    return pd.Series(filtered_data, index=data.index, name=data.name)


def flag_high_leak_periods(leak_rate_series: pd.Series,
                           leak_threshold: float,
                           min_duration_sec: float,
                           sampling_freq_hz: float) -> pd.Series:
    """
    Flags periods of high mask leak.

    Args:
        leak_rate_series (pd.Series): Series containing leak rate data.
        leak_threshold (float): Leak rate above which is considered high.
        min_duration_sec (float): Minimum duration for a high leak period to be flagged.
        sampling_freq_hz (float): Sampling frequency of the data.

    Returns:
        pd.Series: Boolean series, True where leak is considered high and sustained.
    """
    min_samples = int(min_duration_sec * sampling_freq_hz)

    is_high_leak = leak_rate_series > leak_threshold

    # Find groups of consecutive high leak samples
    high_leak_periods = pd.Series(np.nan, index=leak_rate_series.index, dtype=bool) # Start with NaNs or False

    if not is_high_leak.any(): # No high leak at all
        return pd.Series(False, index=leak_rate_series.index, dtype=bool)

    # Identify change points to find blocks of consecutive True/False
    change_points = is_high_leak.ne(is_high_leak.shift()).cumsum()
    # Iterate over blocks of consecutive values
    for _, group in is_high_leak.groupby(change_points):
        if group.iloc[0] and len(group) >= min_samples: # If it's a high leak block and long enough
            high_leak_periods.loc[group.index] = True
        elif high_leak_periods.loc[group.index].isnull().all(): # If not set by a True block and still NaN
             high_leak_periods.loc[group.index] = False


    high_leak_periods.fillna(False, inplace=True) # Fill any remaining NaNs if any (shouldn't be many)
    return high_leak_periods.astype(bool)


def calculate_rolling_baseline(data_series: pd.Series,
                               window_sec: int,
                               sampling_freq_hz: float,
                               center: bool = True,
                               quantile: float = 0.5) -> pd.Series: # quantile 0.5 for median
    """
    Calculates a rolling baseline (e.g., median or other quantile) for a data series.

    Args:
        data_series (pd.Series): Input data series (e.g., flow rate).
        window_sec (int): Duration of the rolling window in seconds.
        sampling_freq_hz (float): Sampling frequency of the data.
        center (bool): If True, the window is centered on the current point.
                       If False, it's a trailing window.
        quantile (float): The quantile to compute (0.5 for median).

    Returns:
        pd.Series: Series containing the calculated rolling baseline.
    """
    window_samples = int(window_sec * sampling_freq_hz)
    if window_samples <= 0:
        raise ValueError("Window size in samples must be positive.")

    # Ensure window_samples is odd for centering if possible, or handle min_periods
    if center and window_samples % 2 == 0:
        window_samples +=1

    min_periods = window_samples // 2 # Require at least half the window to have data

    baseline = data_series.rolling(window=window_samples,
                                   center=center,
                                   min_periods=min_periods).quantile(quantile) # Using quantile for median

    # Rolling functions can produce NaNs at the beginning/end. Fill them.
    baseline = baseline.fillna(method='bfill').fillna(method='ffill') # Backfill then forward fill

    return pd.Series(baseline, index=data_series.index, name=f"{data_series.name}_baseline")


if __name__ == '__main__':
    # Setup for testing
    sampling_freq_hz = 25  # Hz
    duration_sec = 200    # seconds
    num_samples = duration_sec * sampling_freq_hz
    time_index = pd.to_datetime(np.arange(num_samples) / sampling_freq_hz, unit='s')

    # Create dummy flow rate data (sine wave + noise + some events)
    flow_rate = np.sin(2 * np.pi * 0.2 * np.arange(num_samples) / sampling_freq_hz) # Respiratory rate 0.2 Hz = 12/min
    flow_rate += 0.2 * np.random.randn(num_samples) # Add some noise
    # Simulate an apnea
    flow_rate[100*sampling_freq_hz : 115*sampling_freq_hz] = 0.01 * np.random.randn(15*sampling_freq_hz)
    # Simulate a hypopnea
    flow_rate[150*sampling_freq_hz : 165*sampling_freq_hz] *= 0.3
    flow_rate_series = pd.Series(flow_rate, index=time_index, name='flow_rate')

    # Create dummy leak rate data
    leak_rate = np.random.rand(num_samples) * 10 # Random leak up to 10 L/min
    leak_rate[50*sampling_freq_hz : 80*sampling_freq_hz] = 30 # High leak period
    leak_rate_series = pd.Series(leak_rate, index=time_index, name='leak_rate')

    df_test = pd.DataFrame({'flow_rate': flow_rate_series, 'leak_rate': leak_rate_series})
    print("Original Data (first 5 rows):")
    print(df_test.head())

    # 1. Test Butterworth filter
    print("\n--- Testing Butterworth Lowpass Filter ---")
    # Cutoff for typical breath dynamics, e.g., 2 Hz to remove high freq noise but keep breath shape
    flow_filtered_lowpass = butterworth_filter(df_test['flow_rate'],
                                           filter_type='lowpass',
                                           cutoff_freq_hz=2.0,
                                           sampling_freq_hz=sampling_freq_hz,
                                           order=4)
    print("Filtered flow rate (lowpass, first 5 rows):")
    print(flow_filtered_lowpass.head())

    print("\n--- Testing Butterworth Highpass Filter (e.g., to remove DC offset/slow drift) ---")
    # Example: remove very slow drifts below 0.05 Hz
    flow_filtered_highpass = butterworth_filter(df_test['flow_rate'],
                                           filter_type='highpass',
                                           cutoff_freq_hz=0.05,
                                           sampling_freq_hz=sampling_freq_hz,
                                           order=2)
    print("Filtered flow rate (highpass, first 5 rows):")
    print(flow_filtered_highpass.head())

    print("\n--- Testing Butterworth Bandpass Filter (e.g., for specific respiratory frequencies) ---")
    # Example: keep frequencies between 0.1 Hz (6/min) and 0.5 Hz (30/min)
    flow_filtered_bandpass = butterworth_filter(df_test['flow_rate'],
                                           filter_type='bandpass',
                                           cutoff_freq_hz=(0.1, 0.5),
                                           sampling_freq_hz=sampling_freq_hz,
                                           order=2)
    print("Filtered flow rate (bandpass, first 5 rows):")
    print(flow_filtered_bandpass.head())

    # Test filter with edge cases
    print("\n--- Testing Filter Edge Cases ---")
    short_series = pd.Series(np.random.randn(10), name="short") # Shorter than 3*order
    filtered_short = butterworth_filter(short_series, 'lowpass', 2.0, sampling_freq_hz, order=4)
    print(f"Filtered short series (len {len(filtered_short)}): \n{filtered_short.head()}")

    very_short_series = pd.Series(np.random.randn(5), name="v_short") # Shorter than order*2
    filtered_v_short = butterworth_filter(very_short_series, 'lowpass', 2.0, sampling_freq_hz, order=3) # order 3 for N*2=6
    print(f"Filtered very short series (len {len(filtered_v_short)}): \n{filtered_v_short.head()}")

    cutoff_at_nyquist = butterworth_filter(df_test['flow_rate'], 'lowpass', sampling_freq_hz / 2, sampling_freq_hz)
    print(f"Lowpass at Nyquist (should be same as original or near): \n{cutoff_at_nyquist.head()}")
    cutoff_above_nyquist = butterworth_filter(df_test['flow_rate'], 'lowpass', sampling_freq_hz, sampling_freq_hz)
    print(f"Lowpass above Nyquist (should be same as original): \n{cutoff_above_nyquist.head()}")


    # 2. Test High Leak Flagging
    print("\n--- Testing High Leak Flagging ---")
    leak_threshold_val = 20.0  # L/min
    min_duration_val = 5.0     # seconds
    df_test['high_leak_flag'] = flag_high_leak_periods(df_test['leak_rate'],
                                                       leak_threshold=leak_threshold_val,
                                                       min_duration_sec=min_duration_val,
                                                       sampling_freq_hz=sampling_freq_hz)
    print(f"Data with high leak flag (showing period around 50-80s where leak is {leak_threshold_val}+):")
    print(df_test[ (df_test.index >= pd.Timestamp(ts_input=45, unit='s')) & (df_test.index <= pd.Timestamp(ts_input=85, unit='s')) ][['leak_rate', 'high_leak_flag']])
    print(f"Total high leak flagged samples: {df_test['high_leak_flag'].sum()}")
    # Test with no high leak
    no_leak_series = pd.Series(np.ones(100) * 5, index=pd.to_datetime(np.arange(100)/sampling_freq_hz, unit='s'))
    no_leak_flag = flag_high_leak_periods(no_leak_series, 20, 5, sampling_freq_hz)
    print(f"Flagging on no-high-leak series (all False): {no_leak_flag.unique()}")


    # 3. Test Rolling Baseline Calculation
    print("\n--- Testing Rolling Baseline Calculation (Median) ---")
    # Use the lowpass filtered flow for baseline calculation
    df_test['flow_rate_filtered'] = flow_filtered_lowpass
    df_test['flow_baseline_median'] = calculate_rolling_baseline(df_test['flow_rate_filtered'],
                                                              window_sec=120,
                                                              sampling_freq_hz=sampling_freq_hz,
                                                              center=True,
                                                              quantile=0.5) # Median
    print("Data with flow baseline (median, first 5 and around apnea at 100s):")
    print(df_test[['flow_rate_filtered', 'flow_baseline_median']].head())
    print(df_test[['flow_rate_filtered', 'flow_baseline_median']][98*sampling_freq_hz : 102*sampling_freq_hz])

    print("\n--- Testing Rolling Baseline Calculation (25th Percentile) ---")
    df_test['flow_baseline_p25'] = calculate_rolling_baseline(df_test['flow_rate_filtered'],
                                                              window_sec=60,
                                                              sampling_freq_hz=sampling_freq_hz,
                                                              center=True,
                                                              quantile=0.25) # 25th percentile
    print("Data with flow baseline (25th percentile, first 5):")
    print(df_test[['flow_rate_filtered', 'flow_baseline_p25']].head())

    # Optional: Plotting for visual verification if run in an environment that supports it
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    # axs[0].plot(df_test.index, df_test['flow_rate'], label='Raw Flow')
    # axs[0].plot(df_test.index, df_test['flow_rate_filtered'], label='Filtered Flow (Lowpass 2Hz)')
    # axs[0].plot(df_test.index, df_test['flow_baseline_median'], label='Flow Baseline (Median 120s)', linestyle='--')
    # axs[0].legend()
    # axs[0].set_title('Flow Data & Baseline')
    # axs[1].plot(df_test.index, df_test['leak_rate'], label='Leak Rate')
    # axs[1].axhline(leak_threshold_val, color='red', linestyle='--', label=f'Leak Threshold ({leak_threshold_val})')
    # axs[1].fill_between(df_test.index, 0, df_test['leak_rate'].max(), where=df_test['high_leak_flag'],
    #                     color='red', alpha=0.3, label='High Leak Period')
    # axs[1].legend()
    # axs[1].set_title('Leak Data & Flagging')
    # axs[2].plot(df_test.index, flow_filtered_highpass, label='Filtered Flow (Highpass 0.05Hz)')
    # axs[2].plot(df_test.index, flow_filtered_bandpass, label='Filtered Flow (Bandpass 0.1-0.5Hz)')
    # axs[2].legend()
    # axs[2].set_title('Other Filter Examples')
    # plt.tight_layout()
    # plt.show()
    print("\nPreprocessing module tests complete.")
