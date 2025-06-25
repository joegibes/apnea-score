import pandas as pd
from typing import Optional, List
import mne
import numpy as np

def load_edf_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Loads CPAP data from an EDF file into a pandas DataFrame.

    Args:
        filepath (str): Path to the EDF file.

    Returns:
        Optional[pd.DataFrame]: DataFrame with data from the EDF file,
                                 or None if loading fails.
    """
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        df = raw.to_data_frame()
        # Timestamps are not always in a 'timestamp' column, they are often the index
        if 'time' in df.columns:
             df['timestamp'] = pd.to_datetime(raw.info['meas_date']) + pd.to_timedelta(df['time'], unit='s')
             df = df.set_index('timestamp')
             df = df.drop(columns=['time'])
        return df
    except Exception as e:
        print(f"Error loading EDF file {filepath}: {e}")
        return None

def load_cpap_data(filepath: str,
                   timestamp_col: str = 'timestamp',
                   flow_rate_col: str = 'flow_rate',
                   pressure_col: str = 'pressure',
                   leak_rate_col: str = 'leak_rate',
                   minute_vent_col: Optional[str] = 'minute_ventilation',
                   resp_rate_col: Optional[str] = 'respiratory_rate',
                   tidal_vol_col: Optional[str] = 'tidal_volume',
                   custom_col_map: Optional[dict] = None
                  ) -> Optional[pd.DataFrame]:
    """
    Loads CPAP data from a CSV file into a pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.
        timestamp_col (str): Name of the column containing timestamps.
        flow_rate_col (str): Name of the column for flow rate.
        pressure_col (str): Name of the column for pressure.
        leak_rate_col (str): Name of the column for leak rate.
        minute_vent_col (Optional[str]): Name of the column for minute ventilation.
        resp_rate_col (Optional[str]): Name of the column for respiratory rate.
        tidal_vol_col (Optional[str]): Name of the column for tidal volume.
        custom_col_map (Optional[dict]): A dictionary to map expected column names
                                         (e.g., 'flow_rate_std') to actual names
                                         in the CSV (e.g., 'Flow Rate (L/min)').

    Returns:
        Optional[pd.DataFrame]: DataFrame with standardized column names
                                ('timestamp', 'flow_rate', 'pressure', 'leak_rate',
                                 'minute_ventilation', 'respiratory_rate', 'tidal_volume')
                                 or None if loading fails.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    # Standardize column names
    # Base columns that are expected
    expected_to_actual = {
        'timestamp': timestamp_col,
        'flow_rate': flow_rate_col,
        'pressure': pressure_col,
        'leak_rate': leak_rate_col,
    }
    # Optional columns
    if minute_vent_col:
        expected_to_actual['minute_ventilation'] = minute_vent_col
    if resp_rate_col:
        expected_to_actual['respiratory_rate'] = resp_rate_col
    if tidal_vol_col:
        expected_to_actual['tidal_volume'] = tidal_vol_col

    if custom_col_map:
        expected_to_actual.update(custom_col_map)

    # Reverse map for renaming: actual_name -> standard_name
    actual_to_standard = {v: k for k, v in expected_to_actual.items() if v in df.columns}

    # Check if essential columns are present after mapping
    missing_essential_cols = []
    if 'timestamp' not in actual_to_standard.values():
        missing_essential_cols.append(timestamp_col)
    if 'flow_rate' not in actual_to_standard.values():
        missing_essential_cols.append(flow_rate_col)
    if 'pressure' not in actual_to_standard.values():
        missing_essential_cols.append(pressure_col)
    if 'leak_rate' not in actual_to_standard.values():
         missing_essential_cols.append(leak_rate_col)

    if missing_essential_cols:
        print(f"Error: Essential columns missing from CSV or mapping: {', '.join(missing_essential_cols)}")
        print(f"Available columns in CSV: {df.columns.tolist()}")
        print(f"Current mapping: {expected_to_actual}")
        return None

    df_renamed = df.rename(columns=actual_to_standard)

    # Select only the standardized columns we care about
    standard_columns = [
        'timestamp', 'flow_rate', 'pressure', 'leak_rate',
        'minute_ventilation', 'respiratory_rate', 'tidal_volume'
    ]
    final_columns = [col for col in standard_columns if col in df_renamed.columns]
    df_processed = df_renamed[final_columns].copy()


    # Convert timestamp column
    try:
        # Attempt to infer datetime format, robust to various common formats
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
    except Exception as e:
        print(f"Error converting timestamp column: {e}. Attempting common formats.")
        # Try a few common formats if direct conversion fails
        formats_to_try = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %I:%M:%S %p"]
        converted = False
        for fmt in formats_to_try:
            try:
                df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], format=fmt)
                converted = True
                break
            except ValueError:
                continue
        if not converted:
            print("Error: Could not parse timestamp column with common formats. Please ensure it's a recognizable datetime format.")
            return None

    df_processed.set_index('timestamp', inplace=True)

    # Basic cleaning: interpolate small gaps (e.g., up to 1 second) in numeric columns
    numeric_cols = df_processed.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') # Ensure numeric
        df_processed[col] = df_processed[col].interpolate(method='linear', limit=5, limit_direction='both') # Small limit

    return df_processed

def resample_data(df: pd.DataFrame, target_freq_hz: int = 25) -> Optional[pd.DataFrame]:
    """
    Resamples the DataFrame to a target frequency.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.
        target_freq_hz (int): Target sampling frequency in Hz.

    Returns:
        Optional[pd.DataFrame]: Resampled DataFrame, or None if error.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: DataFrame must have a DatetimeIndex to resample.")
        return None

    if df.empty:
        print("Warning: DataFrame is empty, cannot resample.")
        return df

    rule = f"{1000/target_freq_hz:.0f}ms" # Resample rule in milliseconds

    try:
        # Separate numeric and non-numeric columns if any non-numeric exist besides index
        numeric_cols = df.select_dtypes(include=np.number).columns
        df_numeric = df[numeric_cols]

        # Using mean for upsampling/downsampling numeric data. Can also use 'linear' for upsampling.
        df_resampled_numeric = df_numeric.resample(rule).mean()

        # For upsampling, linear interpolation is often preferred for continuous signals
        # For downsampling, mean is generally okay.
        # If upsampling (new frequency > original), interpolate after resampling.
        # This heuristic checks if we are likely upsampling significantly
        original_avg_interval_ms = df.index.to_series().diff().median().total_seconds() * 1000
        target_interval_ms = 1000 / target_freq_hz

        if target_interval_ms < original_avg_interval_ms: # Likely upsampling
             df_resampled_numeric = df_resampled_numeric.interpolate(method='linear')

        # If there were other column types, they'd need specific handling.
        # For now, assuming all relevant CPAP data is numeric.

        return df_resampled_numeric

    except Exception as e:
        print(f"Error during resampling: {e}")
        return None

if __name__ == '__main__':
    pass