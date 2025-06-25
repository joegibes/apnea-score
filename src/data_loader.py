import pandas as pd
from typing import Optional, List

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

    rule = f"{1000/target_freq_hz:.0f}L" # Resample rule in milliseconds (L)

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
    # Create a dummy CSV for testing
    dummy_data = {
        'Time': pd.to_datetime(['2023-01-01 00:00:00.000',
                                '2023-01-01 00:00:00.040',
                                '2023-01-01 00:00:00.080',
                                '2023-01-01 00:00:00.120',
                                '2023-01-01 00:00:00.200']), # Note: uneven sampling for test
        'Flow (L/min)': [0.1, 0.5, 0.2, -0.3, 0.1],
        'Pressure (cmH2O)': [5.0, 5.1, 5.0, 4.9, 5.0],
        'Leak (L/min)': [0.0, 0.1, 0.0, 0.2, 0.1],
        'Events': ['None', 'None', 'Hypopnea', 'None', 'Apnea'] # Example of a non-numeric column
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_csv_path = 'data/dummy_cpap_data.csv'
    dummy_df.to_csv(dummy_csv_path, index=False)

    print(f"Created dummy CSV at {dummy_csv_path}")

    # Test loading
    print("\n--- Testing data loading ---")
    loaded_df = load_cpap_data(dummy_csv_path,
                               timestamp_col='Time',
                               flow_rate_col='Flow (L/min)',
                               pressure_col='Pressure (cmH2O)',
                               leak_rate_col='Leak (L/min)')

    if loaded_df is not None:
        print("Successfully loaded data:")
        print(loaded_df.head())
        print("\nData types:")
        print(loaded_df.dtypes)

        # Test resampling
        print("\n--- Testing resampling to 10 Hz ---")
        resampled_df_10hz = resample_data(loaded_df.copy(), target_freq_hz=10) # Target 100ms interval
        if resampled_df_10hz is not None:
            print("Resampled data (10 Hz):")
            print(resampled_df_10hz)
            print(f"Number of rows: {len(resampled_df_10hz)}")

        print("\n--- Testing resampling to 25 Hz (Original was effectively 25Hz for first few points) ---")
        # Original data is 40ms interval = 25Hz
        resampled_df_25hz = resample_data(loaded_df.copy(), target_freq_hz=25)
        if resampled_df_25hz is not None:
            print("Resampled data (25 Hz):")
            print(resampled_df_25hz)
            print(f"Number of rows: {len(resampled_df_25hz)}")

        print("\n--- Testing resampling to 50 Hz (Upsampling) ---")
        resampled_df_50hz = resample_data(loaded_df.copy(), target_freq_hz=50)
        if resampled_df_50hz is not None:
            print("Resampled data (50 Hz):")
            print(resampled_df_50hz)
            print(f"Number of rows: {len(resampled_df_50hz)}")

    else:
        print("Data loading failed.")

    # Test with missing essential column
    print("\n--- Testing with missing essential column ---")
    missing_col_map = {
        'timestamp': 'Time',
        'flow_rate': 'NonExistentFlowCol', # This column doesn't exist
        'pressure': 'Pressure (cmH2O)',
        'leak_rate': 'Leak (L/min)'
    }
    # Need to save a df that would cause this
    temp_df_missing = dummy_df.rename(columns={'Flow (L/min)': 'Actual Flow'})
    temp_df_missing.to_csv('data/dummy_missing_col.csv', index=False)

    loaded_missing_df = load_cpap_data('data/dummy_missing_col.csv',
                                       timestamp_col='Time',
                                       flow_rate_col='NonExistentFlowCol', # Standard name we expect
                                       pressure_col='Pressure (cmH2O)',
                                       leak_rate_col='Leak (L/min)')
    if loaded_missing_df is None:
        print("Correctly failed to load due to missing mapped essential column.")

    print("\n--- Testing with custom column map ---")
    custom_map = {
        'flow_rate': 'Flow (L/min)', # map standard 'flow_rate' to actual 'Flow (L/min)'
        'pressure': 'Pressure (cmH2O)',
        'leak_rate': 'Leak (L/min)'
    }
    # Here, we rely on the default timestamp_col='timestamp', but our file has 'Time'
    # So, we need to provide timestamp_col explicitly or add it to custom_col_map
    loaded_custom_df = load_cpap_data(dummy_csv_path,
                                      timestamp_col='Time', # Explicitly state the actual timestamp column name
                                      custom_col_map=custom_map) # Provide map for others

    if loaded_custom_df is not None:
        print("Successfully loaded data using custom map:")
        print(loaded_custom_df.head())
    else:
        print("Custom map loading failed.")

    print("\n--- Testing with completely different column names and custom_col_map ---")
    df_alt_names = pd.DataFrame({
        'MyTime': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:00:01']),
        'AirFlow': [0.2, 0.3],
        'AirPressure': [6, 6.1],
        'MaskLeak': [0.1, 0.1]
    })
    alt_csv_path = 'data/dummy_alt_names.csv'
    df_alt_names.to_csv(alt_csv_path, index=False)

    loaded_alt_df = load_cpap_data(alt_csv_path,
                                   custom_col_map={
                                       'timestamp': 'MyTime',
                                       'flow_rate': 'AirFlow',
                                       'pressure': 'AirPressure',
                                       'leak_rate': 'MaskLeak'
                                   })
    if loaded_alt_df is not None:
        print("Successfully loaded data with alternative names via custom_col_map:")
        print(loaded_alt_df.head())
    else:
        print("Loading with alternative names and custom_col_map failed.")

    print("\n--- Testing data loading with specified optional columns ---")
    dummy_data_full = {
        'Time': pd.to_datetime(['2023-01-01 00:00:00.000', '2023-01-01 00:00:00.040']),
        'Flow': [0.1, 0.5],
        'Press': [5.0, 5.1],
        'Lek': [0.0, 0.1],
        'MV': [5.0, 5.2],
        'RR': [12, 13],
        'TV': [0.4, 0.45]
    }
    dummy_df_full = pd.DataFrame(dummy_data_full)
    dummy_full_csv_path = 'data/dummy_cpap_data_full.csv'
    dummy_df_full.to_csv(dummy_full_csv_path, index=False)

    loaded_df_full = load_cpap_data(dummy_full_csv_path,
                               timestamp_col='Time',
                               flow_rate_col='Flow',
                               pressure_col='Press',
                               leak_rate_col='Lek',
                               minute_vent_col='MV',
                               resp_rate_col='RR',
                               tidal_vol_col='TV')

    if loaded_df_full is not None:
        print("Successfully loaded data with all optional columns specified:")
        print(loaded_df_full.head())
        expected_cols = ['flow_rate', 'pressure', 'leak_rate', 'minute_ventilation', 'respiratory_rate', 'tidal_volume']
        missing_in_df = [col for col in expected_cols if col not in loaded_df_full.columns]
        if not missing_in_df:
            print(f"All expected columns present: {loaded_df_full.columns.tolist()}")
        else:
            print(f"Error: Missing expected columns after load: {missing_in_df}")
    else:
        print("Full data loading failed.")
import numpy as np # Add numpy import for the script
