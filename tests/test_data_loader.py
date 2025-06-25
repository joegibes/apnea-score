import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

# Add src directory to Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.data_loader import load_cpap_data, resample_data

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to store dummy CSVs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dummy_data_path = os.path.join(self.temp_dir.name, 'dummy.csv')

        # Basic dummy data
        self.df_basic = pd.DataFrame({
            'Time': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:00:01']),
            'Flow': [0.1, 0.2],
            'Press': [5.0, 5.1],
            'Lek': [1.0, 1.1]
        })
        self.df_basic.to_csv(self.dummy_data_path, index=False)

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_load_cpap_data_basic(self):
        """Test basic loading and column renaming."""
        loaded_df = load_cpap_data(self.dummy_data_path,
                                   timestamp_col='Time',
                                   flow_rate_col='Flow',
                                   pressure_col='Press',
                                   leak_rate_col='Lek')
        self.assertIsNotNone(loaded_df)
        self.assertIn('flow_rate', loaded_df.columns)
        self.assertIn('pressure', loaded_df.columns)
        self.assertIn('leak_rate', loaded_df.columns)
        self.assertEqual(len(loaded_df), 2)
        self.assertTrue(isinstance(loaded_df.index, pd.DatetimeIndex))

    def test_load_cpap_data_missing_essential_col_in_file(self):
        """Test loading when an essential column specified in mapping is missing from CSV."""
        df_missing_col = self.df_basic.drop(columns=['Flow'])
        missing_col_path = os.path.join(self.temp_dir.name, 'missing_flow.csv')
        df_missing_col.to_csv(missing_col_path, index=False)

        loaded_df = load_cpap_data(missing_col_path,
                                   timestamp_col='Time',
                                   flow_rate_col='Flow_NonExistent', # Try to map to a non-existent name
                                   pressure_col='Press',
                                   leak_rate_col='Lek')
        self.assertIsNone(loaded_df) # Should fail because 'Flow_NonExistent' isn't in CSV

        loaded_df_correct_map_missing = load_cpap_data(missing_col_path,
                                   timestamp_col='Time',
                                   flow_rate_col='Flow', # Correct standard name, but 'Flow' is not in this CSV
                                   pressure_col='Press',
                                   leak_rate_col='Lek')
        self.assertIsNone(loaded_df_correct_map_missing)


    def test_load_cpap_data_with_custom_map(self):
        """Test loading using the custom_col_map argument."""
        custom_map = {
            'timestamp': 'Time', # This will be overridden by timestamp_col if both provided
            'flow_rate': 'Flow',
            'pressure': 'Press',
            # 'leak_rate' will use its default 'leak_rate' or explicit arg if provided
        }
        loaded_df = load_cpap_data(self.dummy_data_path,
                                   timestamp_col='Time', # Explicit timestamp col
                                   leak_rate_col='Lek',  # Explicit leak col
                                   custom_col_map=custom_map)
        self.assertIsNotNone(loaded_df)
        self.assertIn('flow_rate', loaded_df.columns)
        self.assertIn('pressure', loaded_df.columns)
        self.assertIn('leak_rate', loaded_df.columns)
        self.assertEqual(pd.Timestamp('2023-01-01 00:00:00'), loaded_df.index[0])

    def test_load_cpap_data_file_not_found(self):
        loaded_df = load_cpap_data('non_existent_file.csv')
        self.assertIsNone(loaded_df)

    def test_load_optional_columns(self):
        df_full = pd.DataFrame({
            'Time': pd.to_datetime(['2023-01-01 00:00:00']),
            'F': [0.1], 'P': [5.0], 'L': [1.0],
            'MV': [6.0], 'RR': [15], 'TV': [0.4]
        })
        full_path = os.path.join(self.temp_dir.name, 'full.csv')
        df_full.to_csv(full_path, index=False)

        loaded_df = load_cpap_data(full_path,
                                   timestamp_col='Time', flow_rate_col='F', pressure_col='P', leak_rate_col='L',
                                   minute_vent_col='MV', resp_rate_col='RR', tidal_vol_col='TV')
        self.assertIsNotNone(loaded_df)
        self.assertIn('minute_ventilation', loaded_df.columns)
        self.assertIn('respiratory_rate', loaded_df.columns)
        self.assertIn('tidal_volume', loaded_df.columns)
        self.assertEqual(loaded_df['minute_ventilation'].iloc[0], 6.0)


    def test_resample_data_basic(self):
        """Test basic resampling functionality."""
        # Original data is 1 sample per second (1 Hz)
        df_to_resample = self.df_basic.set_index(pd.to_datetime(self.df_basic['Time']))
        df_to_resample = df_to_resample[['Flow', 'Press', 'Lek']].rename(
            columns={'Flow':'flow_rate', 'Press':'pressure', 'Lek':'leak_rate'} # Match expected names
        )

        # Resample to 2 Hz (upsample)
        resampled_df = resample_data(df_to_resample, target_freq_hz=2)
        self.assertIsNotNone(resampled_df)
        # Original has 2 points at 0s, 1s. Resampled to 2Hz over 1s duration.
        # Timestamps should be 0s, 0.5s, 1s. (3 points)
        self.assertEqual(len(resampled_df), 3)
        self.assertAlmostEqual((resampled_df.index[1] - resampled_df.index[0]).total_seconds(), 0.5)

        # Check linear interpolation for upsampling
        # Original flow: 0.1 at 0s, 0.2 at 1s.
        # Expected at 0.5s: (0.1+0.2)/2 = 0.15
        self.assertAlmostEqual(resampled_df['flow_rate'].iloc[1], 0.15, places=4)


    def test_resample_data_empty_df(self):
        empty_df = pd.DataFrame(columns=['flow_rate']).set_index(pd.to_datetime([]))
        resampled_empty = resample_data(empty_df, target_freq_hz=10)
        self.assertTrue(resampled_empty.empty)

    def test_resample_data_no_datetimeindex(self):
        df_no_dt_index = pd.DataFrame({'flow_rate': [1,2,3]})
        resampled_df = resample_data(df_no_dt_index, target_freq_hz=10)
        self.assertIsNone(resampled_df) # Should return None as per function logic

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
