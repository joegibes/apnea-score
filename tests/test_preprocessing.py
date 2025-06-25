import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to Python path to import modules
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.preprocessing import butterworth_filter, calculate_rolling_baseline, flag_high_leak_periods

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.sampling_freq_hz = 10.0  # Hz
        self.num_samples = 100
        self.time_index = pd.to_datetime(np.arange(self.num_samples) / self.sampling_freq_hz, unit='s')
        self.test_series = pd.Series(np.sin(2 * np.pi * 1 * np.arange(self.num_samples) / self.sampling_freq_hz) + \
                                     0.5 * np.sin(2 * np.pi * 4 * np.arange(self.num_samples) / self.sampling_freq_hz),
                                     index=self.time_index, name="test_signal")
        self.empty_series = pd.Series([], dtype=float)

    def test_butterworth_lowpass_filter_simple(self):
        """Test lowpass filter with a simple case, checking if high frequencies are attenuated."""
        # Signal is 1Hz sine + 4Hz sine. Lowpass at 2Hz should primarily keep 1Hz component.
        cutoff_freq = 2.0
        order = 4
        filtered_series = butterworth_filter(self.test_series, 'lowpass', cutoff_freq, self.sampling_freq_hz, order)

        # A proper check would involve FFT, but for a simple test:
        # The variance of the filtered signal should be less than original if high freq is removed.
        # And mean should be similar if no DC component is introduced/removed by lowpass.
        self.assertTrue(filtered_series.var() < self.test_series.var())
        self.assertAlmostEqual(filtered_series.mean(), self.test_series.mean(), places=2)
        self.assertEqual(len(filtered_series), len(self.test_series))

    def test_butterworth_filter_empty_series(self):
        """Test filter behavior with an empty series."""
        filtered_empty = butterworth_filter(self.empty_series, 'lowpass', 1.0, self.sampling_freq_hz)
        self.assertTrue(filtered_empty.empty)

    def test_butterworth_filter_cutoff_at_nyquist(self):
        """Test filter behavior when cutoff is at Nyquist frequency (should ideally not alter much for lowpass)."""
        nyquist_cutoff = self.sampling_freq_hz / 2.0
        # For lowpass at Nyquist, it should pass almost everything or be very close to original
        filtered_series = butterworth_filter(self.test_series, 'lowpass', nyquist_cutoff, self.sampling_freq_hz)
        # Depending on filter order and implementation details, it might not be exactly identical
        pd.testing.assert_series_equal(filtered_series, self.test_series, rtol=0.1, check_dtype=False)


    def test_butterworth_filter_invalid_type(self):
        """Test filter with an invalid filter type."""
        with self.assertRaises(ValueError):
            butterworth_filter(self.test_series, 'INVALID_TYPE', 1.0, self.sampling_freq_hz)

    def test_calculate_rolling_baseline_simple_median(self):
        """Test rolling median baseline with a very simple series."""
        data = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1], name="simple_data")
        data.index = pd.to_datetime(np.arange(len(data)) / 1.0, unit='s') # 1Hz sampling

        # Window of 3s (3 samples), median
        baseline = calculate_rolling_baseline(data, window_sec=3, sampling_freq_hz=1.0, center=True, quantile=0.5)

        # Expected: (bfill/ffill for edges)
        # Original: 1, 2, 3, 4, 5, 4, 3, 2, 1
        # Rolling median (3, center): NaN, 2, 3, 4, 5, 4, 3, 2, NaN
        # After ffill/bfill: 2, 2, 3, 4, 5, 4, 3, 2, 2 (approx)
        expected_baseline_values = [2, 2, 3, 4, 5, 4, 3, 2, 2] # Manual calculation based on pandas rolling
        pd.testing.assert_series_equal(baseline, pd.Series(expected_baseline_values, index=data.index, name="simple_data_baseline", dtype=float), check_dtype=False)

    def test_calculate_rolling_baseline_empty(self):
        baseline_empty = calculate_rolling_baseline(self.empty_series, 3, 1.0)
        self.assertTrue(baseline_empty.empty)

    def test_flag_high_leak_periods(self):
        """Test high leak flagging."""
        leak_data = pd.Series([5, 10, 25, 30, 28, 15, 5, 30, 35, 32, 8], name="leak_rate")
        leak_data.index = pd.to_datetime(np.arange(len(leak_data)) / 1.0, unit='s') # 1Hz

        # Threshold 20 L/min, min duration 2s (2 samples)
        flagged_leaks = flag_high_leak_periods(leak_data, leak_threshold=20, min_duration_sec=2, sampling_freq_hz=1.0)

        # Expected: F, F, T, T, T, F, F, T, T, T, F
        # Indices of True: 2,3,4 and 7,8,9
        expected_flags = pd.Series([False, False, True, True, True, False, False, True, True, True, False], index=leak_data.index)
        pd.testing.assert_series_equal(flagged_leaks, expected_flags, check_dtype=False)

    def test_flag_high_leak_no_high_leak(self):
        """Test high leak flagging when no period meets criteria."""
        leak_data = pd.Series([5, 10, 15, 18, 15, 10, 5], name="leak_rate_low")
        leak_data.index = pd.to_datetime(np.arange(len(leak_data)) / 1.0, unit='s')
        flagged_leaks = flag_high_leak_periods(leak_data, leak_threshold=20, min_duration_sec=2, sampling_freq_hz=1.0)
        self.assertFalse(flagged_leaks.any()) # Should all be False

    def test_flag_high_leak_short_duration(self):
        """Test high leak flagging when high leak is too short."""
        leak_data = pd.Series([5, 30, 5], name="leak_rate_short_burst") # High leak for 1s only
        leak_data.index = pd.to_datetime(np.arange(len(leak_data)) / 1.0, unit='s')
        flagged_leaks = flag_high_leak_periods(leak_data, leak_threshold=20, min_duration_sec=2, sampling_freq_hz=1.0)
        self.assertFalse(flagged_leaks.any())


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
