import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.breath_detection import detect_breaths_from_flow
# from src.preprocessing import butterworth_filter # If we want to test with filtered data

class TestBreathDetection(unittest.TestCase):

    def setUp(self):
        self.sampling_freq_hz = 25.0  # Hz, typical for CPAP
        self.scale_factor = 100000 # Simulate large scaled data like in Parquet

    def _generate_synthetic_flow(self, n_breaths=5, breath_duration_s=4,
                                 insp_ratio=0.4, peak_insp_flow=0.5, peak_exp_flow=-0.4,
                                 noise_level=0.01, scale_data=False):
        """Generates a simple synthetic flow signal."""
        insp_duration_samples = int(breath_duration_s * insp_ratio * self.sampling_freq_hz)
        exp_duration_samples = int(breath_duration_s * (1 - insp_ratio) * self.sampling_freq_hz)
        total_breath_samples = insp_duration_samples + exp_duration_samples

        flow = np.array([])
        for _ in range(n_breaths):
            # Inspiration (half sine)
            t_insp = np.linspace(0, np.pi, insp_duration_samples, endpoint=False)
            insp_phase = peak_insp_flow * np.sin(t_insp)
            # Expiration (half sine)
            t_exp = np.linspace(0, np.pi, exp_duration_samples, endpoint=False)
            exp_phase = peak_exp_flow * np.sin(t_exp) # peak_exp_flow is negative
            flow = np.concatenate((flow, insp_phase, exp_phase))

        flow += noise_level * np.random.randn(len(flow))
        if scale_data:
            flow *= self.scale_factor

        time_index = pd.to_datetime(np.arange(len(flow)) / self.sampling_freq_hz, unit='s')
        return pd.Series(flow, index=time_index, name="flow_rate")

    def test_empty_series_input(self):
        """Test with an empty input series."""
        flow_series = pd.Series([], dtype=float, index=pd.to_datetime([]))
        breaths_df = detect_breaths_from_flow(flow_series, self.sampling_freq_hz)
        self.assertTrue(breaths_df.empty, "Should return empty DataFrame for empty input.")

    def test_flat_zero_flow(self):
        """Test with a flatline zero flow signal."""
        flow_series = pd.Series(np.zeros(100),
                                index=pd.to_datetime(np.arange(100) / self.sampling_freq_hz, unit='s'))
        breaths_df = detect_breaths_from_flow(flow_series, self.sampling_freq_hz)
        self.assertTrue(breaths_df.empty, "Should return empty DataFrame for flat zero flow.")

    def test_basic_breath_detection_physiological_scale(self):
        """Test with a few clean breaths on a physiological scale."""
        flow_series = self._generate_synthetic_flow(n_breaths=3, scale_data=False)
        # Pass None for thresholds to test adaptive logic, or small physiological values
        breaths_df = detect_breaths_from_flow(flow_series, self.sampling_freq_hz,
                                              zero_crossing_hysteresis=0.02,
                                              peak_prominence_threshold=0.05)
        self.assertEqual(len(breaths_df), 2, "Should detect 2 full breaths from 3 cycles if ends mid-breath.")
                                             # Loop requires k+2 crossings, so n_breaths-1 typically.
                                             # If last breath is complete in data, might be n_breaths.
                                             # Current logic often results in N-1 or N-2 for N cycles.
                                             # For 3 cycles, we expect 2 full breaths.

        if not breaths_df.empty:
            self.assertGreater(breaths_df['insp_duration_s'].mean(), 0.5) # Basic sanity
            self.assertGreater(breaths_df['exp_duration_s'].mean(), 0.5)  # Basic sanity
            self.assertAlmostEqual(breaths_df['insp_peak_flow'].mean(), 0.5, delta=0.15)
            self.assertAlmostEqual(breaths_df['exp_peak_flow'].mean(), -0.4, delta=0.15)


    def test_breath_detection_scaled_data(self):
        """Test with data scaled to large values, relying on adaptive thresholds."""
        flow_series = self._generate_synthetic_flow(n_breaths=3, scale_data=True)
        # Pass None for thresholds to force adaptive logic based on large scale
        breaths_df = detect_breaths_from_flow(flow_series, self.sampling_freq_hz,
                                              zero_crossing_hysteresis=None,
                                              peak_prominence_threshold=None)
        self.assertEqual(len(breaths_df), 2) # Expect 2 full breaths

        if not breaths_df.empty:
            # Check if peak flows are also scaled
            self.assertAlmostEqual(breaths_df['insp_peak_flow'].mean(), 0.5 * self.scale_factor, delta=0.15 * self.scale_factor)
            self.assertAlmostEqual(breaths_df['exp_peak_flow'].mean(), -0.4 * self.scale_factor, delta=0.15 * self.scale_factor)
            # Tidal volume should also be scaled
            # Expected TV ~ 0.5 * peak_insp_flow * insp_duration_s (for half-sine shape)
            # insp_duration_s = 4s * 0.4 = 1.6s
            # Expected TV ~ 0.5 * (0.5 * scale_factor) * 1.6 ~ 0.4 * scale_factor (very approx for sine)
            # More accurately for half-sine: (peak_flow * duration) / pi
            expected_tv_approx = (0.5 * self.scale_factor * (4*0.4)) / np.pi
            self.assertAlmostEqual(breaths_df['tidal_volume_l'].mean(), expected_tv_approx, delta=expected_tv_approx * 0.3)


    def test_noisy_zero_crossings_robustness(self):
        """Test robustness against noisy zero crossings (potential double counts)."""
        peak_flow_val = 0.5
        flow_cycle = self._generate_synthetic_flow(n_breaths=1, peak_insp_flow=peak_flow_val, peak_exp_flow=-peak_flow_val, noise_level=0)

        # Introduce noise specifically around a zero crossing point
        # Find first insp_to_exp crossing (mid-point of the first breath)
        mid_point_approx = int(len(flow_cycle) * 0.4) # Approx end of first inspiration

        noisy_segment_len = int(0.1 * self.sampling_freq_hz) # 100ms of noise
        start_noise = mid_point_approx - noisy_segment_len // 2
        end_noise = mid_point_approx + noisy_segment_len // 2

        # Add significant noise that crosses zero multiple times in a short window
        # Noise amplitude should be > a small fixed hysteresis but hopefully handled by adaptive or refractory period
        noise_amp = 0.1 # Physiological scale noise amplitude
        if start_noise > 0 and end_noise < len(flow_cycle):
            flow_cycle.iloc[start_noise:end_noise] += noise_amp * np.sin(np.linspace(0, 10*np.pi, end_noise-start_noise)) # High freq noise
            flow_cycle.iloc[start_noise:end_noise] -= flow_cycle.iloc[start_noise:end_noise].mean() # Center noise around 0

        # Test with a small fixed hysteresis that *would* fail without robust logic
        breaths_df_fixed_low_hyst = detect_breaths_from_flow(flow_cycle, self.sampling_freq_hz,
                                                       zero_crossing_hysteresis=0.01,
                                                       peak_prominence_threshold=0.05)

        # Test with adaptive hysteresis (by passing None)
        breaths_df_adaptive = detect_breaths_from_flow(flow_cycle, self.sampling_freq_hz,
                                                       zero_crossing_hysteresis=None,
                                                       peak_prominence_threshold=None)

        # We expect 0 full breaths because we only generated one cycle and it's noisy at crossing.
        # Or, if the noise is overcome, it might find 0 or 1 depending on exact structure.
        # The key is that it shouldn't produce an excessive number of tiny "breaths" from the noise.
        # Given one cycle, it's hard to form a full k, k+1, k+2 sequence.
        # Let's try with 3 cycles and noise in the middle one
        flow_series_3cycle_noisy = self._generate_synthetic_flow(n_breaths=3, scale_data=False, noise_level=0.01)
        mid_cycle_start = int(len(flow_series_3cycle_noisy) / 3)
        mid_cycle_insp_end = mid_cycle_start + int(4 * 0.4 * self.sampling_freq_hz) # Approx end of insp of 2nd breath

        start_noise = mid_cycle_insp_end - noisy_segment_len // 2
        end_noise = mid_cycle_insp_end + noisy_segment_len // 2

        if start_noise > 0 and end_noise < len(flow_series_3cycle_noisy):
            # Add noise that crosses zero to simulate difficult crossing
            flow_series_3cycle_noisy.iloc[start_noise:end_noise] = \
                noise_amp * np.sin(np.linspace(0, 10*np.pi, end_noise-start_noise))

        breaths_df_3cycle_adaptive = detect_breaths_from_flow(flow_series_3cycle_noisy, self.sampling_freq_hz,
                                                       zero_crossing_hysteresis=None,
                                                       peak_prominence_threshold=None)

        # Expect around 2 breaths. If it's much more, double counting might be an issue.
        # If it's less, the noise might be breaking a valid breath.
        self.assertLessEqual(len(breaths_df_3cycle_adaptive), 3, "Should not detect excessive breaths due to noise at one crossing.")
        self.assertGreaterEqual(len(breaths_df_3cycle_adaptive), 1, "Should detect at least one breath despite noise.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
