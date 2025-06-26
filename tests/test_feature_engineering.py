import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.feature_engineering import calculate_flow_limitation_features
# We might also test extract_features_for_event later, but it's more of an integration test.

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.sampling_freq_hz = 25.0  # Hz
        self.duration_insp_s = 2.0 # Standard inspiratory duration for tests
        self.num_samples_insp = int(self.duration_insp_s * self.sampling_freq_hz)
        self.time_insp = np.linspace(0, self.duration_insp_s, self.num_samples_insp, endpoint=False)
        self.time_index_insp = pd.to_datetime(self.time_insp, unit='s')

    def _create_insp_segment(self, shape_type="sine", peak_flow=1.0):
        """Helper to create different inspiratory flow shapes."""
        flow = np.zeros(self.num_samples_insp)
        if shape_type == "sine": # Rounded breath
            flow = peak_flow * np.sin(np.pi * self.time_insp / self.duration_insp_s)
        elif shape_type == "flat": # Perfectly flat top (maximal limitation)
            flow = np.ones(self.num_samples_insp) * peak_flow
        elif shape_type == "triangle": # Triangular breath
            mid_point = self.duration_insp_s / 2
            for i, t in enumerate(self.time_insp):
                if t <= mid_point:
                    flow[i] = (peak_flow / mid_point) * t
                else:
                    flow[i] = peak_flow - (peak_flow / (self.duration_insp_s - mid_point)) * (t - mid_point)
            flow = np.clip(flow, 0, peak_flow) # Ensure it doesn't go below 0 due to slope calc
        elif shape_type == "moderately_flat": # Plateau for part of insp
            plateau_start_ratio = 0.3
            plateau_end_ratio = 0.7
            t_plateau_start = self.duration_insp_s * plateau_start_ratio
            t_plateau_end = self.duration_insp_s * plateau_end_ratio

            for i, t_val in enumerate(self.time_insp):
                if t_val < t_plateau_start:
                    flow[i] = peak_flow * (t_val / t_plateau_start)
                elif t_val <= t_plateau_end:
                    flow[i] = peak_flow
                else: # Ramp down
                    flow[i] = peak_flow * (1 - (t_val - t_plateau_end) / (self.duration_insp_s - t_plateau_end))
            flow = np.clip(flow, 0, peak_flow)

        return pd.Series(flow, index=self.time_index_insp, name="flow_rate")

    def test_ifl_perfectly_rounded_sine(self):
        """Test IFL features on a synthetic half-sine wave (rounded) inspiration."""
        insp_flow = self._create_insp_segment(shape_type="sine", peak_flow=1.0)
        features = calculate_flow_limitation_features(insp_flow, self.sampling_freq_hz)

        # For a half-sine normalized from 0-1 in time and amplitude:
        # Area ratio (mean of normalized sine over half period) = (Integral_0^pi sin(x) dx) / pi = 2/pi ~ 0.6366
        self.assertAlmostEqual(features['ifl_shape_area_ratio'], 2/np.pi, delta=0.02)
        # Mid-point flow (normalized) for sine is 1.0
        self.assertAlmostEqual(features['ifl_shape_mid_point_flow'], 1.0, delta=0.02)
        # Peak broadness for sine ((sin(pi/4)+sin(3pi/4))/2) = sin(pi/4) = ~0.707
        self.assertAlmostEqual(features['ifl_shape_peak_broadness'], np.sin(np.pi/4), delta=0.02)
        # Quadratic coeff 'a' should be significantly negative for a rounded peak
        self.assertLess(features['ifl_poly_coeff_a'], -1.5) # Heuristic, typically around -4 for y=-4x(x-1)
        # print(f"Sine: Area Ratio={features['ifl_shape_area_ratio']:.3f}, PolyA={features['ifl_poly_coeff_a']:.3f}")


    def test_ifl_perfectly_flat_top(self):
        """Test IFL features on a synthetic flat top (square wave) inspiration."""
        insp_flow = self._create_insp_segment(shape_type="flat", peak_flow=1.0)
        features = calculate_flow_limitation_features(insp_flow, self.sampling_freq_hz)

        # For a perfect square wave normalized:
        self.assertAlmostEqual(features['ifl_shape_area_ratio'], 1.0, delta=0.01)
        self.assertAlmostEqual(features['ifl_shape_mid_point_flow'], 1.0, delta=0.01)
        self.assertAlmostEqual(features['ifl_shape_peak_broadness'], 1.0, delta=0.01)
        # Quadratic coeff 'a' should be close to 0 for a flat line
        self.assertAlmostEqual(features['ifl_poly_coeff_a'], 0.0, delta=0.1) # Allow some minor fitting error
        # print(f"Flat: Area Ratio={features['ifl_shape_area_ratio']:.3f}, PolyA={features['ifl_poly_coeff_a']:.3f}")

    def test_ifl_triangular_breath(self):
        """Test IFL features on a synthetic triangular inspiration."""
        insp_flow = self._create_insp_segment(shape_type="triangle", peak_flow=1.0)
        features = calculate_flow_limitation_features(insp_flow, self.sampling_freq_hz)
        # For a perfect triangle normalized (0,0) -> (0.5,1) -> (1,0)
        # Area ratio (mean of normalized triangle) = 0.5
        self.assertAlmostEqual(features['ifl_shape_area_ratio'], 0.5, delta=0.02)
        # Mid-point flow (normalized) for triangle peaking at 0.5 is 1.0
        self.assertAlmostEqual(features['ifl_shape_mid_point_flow'], 1.0, delta=0.02)
        # Peak broadness for triangle (0.5+0.5)/2 = 0.5
        self.assertAlmostEqual(features['ifl_shape_peak_broadness'], 0.5, delta=0.02)
        # Quadratic coeff 'a' should be negative. For y = -4(x-0.5)^2+1 -> a = -4
        # For y = -|2x-1|+1 on [0,1] (approx), poly fit for a triangle is often around -8x^2+8x+0
        self.assertLess(features['ifl_poly_coeff_a'], -1.0) # Expect negative 'a'
        # print(f"Triangle: Area Ratio={features['ifl_shape_area_ratio']:.3f}, PolyA={features['ifl_poly_coeff_a']:.3f}")


    def test_ifl_moderately_flat_breath(self):
        """Test IFL features on a moderately flattened inspiration."""
        insp_flow = self._create_insp_segment(shape_type="moderately_flat", peak_flow=1.0)
        features = calculate_flow_limitation_features(insp_flow, self.sampling_freq_hz)
        # Expected values should be between sine/triangle and flat
        self.assertGreater(features['ifl_shape_area_ratio'], 0.6) # Greater than sine/triangle
        self.assertLess(features['ifl_shape_area_ratio'], 1.0)    # Less than perfectly flat
        self.assertAlmostEqual(features['ifl_shape_mid_point_flow'], 1.0, delta=0.02) # Plateau at peak
        self.assertGreater(features['ifl_shape_peak_broadness'], 0.7) # Broader than sine
        self.assertLess(features['ifl_shape_peak_broadness'], 1.0)
        # Poly 'a' should be less negative than sine, but not zero
        self.assertGreater(features['ifl_poly_coeff_a'], -3.0) # Heuristic, less negative than sine
        self.assertLess(features['ifl_poly_coeff_a'], -0.1)  # Not as flat as 0
        # print(f"ModFlat: Area Ratio={features['ifl_shape_area_ratio']:.3f}, PolyA={features['ifl_poly_coeff_a']:.3f}")

    def test_ifl_with_device_signal(self):
        insp_flow = self._create_insp_segment(shape_type="sine")
        # Device IFL: 0=no limit, 0.1=mild, 0.2=mod, 0.3=severe (example scale)
        device_ifl_vals = np.zeros(self.num_samples_insp)
        device_ifl_vals[self.num_samples_insp//3 : 2*self.num_samples_insp//3] = 0.2 # Moderate IFL for middle third
        device_ifl_signal = pd.Series(device_ifl_vals, index=self.time_index_insp)

        features = calculate_flow_limitation_features(insp_flow, self.sampling_freq_hz, device_ifl_signal)

        self.assertIsNotNone(features['device_ifl_mean'])
        self.assertAlmostEqual(features['device_ifl_mean'], 0.2 / 3, delta=0.02) # 0.2 for 1/3 of duration
        self.assertEqual(features['device_ifl_max'], 0.2)
        self.assertEqual(features['device_ifl_median'], 0.0) # Median should be 0 if more than half is 0
        self.assertAlmostEqual(features['device_ifl_percent_high'], (1/3)*100, delta=2) # Assuming threshold 0.1

    def test_ifl_short_inspiration(self):
        short_insp_samples = int(0.05 * self.sampling_freq_hz) # 50ms, too short
        if short_insp_samples < 5: short_insp_samples = 5 # Ensure at least 5 for some edge cases in function

        short_time_index = pd.to_datetime(np.arange(short_insp_samples) / self.sampling_freq_hz, unit='s')
        insp_flow = pd.Series(np.linspace(0,1,short_insp_samples), index=short_time_index)
        features = calculate_flow_limitation_features(insp_flow, self.sampling_freq_hz)
        # Expect all NaNs due to short duration or too few points for reliable analysis
        for key, value in features.items():
            self.assertTrue(pd.isna(value), f"Feature {key} should be NaN for very short inspiration.")

    def test_ifl_zero_flow_inspiration(self):
        insp_flow = pd.Series(np.zeros(self.num_samples_insp), index=self.time_index_insp)
        features = calculate_flow_limitation_features(insp_flow, self.sampling_freq_hz)
        for key, value in features.items():
            self.assertTrue(pd.isna(value), f"Feature {key} should be NaN for zero flow inspiration.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

from typing import Optional, Dict # Add these if not already present at top of feature_engineering.py
# This is for the return type hint of calculate_flow_limitation_features
# And for Optional type hint for device_flow_lim_segment
