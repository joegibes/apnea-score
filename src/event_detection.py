import pandas as pd
import numpy as np
from typing import Optional, Tuple

def detect_apneas_hypopneas(
    flow_series: pd.Series,
    baseline_flow_series: pd.Series,
    sampling_freq_hz: float,
    apnea_threshold_ratio: float = 0.1, # Flow < 10% of baseline
    hypopnea_upper_threshold_ratio: float = 0.7, # Flow < 70% of baseline (AASM often 50% for alternative, 30% for primary)
    hypopnea_lower_threshold_ratio: float = 0.1, # Flow > 10% of baseline (to distinguish from apnea) AASM uses > reduction
                                                # More directly, reduction of 30-90% is hypopnea.
                                                # So, if flow is between 10% and 70% of baseline.
    min_event_duration_s: float = 10.0,
    high_leak_flags: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Detects apnea and hypopnea candidate events from a flow rate signal
    based on reductions relative to a dynamic baseline.

    Args:
        flow_series (pd.Series): Time series of flow rate (should be positive, e.g., envelope or rectified insp flow).
                                 For this version, we'll use the raw flow and consider its magnitude.
        baseline_flow_series (pd.Series): Time series of the calculated baseline flow (e.g., rolling median of breath amplitudes or similar).
                                          This should represent the expected typical peak flow or flow amplitude.
        sampling_freq_hz (float): Sampling frequency of the flow signal in Hz.
        apnea_threshold_ratio (float): Flow magnitude drops below this ratio of baseline to be considered apnea.
                                       E.g., 0.1 means < 10% of baseline flow amplitude.
        hypopnea_upper_threshold_ratio (float): Flow magnitude is below this ratio of baseline for hypopnea.
                                                E.g., 0.7 means < 70% of baseline. (AASM: reduction >30%)
        hypopnea_lower_threshold_ratio (float): Flow magnitude is above this ratio of baseline for hypopnea
                                                (to distinguish from apnea). E.g., 0.1 means > 10% of baseline.
                                                (AASM: reduction <90%)
        min_event_duration_s (float): Minimum duration in seconds for an event to be classified.
        high_leak_flags (Optional[pd.Series]): Boolean series indicating periods of high mask leak.
                                              Events during high leak may be excluded or flagged.

    Returns:
        pd.DataFrame: DataFrame with columns:
            'event_start_time': Timestamp of the event start.
            'event_end_time': Timestamp of the event end.
            'event_duration_s': Duration of the event in seconds.
            'event_type': 'apnea_candidate' or 'hypopnea_candidate'.
            'avg_flow_reduction_percent': Average flow reduction during the event relative to baseline.
            'min_flow_during_event_ratio': Minimum flow during event as ratio to baseline.
            'excluded_due_to_leak': Boolean, True if event overlapped with high leak.
    """
    if not isinstance(flow_series, pd.Series) or not isinstance(baseline_flow_series, pd.Series):
        raise TypeError("flow_series and baseline_flow_series must be pandas Series.")
    if flow_series.empty or baseline_flow_series.empty:
        return pd.DataFrame()
    if not flow_series.index.equals(baseline_flow_series.index):
        raise ValueError("flow_series and baseline_flow_series must have the same index.")

    min_samples_for_event = int(min_event_duration_s * sampling_freq_hz)
    dt_seconds = 1.0 / sampling_freq_hz

    # Use absolute flow for magnitude comparison, baseline should also be magnitude based (e.g. median of peak insp flows)
    # For simplicity here, assume baseline_flow_series represents a comparable amplitude measure.
    # If baseline is of actual flow, it might hover near zero.
    # A better baseline for this purpose would be e.g. median of recent breath peak inspiratory flows.
    # For now, let's assume baseline_flow_series is appropriately positive and represents typical breath amplitude.
    # We will take abs(flow_series) for comparison if baseline is amplitude-based.
    # If baseline is also raw flow, then direct comparison is okay.
    # Let's assume baseline_flow_series is a positive amplitude measure.

    # Calculate flow as a ratio of baseline. Baseline should not be zero.
    # Add a small epsilon to baseline to prevent division by zero if baseline is actual flow.
    baseline_adjusted = baseline_flow_series.copy()
    if (baseline_adjusted <= 0).any(): # If baseline can be 0 or negative (e.g. it's raw flow median)
        # This indicates the baseline might not be an "amplitude" baseline.
        # For now, we'll use a small positive floor for baseline in ratio calculation
        # print("Warning: Baseline contains zero or negative values. Using a small floor for ratio calculation.")
        baseline_adjusted[baseline_adjusted <= 1e-3] = 1e-3
        # And compare absolute flow to this baseline.
        flow_to_compare = np.abs(flow_series)
    else:
        # If baseline is always positive (an amplitude measure), compare abs flow to it.
        flow_to_compare = np.abs(flow_series)


    flow_ratio = flow_to_compare / baseline_adjusted
    flow_ratio = flow_ratio.clip(upper=2.0) # Cap ratio to avoid extreme values if baseline is tiny

    # 1. Identify periods satisfying apnea criteria
    is_apnea_potential = flow_ratio < apnea_threshold_ratio

    # 2. Identify periods satisfying hypopnea criteria
    # Flow reduction of X% means flow is (1-X) of baseline.
    # E.g. 30% reduction -> flow is 70% of baseline. (upper limit for hypopnea flow)
    # E.g. 90% reduction -> flow is 10% of baseline. (lower limit for hypopnea flow)
    # So, hypopnea_lower_threshold_ratio < flow_ratio < hypopnea_upper_threshold_ratio
    is_hypopnea_potential = (flow_ratio >= apnea_threshold_ratio) & \
                            (flow_ratio < hypopnea_upper_threshold_ratio)
                            # This definition means flow is between 10% and 70% of baseline.

    events = []

    for event_type_name, is_event_series in [('apnea_candidate', is_apnea_potential),
                                             ('hypopnea_candidate', is_hypopnea_potential)]:

        # Find blocks of consecutive True values
        if not is_event_series.any():
            continue

        change_points = is_event_series.ne(is_event_series.shift()).cumsum()
        for _, group in is_event_series.groupby(change_points):
            if group.iloc[0] and len(group) >= min_samples_for_event: # If it's an event block and long enough
                start_time = group.index[0]
                end_time = group.index[-1] + pd.Timedelta(seconds=dt_seconds) # End time is start of next sample
                duration_s = (end_time - start_time).total_seconds()

                event_flow_segment = flow_series.loc[start_time : group.index[-1]]
                event_baseline_segment = baseline_adjusted.loc[start_time : group.index[-1]]

                # Calculate average flow reduction: 1 - (avg_event_flow / avg_event_baseline)
                # Use absolute flow for average magnitude during event
                avg_event_flow_abs = np.abs(event_flow_segment).mean()
                avg_event_baseline = event_baseline_segment.mean()

                if avg_event_baseline > 1e-3: # Avoid division by zero
                    avg_flow_reduction_percent = (1 - (avg_event_flow_abs / avg_event_baseline)) * 100
                else:
                    avg_flow_reduction_percent = 100.0 if avg_event_flow_abs < 1e-3 else 0.0

                min_flow_val = np.abs(event_flow_segment).min()
                # Use baseline at the point of min flow, or average baseline if that's problematic
                # For simplicity, use average baseline over event
                min_flow_during_event_ratio = min_flow_val / avg_event_baseline if avg_event_baseline > 1e-3 else (0.0 if min_flow_val < 1e-3 else 1.0)


                excluded = False
                if high_leak_flags is not None:
                    # Check for any overlap with high leak
                    leak_during_event = high_leak_flags.loc[start_time : group.index[-1]]
                    if leak_during_event.any():
                        excluded = True

                events.append({
                    'event_start_time': start_time,
                    'event_end_time': end_time,
                    'event_duration_s': duration_s,
                    'event_type': event_type_name,
                    'avg_flow_reduction_percent': avg_flow_reduction_percent,
                    'min_flow_during_event_ratio': min_flow_during_event_ratio,
                    'excluded_due_to_leak': excluded
                })

    if not events:
        return pd.DataFrame()

    events_df = pd.DataFrame(events)
    if events_df.empty:
        return events_df

    # Sort events by start time
    events_df.sort_values(by='event_start_time', inplace=True)
    events_df.reset_index(drop=True, inplace=True)

    # Resolve overlapping events: prioritize apneas over hypopneas, then longer events.
    # This is a simple approach; more sophisticated merging might be needed.
    final_events = []
    last_event_end_time = pd.Timestamp.min.tz_localize('UTC') # Ensure timezone aware if index is

    # If index is not localized, use naive timestamp
    if events_df['event_start_time'].iloc[0].tzinfo is None:
         last_event_end_time = pd.Timestamp.min


    for i, current_event in events_df.iterrows():
        # If this event starts before the last one ended, it's an overlap or very close
        if current_event['event_start_time'] < last_event_end_time:
            # Potential overlap with the *last added* event in final_events
            if not final_events: # Should not happen if last_event_end_time is updated
                final_events.append(current_event.to_dict())
                last_event_end_time = current_event['event_end_time']
                continue

            prev_event = final_events[-1] # Get the last event *added to final_events*

            # Overlap resolution logic:
            # 1. If current is apnea and previous is hypopnea, and they overlap significantly:
            #    Consider replacing prev hypopnea or merging if apnea covers most of it.
            #    For now, a simpler rule: if types differ, apnea wins.
            #    If an apnea starts during a hypopnea, the hypopnea should end at apnea start.
            #    If current event is 'apnea' and previous is 'hypopnea'
            if current_event['event_type'] == 'apnea_candidate' and prev_event['event_type'] == 'hypopnea_candidate':
                # If current apnea starts before previous hypopnea ends, shorten previous hypopnea
                if current_event['event_start_time'] < prev_event['event_end_time']:
                    prev_event['event_end_time'] = current_event['event_start_time']
                    prev_event['event_duration_s'] = (prev_event['event_end_time'] - prev_event['event_start_time']).total_seconds()
                    # If hypopnea becomes too short, remove it
                    if prev_event['event_duration_s'] < min_event_duration_s:
                        final_events.pop()
                    # Add the apnea
                    final_events.append(current_event.to_dict())
                    last_event_end_time = current_event['event_end_time']
                else: # No actual overlap in time, just processing order
                    final_events.append(current_event.to_dict())
                    last_event_end_time = current_event['event_end_time']

            elif current_event['event_type'] == 'hypopnea_candidate' and prev_event['event_type'] == 'apnea_candidate':
                # If current hypopnea starts before previous apnea ends, this hypopnea is likely invalid or part of apnea recovery.
                # Ignore this hypopnea if it's fully contained or starts within the apnea.
                if current_event['event_start_time'] < prev_event['event_end_time']:
                    # Hypopnea is consumed by previous apnea, so skip adding it
                    pass
                else: # No actual overlap
                    final_events.append(current_event.to_dict())
                    last_event_end_time = current_event['event_end_time']

            elif current_event['event_type'] == prev_event['event_type']:
                 # Same type overlap: merge or choose longer one.
                 # For now, if they overlap, extend previous event if current ends later.
                 if current_event['event_start_time'] < prev_event['event_end_time']: # They overlap
                    if current_event['event_end_time'] > prev_event['event_end_time']:
                        prev_event['event_end_time'] = current_event['event_end_time']
                        prev_event['event_duration_s'] = (prev_event['event_end_time'] - prev_event['event_start_time']).total_seconds()
                        # Recalculate metrics for merged event would be good, but complex here.
                    # else current is contained within prev, so ignore current.
                    last_event_end_time = prev_event['event_end_time'] # Update with potentially new end time
                 else: # No actual overlap
                    final_events.append(current_event.to_dict())
                    last_event_end_time = current_event['event_end_time']
            else: # Should not happen if types are only apnea/hypopnea
                final_events.append(current_event.to_dict())
                last_event_end_time = current_event['event_end_time']

        else: # No overlap with the previous event added to final_events
            final_events.append(current_event.to_dict())
            last_event_end_time = current_event['event_end_time']

    if not final_events:
        return pd.DataFrame()

    final_events_df = pd.DataFrame(final_events)
    # Re-filter for duration in case merges made some too short (though logic above tries to handle it)
    final_events_df = final_events_df[final_events_df['event_duration_s'] >= min_event_duration_s]

    return final_events_df.sort_values(by='event_start_time').reset_index(drop=True)


if __name__ == '__main__':
    # Setup for testing
    sampling_freq_hz = 25  # Hz
    duration_sec = 300    # seconds. Increased duration for more event types.
    num_samples = duration_sec * sampling_freq_hz
    time_stamps = np.arange(num_samples) / sampling_freq_hz
    time_index = pd.to_datetime(time_stamps, unit='s')

    # Create dummy flow rate data
    base_amplitude = 0.5 # L/s peak insp flow for normal breaths
    flow_rate = np.zeros(num_samples)
    # Simulate some breaths (simplistic positive flow envelope)
    for i in range(num_samples):
        # Simple periodic breathing (approx 15 bpm)
        if (i // int(sampling_freq_hz * 2)) % 2 == 0 : # Inspiratory phase (2s)
            flow_rate[i] = base_amplitude * np.sin(np.pi * (i % (sampling_freq_hz * 2)) / (sampling_freq_hz*2) )
        else: # Expiratory phase (2s) - keep it simple, make it small positive or zero for this test
            flow_rate[i] = 0.05 * np.sin(np.pi * (i % (sampling_freq_hz * 2)) / (sampling_freq_hz*2) )

    flow_rate += 0.02 * np.random.randn(num_samples) # Add some noise
    flow_rate = np.abs(flow_rate) # Take absolute value as if it's a flow envelope

    # Simulate a baseline (e.g., rolling median of breath amplitudes - simplified here)
    # For testing, make it relatively stable then drop it during events.
    baseline_flow = pd.Series(np.ones(num_samples) * base_amplitude, index=time_index)


    # Introduce an Apnea: 15 seconds of very low flow
    apnea_start_sec = 30
    apnea_end_sec = 45
    apnea_start_idx = int(apnea_start_sec * sampling_freq_hz)
    apnea_end_idx = int(apnea_end_sec * sampling_freq_hz)
    flow_rate[apnea_start_idx:apnea_end_idx] = base_amplitude * 0.05 # 5% of base_amplitude
    # baseline_flow[apnea_start_idx:apnea_end_idx] = base_amplitude # Baseline remains high

    # Introduce a Hypopnea: 20 seconds of reduced flow
    hyp_start_sec = 80
    hyp_end_sec = 100
    hyp_start_idx = int(hyp_start_sec * sampling_freq_hz)
    hyp_end_idx = int(hyp_end_sec * sampling_freq_hz)
    flow_rate[hyp_start_idx:hyp_end_idx] = base_amplitude * 0.4 # 40% of base_amplitude
    # baseline_flow[hyp_start_idx:hyp_end_idx] = base_amplitude # Baseline remains high

    # Introduce another Apnea, overlapping start of a later Hypopnea
    apnea2_start_sec = 120
    apnea2_end_sec = 135
    apnea2_start_idx = int(apnea2_start_sec * sampling_freq_hz)
    apnea2_end_idx = int(apnea2_end_sec * sampling_freq_hz)
    flow_rate[apnea2_start_idx:apnea2_end_idx] = base_amplitude * 0.03

    hyp2_start_sec = 130 # This hypopnea starts during apnea2
    hyp2_end_sec = 145
    hyp2_start_idx = int(hyp2_start_sec * sampling_freq_hz)
    hyp2_end_idx = int(hyp2_end_sec * sampling_freq_hz)
    # This hypopnea is partially masked by the preceding apnea.
    # We apply the reduction on top of whatever flow_rate has there.
    # To make it distinct, ensure the flow for hypopnea is set correctly.
    flow_rate[max(apnea2_end_idx, hyp2_start_idx) : hyp2_end_idx] = base_amplitude * 0.5


    flow_series = pd.Series(flow_rate, index=time_index, name='flow_rate_abs_envelope')

    # Simulate high leak periods
    leak_flags = pd.Series(False, index=time_index)
    leak_start_sec = 40 # This leak overlaps with the first apnea
    leak_end_sec = 50
    leak_flags.iloc[int(leak_start_sec*sampling_freq_hz) : int(leak_end_sec*sampling_freq_hz)] = True


    print("--- Testing Apnea/Hypopnea Detection ---")

    # Using default thresholds:
    # Apnea: <10% of baseline
    # Hypopnea: 10% to 70% of baseline (reduction of 30% to 90%)
    detected_events_df = detect_apneas_hypopneas(flow_series,
                                                 baseline_flow_series=baseline_flow,
                                                 sampling_freq_hz=sampling_freq_hz,
                                                 min_event_duration_s=10.0,
                                                 high_leak_flags=leak_flags)

    print(f"Detected {len(detected_events_df)} events:")
    if not detected_events_df.empty:
        print(detected_events_df[['event_start_time', 'event_end_time', 'event_type', 'event_duration_s', 'avg_flow_reduction_percent', 'excluded_due_to_leak']])

        # Expected events:
        # 1. Apnea @ 30s-45s (15s), flow 5%. Expected reduction ~95%. Overlaps leak.
        # 2. Hypopnea @ 80s-100s (20s), flow 40%. Expected reduction ~60%.
        # 3. Apnea @ 120s-135s (15s), flow 3%. Expected reduction ~97%.
        # 4. Hypopnea @ 135s-145s (10s), flow 50%. Expected reduction ~50%. (Starts after apnea2 ends)

        num_apneas = len(detected_events_df[detected_events_df['event_type'] == 'apnea_candidate'])
        num_hypopneas = len(detected_events_df[detected_events_df['event_type'] == 'hypopnea_candidate'])
        print(f"Number of apneas: {num_apneas}, Number of hypopneas: {num_hypopneas}")

        assert num_apneas >= 2, f"Expected at least 2 apneas, got {num_apneas}"
        assert num_hypopneas >= 2, f"Expected at least 2 hypopneas, got {num_hypopneas}"

        first_apnea = detected_events_df[detected_events_df['event_type'] == 'apnea_candidate'].iloc[0]
        assert first_apnea['excluded_due_to_leak'] == True, "First apnea should be flagged for leak."
        assert abs(first_apnea['event_duration_s'] - 15) < 1, "First apnea duration check."
        assert abs(first_apnea['avg_flow_reduction_percent'] - 95) < 5, "First apnea reduction check."

        first_hypopnea = detected_events_df[
            (detected_events_df['event_type'] == 'hypopnea_candidate') &
            (detected_events_df['excluded_due_to_leak'] == False)
        ].iloc[0]
        assert abs(first_hypopnea['event_duration_s'] - 20) < 1, "First hypopnea duration check."
        assert abs(first_hypopnea['avg_flow_reduction_percent'] - 60) < 5, "First hypopnea reduction check."

        # Check the apnea-hypopnea overlap resolution
        # Apnea2 from 120-135s. Hyp2 intended from 130-145s.
        # Expect Apnea2 (120-135s) and Hyp2 (135-145s)
        apnea2_event = detected_events_df[
            (detected_events_df['event_type'] == 'apnea_candidate') &
            (detected_events_df['event_start_time'].dt.total_seconds() > 100)
        ].iloc[0]
        hypopnea2_event = detected_events_df[
            (detected_events_df['event_type'] == 'hypopnea_candidate') &
            (detected_events_df['event_start_time'].dt.total_seconds() > apnea2_event['event_end_time'].timestamp() - time_index[0].timestamp() -1 ) # starts near end of apnea2
        ].iloc[0]

        assert abs(apnea2_event['event_start_time'].timestamp() - time_index[0].timestamp() - 120) < 1
        assert abs(apnea2_event['event_end_time'].timestamp() - time_index[0].timestamp() - 135) < 1

        assert abs(hypopnea2_event['event_start_time'].timestamp() - apnea2_event['event_end_time'].timestamp()) < 1 , \
            f"Hypopnea2 should start where Apnea2 ends. Apnea2 ends: {apnea2_event['event_end_time']}, Hypopnea2 starts: {hypopnea2_event['event_start_time']}"
        assert abs(hypopnea2_event['event_duration_s'] - 10) < 1, \
            f"Hypopnea2 duration check. Expected 10s, got {hypopnea2_event['event_duration_s']}"


        print("Basic event count and property assertions passed.")

    else:
        print("No events detected. Check parameters or input signal generation.")


    print("\n--- Test with no significant events ---")
    normal_flow = pd.Series(np.ones(num_samples) * base_amplitude * 0.9, index=time_index) # Always 90%
    normal_baseline = pd.Series(np.ones(num_samples) * base_amplitude, index=time_index)
    no_events_df = detect_apneas_hypopneas(normal_flow, normal_baseline, sampling_freq_hz)
    assert no_events_df.empty, f"Should detect no events in normal flow, but got {len(no_events_df)}"
    print("Correctly detected no events in normal flow.")

    print("\n--- Test with all apnea (prolonged low flow) ---")
    all_apnea_flow = pd.Series(np.ones(num_samples) * base_amplitude * 0.05, index=time_index)
    all_apnea_df = detect_apneas_hypopneas(all_apnea_flow, normal_baseline, sampling_freq_hz, min_event_duration_s=duration_sec -1) # ensure it can make one long event
    assert len(all_apnea_df) == 1 and all_apnea_df['event_type'].iloc[0] == 'apnea_candidate', "Should detect one long apnea."
    assert abs(all_apnea_df['event_duration_s'].iloc[0] - duration_sec) < 1 , "Long apnea duration check"
    print("Correctly detected one long apnea.")

    print("\nEvent detection module tests complete.")
from typing import Optional, Tuple # Add this import
