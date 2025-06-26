# Project Goal: Automated Apnea Classification

The primary objective is to replicate the automated, breath-by-breath classification of sleep-disordered breathing (obstructive vs. central) as described in the Parekh et al. (2021) paper, "Endotyping Sleep Apnea One Breath at a Time."

This will be accomplished by creating a Python-based system that analyzes raw CPAP data signals (flow, pressure, etc.) to generate a "probability of obstruction" (Pobs) score for each breath.

## Key Challenges & Strategy

The Parekh et al. paper uses five signals: airflow, thoracic effort, abdominal effort, SpO2, and snore. We only have reliable access to **airflow (flow rate) and pressure**. The critical "effort", "snore", and "SpO2" signals are missing.

**User Instruction (2025-06-25):** Ignore `_SA2` (SpO2/Pulse) files, as the oximeter sensor is not in use. Focus only on signals that are actually present.

**User Instruction (2025-06-26):** For all future data processing, only the BRP and PLD EDF files are relevant. For BRP, the primary signal of interest is 'Flow' (specifically 'Flow.40ms'). For PLD, 'mask pressure' (MaskPress.2s) and 'pressure' (Press.2s, aka set pressure) are the most useful. Other EDF files and signals can be disregarded unless otherwise specified.

## Data Import, Processing, and Storage: Best Practices (2025-06-26)

- **Import:**
  - Use the `load_edf_data` function in `src/data_loader.py` to load EDF files into pandas DataFrames.
  - Only load the relevant BRP and PLD files for each session (ignore other file types).
  - For BRP, extract the 'Flow.40ms' signal. For PLD, extract 'MaskPress.2s', 'Press.2s', 'EprPress.2s', 'Leak.2s', 'RespRate.2s', 'MinVent.2s', and 'FlowLim.2s'.
- **Processing:**
  - Clean PLD data by removing checksum columns (columns containing 'Crc').
  - Merge BRP and PLD DataFrames on their timestamp index using an outer join, then forward-fill and drop any remaining NaNs.
  - Ensure all signals are time-aligned and indexed by timestamp.
- **Storage:**
  - Store merged session DataFrames in a standardized format (e.g., Parquet or HDF5) for efficient future access and analysis.
  - Only retain the columns/signals of interest to minimize storage and processing overhead.

## Workflow for Efficient Feature Extraction (2025-06-26)

1. **User Input:** User selects/uploads the BRP and PLD EDF files for a single night/session.
2. **Efficient Data Processing:**
   - Load and merge the required signals (see above) using optimized, vectorized pandas/numpy operations.
   - Store the merged DataFrame in an efficient format (Parquet/HDF5) for downstream analysis.
3. **Fast Breath Separation:**
   - Run a highly efficient breath separation algorithm (optimized for speed and memory usage) on the Flow.40ms signal.
   - The algorithm should be faster than the current notebook implementation, suitable for 5-8 hours of 25Hz data.
4. **Feature Extraction:**
   - Extract features for each breath, using Flow.40ms as the main signal, but leveraging additional signals (e.g., FlowLim.2s, MinVent.2s) for advanced features like flow limitation and periodic/cyclic breathing.

## Efficiency Requirements
- All algorithms must be highly efficient and scalable to handle high-frequency (25Hz), long-duration (5-8 hour) data.
- Use vectorized operations and avoid unnecessary data copies or slow Python loops wherever possible.

Our strategy is to overcome this limitation by:
1.  **Implementing Directly Possible Features:** We will implement the features from the paper that can be derived from airflow and SpO2.
2.  **Developing Proxy Features:** We will create intelligent proxy features for the missing signals. For example, we can analyze changes in the **mask pressure signal** as a surrogate for respiratory effort. An increasing pressure against a limited flow is a strong indicator of an obstructive event.
3.  **Building a Weighted Model:** We will create a model that, like the paper, combines these features to produce a Pobs score. The weights will be adapted for our unique feature set.

## To-Do List

### Phase 1: Foundational Feature Engineering (In Progress)

-   [x] **Load & Parse EDF Data:** Develop a robust function to load and resample EDF data from `_BRP` (flow/pressure) and `_SA2` (SpO2) files.
-   [x] **Breath Segmentation:** Implement and refine the `detect_breath_starts` function to accurately segment the continuous flow signal into individual breaths.
-   [ ] **Data Synchronization:** Create a utility to load data from multiple EDF files (e.g., a `_BRP` and `_SA2` file from the same session) and align them on a common timestamp index. This is critical for linking flow events to oxygen desaturations.

### Phase 2: Feature Extraction (Next Steps)

This will be implemented in `src/feature_engineering.py`.

-   [ ] **Flow Limitation Feature:**
    -   Research and implement an algorithm to quantify inspiratory flow limitation (i.e., a "flattened" top of the breath). This is the most heavily weighted feature in the paper.
-   [ ] **Flow-Based Timing Features:**
    -   `Prolonged Inspiration`: Calculate inspiratory time for each breath and compare it to a rolling baseline.
    -   `Periodic Breathing`: Implement an algorithm to detect cyclical, crescendo-decrescendo patterns in breath amplitude over longer windows (e.g., 1-2 minutes).
-   [ ] **SpO2 Feature:**
    -   `Type of Desaturation`: Analyze the shape and timing of SpO2 dips relative to breath events to classify them as symmetric (likely central) or asymmetric (likely obstructive).
-   [ ] **Proxy Effort Feature (Pressure-Based):**
    -   Analyze the `Press.40ms` signal. #user note: No, this is the raw pressure. Focus on the flow, mask pressure, etc as better data... I'm not sure how this raw pressure translates to "effort". Also note that this Resmed Airsense 11 uses "Forced Oscillation Technique" to detect clear airway events during apnea - 4s after apnea start is detected, it oscillates 1cmh2O at 4hz (idk) and can somehow tell based on the reflections (pressure signal?). However, IMO this isn't very helpful for our purposes, since we're rating each BREATH, not merely apnea events.
    -   For each breath, calculate the change in pressure during the inspiratory phase.
    -   Hypothesis: A significant *increase* in pressure during a flow-limited breath is a strong proxy for "effort."

### Phase 3: Model Development & Validation

-   [ ] **Build the Pobs Model:**
    -   Create a function in `src/classification_model.py` that takes the features from a single breath as input.
    -   Implement a weighted sum, similar to the paper's method, to calculate an "obstructive score" and a "central score."
    -   Use a logistic function to convert these scores into a final `Pobs` value between 0 and 1.
-   [ ] **Manual Labeling (If Needed):** Use the interactive notebook tool to create a small, high-quality set of labeled breaths. This will be our "gold standard" for validation.
-   [ ] **Model Validation:**
    -   Compare the model's automated `Pobs` scores against the manually labeled data.
    -   Tune the feature weights to maximize accuracy.
-   [ ] **Visualization:** Create plots to visualize the `Pobs` score over time, overlaid on the flow and pressure signals, similar to Figure 3 in the paper.

