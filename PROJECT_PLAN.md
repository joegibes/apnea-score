# Project Goal: Automated Apnea Classification

The primary objective is to replicate the automated, breath-by-breath classification of sleep-disordered breathing (obstructive vs. central) as described in the Parekh et al. (2021) paper, "Endotyping Sleep Apnea One Breath at a Time."

This will be accomplished by creating a Python-based system that analyzes raw CPAP data signals (flow, pressure, etc.) to generate a "probability of obstruction" (Pobs) score for each breath.

## Key Challenges & Strategy

The Parekh et al. paper uses five signals: airflow, thoracic effort, abdominal effort, SpO2, and snore. We only have reliable access to **airflow (flow rate) and pressure**. The critical "effort", "snore", and "SpO2" signals are missing.

**User Instruction (2025-06-25):** Ignore `_SA2` (SpO2/Pulse) files, as the oximeter sensor is not in use. Focus only on signals that are actually present.

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
    -   Analyze the `Press.40ms` signal.
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

