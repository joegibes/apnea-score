import pandas as pd
from src.breath_detection import detect_breaths_from_flow

# Load trimmed 10-minute session data
merged_path = 'data/2025/20250617_023551_merged_10min.parquet'
df = pd.read_parquet(merged_path)

# Run breath detection on Flow.40ms
sampling_freq_hz = 25  # Known from your data
breaths = detect_breaths_from_flow(df['Flow.40ms'], sampling_freq_hz)

print('Detected breaths:', len(breaths))
print(breaths.info())
print(breaths.head()) 