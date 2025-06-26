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

# --- Quick plot: Flow signal with detected breath start times ---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 5))
df['Flow.40ms'].plot(ax=ax, lw=1, color='blue', label='Flow.40ms')
if not breaths.empty:
    # Convert breath_start_time to numpy datetime64 for matplotlib compatibility
    breath_starts = breaths['breath_start_time']
    if hasattr(breath_starts, 'dt'):
        breath_starts = breath_starts.dt.tz_convert(None)  # Remove timezone if present
        breath_starts = breath_starts.dt.to_pydatetime()
    ax.vlines(breath_starts, ymin=df['Flow.40ms'].min(), ymax=df['Flow.40ms'].max(), color='red', alpha=0.5, lw=1, label='Breath Start')
ax.set_title('Flow Signal with Detected Breath Start Times')
ax.set_xlabel('Time')
ax.set_ylabel('Flow (arbitrary units)')
ax.legend()
plt.tight_layout()
plt.show()