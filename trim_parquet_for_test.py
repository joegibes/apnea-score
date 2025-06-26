import pandas as pd

# Load the full merged session
in_path = 'data/2025/20250617_023551_merged.parquet'
out_path = 'data/2025/20250617_023551_merged_10min.parquet'

df = pd.read_parquet(in_path)

# Trim to first 10 minutes
start_time = df.index[0]
end_time = start_time + pd.Timedelta(minutes=10)
df_10min = df.loc[start_time:end_time]

# Save trimmed DataFrame
print(f"Saving {len(df_10min)} rows to {out_path}")
df_10min.to_parquet(out_path) 