import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data_loader import load_edf_data

REQUIRED_SIGNALS = [
    'Flow.40ms',
    'MaskPress.2s',
    'Press.2s',
    'EprPress.2s',
    'Leak.2s',
    'RespRate.2s',
    'MinVent.2s',
    'FlowLim.2s',
]

def load_and_merge_signals(brp_path, pld_path):
    """
    Loads and merges required signals from BRP and PLD EDF files.
    Returns a DataFrame indexed by timestamp with only the required columns.
    """
    if not os.path.exists(brp_path) or not os.path.exists(pld_path):
        raise FileNotFoundError(f"BRP or PLD file not found: {brp_path}, {pld_path}")

    df_brp = load_edf_data(brp_path)
    df_pld = load_edf_data(pld_path)
    if df_brp is None or df_pld is None:
        raise ValueError("Failed to load BRP or PLD data.")

    # Clean PLD: drop checksum columns
    cols_to_drop = [c for c in df_pld.columns if 'Crc' in c]
    df_pld = df_pld.drop(columns=cols_to_drop, errors='ignore')

    # Select only required columns
    brp_cols = [col for col in REQUIRED_SIGNALS if col in df_brp.columns]
    pld_cols = [col for col in REQUIRED_SIGNALS if col in df_pld.columns]
    df_brp = df_brp[brp_cols]
    df_pld = df_pld[pld_cols]

    # Merge on index (timestamp)
    merged = pd.merge(df_brp, df_pld, left_index=True, right_index=True, how='outer')
    merged = merged.ffill().dropna()
    merged = merged.loc[:,~merged.columns.duplicated()]  # Remove duplicate columns if any
    return merged

def save_merged_signals(df, out_path):
    """
    Save merged DataFrame as Parquet or HDF5 based on file extension.
    """
    if out_path.endswith('.parquet'):
        df.to_parquet(out_path)
    elif out_path.endswith('.h5') or out_path.endswith('.hdf5'):
        df.to_hdf(out_path, key='signals', mode='w')
    else:
        raise ValueError("Output file must end with .parquet, .h5, or .hdf5")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge and save BRP/PLD EDF signals for a session.")
    parser.add_argument('--brp', required=True, help="Path to BRP EDF file")
    parser.add_argument('--pld', required=True, help="Path to PLD EDF file")
    parser.add_argument('--out', required=True, help="Output file (.parquet or .h5)")
    args = parser.parse_args()

    merged = load_and_merge_signals(args.brp, args.pld)
    save_merged_signals(merged, args.out)
    print(f"Merged signals saved to {args.out}") 