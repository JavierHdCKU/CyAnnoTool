#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

# Configuration - now using the FlowSOM example data
OUT_DIR = Path("/home/javier/scina_module/test_data")
FCS_DIR = OUT_DIR / "fcs"

HANDGATED_CSV = OUT_DIR / "Handgated.csv"
LIVE_TRAIN_CSV = OUT_DIR / "LivecellsTraining.csv"
LIVE_UNLAB_CSV = OUT_DIR / "Livecells.csv"


def main():
    print("Using FlowSOM example data...")
    
    # Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all FCS files in the fcs directory
    all_fcs = sorted(FCS_DIR.rglob("*.fcs"))

    if len(all_fcs) == 0:
        raise SystemExit("ERROR: No FCS files found after extraction!")

    print(f"Found {len(all_fcs)} FCS files")

    # LivecellsTraining.csv
    live_train_rows = []
    for f in all_fcs:
        sample_id = f.stem
        live_train_rows.append((str(f.resolve()), sample_id))

    df_live_train = pd.DataFrame(live_train_rows, columns=["file", "sample_id"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_live_train.to_csv(LIVE_TRAIN_CSV, index=False)
    print(f"Generated: {LIVE_TRAIN_CSV}")

    # Livecells.csv
    df_live_unlab = df_live_train.copy()
    df_live_unlab.to_csv(LIVE_UNLAB_CSV, index=False)
    print(f"Generated: {LIVE_UNLAB_CSV}")

    # Handgated.csv (dummy)
    first_fcs = all_fcs[0]
    handgated_rows = [
        (str(first_fcs.resolve()), "Unknown", first_fcs.stem)
    ]

    df_hand = pd.DataFrame(
        handgated_rows,
        columns=["file", "cell_type", "sample_id"]
    )
    df_hand.to_csv(HANDGATED_CSV, index=False)

    print(f"Generated (SIMULATED!) hand-gated file: {HANDGATED_CSV}")
    print("WARNING: This is a dummy hand-gated entry. Replace with real gated data for real analysis.")
    print("\nDone!")


if __name__ == "__main__":
    main()
