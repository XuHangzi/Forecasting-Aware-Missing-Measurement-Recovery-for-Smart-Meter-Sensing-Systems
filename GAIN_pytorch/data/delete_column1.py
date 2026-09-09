from pathlib import Path

import pandas as pd


USER_COUNT = 4
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "electricity_data_2013_processed.csv"


def find_source_file() -> Path:
    active_file = SCRIPT_DIR / "electricity_data_2013.csv"
    if active_file.exists():
        return active_file

    candidates = [
        path
        for path in SCRIPT_DIR.glob("electricity_data_2013*.csv")
        if "processed" not in path.stem and "complete" not in path.stem
    ]
    if not candidates:
        raise FileNotFoundError("No electricity_data_2013 source CSV file was found.")
    return candidates[0]


try:
    input_file = find_source_file()
    df = pd.read_csv(input_file)
    df_processed = df.iloc[:, 1 : 1 + USER_COUNT]
    df_processed.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Generated processed dataset successfully.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Kept user count: {len(df_processed.columns)}")
    print(f"Kept columns: {list(df_processed.columns)}")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Processing failed: {str(e)}")
