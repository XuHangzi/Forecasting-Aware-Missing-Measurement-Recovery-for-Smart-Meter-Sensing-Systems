from pathlib import Path

import pandas as pd


TIME_COLUMN_NAME = "hour"
USER_COUNT = 4
SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_FILE = SCRIPT_DIR / "electricity_data_2013_processed.csv"
OUTPUT_FILE = SCRIPT_DIR / "electricity_data_2013_4users_complete.csv"


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
    original_file = find_source_file()
    df_original = pd.read_csv(original_file)
    df_processed = pd.read_csv(PROCESSED_FILE)

    if TIME_COLUMN_NAME not in df_original.columns:
        raise KeyError(TIME_COLUMN_NAME)

    expected_data_columns = list(df_original.columns[1 : 1 + USER_COUNT])
    if list(df_processed.columns) != expected_data_columns:
        raise ValueError(
            f"Processed data columns {list(df_processed.columns)} do not match the first "
            f"{USER_COUNT} data columns in the source file {expected_data_columns}."
        )

    df_complete = df_processed.copy()
    df_complete[TIME_COLUMN_NAME] = df_original[TIME_COLUMN_NAME]
    df_complete = df_complete[[TIME_COLUMN_NAME] + expected_data_columns]
    df_complete.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated time-restored dataset: {OUTPUT_FILE}")
    print(f"Output column count: {len(df_complete.columns)}")
    print(f"Output columns: {list(df_complete.columns)}")

except FileNotFoundError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"Error: {str(e)}")
except KeyError:
    print(f"Error: time column '{TIME_COLUMN_NAME}' was not found in the source file.")
except Exception as e:
    print(f"Processing failed: {str(e)}")
