
import pandas as pd
import os


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "events.csv"
REPORT_FILE = BASE_DIR / "reports" / "validation_report.txt"

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)


try:
    df = pd.read_csv(INPUT_FILE)

except Exception as e:
    print("Dataset could not be loaded.")
    print(e)
    exit()


report = []

report.append("=" * 60)
report.append("EVENTWISE AI DATA VALIDATION REPORT")
report.append("=" * 60)
report.append("\n")



report.append(f"Rows : {df.shape[0]}")
report.append(f"Columns : {df.shape[1]}")
report.append("\n")


report.append("Columns Present:")

for col in df.columns:
    report.append(f" - {col}")

report.append("\n")



duplicates = df.duplicated().sum()

report.append("Duplicate Rows")
report.append("----------------------")
report.append(f"Duplicate count : {duplicates}")
report.append("\n")



if "event_id" in df.columns:

    duplicate_ids = df["event_id"].duplicated().sum()

    report.append("Duplicate Event IDs")
    report.append("----------------------")
    report.append(f"Duplicate IDs : {duplicate_ids}")
    report.append("\n")


report.append("Missing Values")
report.append("----------------------")

missing = df.isnull().sum()

for col in missing.index:
    report.append(f"{col} : {missing[col]}")

report.append("\n")



if "latitude" in df.columns:

    invalid_lat = df[
        (df["latitude"] < -90) |
        (df["latitude"] > 90)
    ]

    report.append("Latitude Validation")
    report.append("----------------------")
    report.append(f"Invalid Latitude Count : {len(invalid_lat)}")
    report.append("\n")



if "longitude" in df.columns:

    invalid_lon = df[
        (df["longitude"] < -180) |
        (df["longitude"] > 180)
    ]

    report.append("Longitude Validation")
    report.append("----------------------")
    report.append(f"Invalid Longitude Count : {len(invalid_lon)}")
    report.append("\n")


if "timestamp" in df.columns:

    parsed = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    invalid_time = parsed.isnull().sum()

    report.append("Timestamp Validation")
    report.append("----------------------")
    report.append(f"Invalid timestamps : {invalid_time}")
    report.append("\n")


if "cause" in df.columns:

    empty_causes = (
        df["cause"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    null_causes = df["cause"].isnull().sum()

    report.append("Cause Validation")
    report.append("----------------------")
    report.append(f"NULL causes : {null_causes}")
    report.append(f"Empty causes : {empty_causes}")
    report.append("\n")


if "severity" in df.columns:

    invalid = df[
        ~df["severity"].isin(
            ["Low", "Medium", "High"]
        )
    ]

    report.append("Severity Validation")
    report.append("----------------------")
    report.append(f"Invalid severity entries : {len(invalid)}")
    report.append("\n")


if "congestion_level" in df.columns:

    invalid = df[
        ~df["congestion_level"].isin(
            ["Low", "Medium", "High"]
        )
    ]

    report.append("Congestion Validation")
    report.append("----------------------")
    report.append(f"Invalid congestion labels : {len(invalid)}")
    report.append("\n")



if "latitude" in df.columns and "longitude" in df.columns:

    coord_missing = df[
        df["latitude"].isnull() |
        df["longitude"].isnull()
    ]

    report.append("Coordinate Completeness")
    report.append("----------------------")
    report.append(f"Rows missing coordinates : {len(coord_missing)}")
    report.append("\n")



report.append("Numerical Summary")
report.append("----------------------")

try:

    summary = df.describe(include="all")

    report.append(summary.to_string())

except:

    report.append("Summary unavailable.")

report.append("\n")



with open(REPORT_FILE, "w") as f:

    for line in report:
        f.write(str(line))
        f.write("\n")

print("=" * 50)
print("Validation Complete")
print(f"Report saved to {REPORT_FILE}")
print("=" * 50)