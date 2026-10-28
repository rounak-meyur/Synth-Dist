#!/usr/bin/env python3
"""
Build the pipeline's home load files (<county>-home-load.csv) from the load files downloaded
by scripts/downloader.py (<PREFIX><county>-<date>.csv) and the synthetic household files
with the home locations (household_<census tract>.csv).

Each output row is a home: hid, longitude, latitude and its hourly load hour1..hour24 in W
(total_kwh_1..total_kwh_24 of the load file times 1000). The location of a home is looked
up in the household file of its census tract (the first 11 digits of the hid), matching the
hid with SYNTHETIC_HID without dashes. Homes without a location are left out.

Usage (from the repository root):
    uv run python scripts/get_load_csv.py -i data/raw/load_VA --household-dir data/raw/households_VA \
        -o data/load_51 --prefix VA
"""
import argparse
import re
from pathlib import Path

import pandas as pd

DATE = "20140523"
HOURS = range(1, 25)


def read_household_locations(household_directory: Path, tracts) -> pd.DataFrame:
    """Locations (hid, longitude, latitude) of the homes of the given census tracts."""
    frames = []
    for tract in sorted(tracts):
        household_csv_path = household_directory / f"household_{tract}.csv"
        if not household_csv_path.exists():
            print(f"  Warning: File not found: {household_csv_path}")
            continue
        household_df = pd.read_csv(
            household_csv_path, dtype={"SYNTHETIC_HID": str}, usecols=["SYNTHETIC_HID", "longitude", "latitude"]
        )
        # Remove dashes from SYNTHETIC_HID to match the hid of the load file
        household_df["hid"] = household_df["SYNTHETIC_HID"].str.replace("-", "")
        frames.append(household_df[["hid", "longitude", "latitude"]])
    if not frames:
        return pd.DataFrame(columns=["hid", "longitude", "latitude"])
    locations = pd.concat(frames, ignore_index=True)
    duplicates = locations["hid"].duplicated()
    if duplicates.any():
        print(f"  Warning: {duplicates.sum()} homes have several locations; using the first one")
    return locations[~duplicates].set_index("hid")


def process_single_csv(original_csv_path, household_directory, output_csv_path) -> pd.DataFrame:
    """
    Build the home load file of one county.

    Args:
        original_csv_path: Load file downloaded by scripts/downloader.py
        household_directory: Directory containing household_<census tract>.csv files
        output_csv_path: Path for the output CSV file

    Returns:
        pd.DataFrame: The saved home load data
    """
    print(f"Reading {original_csv_path}...")
    original_df = pd.read_csv(original_csv_path, dtype={"hid": str})
    print(f"Processing {len(original_df)} homes...")

    # census tract: first 11 characters of hid
    locations = read_household_locations(Path(household_directory), set(original_df["hid"].str[:11]))
    result_df = pd.DataFrame({
        "hid": original_df["hid"],
        "longitude": original_df["hid"].map(locations["longitude"]),
        "latitude": original_df["hid"].map(locations["latitude"]),
    })
    for i in HOURS:
        kwh_col = f"total_kwh_{i}"
        result_df[f"hour{i}"] = original_df[kwh_col] * 1000 if kwh_col in original_df else 0.0

    missing = result_df["longitude"].isna() | result_df["latitude"].isna()
    if missing.any():
        print(f"  Warning: {missing.sum()} homes without a location are left out")
        result_df = result_df[~missing]

    print(f"  Saving results to {output_csv_path}...")
    result_df.to_csv(output_csv_path, index=False)
    print(f"  Processing complete! {len(result_df)} homes saved to {output_csv_path}")
    return result_df


def process_multiple_csv_files(
        input_csv_directory, household_directory, output_directory,
        prefix="VA", date=DATE, regions=None, overwrite=False,
    ):
    """
    Process the load files "<prefix><county>-<date>.csv" in a directory and save the outputs
    as "<county>-home-load.csv".

    Args:
        input_csv_directory: Directory containing the downloaded load files
        household_directory: Directory containing household CSV files
        output_directory: Directory to save output CSV files
        prefix: State abbreviation in the input file names
        date: Date in the input file names
        regions: Counties to process (default: every load file in the input directory)
        overwrite: Process counties whose output file already exists again
    """
    input_csv_directory, output_directory = Path(input_csv_directory), Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"{re.escape(prefix)}(\d+)-{re.escape(date)}\.csv")
    input_files = {}
    for path in sorted(input_csv_directory.glob(f"{prefix}*-{date}.csv")):
        match = pattern.fullmatch(path.name)
        if match:
            input_files[match.group(1)] = path
    if regions:
        missing = [r for r in regions if r not in input_files]
        if missing:
            print(f"Warning: no load files for counties {missing} in {input_csv_directory}")
        input_files = {r: input_files[r] for r in regions if r in input_files}
    if not input_files:
        print(f"No files found matching {input_csv_directory / f'{prefix}*-{date}.csv'}")
        return

    print(f"Found {len(input_files)} files to process")
    processed_files, failed_files = [], []
    for region, input_file in input_files.items():
        output_path = output_directory / f"{region}-home-load.csv"
        if output_path.exists() and not overwrite:
            print(f"{output_path} already exists; skipping it")
            continue
        print(f"\n{'='*60}\nProcessing: {input_file.name} -> {output_path.name}\n{'='*60}")
        try:
            result_df = process_single_csv(input_file, household_directory, output_path)
            processed_files.append((input_file.name, output_path.name, len(result_df)))
        except Exception as e:
            print(f"Error processing {input_file.name}: {e}")
            failed_files.append((input_file.name, str(e)))

    print(f"\n{'='*60}\nPROCESSING SUMMARY\n{'='*60}")
    print(f"Successfully processed: {len(processed_files)}")
    for input_name, output_name, rows in processed_files:
        print(f"  {input_name} -> {output_name} ({rows} homes)")
    print(f"Failed: {len(failed_files)}")
    for input_name, error in failed_files:
        print(f"  {input_name}: {error}")
    print(f"\nAll output files saved to: {output_directory}")
    return failed_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input-dir", default="data/raw/load_VA",
                        help="directory with the downloaded load files")
    parser.add_argument("--household-dir", required=True,
                        help="directory with the household_<census tract>.csv files")
    parser.add_argument("-o", "--output-dir", default="data/load_51",
                        help="directory for the home load files (the pipeline's inputs.home_csv_dir)")
    parser.add_argument("--prefix", default="VA", help="state abbreviation in the input file names (default: VA)")
    parser.add_argument("--date", default=DATE, help=f"date in the input file names (default: {DATE})")
    parser.add_argument("-r", "--region", nargs="+", help="counties to process (default: all downloaded files)")
    parser.add_argument("--overwrite", action="store_true", help="rebuild home load files that already exist")
    args = parser.parse_args()

    failed = process_multiple_csv_files(
        args.input_dir, args.household_dir, args.output_dir,
        prefix=args.prefix, date=args.date, regions=args.region, overwrite=args.overwrite,
    )
    raise SystemExit(1 if failed else 0)
