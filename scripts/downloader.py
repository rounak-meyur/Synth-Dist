#!/usr/bin/env python3
"""
Download the per-county residential load files ("<PREFIX><county>-<date>.csv", e.g.
VA013-20140523.csv) from the net.science (sciduct) file service.

The counties are the regions of a group configuration file (default: all of Virginia) or
the regions given with -r. Files that were already downloaded are skipped. The downloaded
files are turned into the pipeline's home load files by scripts/get_load_csv.py.

Usage (from the repository root):
    uv run python scripts/downloader.py -g configs/regions/virginia.yaml --prefix VA -o data/raw/load_VA
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

import requests
from omegaconf import OmegaConf

from utils.logging_utils import LogManager

BASE_URL = "https://sciduct.bii.virginia.edu/fs/file/47bc88de-752b-4514-8272-60bdbed29175/"
DATE = "20140523"


def load_file_name(prefix: str, region: str, date: str = DATE) -> str:
    """Name of the load file of a county, e.g. VA013-20140523.csv."""
    return f"{prefix}{region}-{date}.csv"


def download_file(file_url: str, file_path: Path, timeout: float = 60) -> bool:
    """Download a file; it is written under a temporary name so a failed download leaves no partial file."""
    logger.info(f"Downloading {file_path.name}...")
    try:
        with requests.get(file_url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                logger.error(f"Failed to download {file_path.name}: HTTP {response.status_code}")
                return False
            tmp = file_path.with_suffix(".part")
            with open(tmp, "wb") as file:
                for chunk in response.iter_content(1 << 20):
                    file.write(chunk)
            os.replace(tmp, file_path)
    except requests.RequestException as e:
        logger.error(f"Failed to download {file_path.name}: {e}")
        return False
    logger.info(f"{file_path} saved.")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-g", "--group", default="configs/regions/virginia.yaml",
                        help="group configuration file with the counties to download")
    parser.add_argument("-r", "--region", nargs="+", help="county FIPS codes to download instead of the group's")
    parser.add_argument("--prefix", default="VA", help="state abbreviation in the file names (default: VA)")
    parser.add_argument("--date", default=DATE, help=f"date in the file names (default: {DATE})")
    parser.add_argument("--base-url", default=BASE_URL, help="URL of the folder with the load files")
    parser.add_argument("-o", "--out-dir", default="data/raw/load_VA", help="directory for the downloaded files")
    parser.add_argument("--overwrite", action="store_true", help="download files that already exist again")
    args = parser.parse_args(argv)

    regions = args.region or [str(r) for r in OmegaConf.load(args.group)["group"]["regions"]]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for region in regions:
        name = load_file_name(args.prefix, region, args.date)
        path = out_dir / name
        if path.exists() and not args.overwrite:
            logger.info(f"{path} already exists; skipping it")
            continue
        url = f"{args.base_url.rstrip('/')}/{name}?&http_accept=application/octet-stream"
        if not download_file(url, path):
            failed.append(name)

    logger.info(f"Downloaded load files of {len(regions) - len(failed)} of {len(regions)} counties to {out_dir}")
    if failed:
        logger.error(f"Failed downloads: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    LogManager.initialize(
        log_file_path=str(REPO_ROOT / "logs/csv_downloader.log"),
        log_config_path=str(REPO_ROOT / "configs/logging_config.ini")
    )
logger = LogManager.get_logger("downloader")

if __name__ == "__main__":
    main()
