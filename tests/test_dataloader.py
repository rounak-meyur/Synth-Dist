import csv

import pytest

from conftest import REPO_ROOT, write_homes_csv
from utils.dataloader import Home, Substation, load_homes, load_substations


@pytest.fixture
def valid_csv(tmp_path):
    path = tmp_path / "valid.csv"
    homes = [
        Home(id=1, cord=(-73.935242, 40.730610), profile=[i * 10.0 for i in range(1, 25)], peak=0, load=0),
        Home(id=2, cord=(-74.006015, 40.712775), profile=[i * 5.0 for i in range(1, 25)], peak=0, load=0),
    ]
    write_homes_csv(path, homes)
    return path


def test_load_homes_valid(valid_csv):
    homes = load_homes(valid_csv)
    assert len(homes) == 2
    assert all(isinstance(h, Home) for h in homes)
    assert homes[0].id == 1
    assert homes[0].cord == (-73.935242, 40.730610)
    assert len(homes[0].profile) == 24
    assert homes[0].peak == 240
    assert homes[0].load == 125


def test_load_homes_accepts_str_path(valid_csv):
    assert len(load_homes(str(valid_csv))) == 2


def test_load_homes_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_homes(tmp_path / "missing.csv")


def test_load_homes_missing_hour_columns(tmp_path):
    path = tmp_path / "invalid.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hid", "longitude", "latitude"])
        writer.writerow(["1", "-73.935242", "40.730610"])
    with pytest.raises(ValueError):
        load_homes(path)


def test_load_homes_non_numeric_value(tmp_path, homes):
    path = tmp_path / "bad_value.csv"
    write_homes_csv(path, homes)
    text = path.read_text().splitlines()
    text[1] = text[1].replace(str(homes[0].cord[0]), "not-a-number", 1)
    path.write_text("\n".join(text) + "\n")
    with pytest.raises(ValueError):
        load_homes(path)


def test_load_homes_repo_region_999():
    """The default region shipped in data/load/ must load cleanly."""
    homes = load_homes(REPO_ROOT / "data/load/999-home-load.csv")
    assert len(homes) > 0
    assert len({h.id for h in homes}) == len(homes), "home ids must be unique"
    for h in homes:
        assert -180 <= h.cord[0] <= 180 and -90 <= h.cord[1] <= 90
        assert h.peak >= h.load >= 0


# load_substations

def _write_substations(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "X", "Y", "NAME"])
        writer.writerows(rows)


def test_load_substations_filters_to_convex_hull(tmp_path, homes):
    xs = [h.cord[0] for h in homes]
    ys = [h.cord[1] for h in homes]
    inside = (sum(xs) / len(xs), sum(ys) / len(ys))
    path = tmp_path / "subs.csv"
    _write_substations(path, [
        ["0042", inside[0], inside[1], "inside"],
        ["0099", 0.0, 0.0, "outside"],
    ])

    subs = load_substations(path, homes)
    assert subs == [Substation(id="0042", cord=inside)]


def test_load_substations_requires_homes(tmp_path):
    with pytest.raises(ValueError):
        load_substations(tmp_path / "subs.csv", [])


def test_load_substations_file_not_found(tmp_path, homes):
    with pytest.raises(FileNotFoundError):
        load_substations(tmp_path / "missing.csv", homes)
