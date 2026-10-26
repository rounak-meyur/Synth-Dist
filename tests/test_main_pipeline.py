"""
End-to-end check of main.py: config + logging, home loading, road loading/caching,
and home-to-road mapping (the part of main() that is currently active).

OpenStreetMap is replaced by a synthetic road grid, and every output goes to tmp_path.
"""
import sys
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import main as main_module
from conftest import REPO_ROOT, write_homes_csv
from utils import osm_utils

REGION = "777"


@pytest.fixture
def run_main(tmp_path, monkeypatch, homes, restore_logging):
    conf = OmegaConf.load(REPO_ROOT / "configs/config.yaml")
    # main.py builds paths by string concatenation, so directories need a trailing slash
    conf.inputs.home_csv_dir = f"{tmp_path}/data/load/"
    conf.miscellaneous.log_dir = f"{tmp_path}/logs/"
    conf.miscellaneous.intermediate_dir = f"{tmp_path}/out/interim/"
    conf.miscellaneous.mapping_dir = f"{tmp_path}/out/mapping/"
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(conf, config_path)

    write_homes_csv(tmp_path / f"data/load/{REGION}-home-load.csv", homes)

    # main.py reads configs/logging_config.ini relative to the working directory
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(sys, "argv", ["main.py", "-c", str(config_path), "-r", REGION])

    def _run(osm_mock):
        monkeypatch.setattr(osm_utils.ox, "graph_from_polygon", osm_mock)
        main_module.main()

    return _run


def outputs(tmp_path):
    return {
        "edges": tmp_path / f"out/interim/{REGION}_road_edges.csv",
        "nodes": tmp_path / f"out/interim/{REGION}_road_nodes.csv",
        "h2r": tmp_path / f"out/mapping/{REGION}_map_h2r.txt",
        "log": tmp_path / f"logs/application_{REGION}.log",
    }


def test_main_first_run_creates_outputs(tmp_path, run_main, road_grid, homes):
    osm = MagicMock(return_value=road_grid)
    run_main(osm)

    osm.assert_called_once()
    out = outputs(tmp_path)
    for name, path in out.items():
        assert path.exists(), f"{name} output missing: {path}"

    # every home is mapped to a road edge
    lines = out["h2r"].read_text().splitlines()
    assert len(lines) == len(homes)
    assert {int(line.split("id=")[1].split(",")[0]) for line in lines} == {h.id for h in homes}

    log = out["log"].read_text()
    assert "Residence and road data loading complete" in log
    assert "Residence and road data mapping complete" in log


def test_main_second_run_uses_cached_files(tmp_path, run_main, road_grid):
    run_main(MagicMock(return_value=road_grid))
    out = outputs(tmp_path)
    cached = {k: out[k].read_text() for k in ("edges", "nodes", "h2r")}

    # A second run must not touch OpenStreetMap and must reuse the cached outputs
    osm = MagicMock(side_effect=AssertionError("OSM should not be queried when roads are cached"))
    run_main(osm)

    osm.assert_not_called()
    assert {k: out[k].read_text() for k in cached} == cached
    assert out["log"].read_text().count("Residence and road data mapping complete") == 2


def test_main_missing_home_csv(tmp_path, run_main, road_grid):
    (tmp_path / f"data/load/{REGION}-home-load.csv").unlink()
    with pytest.raises(FileNotFoundError):
        run_main(MagicMock(return_value=road_grid))
