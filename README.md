# SYNGRID: Synthetic Distribution Grid Generator


[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Description
SYNGRID is a Python-based toolkit for generating synthetic electrical distribution networks that mirror the characteristics of real-world power distribution systems. The toolkit uses geospatial data, demographic information, and power system constraints to create realistic secondary distribution networks for specified regions.

## 🎯 Key Features

- **Optimized Network Generation**:
  - Secondary network optimization with transformer placement
  - Primary feeder optimization with voltage and power flow constraints
  - Multi-stage optimization for large-scale networks

- **Geospatial Integration**:
  - Integration with OpenStreetMap for road network data
  - Real-world coordinate system support
  - Automated mapping of homes to road networks

- **Network Partitioning**:
  - Voronoi-based network partitioning
  - Automated substation service area determination
  - Load-based transformer placement

## 🏗️ Project Structure
```
Synth-Dist/
├── .github/workflows/
│   └── tests.yml            # GitHub Actions CI (runs pytest via uv)
├── models/                  # Core network construction models
│   ├── primnet.py           # Primary network optimization (MILP)
│   ├── primnet_heuristic.py # Primary network heuristic
│   ├── secnet.py            # Secondary network optimization (MILP)
│   └── secnet_heuristic.py  # Secondary network heuristic
├── utils/                   # Utility functions
│   ├── checkpoint_utils.py  # Pipeline checkpointing
│   ├── dataloader.py        # Data loading utilities
│   ├── drawings.py          # Visualization functions
│   ├── finalnet_utils.py    # Combining primary/secondary into the final network
│   ├── logging_utils.py     # Logging configuration
│   ├── mapping.py           # Geospatial mapping utilities
│   ├── osm_utils.py         # OpenStreetMap integration
│   ├── partition_utils.py   # Network partitioning
│   ├── primnet_utils.py     # Primary network utilities
│   ├── secnet_utils.py      # Secondary network utilities
│   └── stats_utils.py       # Run statistics
├── scripts/                 # Standalone helper scripts
│   ├── downloader.py        # Download per-county load files from net.science
│   ├── get_load_csv.py      # Build the home load CSVs (<region>-home-load.csv) from them
│   ├── homes.py             # Collect home IDs into out/homes.json
│   └── slurm/submit.sh      # Submit the pipeline for a group of regions to SLURM
├── tests/                   # pytest suite
├── configs/                 # Configuration files
│   ├── config_heuristic.yaml
│   ├── config_milp.yaml
│   ├── logging_config.ini
│   └── regions/             # Groups of regions
│       ├── virginia.yaml           # all of Virginia (default)
│       ├── northern_virginia.yaml  # Northern Virginia
│       └── roanoke_montgomery.yaml # Montgomery and Roanoke counties, Roanoke and Salem cities
├── data/                    # Input data
│   ├── load/                # Home load CSVs of regions 121, 161, 770, 775 and sample 999 (<region>-home-load.csv)
│   ├── load_51/             # Home load CSVs of every Virginia county (not in git, see below)
│   └── substations.csv      # Substation locations
├── out/, figs/, logs/, cache/  # Generated outputs (git-ignored)
├── main.py                  # Main execution script
├── pyproject.toml           # Project metadata and dependencies (uv)
├── uv.lock                  # Locked dependency versions
├── .python-version          # Python version for uv (3.12)
├── Dockerfile               # Container build
├── syndist.yml              # Conda environment file
└── README.md                # This file
```

## 🛠️ Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/syngrid.git
cd syngrid
```

2. Create the environment with [uv](https://docs.astral.sh/uv/) (dependencies are in `pyproject.toml`, pinned in `uv.lock`):
```bash
uv sync
```

3. Run the tests:
```bash
uv run pytest
```
Tests marked `integration` need the cached road network in `out/interim/` and are skipped when it is absent (e.g. in CI).

## 📦 Dependencies
- Python 3.11+
- NetworkX
- CVXPY
- PySCIPOpt
- NumPy
- Other dependencies listed in `pyproject.toml`

## 🚀 Usage

1. Prepare your input data:
   - Home locations and load data (CSV), one file per region (county)
   - Substation locations (CSV)
   - Configuration file (YAML) and group configuration file (YAML)

2. Run the main script:
```bash
uv run main.py -c configs/config_heuristic.yaml -g configs/regions/northern_virginia.yaml
```

### Home load data
The repository only contains the home load files of regions 121 (Montgomery County) and
161 (Roanoke County), and of the sample region 999, in `data/load/`. The `data/load_51/`
directory with the files of every Virginia county is **not committed to git**; neither are
the files of the cities of Roanoke (770) and Salem (775), which are copied from it to
`data/load/`. To run the pipeline on any other region:

1. Download the load files of the counties with `scripts/downloader.py`. It downloads
   `<PREFIX><county>-20140523.csv` (e.g. `VA013-20140523.csv`, hourly loads of every home
   in kWh) from the net.science file service for every region of a group configuration
   file, or for the regions given with `-r`, and skips files already downloaded:
   ```bash
   uv run python scripts/downloader.py -g configs/regions/virginia.yaml --prefix VA -o data/raw/load_VA
   ```
   Use `--base-url` if the files are in another net.science folder, and `--date` for
   another date in the file names.
2. Build the home load files with `scripts/get_load_csv.py`. It adds the location of every
   home from the synthetic household files (`household_<census tract>.csv`, with columns
   `SYNTHETIC_HID`, `longitude` and `latitude`), converts the loads to W and writes
   `<county>-home-load.csv` (format in [Homes CSV](#homes-csv)). Homes without a location
   are left out:
   ```bash
   uv run python scripts/get_load_csv.py -i data/raw/load_VA --household-dir data/raw/households_VA \
       -o data/load_51 --prefix VA
   ```
3. Point `inputs.home_csv_dir` in the group configuration file to the output directory
   (with a trailing `/`).

For another state, use its abbreviation as `--prefix`, its counties in the group file
(`-g`) and its own output directory.

`configs/regions/virginia.yaml` (the default) and `configs/regions/northern_virginia.yaml`
point to `data/load_51/`, so they need these files. `configs/regions/roanoke_montgomery.yaml`
(Montgomery and Roanoke counties with the cities of Roanoke and Salem) points to
`data/load/`; it needs the files of 770 and 775 copied there:
```bash
cp data/load_51/770-home-load.csv data/load_51/775-home-load.csv data/load/
uv run main.py -g configs/regions/roanoke_montgomery.yaml
```

To run only the committed regions (Montgomery and Roanoke counties), use a group file that
points to `data/load/`:
```yaml
# configs/regions/montgomery_roanoke.yaml
state: 51
inputs:
  home_csv_dir: "data/load/"
group:
  name: montgomery_roanoke
  regions: [
    "121",  # Montgomery County
    "161",  # Roanoke County
  ]
```
```bash
uv run main.py -g configs/regions/montgomery_roanoke.yaml
```

### Groups of regions and pipeline stages
The pipeline runs on a **group** of regions listed in a group configuration file
(`configs/regions/*.yaml`, default: all of Virginia). The group file sets the state (FIPS
code), the directory of the home load files and the regions; all three are required. It
is merged over the configuration file, so it can also override any other setting:

```yaml
state: 51
inputs:
  home_csv_dir: "data/load_51/"
group:
  name: northern_virginia
  regions: ["013", "059", "107", "153", "510", "600", "610", "683", "685"]
```

The pipeline has four stages:

| Stage | Runs once per | Does |
|-------|---------------|------|
| `secnet` | region | Road network, home to road mapping and secondary network |
| `partition` | group | Merges the secondary networks of all regions and assigns every transformer to a substation serving the group |
| `primnet` | substation | Primary network of the substation |
| `finalnet` | group | Combines the primary networks with the secondary networks of all regions |

Secondary networks of different regions, and primary networks of different substations,
are independent of each other, so they can run in parallel. Because the partitioning
sees all regions at once, a substation near a county border serves homes on both sides
of it.

`all` (the default stage) runs every stage in order in one process. The stages can also be
run separately; the stage name comes first:
```bash
G=configs/regions/northern_virginia.yaml
uv run main.py secnet -g $G -r 013              # one region (default: every region of the group)
uv run main.py partition -g $G                  # after secnet finished for every region
uv run main.py primnet -g $G -s 123456          # one substation (default: every substation)
uv run main.py finalnet -g $G                   # after primnet finished for every substation
```

- `-r` runs the given regions instead of the group's regions. With `all`, `partition` and
  `finalnet`, they form a group named after them, e.g. `uv run main.py -r 999` runs the
  sample region on its own.
- `--task-index i --task-count n` makes `secnet`/`primnet` run every n-th region or
  substation starting at the i-th, so that n jobs share the work.
- `--list` prints the regions (`secnet`) or substations (`primnet`) a task would run.
- `-p` generates plots of the merged network (after `partition`) and of the final
  network (after `finalnet`).

A group stage stops with an error if an earlier stage has not finished, or was run with
different parameters, for some region or substation.

#### Tracking progress of the primnet stage
- On the console, a progress bar counts the finished substations of the job. A second bar
  counts the solved feeder areas of the current substation. Before that bar appears, the
  substation's network is split into feeder areas, which can take a while for a big one.
- In the log file (`logs/application_{group}_primnet[_{task}].log`), a line
  `Primary networks finished for i of n substations` is written after each substation.
  Use `tail -f` on it, or on the SLURM output files in `logs/slurm/`.
- Across all jobs of a group, compare the number of finished primary networks with the
  number of substations:
  ```bash
  ls out/primnet/{group}_*_edges.csv | wc -l           # finished substations
  wc -l < out/interim/{group}_partition/substations.txt  # all substations
  ```

### Running on a SLURM cluster
`scripts/slurm/submit.sh` submits a group as four dependent jobs: a job array with one
task per region for `secnet`, one `partition` job, a job array for `primnet` and one
`finalnet` job. Each job starts only after the previous one succeeded.

```bash
scripts/slurm/submit.sh [GROUP_CONFIG] [CONFIG] [-- EXTRA_MAIN_ARGS...]
scripts/slurm/submit.sh configs/regions/northern_virginia.yaml configs/config_heuristic.yaml
scripts/slurm/submit.sh configs/regions/virginia.yaml configs/config_milp.yaml -- --from-scratch
```

Configure the script for your cluster first, by editing the settings at its top or by
setting them as environment variables:

| Setting | Meaning |
|---------|---------|
| `ACCOUNT` | Allocation/account charged for the jobs (`sbatch --account`). Empty by default; set it if your cluster requires one. |
| `PARTITION` | Partition/queue the jobs run in (`sbatch --partition`). Empty: the cluster's default partition. |
| `QOS` | Quality of service (`sbatch --qos`), if your cluster uses one |
| `SBATCH_EXTRA` | Any other `sbatch` options for every job, e.g. `"--constraint=cpu"` |
| `RUN` | Command that runs Python in the project environment (default `uv run`; e.g. `"conda run -n synth python"`) |
| `SECNET_TIME`, `SECNET_MEM`, `SECNET_CPUS`, `SECNET_MAX_RUNNING` | Time limit, memory, CPUs and maximum number of concurrently running tasks of the secnet array (default 4: every task downloads its roads from the Overpass API, which limits concurrent requests) |
| `PARTITION_TIME`, `PARTITION_MEM` | Resources of the partition job, which holds the merged network of the whole group |
| `PRIMNET_TASKS`, `PRIMNET_TIME`, `PRIMNET_MEM`, `PRIMNET_CPUS`, `PRIMNET_MAX_RUNNING` | Number of tasks the substations are split over, and their resources (raise `PRIMNET_CPUS` for MILP solver threads) |
| `FINALNET_TIME`, `FINALNET_MEM` | Resources of the finalnet job |

Empty settings are not passed to `sbatch`. For example:
```bash
ACCOUNT=my_allocation PARTITION=compute PARTITION_MEM=256G \
    scripts/slurm/submit.sh configs/regions/virginia.yaml
```

The project environment must be available on the compute nodes (e.g. run `uv sync` in the
repository once). Jobs run in the repository root, and SLURM output goes to `logs/slurm/`.
Road networks are downloaded from OpenStreetMap in the `secnet` jobs, so compute nodes
need internet access; otherwise run the `secnet` stage where internet is available
and submit the remaining stages. A failed job can simply be resubmitted: finished
regions and substations are reused from their checkpoints.

### Overpass API errors
Road networks are downloaded from the public Overpass API (OpenStreetMap), which rejects
requests when it is busy ("This Overpass API server is currently at capacity"). A failed
download is retried with exponential backoff, configured in the `osm` section of the
configuration file:

```yaml
osm:
  overpass_url: "https://overpass-api.de/api"
  max_retries: 6    # retries of a failed download
  retry_wait: 30    # seconds before the first retry, doubled after every failure
```

If downloads keep failing, run fewer regions at the same time (`SECNET_MAX_RUNNING` for
SLURM), rerun later (downloaded road networks are saved and reused), or point
`overpass_url` to another public Overpass instance or to your own
([Overpass API installation](https://github.com/drolbr/Overpass-API)).

### Checkpoints and run statistics
Each stage saves its outputs and is reused by later runs. A stage is rerun when its
inputs or parameters change, or when its outputs are missing. The region stages (roads,
mapping, secondary network) are recorded in `out/interim/{region}_checkpoints.json`; an
interrupted secondary network stage resumes from the road links that already finished.
The group stages are recorded in `out/interim/{group}_group_checkpoints.json` and every
substation's primary network in `out/interim/{group}_primnet/{substation}_checkpoints.json`.
Group stages compare the contents of the files they read, so they are reused when a rerun
of an earlier stage produced the same outputs.

To ignore saved outputs and run the stages again, use `--from-scratch` (or set
`miscellaneous.from_scratch: true` in the config):
```bash
uv run main.py -g configs/regions/northern_virginia.yaml --from-scratch
```

The final distribution network (substations, road nodes, transformers and homes with
feeder, primary and secondary lines) is written to
`out/finalnet/{group}_distribution_network_nodes.csv` and `_edges.csv`.

Compute time, peak memory and data sizes of every stage are written to
`out/stats/{region}_run_stats.json` (homes, road nodes/edges, transformers, secondary
edges, ...), `out/stats/{group}_group_run_stats.json` (partitions, primary and final
networks) and `out/stats/{group}_primnet/{substation}_run_stats.json`.

## 📊 Configuration file

The program uses YAML configuration files with the following key sections. The state
(FIPS code) and the directory of the home load files are not set here: they come from the
group configuration file, together with the regions (see [Groups of regions and pipeline stages](#groups-of-regions-and-pipeline-stages)).

```yaml
inputs:
  substation_csv: "data/substations.csv"

miscellaneous:
  log_dir: "logs/"
  intermediate_dir: "out/interim/"
  mapping_dir: "out/mapping/"
  stats_dir: "out/stats/"
  padding: 0.001

secnet:
  out_dir: "out/secnet/"
  base_transformer_id: 000000000000
  secnet_args:
    separation: 50.0
    penalty: 0.5
    max_rating: 25000
    max_hops: 10

primnet:
  out_dir: "out/primnet/"
  primnet_args:
    voltage_max: 1.05
    voltage_min: 0.90
    maximum_voltage_drop: 0.15
    conductor_resistance_per_km: 0.8625
    base_impedance: 39690
    maximum_branch_flow: 400
    max_feeder_number: 10
    max_feeder_capacity: 400
    relative_gap: 0.05
```

## Input Data Format

### Homes CSV
```csv
hid,longitude,latitude,hour1,hour2,...,hour24
1,-73.985,40.748,5.2,4.8,...,6.1
```

### Substations CSV
```csv
ID,X,Y
SUB1,-73.982,40.745
```

## Output Files

The program generates several output files:

- **Secondary Network**:
  - `{region}_transformers.csv`: Transformer locations and loads
  - `{region}_secondary_edges.csv`: Secondary network connections
  - `{region}_road_transformer_edges.txt`: Road-transformer sequences

- **Partitioning** (`out/interim/{group}_partition/`):
  - `{substation_id}.json`: Road network and transformers assigned to a substation
  - `substations.txt`: Substations serving the group

- **Primary Network**:
  - `{group}_{substation_id}_nodes.csv`: Primary network nodes
  - `{group}_{substation_id}_edges.csv`: Primary network connections

- **Final Network**:
  - `{group}_distribution_network_nodes.csv` / `_edges.csv`

- **Visualizations** (with `-p` flag):
  - Network plots in the `figs/` directory

## 🔍 Logging
The project uses a comprehensive logging system:
- Log files are stored in the `logs/` directory
- Configuration in `configs/logging_config.ini`
- Different log levels for console and file output
- Automatic run separation in log files

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors
- [Rounak Meyur](https://www.pnnl.gov/people/rounak-meyur)
- [Henning S. Mortveit](https://engineering.virginia.edu/faculty/henning-s-mortveit)

## 🙏 Acknowledgments
- SCIP Optimization Suite
- NetworkX developers
- OpenStreetMap contributors
- All contributors to this project

## 📞 Contact
For questions and feedback:
- Email: [rounak.meyur@pnnl.gov](rounak.meyur@pnnl.gov)
- Issue Tracker: [GitHub Issues](https://github.com/rounak-meyur/SynthDist/issues)

## 🗺️ Roadmap
- [ ] Implement parallel optimization for large networks
- [ ] Add visualization tools
- [ ] Include more realistic power flow constraints
- [ ] Add support for different voltage levels
- [ ] Create web interface for network generation

## 📚 Citation
If you use this software in your research, please cite:
```bibtex
@article{pnas2022,
  author = {Rounak Meyur  and Anil Vullikanti  and Samarth Swarup  and Henning S. Mortveit  and Virgilio Centeno  and Arun Phadke  and H. Vincent Poor  and Madhav V. Marathe },
  title = {Ensembles of realistic power distribution networks},
  journal = {Proceedings of the National Academy of Sciences},
  volume = {119},
  number = {42},
  pages = {e2205772119},
  year = {2022},
  url = {https://doi.org/10.1073/pnas.2205772119}
}
```