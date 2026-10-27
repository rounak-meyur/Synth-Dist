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
syngrid/
├── models/                  # Core optimization models
│   ├── primnet.py           # Primary network optimization
│   └── secnet.py            # Secondary network optimization
├── utils/                   # Utility functions
│   ├── dataloader.py        # Data loading utilities
│   ├── drawings.py          # Visualization functions
│   ├── logging_utils.py     # Logging configuration
│   ├── mapping.py           # Geospatial mapping utilities
│   ├── osm_utils.py         # OpenStreetMap integration
│   ├── partition_utils.py   # Network partitioning
│   ├── primnet_utils.py     # Primary network utilities
│   └── secnet_utils.py      # Secondary network utilities
├── configs/                 # Configuration files
├── data/                    # Input data directory
├── figs/                    # Output figures
├── logs/                    # Log files
├── main.py                  # Main execution script
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
   - Home locations and load data (CSV)
   - Substation locations (CSV)
   - Configuration file (YAML)

2. Run the main script:
```bash
uv run main.py -c configs/config_heuristic.yaml -r region_ID
```

To generate visualization plots, add the `-p` flag:
```bash
uv run main.py -c configs/config_heuristic.yaml -r region_ID -p
```

### Checkpoints and run statistics
Each stage (roads, mapping, secondary network, partitioning, primary network, final
network) saves its outputs and is
reused by later runs. A stage is rerun when its inputs or parameters change, when an
earlier stage was rerun, or when its outputs are missing; the record is kept in
`out/interim/{region}_checkpoints.json`. An interrupted secondary network stage resumes
from the road links that already finished.

To ignore saved outputs and run every stage again, use `--from-scratch` (or set
`miscellaneous.from_scratch: true` in the config):
```bash
uv run main.py -c configs/config_heuristic.yaml -r region_ID --from-scratch
```

The final distribution network (substations, road nodes, transformers and homes with
feeder, primary and secondary lines) is written to
`out/finalnet/{region}_distribution_network_nodes.csv` and `_edges.csv`.

Compute time, peak memory and data sizes (homes, road nodes/edges, transformers,
secondary edges, per-substation partitions, ...) of every stage are written to
`out/stats/{region}_run_stats.json`.

## 📊 Configuration file

The program uses YAML configuration files with the following key sections:

```yaml
state: 51

inputs:
  home_csv_dir: "data/load/"
  substation_csv: "data/substations.csv"

miscellaneous:
  log_dir: "logs/"
  intermediate_dir: "out/interim/"
  mapping_dir: "out/mapping/"
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

- **Primary Network**:
  - `{substation_id}_nodes.csv`: Primary network nodes
  - `{substation_id}_edges.csv`: Primary network connections

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