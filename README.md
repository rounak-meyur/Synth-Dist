# SYNGRID: Synthetic Distribution Grid Generator

## 📝 Description
SYNGRID is a Python-based toolkit for generating synthetic electrical distribution networks that mirror the characteristics of real-world power distribution systems. The toolkit uses geospatial data, demographic information, and power system constraints to create realistic secondary distribution networks for specified regions.

## 🎯 Key Features
- Optimization-based secondary distribution network generation
- Geospatial integration with real-world coordinates
- Power flow constraints consideration
- Customizable network parameters
- Detailed logging system for tracking network generation process
- Support for multiple regions and scales

## 🏗️ Project Structure
```
syngrid/
├── configs/
│   ├── logging_config.ini      # Logging configuration
│   └── config.yaml            # Main configuration file
├── logs/                      # Log files directory
├── utils/
│   ├── __init__.py
│   └── logging_utils.py       # Logging utilities
├── models/
│   ├── __init__.py
│   └── secnet.py             # Secondary network optimization model
├── docs/
│   └── secondary_network.pdf  # Technical documentation
├── tests/                    # Test files
├── requirements.txt          # Project dependencies
└── README.md                # This file
```

## 🛠️ Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/syngrid.git
cd syngrid
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Dependencies
- Python 3.8+
- NetworkX
- CVXPY
- PySCIPOpt
- NumPy
- Other dependencies listed in requirements.txt

## 🚀 Usage
Here's a basic example of generating a secondary distribution network:

```python
from models.secnet import create_secondary_distribution_network
import networkx as nx

# Create input graph with required attributes
graph = nx.Graph()

# Add nodes (homes and transformers)
graph.add_node(1, cord=(longitude1, latitude1), load=5.5, label='H')
graph.add_node(2, cord=(longitude2, latitude2), load=0, label='T')

# Add potential edges
graph.add_edge(1, 2, length=100, crossing=0.5)

# Generate optimized network
result = create_secondary_distribution_network(
    graph=graph,
    penalty=1.0,
    max_rating=100,
    max_hops=5
)
```

## 📊 Network Optimization Parameters
- `penalty`: Cost multiplier for edge crossings
- `max_rating`: Maximum power rating for transformers
- `max_hops`: Maximum allowed hops from transformer to home
- Edge cost = `length + (penalty * crossing)`

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
- All contributors to this project

## 📞 Contact
For questions and feedback:
- Email: [rounak.meyur@pnnl.gov](rounak.meyur@pnnl.gov)
- Issue Tracker: [GitHub Issues](https://github.com/rounak-meyur/SynthDist/issues)

## 🗺️ Roadmap
- [ ] Add support for primary distribution network generation
- [ ] Implement parallel optimization for large networks
- [ ] Add visualization tools
- [ ] Include more realistic power flow constraints
- [ ] Add support for different voltage levels
- [ ] Create web interface for network generation

## 📚 Citation
If you use this software in your research, please cite:
```bibtex
@software{syngrid2024,
  title = {SYNGRID: Synthetic Grid Distribution Network Generator},
  author = {[Your Name]},
  year = {2024},
  url = {https://github.com/yourusername/syngrid}
}
```