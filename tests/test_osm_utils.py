import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
import pandas as pd
import networkx as nx
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from geopy.distance import geodesic

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.osm_utils import load_roads, plot_network
from utils.dataloader import Home, load_homes
from utils.logging_utils import LogManager

if __name__ == '__main__':
    LogManager.initialize(
        log_file_path="logs/test_osm_utils.log", 
        log_config_path="configs/logging_config.ini"
    )

logger = LogManager.get_logger("unittest_osm_utils")

class TestOSMUtils(unittest.TestCase):

    def setUp(self):
        # Load test homes from the CSV file
        csv_path = Path('data/load/test-home-load.csv')
        self.test_homes = load_homes(str(csv_path))

    @patch('utils.osm_utils.ox.graph_from_polygon')
    def test_load_roads(self, mock_graph_from_polygon):
        # Create points from home coordinates
        points = [Point(home.cord[0], home.cord[1]) for home in self.test_homes]

        # Create a polygon from the points
        polygon = Polygon(unary_union(points).convex_hull)

        # Create a mock graph representing the road network
        mock_graph = nx.MultiDiGraph()

        # Add nodes for the road network (not directly corresponding to homes)
        bounds = polygon.bounds
        step_lat = (bounds[3] - bounds[1]) / 10  # Divide the bounding box into a 10x10 grid
        step_lon = (bounds[2] - bounds[0]) / 10
        for i in range(11):
            for j in range(11):
                lon = bounds[0] + i * step_lon
                lat = bounds[1] + j * step_lat
                if polygon.contains(Point(lon, lat)):
                    mock_graph.add_node(f"{i}_{j}", x=lon, y=lat)

        # Add some edges to connect the nodes
        nodes = list(mock_graph.nodes(data=True))
        for i in range(len(nodes) - 1):
            u, u_data = nodes[i]
            v, v_data = nodes[i+1]
            mock_graph.add_edge(u, v, geometry=LineString([(u_data['x'], u_data['y']), (v_data['x'], v_data['y'])]))

        # Add an edge without geometry to test that case
        u, v = list(mock_graph.nodes())[:2]
        mock_graph.add_edge(u, v)

        mock_graph_from_polygon.return_value = mock_graph

        # Call the function with actual test data
        result = load_roads(self.test_homes)

        # Check if the graph was created
        self.assertIsNotNone(result)

        # Check if all edges have a geometry
        for u, v, data in result.edges(data=True):
            self.assertIn('geometry', data)

        # Check if the graph covers the area containing all homes
        max_distance = geodesic((0, 0), (0, step_lat)).meters  # Convert step to meters
        for home in self.test_homes:
            home_point = (home.cord[1], home.cord[0])  # Note: geodesic uses (lat, lon)
            self.assertTrue(any(geodesic(home_point, (data['y'], data['x'])).meters < max_distance
                                for _, data in result.nodes(data=True)))

        # Check if the function was called with the correct parameters
        mock_graph_from_polygon.assert_called_once()
        args, kwargs = mock_graph_from_polygon.call_args
        self.assertEqual(kwargs['network_type'], "all")
        self.assertEqual(kwargs['retain_all'], True)
        self.assertEqual(kwargs['truncate_by_edge'], False)

    def test_load_roads_no_homes(self):
        with self.assertRaises(ValueError):
            load_roads([])

    @patch('utils.osm_utils.ox.graph_from_polygon')
    def test_load_roads_no_edges(self, mock_graph_from_polygon):
        mock_graph = nx.MultiDiGraph()
        mock_graph_from_polygon.return_value = mock_graph

        with self.assertRaises(ValueError):
            load_roads(self.test_homes)

    @patch('utils.osm_utils.ox.plot_graph')
    @patch('utils.osm_utils.plt.savefig')
    def test_plot_network(self, mock_savefig, mock_plot_graph):
        # Mock the graph and plot
        mock_graph = nx.MultiDiGraph()
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plot_graph.return_value = (mock_fig, mock_ax)

        # Remove 'figs' directory if it exists
        figs_dir = Path('figs')
        if figs_dir.exists():
            for file in figs_dir.iterdir():
                file.unlink()
            figs_dir.rmdir()

        # Call the function
        plot_network(mock_graph, self.test_homes, 'test_plot.png')

        # Check if 'figs' directory was created
        self.assertTrue(figs_dir.exists())

        # Check if savefig was called with the correct path
        mock_savefig.assert_called_once_with(figs_dir / 'test_plot.png', dpi=300, bbox_inches='tight')

        # Clean up: remove the 'figs' directory
        for file in figs_dir.iterdir():
            file.unlink()
        figs_dir.rmdir()

if __name__ == '__main__':
    unittest.main()