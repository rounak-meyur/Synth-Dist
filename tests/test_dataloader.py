import unittest
from pathlib import Path
import csv
import tempfile
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataloader import load_homes, Home

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.valid_csv_path = Path(self.test_dir) / "valid_test.csv"
        self.invalid_csv_path = Path(self.test_dir) / "invalid_test.csv"

        # Create a valid test CSV file
        with open(self.valid_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['hid', 'longitude', 'latitude'] + [f'hour{i}' for i in range(1, 25)])
            writer.writerow(['1', '-73.935242', '40.730610'] + [str(i * 10) for i in range(1, 25)])
            writer.writerow(['2', '-74.006015', '40.712775'] + [str(i * 5) for i in range(1, 25)])

        # Create an invalid test CSV file (missing columns)
        with open(self.invalid_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['hid', 'longitude', 'latitude'])
            writer.writerow(['1', '-73.935242', '40.730610'])

    def test_load_homes_valid(self):
        homes = load_homes(self.valid_csv_path)
        self.assertEqual(len(homes), 2)
        self.assertIsInstance(homes[0], Home)
        self.assertEqual(homes[0].cord, (-73.935242, 40.730610))
        self.assertEqual(len(homes[0].profile), 24)
        self.assertEqual(homes[0].peak, 240)
        self.assertEqual(homes[0].load, 125)

    def test_load_homes_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_homes("non_existent_file.csv")

    def test_load_homes_invalid_format(self):
        with self.assertRaises(ValueError):
            load_homes(self.invalid_csv_path)

if __name__ == '__main__':
    unittest.main()