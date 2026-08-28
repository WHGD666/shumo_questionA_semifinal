import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_spline_tuning import parameter_grid, model_name


class Task1SplineTuningTests(unittest.TestCase):
    def test_grid_has_twenty_four_unique_candidates(self):
        grid = parameter_grid()
        names = [model_name(row["n_knots"], row["degree"], row["alpha"]) for row in grid]
        self.assertEqual(len(grid), 24)
        self.assertEqual(len(names), len(set(names)))

    def test_control_candidate_is_in_grid(self):
        names = {
            model_name(row["n_knots"], row["degree"], row["alpha"])
            for row in parameter_grid()
        }
        self.assertIn("spline_k5_d3_a10p0", names)


if __name__ == "__main__":
    unittest.main()
