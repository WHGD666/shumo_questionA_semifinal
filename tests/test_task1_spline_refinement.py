import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_spline_tuning import model_name


class Task1SplineRefinementTests(unittest.TestCase):
    def test_refinement_grid_has_eight_unique_candidates(self):
        names = [
            model_name(knots, 2, alpha)
            for knots in (3, 4)
            for alpha in (0.3, 1.0, 3.0, 10.0)
        ]
        self.assertEqual(len(names), 8)
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("spline_k4_d2_a3p0", names)


if __name__ == "__main__":
    unittest.main()
