import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_residual_stability import INNER_SEEDS, RESIDUAL_WEIGHT, stability_acceptance


class Task1ResidualStabilityTests(unittest.TestCase):
    def test_candidate_is_frozen(self):
        self.assertEqual(RESIDUAL_WEIGHT, 0.75)
        self.assertEqual(INNER_SEEDS, (2026, 2037, 2048, 2059, 2070))

    def test_acceptance_requires_positive_and_material_mean_delta(self):
        self.assertTrue(stability_acceptance([0.002, 0.0015, 0.001]))
        self.assertFalse(stability_acceptance([0.002, -0.0001, 0.002]))
        self.assertFalse(stability_acceptance([0.0005, 0.0006, 0.0007]))


if __name__ == "__main__":
    unittest.main()
