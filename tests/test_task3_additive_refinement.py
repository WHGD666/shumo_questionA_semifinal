import unittest

from src.task3_additive_refinement import (
    COMPETITION_ALPHAS,
    KNOTS,
    SCIENTIFIC_ALPHAS,
    candidate_grid,
)


class Task3AdditiveRefinementTests(unittest.TestCase):
    def test_competition_grid_is_bounded_and_contains_control(self):
        grid = candidate_grid("competition_proxy_inclusive")
        self.assertEqual(len(grid), len(KNOTS) * len(COMPETITION_ALPHAS))
        self.assertEqual({item["penalty"] for item in grid}, {"elastic"})
        self.assertIn("additive_elastic_k4_d2_a0p003", {item["model"] for item in grid})

    def test_scientific_grid_is_bounded_and_contains_control(self):
        grid = candidate_grid("scientific_proxy_removed")
        self.assertEqual(len(grid), len(KNOTS) * len(SCIENTIFIC_ALPHAS))
        self.assertEqual({item["penalty"] for item in grid}, {"ridge"})
        self.assertIn("additive_ridge_k4_d2_a0p3", {item["model"] for item in grid})

    def test_candidate_names_are_unique_within_each_contract(self):
        for contract in ("competition_proxy_inclusive", "scientific_proxy_removed"):
            grid = candidate_grid(contract)
            self.assertEqual(len({item["model"] for item in grid}), len(grid))

    def test_unknown_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            candidate_grid("unknown")


if __name__ == "__main__":
    unittest.main()
