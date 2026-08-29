import unittest

from src.task2_latent_factor_models import (
    PCR_COMPONENTS,
    PLS_COMPONENTS,
    candidate_grid,
    make_model,
)


class Task2LatentFactorModelTests(unittest.TestCase):
    def test_grid_is_bounded_unique_and_contains_pls8_control(self):
        grid = candidate_grid()
        self.assertEqual(len(grid), len(PLS_COMPONENTS) + len(PCR_COMPONENTS))
        self.assertEqual(len({item["model"] for item in grid}), len(grid))
        self.assertIn(
            {"model": "pls_8", "family": "pls", "component_count": 8},
            grid,
        )

    def test_components_are_positive_and_bounded(self):
        self.assertGreater(min(PLS_COMPONENTS + PCR_COMPONENTS), 0)
        self.assertLessEqual(max(PLS_COMPONENTS + PCR_COMPONENTS), 80)

    def test_model_factory_preserves_component_count(self):
        pls = make_model("pls", components=12, seed=2026)
        pcr = make_model("pcr", components=24, seed=2026)
        self.assertEqual(pls.n_components, 12)
        self.assertEqual(pcr.named_steps["pca"].n_components, 24)
        self.assertEqual(pcr.named_steps["ridge"].alpha, 10.0)


if __name__ == "__main__":
    unittest.main()
