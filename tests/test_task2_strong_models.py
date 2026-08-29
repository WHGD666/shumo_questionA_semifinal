import unittest

from src.task2_strong_models import CONTROL_ADJUSTED_R2, model_factories


class Task2StrongModelTests(unittest.TestCase):
    def test_factory_contains_only_frozen_candidates(self):
        self.assertEqual(set(model_factories()), {"lightgbm_fixed", "catboost_fixed"})

    def test_models_are_regressors_with_fixed_seeds(self):
        models = model_factories(seed=2037)
        self.assertEqual(models["lightgbm_fixed"].random_state, 2037)
        self.assertEqual(models["catboost_fixed"].get_param("random_seed"), 2037)

    def test_controls_are_frozen_for_both_contracts(self):
        self.assertEqual(set(CONTROL_ADJUSTED_R2), {"competition", "scientific_proxy_removed"})
        self.assertGreater(CONTROL_ADJUSTED_R2["scientific_proxy_removed"], 0.62)


if __name__ == "__main__":
    unittest.main()
