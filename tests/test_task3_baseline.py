import unittest

from sklearn.base import is_regressor

from src.task3_baseline import SEED, _metrics, model_factories


class Task3BaselineTests(unittest.TestCase):
    def test_required_baseline_families_are_present(self):
        self.assertEqual(set(model_factories()), {
            "dummy_mean",
            "ordinary_least_squares",
            "ridge_fixed",
            "elastic_net_fixed",
            "pls_16",
            "extra_trees_fixed",
        })

    def test_models_are_fresh_regressor_instances(self):
        first = model_factories(SEED)
        second = model_factories(SEED)
        for name in first:
            self.assertIsNot(first[name], second[name])
            self.assertTrue(is_regressor(first[name]))

    def test_frozen_parameters_are_preserved(self):
        models = model_factories(SEED)
        self.assertEqual(models["ridge_fixed"].alpha, 10.0)
        self.assertEqual(models["elastic_net_fixed"].alpha, 0.001)
        self.assertEqual(models["elastic_net_fixed"].l1_ratio, 0.5)
        self.assertEqual(models["pls_16"].n_components, 16)
        self.assertEqual(models["extra_trees_fixed"].n_estimators, 400)
        self.assertEqual(models["extra_trees_fixed"].random_state, SEED)

    def test_regression_metric_outputs(self):
        metric = _metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(metric["r2"], 1.0)
        self.assertEqual(metric["mae"], 0.0)
        self.assertEqual(metric["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
