import unittest

from sklearn.kernel_approximation import Nystroem, RBFSampler

from src.task2_kernel_models import (
    GAMMAS,
    NYSTROEM_COMPONENTS,
    RBF_COMPONENTS,
    RIDGE_ALPHAS,
    candidate_grid,
    make_feature_map,
)


class Task2KernelModelTests(unittest.TestCase):
    def test_grid_is_bounded_and_unique(self):
        grid = candidate_grid()
        expected = len(GAMMAS) * (
            len(RBF_COMPONENTS) + len(NYSTROEM_COMPONENTS)
        ) * len(RIDGE_ALPHAS)
        self.assertEqual(len(grid), expected)
        self.assertEqual(len({item["model"] for item in grid}), len(grid))

    def test_feature_map_factory_preserves_seed_and_parameters(self):
        rbf = make_feature_map("rbf_sampler", gamma=0.003, components=256, seed=2037)
        nystroem = make_feature_map("nystroem", gamma=0.01, components=128, seed=2037)
        self.assertIsInstance(rbf, RBFSampler)
        self.assertIsInstance(nystroem, Nystroem)
        self.assertEqual(rbf.random_state, 2037)
        self.assertEqual(nystroem.n_components, 128)
        self.assertEqual(nystroem.gamma, 0.01)

    def test_kernel_grid_does_not_expand_without_config_change(self):
        self.assertEqual(GAMMAS, (0.001, 0.003, 0.01))
        self.assertEqual(RIDGE_ALPHAS, (1.0, 10.0))


if __name__ == "__main__":
    unittest.main()
