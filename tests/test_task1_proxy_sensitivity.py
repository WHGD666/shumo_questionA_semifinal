import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_proxy_sensitivity import PROXY_FIELDS, feature_contracts, model_factories


class Task1ProxySensitivityTests(unittest.TestCase):
    def test_proxy_removed_contract_excludes_all_proxies(self):
        columns = ["Age", "Sleep_Quality_Score", *sorted(PROXY_FIELDS)]
        contracts = feature_contracts(columns)
        self.assertTrue(PROXY_FIELDS.isdisjoint(contracts["proxy_removed"]))
        self.assertTrue(PROXY_FIELDS.issubset(contracts["full_non_sleep"]))

    def test_target_is_excluded_from_both_contracts(self):
        contracts = feature_contracts(["Age", "Sleep_Quality_Score"])
        for features in contracts.values():
            self.assertNotIn("Sleep_Quality_Score", features)

    def test_only_frozen_linear_candidates_are_used(self):
        self.assertEqual(
            set(model_factories()),
            {"ridge_alpha_3", "elastic_alpha_0p001_l1_0p9"},
        )


if __name__ == "__main__":
    unittest.main()
