import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task2_protocol import COMPOSITE_PROXIES, TARGET, adjusted_r2, feature_contracts


class Task2ProtocolTests(unittest.TestCase):
    def test_adjusted_r2_formula(self):
        self.assertAlmostEqual(adjusted_r2(0.8, n=100, p=9), 0.78)

    def test_adjusted_r2_rejects_invalid_degrees_of_freedom(self):
        with self.assertRaises(ValueError):
            adjusted_r2(0.8, n=10, p=9)

    def test_contracts_exclude_target_and_identifier(self):
        columns = ["Person_ID", TARGET, "Age", *sorted(COMPOSITE_PROXIES)]
        contracts = feature_contracts(columns)
        for features in contracts.values():
            self.assertNotIn("Person_ID", features)
            self.assertNotIn(TARGET, features)

    def test_proxy_removed_contract_differs_only_by_declared_proxies(self):
        columns = ["Person_ID", TARGET, "Age", *sorted(COMPOSITE_PROXIES)]
        contracts = feature_contracts(columns)
        self.assertEqual(
            set(contracts["competition"]) - set(contracts["scientific_proxy_removed"]),
            COMPOSITE_PROXIES,
        )


if __name__ == "__main__":
    unittest.main()
