import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task1_proxy_sensitivity import PROXY_FIELDS, feature_contracts


class Task1SplineProxyValidationTests(unittest.TestCase):
    def test_contracts_differ_only_by_declared_proxies(self):
        columns = ["Age", "BMI", "Sleep_Quality_Score", *sorted(PROXY_FIELDS)]
        contracts = feature_contracts(columns)
        self.assertEqual(
            set(contracts["full_non_sleep"]) - set(contracts["proxy_removed"]),
            PROXY_FIELDS,
        )


if __name__ == "__main__":
    unittest.main()
