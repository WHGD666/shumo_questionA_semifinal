import unittest

import pandas as pd

from src.task3_protocol import (
    DETERMINISTIC_TARGET_PROXIES,
    LABEL_ORDER,
    STRONG_COMPOSITE_PROXIES,
    TARGET,
    feature_contracts,
    health_score_to_category,
    validate_deterministic_target_proxies,
)


class Task3ProtocolTests(unittest.TestCase):
    def test_health_score_bins_use_frozen_boundaries(self):
        values = pd.Series([27.3, 44.9, 45.0, 64.9, 65.0, 79.9, 80.0, 100.0])
        expected = ["Poor", "Poor", "Average", "Average", "Good", "Good", "Excellent", "Excellent"]
        self.assertEqual(health_score_to_category(values).tolist(), expected)
        self.assertEqual(LABEL_ORDER, ("Poor", "Average", "Good", "Excellent"))

    def test_deterministic_target_proxies_are_forbidden_in_both_contracts(self):
        columns = [
            "Person_ID", TARGET, "Age", "Healthy_Aging_Score", "Fitness_Level", "Wellness_Category"
        ]
        contracts = feature_contracts(columns)
        for features in contracts.values():
            self.assertNotIn("Person_ID", features)
            self.assertNotIn(TARGET, features)
            self.assertTrue(DETERMINISTIC_TARGET_PROXIES.isdisjoint(features))

    def test_contracts_differ_only_by_strong_composite_proxy(self):
        columns = [
            "Person_ID", TARGET, "Age", "Healthy_Aging_Score", "Fitness_Level", "Wellness_Category"
        ]
        contracts = feature_contracts(columns)
        self.assertEqual(
            set(contracts["competition_proxy_inclusive"]) - set(contracts["scientific_proxy_removed"]),
            STRONG_COMPOSITE_PROXIES,
        )

    def test_proxy_validation_detects_exact_binned_labels(self):
        frame = pd.DataFrame({
            TARGET: [40.0, 50.0, 70.0, 90.0],
            "Fitness_Level": ["Poor", "Average", "Good", "Excellent"],
            "Wellness_Category": ["Poor", "Average", "Good", "Excellent"],
        })
        self.assertTrue(all(validate_deterministic_target_proxies(frame).values()))

    def test_proxy_validation_rejects_one_incorrect_label(self):
        frame = pd.DataFrame({
            TARGET: [40.0, 50.0, 70.0, 90.0],
            "Fitness_Level": ["Poor", "Average", "Good", "Excellent"],
            "Wellness_Category": ["Poor", "Average", "Excellent", "Excellent"],
        })
        checks = validate_deterministic_target_proxies(frame)
        self.assertFalse(checks["wellness_matches_health_score_bins"])
        self.assertFalse(checks["fitness_equals_wellness"])


if __name__ == "__main__":
    unittest.main()
