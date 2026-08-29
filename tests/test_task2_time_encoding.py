import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task2_time_encoding import engineer_cyclic_times, time_to_minutes


class Task2TimeEncodingTests(unittest.TestCase):
    def test_time_to_minutes_parses_boundaries_and_rejects_invalid_values(self):
        actual = time_to_minutes(pd.Series(["00:00", "23:59", "24:00", "bad"]))
        self.assertEqual(actual.iloc[0], 0.0)
        self.assertEqual(actual.iloc[1], 1439.0)
        self.assertTrue(np.isnan(actual.iloc[2]))
        self.assertTrue(np.isnan(actual.iloc[3]))

    def test_cyclic_encoding_drops_raw_time_and_preserves_midnight_geometry(self):
        frame = pd.DataFrame({"Sleep_Time": ["00:00"], "Wake_Up_Time": ["06:00"], "Age": [30]})
        transformed = engineer_cyclic_times(frame)
        self.assertNotIn("Sleep_Time", transformed)
        self.assertNotIn("Wake_Up_Time", transformed)
        self.assertAlmostEqual(transformed.loc[0, "Sleep_Time_Sin"], 0.0, places=12)
        self.assertAlmostEqual(transformed.loc[0, "Sleep_Time_Cos"], 1.0, places=12)
        self.assertEqual(transformed.loc[0, "Age"], 30)


if __name__ == "__main__":
    unittest.main()
