import unittest
import numpy as np
import pandas as pd
from target_info import TargetInfo
from numerical_features import NumericalFeatures, DistanceInfo, ClustersInfo

class TestTargetInfo(unittest.TestCase):

    def test_default_categorical_targets(self):
        """Test default categorical target generation with integer labels."""
        target_generator = TargetInfo(
            levels=[2, 3],
            distributions=["categorical", "categorical"],
            balance_weights={0: [0.7, 0.3], 1: [0.3, 0.4, 0.3]}
        )
        targets_df = target_generator.generate_targets(n_samples=10)
        self.assertEqual(targets_df.shape, (10, 2))
        self.assertListEqual(targets_df.columns.tolist(), ["T_0", "T_1"])

    def test_normal_distribution_targets(self):
        """Test normal distribution target generation."""
        target_generator = TargetInfo(
            levels=[1],
            distributions=["normal"],
            distribution_params={0: {"mean": 50, "std_dev": 10}}
        )
        targets_df = target_generator.generate_targets(n_samples=10)
        self.assertEqual(targets_df.shape, (10, 1))
        self.assertEqual(targets_df.columns.tolist(), ["T_0"])
        self.assertTrue(np.all(targets_df["T_0"].between(20, 80)))  # Approx range check for mean ± 3*std_dev

    def test_uniform_distribution_targets(self):
        """Test uniform distribution target generation."""
        target_generator = TargetInfo(
            levels=[1],
            distributions=["uniform"],
            distribution_params={0: {"min": 10, "max": 20}}
        )
        targets_df = target_generator.generate_targets(n_samples=10)
        self.assertEqual(targets_df.shape, (10, 1))
        self.assertEqual(targets_df.columns.tolist(), ["T_0"])
        self.assertTrue(np.all(targets_df["T_0"].between(10, 20)))

    def test_lognormal_distribution_targets(self):
        """Test lognormal distribution target generation."""
        target_generator = TargetInfo(
            levels=[1],
            distributions=["lognormal"],
            distribution_params={0: {"mean": 1, "std_dev": 0.5}}
        )
        targets_df = target_generator.generate_targets(n_samples=10)
        self.assertEqual(targets_df.shape, (10, 1))
        self.assertEqual(targets_df.columns.tolist(), ["T_0"])
        self.assertTrue(np.all(targets_df["T_0"] > 0))  # Lognormal values should be positive

    def test_hybrid_categorical_and_normal(self):
        """Test hybrid target generation with categorical and normal distributions."""
        target_generator = TargetInfo(
            levels=[2, 1],
            distributions=["categorical", "normal"],
            balance_weights={0: [0.6, 0.4]},
            custom_labels={0: ["low", "high"]},
            distribution_params={1: {"mean": 100, "std_dev": 15}}
        )
        targets_df = target_generator.generate_targets(n_samples=10)
        self.assertEqual(targets_df.shape, (10, 2))
        self.assertEqual(targets_df.columns.tolist(), ["T_0", "T_1"])
        self.assertIn("low", targets_df["T_0"].values)
        self.assertIn("high", targets_df["T_0"].values)
        self.assertTrue(np.all(targets_df["T_1"].between(55, 145)))  # Range based on 3 std devs

    def test_custom_labels_mismatch(self):
        """Test custom labels with a mismatch in level categories, expecting a ValueError."""
        with self.assertRaises(ValueError):
            TargetInfo(
                levels=[2, 3],
                distributions=["categorical", "categorical"],
                custom_labels={0: ["only_one_label"], 1: ["class_1", "class_2", "class_3"]}
            )

if __name__ == "__main__":
    unittest.main()
