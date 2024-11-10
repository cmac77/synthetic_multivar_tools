"""
categorical_features.py

This module defines classes and functions for generating synthetic categorical features within a hierarchical synthetic
data generation framework. It allows users to specify levels, distributions, and custom configurations for categorical
feature generation.

Classes:
    - CategoricalFeatures: Class to generate categorical features based on the specified levels and distributions.

Functions:
    - add_categorical_features(n_samples: int): Generates categorical features as a pandas DataFrame based on defined
      feature levels and distributions.

Example Usage:
    # Initialize CategoricalFeatures with 2 levels and uniform distribution:
    categorical_feature_generator = CategoricalFeatures(levels=[2, 3], distribution="uniform")
    
    # Generate categorical features for 100 samples:
    df = categorical_feature_generator.add_categorical_features(n_samples=100)
    
    # Example DataFrame Output:
    print(df)
    
    #     feature1 feature2
    # 0          A        X
    # 1          B        Y
    # 2          A        Z
    # 3          B        X
    # ...

Requirements:
    pandas, numpy
"""

# %%
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class CategoricalFeatures:
    """
    A class representing categorical information for synthetic data generation.

    Attributes:
        n_features (int): Number of categorical features to generate.
        levels (List[int]): Number of categories (levels) for each feature. Defaults to binary features if not provided.
        distribution_weights (Optional[Dict[int, List[float]]]): Optional weights for each categorical feature.
            This allows for specifying non-uniform probabilities for category selection.
    """

    n_features: int
    levels: List[int] = field(
        default_factory=lambda: [2]
    )  # Defaults to binary features if not provided
    distribution_weights: Optional[Dict[int, List[float]]] = (
        None  # Optional weights for each categorical feature
    )

    def __post_init__(self):
        """
        Post-initialization to ensure parameters are correctly set.

        - Pads the `levels` list to match `n_features` if necessary.
        - Initializes `distribution_weights` to `None` for all features if not provided.
        """
        if len(self.levels) < self.n_features:
            self.levels = (self.levels + [2] * self.n_features)[
                : self.n_features
            ]  # Pad levels to match n_features
        if self.distribution_weights is None:
            self.distribution_weights = {
                i: None for i in range(self.n_features)
            }

        # Validation to ensure that levels and distribution weights are consistent
        if any(level <= 0 for level in self.levels):
            raise ValueError(
                "All values in `levels` must be greater than zero."
            )

        for feature_idx, weights in (self.distribution_weights or {}).items():
            if weights and len(weights) != self.levels[feature_idx]:
                raise ValueError(
                    f"Distribution weights for feature {feature_idx} must match the number of levels ({self.levels[feature_idx]})."
                )

    def add_categorical_features(self, n_samples: int) -> pd.DataFrame:
        """
        Adds categorical features to the dataset based on CategoricalFeatures.

        Args:
            n_samples (int): Number of data points to generate.

        Returns:
            pd.DataFrame: A DataFrame containing the generated synthetic categorical features.
        """
        # Generate data for each categorical feature
        categorical_data = {}
        for i in range(self.n_features):
            feature_name = f"categorical_feature_{i+1}"
            num_levels = self.levels[i]
            categories = [f"C_{j}" for j in range(num_levels)]

            # If weights are provided for the feature distribution, use them
            weights = self.distribution_weights.get(i)
            if weights and len(weights) != num_levels:
                raise ValueError(
                    f"Distribution weights for {feature_name} do not match the number of levels ({num_levels})."
                )

            # Generate categorical data using specified weights (or uniform if weights is None)
            categorical_data[feature_name] = np.random.choice(
                categories, size=n_samples, p=weights
            )

        # Convert categorical data into a pandas DataFrame
        categorical_df = pd.DataFrame(categorical_data)
        return categorical_df
