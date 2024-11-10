#%%
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union
from dataclasses import dataclass, field

#%%
@dataclass
class TargetInfo:
    levels: Optional[List[int]] = None
    distributions: Optional[List[str]] = None
    balance_weights: Optional[Dict[int, List[float]]] = None
    custom_labels: Optional[Dict[int, List[Union[int, str]]]] = None
    features_source: Optional[object] = None
    distribution_params: Optional[Dict[int, Dict[str, Union[float, List[float]]]]] = field(default_factory=dict)

    def __post_init__(self):
        if self.custom_labels is None:
            self.custom_labels = {}
        if self.balance_weights is None:
            self.balance_weights = {}
        # Ensure distribution_params is a dictionary
        if self.distribution_params is None:
            self.distribution_params = {}

        if self.features_source:
            self._extract_from_source()

        if self.distributions is None:
            self.distributions = ["categorical"] * len(self.levels)

        if self.balance_weights:
            self._validate_balance_weights()
        if self.custom_labels:
            self._validate_custom_labels()
    
    def _extract_from_source(self):
        if hasattr(self.features_source, 'levels'):
            self.levels = self.features_source.levels
        if hasattr(self.features_source, 'distributions'):
            self.distributions = self.features_source.distributions

    def _validate_balance_weights(self):
        for level, weights in self.balance_weights.items():
            if level >= len(self.levels):
                raise ValueError(f"Level {level} in balance_weights exceeds available levels.")
            if len(weights) != self.levels[level]:
                raise ValueError(f"Level {level} weights do not match the number of categories.")

    def _validate_custom_labels(self):
        for level, labels in self.custom_labels.items():
            if level >= len(self.levels):
                raise ValueError(f"Level {level} in custom_labels exceeds available levels.")
            if len(labels) != self.levels[level]:
                raise ValueError(f"Level {level} labels do not match the number of categories.")
    
    def generate_targets(self, n_samples: int) -> pd.DataFrame:
        targets = []
        for level_idx, level in enumerate(self.levels):
            distribution = self.distributions[level_idx]
            labels = self.custom_labels.get(level_idx, list(range(level)))
            weights = self.balance_weights.get(level_idx, None)

            if distribution == "categorical":
                target_data = np.random.choice(labels, size=n_samples, p=weights)

            elif distribution == "normal":
                mean = self.distribution_params.get(level_idx, {}).get("mean", 0)
                std_dev = self.distribution_params.get(level_idx, {}).get("std_dev", 1)
                target_data = np.random.normal(mean, std_dev, size=n_samples)

            elif distribution == "uniform":
                min_val = self.distribution_params.get(level_idx, {}).get("min", 0)
                max_val = self.distribution_params.get(level_idx, {}).get("max", 1)
                target_data = np.random.uniform(min_val, max_val, size=n_samples)

            elif distribution == "lognormal":
                mean = self.distribution_params.get(level_idx, {}).get("mean", 0)
                std_dev = self.distribution_params.get(level_idx, {}).get("std_dev", 1)
                target_data = np.random.lognormal(mean, std_dev, size=n_samples)

            else:
                raise NotImplementedError(f"Distribution type '{distribution}' is not supported.")

            targets.append(target_data)

        targets_df = pd.DataFrame({f"T_{i}": targets[i] for i in range(len(targets))})
        return targets_df
#%%

