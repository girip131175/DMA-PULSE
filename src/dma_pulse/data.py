"""Data utilities for the published PULSE representation.

The implementation follows the preprocessing operations described in the
DMA-PULSE paper. It intentionally does not ship the PULSE dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PulseConfig:
    """Expected temporal representation from the paper."""

    sequence_length: int = 5
    feature_dim: int = 11


class PulsePreprocessor:
    """Preprocess tabular activity records before sequence construction.

    The caller supplies the dataset-specific column names because the public
    repository does not contain the original PULSE data dictionary.
    """

    def __init__(
        self,
        categorical_columns: Sequence[str] = (),
        continuous_columns: Sequence[str] = (),
        timestamp_column: str | None = None,
        user_column: str | None = None,
    ) -> None:
        self.categorical_columns = list(categorical_columns)
        self.continuous_columns = list(continuous_columns)
        self.timestamp_column = timestamp_column
        self.user_column = user_column
        self.scaler = MinMaxScaler()
        self._fitted = False

    def clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove incomplete rows and normalize timestamp/user identity fields."""
        out = frame.dropna(how="any").copy()

        if self.timestamp_column and self.timestamp_column in out:
            out[self.timestamp_column] = pd.to_datetime(
                out[self.timestamp_column], errors="coerce", utc=True
            )
            out = out.dropna(subset=[self.timestamp_column])
            out[self.timestamp_column] = (
                out[self.timestamp_column].astype("int64") / 1e9
            )

        if self.user_column and self.user_column in out:
            out[self.user_column] = pd.factorize(out[self.user_column])[0]

        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = self.clean(frame)

        if self.continuous_columns:
            out[self.continuous_columns] = self.scaler.fit_transform(
                out[self.continuous_columns]
            )
            self._fitted = True

        if self.categorical_columns:
            out = pd.get_dummies(
                out,
                columns=self.categorical_columns,
                dtype=float,
            )

        return out

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = self.clean(frame)
        if self.continuous_columns:
            if not self._fitted:
                raise RuntimeError("Call fit_transform before transform.")
            out[self.continuous_columns] = self.scaler.transform(
                out[self.continuous_columns]
            )
        if self.categorical_columns:
            out = pd.get_dummies(
                out,
                columns=self.categorical_columns,
                dtype=float,
            )
        return out


def validate_representation(
    x: np.ndarray,
    config: PulseConfig = PulseConfig(),
) -> None:
    """Validate the paper's [samples, 5, 11] sequence convention."""
    if x.ndim != 3:
        raise ValueError(f"Expected a 3-D tensor; received shape {x.shape}.")
    if x.shape[1:] != (config.sequence_length, config.feature_dim):
        raise ValueError(
            "Expected temporal representation "
            f"[N, {config.sequence_length}, {config.feature_dim}], "
            f"received {x.shape}."
        )


def make_sequences(
    values: Iterable[np.ndarray],
    sequence_length: int = 5,
) -> np.ndarray:
    """Group consecutive feature rows into fixed-length temporal sequences."""
    rows = np.asarray(list(values), dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("values must contain 2-D feature rows")
    usable = len(rows) - (len(rows) % sequence_length)
    if usable == 0:
        return np.empty((0, sequence_length, rows.shape[1]), dtype=np.float32)
    return rows[:usable].reshape(-1, sequence_length, rows.shape[1])
