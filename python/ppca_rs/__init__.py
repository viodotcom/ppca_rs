from __future__ import annotations

from .ppca_rs import *

from dataclasses import dataclass
from typing import Any, List, Literal, Optional

import numpy as np


__version__ = "0.2.0"


@dataclass(frozen=True)
class TrainMetrics:
    llk: float
    aic: float
    bic: float


@dataclass
class PPCATrainer:
    """A trainer for a PPCA Model over masked data."""

    dataset: Dataset
    """The list of masked samples against which the PPCA will be trained."""

    def __init(self, state_size: int) -> PPCAModel:
        """Initializes a first guess for the model."""
        init = PPCAModel.init(state_size, self.dataset)
        return init

    def train(
        self,
        *,
        start: Optional[PPCAModel] = None,
        state_size: int,
        n_iters: int = 10,
        metric: Literal["aic"] | Literal["bic"] | Literal["llk"] = "aic",
        quiet: bool = False,
    ) -> PPCAModel:
        """Trains a PPCA model for a given state size for a given number of iterations."""
        model = start or self.__init(state_size)

        for idx in range(n_iters):
            if not quiet:
                llk = model.llk(self.dataset)
                metrics = TrainMetrics(
                    llk=llk / len(self.dataset),
                    aic=2.0 * (model.n_parameters - llk) / len(self.dataset),
                    bic=(llk - model.n_parameters * np.log(len(self.dataset)))
                    / len(self.dataset),
                )
                print(
                    f"Masked PPCA iteration {idx + 1}: {metric}={getattr(metrics, metric)}"
                )

            model = model.iterate(self.dataset)

        return model.to_canonical()


@dataclass
class PPCAMixTrainer:
    """A trainer for a PPCA Mixture Model over masked data."""

    dataset: Dataset
    """The list of masked samples against which the PPCA mixture model will be trained."""

    def __init(self, n_models: int, state_size: int) -> PPCAMix:
        """Initializes a first guess for the model."""
        init = PPCAMix.init(n_models, state_size, self.dataset)
        return init

    def train(
        self,
        *,
        start: Optional[PPCAMix] = None,
        state_size: int,
        n_iters: int = 10,
        metric: Literal["aic"] | Literal["bic"] | Literal["llk"] = "aic",
        quiet: bool = False,
    ) -> PPCAMix:
        """Trains a PPCA model for a given state size for a given number of iterations."""
        model = start or self.__init(state_size)

        for idx in range(n_iters):
            if not quiet:
                llk = model.llk(self.dataset)
                metrics = TrainMetrics(
                    llk=llk / len(self.dataset),
                    aic=2.0 * (model.n_parameters - llk) / len(self.dataset),
                    bic=(llk - model.n_parameters * np.log(len(self.dataset)))
                    / len(self.dataset),
                )
                print(
                    f"Masked PPCA mix iteration {idx + 1}: {metric}={getattr(metrics, metric)}"
                )

            model = model.iterate(self.dataset)

        return model.to_canonical()


@dataclass
class DataFrameAdapter:
    """Utility class to facilitate the transformation of DataFrames into Datasets."""

    key: List[str]
    """A key that will uniquely define a sample inside the DataFrame"""
    dimensions: List[str]
    """The columns that will define the dimensions of the output space"""
    metric: str
    """The metric that will populate the output space"""
    dimensions_idx: Any
    """The mapping between dimensions and dimension indexes."""
    dataset: Dataset
    """The mapped dataset."""

    @classmethod
    def from_pandas(
        cls, df, *, keys: List[str], dimensions: List[str], metric: str
    ) -> DataFrameAdapter:
        """
        Adapts a Pandas DataFrame into a Dataset, given the specification. Since `ppca_rs`
        does not explicitely depend on `pandas`, it uses ducktyping. Be sure you have
        `pandas` installed before using this function.
        """
        # This creates a dimension indexing that is __hopefully__ reproducible.
        dimensions_idx = (
            df[dimensions]
            .drop_duplicates()
            .sort_values(dimensions)
            .reset_index(drop=True)
        )
        dimensions_idx.index.name = "__idx"
        dimensions_idx = dimensions_idx.reset_index()

        # Join and group!
        grouped = df.merge(dimensions_idx, on=dimensions).groupby(keys)

        # Create an empty dataset...
        output_size = len(dimensions_idx)
        dataset_len = len(grouped)
        dataset = np.repeat(np.nan, dataset_len * output_size).reshape((dataset_len, -1))

        # ... then populate it!
        for i, (_, chunk) in enumerate(grouped):
            dataset[i, chunk["__idx"]] = chunk[metric]

        # done.
        return cls(keys, dimensions, metric, dimensions_idx, Dataset(dataset))
