from __future__ import annotations

from .ppca_rs import *

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np


__version__ = "0.5.1"


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
        prior: Optional[Prior] = None,
        state_size: int,
        n_iters: int = 10,
        metric: Literal["aic"] | Literal["bic"] | Literal["llk"] = "aic",
        quiet: bool = False,
    ) -> PPCAModel:
        """
        Trains a PPCA model for a given state size for a given number of iterations. Use
        `smooth_factor` to control for overfit of dimensions with few samples.
        """
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
            model = (
                model.iterate_with_prior(self.dataset, prior)
                if prior is not None
                else model.iterate(self.dataset)
            )

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
        prior: Optional[Prior] = None,
        n_models: int,
        state_size: int,
        n_iters: int = 10,
        metric: Literal["aic"] | Literal["bic"] | Literal["llk"] = "aic",
        quiet: bool = False,
    ) -> PPCAMix:
        """
        Trains a PPCA mix model for a given state size for a given number of iterations. Use
        `smooth_factor` to control for overfit of dimensions with few samples.
        """
        model = start or self.__init(n_models, state_size)

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

            model = (
                model.iterate_with_prior(self.dataset, prior)
                if prior is not None
                else model.iterate(self.dataset)
            )

        return model.to_canonical()


@dataclass
class DataFrameAdapter:
    """Utility class to facilitate the transformation of DataFrames into Datasets."""

    keys: List[str]
    """A key that will uniquely define a sample inside the DataFrame"""
    dimensions: List[str]
    """The columns that will define the dimensions of the output space."""
    metric: str
    """The metric that will populate the output space"""
    dimension_idx: Any
    """
    The mapping between dimensions and dimension indexes. A column called `__dim_idx`
    contains the array index for a given set of dimensions.
    """
    sample_idx: Any
    """
    The mapping between key values and sample indexes.  A column called `__sample_idx`
    contains the array index for a given sample.
    """
    dataset: Dataset
    """The mapped dataset."""
    origin: Literal["pandas"] | Literal["polars"]
    """Which library was used for the adaption"""

    @classmethod
    def from_pandas(
        cls,
        df,
        *,
        keys: List[str],
        dimensions: Optional[List[str]] = None,
        dimension_idx=None,
        metric: str,
    ) -> DataFrameAdapter:
        """
        Adapts a Pandas DataFrame into a Dataset, given the specification. Since `ppca_rs`
        does not explicitely depend on `pandas`, it uses ducktyping. Be sure you have
        `pandas` installed before using this function.
        """
        import pandas as pd

        # This creates a dimension indexing that is __hopefully__ reproducible.
        if dimension_idx is None:
            dimension_idx = (
                df[dimensions]
                .drop_duplicates()
                .sort_values(dimensions)
                .reset_index(drop=True)
            )
            dimension_idx.index.name = "__dim_idx"
            dimension_idx = dimension_idx.reset_index()
        elif dimensions is None:
            dimensions = [
                column for column in dimension_idx.columns if column != "__dim_idx"
            ]

        # Join and group!
        grouped = df.merge(dimension_idx, on=dimensions).groupby(keys)

        # Create an empty dataset...
        output_size = len(dimension_idx)
        dataset_len = len(grouped)
        dataset = np.repeat(np.nan, dataset_len * output_size).reshape(
            (dataset_len, -1)
        )
        sample_idx = []

        # ... then populate it!
        for i, (_, chunk) in enumerate(grouped):
            dataset[i, chunk["__dim_idx"]] = chunk[metric]

        sample_idx = grouped[[]].count().reset_index()
        sample_idx.index.name = "__sample_idx"
        sample_idx = sample_idx.reset_index()[[*keys, "__sample_idx"]]

        # done.
        return cls(
            keys,
            dimensions,
            metric,
            dimension_idx,
            sample_idx,
            Dataset(dataset),
            origin="pandas",
        )

    @classmethod
    def from_polars(
        cls,
        df,
        *,
        keys: List[str],
        dimensions: Optional[List[str]] = None,
        dimension_idx=None,
        metric: str,
    ) -> DataFrameAdapter:
        import polars as pl

        if dimension_idx is None:
            dimension_idx = (
                df.lazy()
                .select(dimensions)
                .unique(maintain_order=False)
                .sort(dimensions)
                .with_row_count("__dim_idx")
                .collect()
            )
        elif dimensions is None:
            dimensions = [
                column for column in dimension_idx.columns if column != "__dim_idx"
            ]

        samples = (
            df.lazy()
            .join(
                dimension_idx.lazy(),
                on=dimensions,
            )
            .groupby(keys)
            .agg([pl.col("__dim_idx"), pl.col(metric)])
            .with_row_count("__sample_idx")
            .collect()
        )

        sample_idx = samples.select([*keys, "__sample_idx"])

        # Create an empty dataset...
        output_size = len(dimension_idx)
        dataset_len = len(samples)
        dataset = np.repeat(np.nan, dataset_len * output_size).reshape(
            (dataset_len, -1)
        )

        # ... then populate it!
        for i, dims, vals in zip(
            samples["__sample_idx"], samples["__dim_idx"], samples[metric]
        ):
            dataset[i, dims] = vals

        # done.
        return cls(
            keys,
            dimensions,
            metric,
            dimension_idx,
            sample_idx,
            Dataset(dataset),
            origin="polars",
        )

    def description(self) -> DataFrameAdapterDescription:
        """Creates a description of this adapter that is suitable for"""
        if self.origin == "pandas":
            return DataFrameAdapterDescription(
                keys=self.keys,
                dimensions=self.dimensions,
                metric=self.metric,
                dimension_idx=[
                    [getattr(tup, column) for column in self.dimensions]
                    for tup in self.dimension_idx.sort_values("__dim_idx").itertuples()
                ],
            )
        elif self.origin == "polars":
            dimension_idx = self.dimension_idx.sort("__dim_idx")
            return DataFrameAdapterDescription(
                keys=self.keys,
                dimensions=self.dimensions,
                metric=self.metric,
                dimension_idx=[
                    [dimension_idx[column][i] for column in self.dimensions]
                    for i in range(len(dimension_idx))
                ],
            )
        else:
            raise Exception(f"Unknown origin {self.origin}")

    def convert_dataset(self, dataset: Dataset, *, column_name: str):
        return self.convert_datasets({column_name: dataset})

    def convert_datasets(self, datasets: Dict[str, Dataset]):
        data = {
            name: dataset.numpy().reshape((-1,)) for name, dataset in datasets.items()
        }
        sample_idx = np.repeat(
            np.arange(0, len(self.sample_idx), dtype="uint32"), len(self.dimension_idx)
        )
        dim_idx = np.tile(
            np.arange(0, len(self.dimension_idx), dtype="uint32"), len(self.sample_idx)
        )

        if self.origin == "pandas":
            import pandas as pd

            return (
                pd.DataFrame(
                    {
                        **data,
                        "__sample_idx": sample_idx,
                        "__dim_idx": dim_idx,
                    }
                )
                .merge(self.dimension_idx, on="__dim_idx")
                .merge(self.sample_idx, on="__sample_idx")[
                    [
                        *self.keys,
                        *self.dimensions,
                        *datasets.keys(),
                    ]
                ]
            )
        elif self.origin == "polars":
            import polars as pl

            return (
                pl.DataFrame(
                    {
                        **data,
                        "__sample_idx": sample_idx,
                        "__dim_idx": dim_idx,
                    }
                )
                .join(self.dimension_idx, on="__dim_idx")
                .join(self.sample_idx, on="__sample_idx")
                .select(
                    [
                        *self.keys,
                        *self.dimensions,
                        *data.keys(),
                    ]
                )
            )
        else:
            raise Exception(f"Unknown origin {self.origin}")


@dataclass
class DataFrameAdapterDescription:
    """
    Shows how to adapt a DataFrame to a Dataset. This class is suitable for serializing
    and storing, not being constrained by actual data.
    """

    keys: List[str]
    """A key that will uniquely define a sample inside the DataFrame"""
    dimensions: List[str]
    """The columns that will define the dimensions of the output space"""
    metric: str
    """The metric that will populate the output space"""
    dimension_idx: List[List]
    """The index of the values that correspond to each dimension in the output space."""

    @property
    def dimension_idx_pandas(self) -> Any:
        import pandas as pd

        return pd.DataFrame(
            {
                "__dim_idx": np.arange(0, len(self.dimension_idx), dtype="uint32"),
                **{
                    dimension: [item[i] for item in self.dimension_idx]
                    for i, dimension in enumerate(self.dimensions)
                },
            }
        )

    @property
    def dimension_idx_polars(self) -> Any:
        import polars as pl

        return pl.DataFrame(
            {
                "__dim_idx": np.arange(0, len(self.dimension_idx), dtype="uint32"),
                **{
                    dimension: [item[i] for item in self.dimension_idx]
                    for i, dimension in enumerate(self.dimensions)
                },
            }
        )

    @classmethod
    def from_json(cls, value: dict) -> DataFrameAdapterDescription:
        return cls(**value)

    def to_json(self) -> dict:
        return {
            "keys": self.keys,
            "dimensions": self.dimensions,
            "metric": self.metric,
            "dimension_idx": self.dimension_idx,
        }

    def adapt_pandas(
        self,
        df,
    ) -> DataFrameAdapter:
        return DataFrameAdapter.from_pandas(
            df,
            keys=self.keys,
            dimension_idx=self.dimension_idx_pandas,
            metric=self.metric,
        )

    def adapt_polars(
        self,
        df,
    ) -> DataFrameAdapter:
        return DataFrameAdapter.from_polars(
            df,
            keys=self.keys,
            dimension_idx=self.dimension_idx_polars,
            metric=self.metric,
        )
