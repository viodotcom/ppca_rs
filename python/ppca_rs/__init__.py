from __future__ import annotations

from .ppca_rs import *

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


__version__ = "0.2.0"


@dataclass(frozen=True)
class TrainMetrics:
    llk_per_sample: float
    aic: float
    bic: float


@dataclass
class PPCATrainer:
    """A trainer for a PPCA Model over masked data."""

    # The list of masked samples against which the PPCA will be trained.
    dataset: Dataset

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
        quiet: bool = False,
    ) -> PPCAModel:
        """Trains a PPCA model for a given state size for a given number of iterations."""
        model = start or self.__init(state_size)

        for idx in range(n_iters):
            if not quiet:
                llk = model.llk(self.dataset)
                metrics = TrainMetrics(
                    llk_per_sample=llk / len(self.dataset),
                    aic=2.0 * (model.n_parameters - llk) / len(self.dataset),
                    bic=(llk - model.n_parameters * np.log(len(self.dataset)))
                    / len(self.dataset),
                )
                print(f"Masked PPCA iteration {idx + 1}: aic={metrics.aic}")

            model = model.iterate(self.dataset)

        return model.to_canonical()
