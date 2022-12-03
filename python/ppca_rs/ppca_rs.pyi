from typing import List, Optional

import numpy as np

class Dataset:
    """
    Represents a dataset. This is a wrapper over a 2D array of dimensions
    (n_samples, n_features).

    ## Note

    All arrays involved have to be of data type `float32`.
    """

    def __init__(self, ndarray: np.ndarray) -> None: ...
    def numpy(self) -> np.ndarray:
        """Returns the underlying dataset as a 2D numpy array."""
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    def empty_dimensions(self) -> List[int]:
        """
        Returns the dimensions which have only masked values for all samples in this
        dataset.
        """
    def output_size(self) -> Optional[int]:
        """
        Returns the size of each sample in this dataset, if the dataset is not empty.
        Else, returns `None`.
        """

class InferredMasked:
    """
    A class containing the result of the Bayesian inference step in `PPCAModel.infer`.
    """

    def states(self) -> np.ndarray:
        """The inferred mean value for each sample."""
    def covariances(self) -> List[np.ndarray]:
        """
        The covariance matrices for each sample. The covariances here can change from
        sample to sample, depending on the mask. If there is lots of masking in a sample,
        the covariance will be overall bigger.
        """
    def smoothed(self, model: PPCAModel) -> Dataset:
        """
        The smoothed output values.
        """
    def smoothed_covariances(self, model: PPCAModel) -> Dataset:
        """
        The covariance for the smoothed output values.
        """
    def smoothed_covariances_diagonal(self, model: PPCAModel) -> Dataset:
        """
        Returns an _approximation_ of the smoothed output covariance matrix, treating each masked
        output as an independent normal distribution.

        # Note

        Use this not to get lost with big matrices in the output, losing CPU, memory and
        hair.
        """
    def extrapolated(self, model: PPCAModel, dataset: Dataset) -> Dataset:
        """
        The extrapolated output values.
        """
    def extrapolated_covariances(self, model: PPCAModel, dataset: Dataset) -> Dataset:
        """
        The covariance for the extraplated values.
        """
    def extrapolated_covariances_diagonal(
        self, model: PPCAModel, dataset: Dataset
    ) -> Dataset:
        """
        Returns an _approximation_ of the extrapolated output covariance matrix, treating each masked
        output as an independent normal distribution.

        # Note

        Use this not to get lost with big matrices in the output, losing CPU, memory and
        hair.
        """

class PPCAModel:
    """
    A PPCA model: each sample for this model behaves according to the following
    statistical latent variable model.
    ```
    x ~ N(0; I(nxn))
    y = C * x + y0 + noise
    noise ~ N(0; sgima ^ 2 * I(mxm))
    ```
    Here, `x` is the latent state, y is the observed sample, that is an affine
    transformation of the hidden state contaminated by isotropic noise.

    ## Note

    All arrays involved have to be of data type `float32`.
    """

    def __init__(
        self, isotropic_noise: float, transform: np.ndarray, mean: np.ndarray
    ) -> None: ...

    transform: np.ndarray
    """The linear transformation from hidden state space to output space."""
    isotropic_noise: float
    """The standard deviation of the noise in the output space."""
    mean: np.ndarray
    """Then center of mass of the distribution in the output space."""
    singular_values: np.ndarray
    """
    The relative strength of each hidden variable on the output. This is equivalent to the
    eigenvalues in the standard PCA.
    """
    output_size: int
    """The number of features for this model."""
    state_size: int
    """The number of hidden values for this model."""
    n_parameters: int
    """The total number of parameters involved in training (used for information criteria)."""
    transform: np.ndarray

    @staticmethod
    def load(b: bytes) -> PPCAModel:
        """Loads a PPCA model from binary data. Use this if you want to avoid picking."""
    def dump(self) -> bytes:
        """Encodes the PPCA model into binary data. Use this if you want to avoid picking."""
    @classmethod
    def init(cls, n_states: int) -> PPCAModel:
        """Creates an uninformed model to seed the trainment."""
    def __repr__(self) -> str: ...
    def llk(self, dataset: Dataset) -> float:
        """
        Calculates the log-probability of a given masked dataset according to the current
        model.
        """
    def sample(self, dataset_size: int, mask_prob: float) -> Dataset:
        """
        Samples random outputs from the model and masks each entry according to a
        Bernoulli (coin-toss) distribution of proability `mask_prob` of erasing the
        generated value.
        """
    def infer(self, dataset: Dataset) -> InferredMasked:
        """
        Infers the hidden components for each sample in the dataset. Use this method for
        fine-grain control on the properties you want to extract from the model.
        """
    def smooth(self, dataset: Dataset) -> Dataset:
        """
        Filters a dataset of samples, removing noise from the extant samples and
        inferring the missing samples.
        """
    def extrapolate(self, dataset: Dataset) -> Dataset:
        """Extrapolates the missing values with the most probable values."""
    def iterate(self, dataset: Dataset) -> PPCAModel:
        """
        Makes one iteration of the EM algorithm for the PPCA over an onbserved dataset,
        returning the improved model.
        """
    def to_canonical(self) -> PPCAModel:
        """
        Returns a canonical version of this model. This does not alter the log-probablility
        function nor the quality of the training. All it does is to transform the hidden
        variables.
        """
