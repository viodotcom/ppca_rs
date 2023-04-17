use nalgebra::DMatrix;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{prelude::*, types::PyBytes};
use rand_distr::Distribution;
use rayon::prelude::*;

use ppca::{
    Dataset, InferredMasked, InferredMaskedMix, MaskedSample, PPCAMix, PPCAModel, PosteriorSampler,
    PosteriorSamplerMix, Prior,
};

use crate::utils::{to_nalgebra, to_nalgebra_vector};

/// This module is implemented in Rust.
#[pymodule]
pub fn ppca_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PriorWrapper>()?;
    m.add_class::<PPCAModelWrapper>()?;
    m.add_class::<DatasetWrapper>()?;
    m.add_class::<InferredMaskedBatch>()?;
    m.add_class::<PosteriorSamplerBatch>()?;
    m.add_class::<PPCAMixWrapper>()?;
    m.add_class::<InferredMaskedMixBatch>()?;
    m.add_class::<PosteriorSamplerMixBatch>()?;
    Ok(())
}

#[pyclass]
#[pyo3(name = "Dataset", module = "ppca_rs")]
struct DatasetWrapper(Dataset);

#[pymethods]
impl DatasetWrapper {
    #[new]
    #[pyo3(signature = (ndarray, weights = None))]
    fn new(
        py: Python,
        ndarray: PyReadonlyArray2<f64>,
        weights: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<DatasetWrapper> {
        let n_samples = ndarray.shape()[0];
        let output_size = ndarray.shape()[1];
        let array_view = ndarray.as_array();
        let iter_sample =
            |sample_id| (0..output_size).map(move |dimension| array_view[(sample_id, dimension)]);

        let data = py.allow_threads(|| {
            (0..n_samples)
                .map(|sample_id| {
                    let data = iter_sample(sample_id).collect::<Vec<_>>().into();
                    MaskedSample::mask_non_finite(data)
                })
                .collect()
        });

        if let Some(weights) = weights {
            Ok(DatasetWrapper(Dataset::new_with_weights(
                data,
                weights.as_array().iter().copied().collect(),
            )))
        } else {
            Ok(DatasetWrapper(Dataset::new(data)))
        }
    }

    #[staticmethod]
    fn load(bytes: &PyBytes) -> PyResult<DatasetWrapper> {
        Ok(DatasetWrapper(
            bincode::deserialize(bytes.as_bytes())
                .map_err(|err| pyo3::exceptions::PyException::new_err(err.to_string()))?,
        ))
    }

    fn dump<'a>(&self, py: Python<'a>) -> &'a PyBytes {
        PyBytes::new(
            py,
            &bincode::serialize(&self.0).expect("can always serialize PPCA model"),
        )
    }

    fn numpy(&self, py: Python) -> Py<PyArray2<f64>> {
        let rows = py.allow_threads(|| {
            self.0
                .data
                .par_iter()
                .map(MaskedSample::masked_vector)
                .collect::<Vec<_>>()
        });

        let matrix = DMatrix::from_columns(&rows).transpose();
        matrix.to_pyarray(py).to_owned()
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn output_size(&self) -> Option<usize> {
        self.0.output_size()
    }

    fn empty_dimensions(&self) -> Vec<usize> {
        self.0.empty_dimensions()
    }

    fn weights(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.0.weights.to_pyarray(py).to_owned()
    }

    fn chunks(slf: Py<Self>, py: Python, chunks: usize) -> DatasetChunks {
        let length = slf.borrow(py).0.len();
        DatasetChunks {
            stride: (length as f64 / chunks as f64).ceil() as usize,
            length,
            position: 0,
            dataset: slf,
        }
    }

    #[staticmethod]
    fn concat(list: Vec<Py<DatasetWrapper>>, py: Python) -> DatasetWrapper {
        let length = list.iter().map(|item| item.borrow(py).0.len()).sum();
        let mut data = Vec::with_capacity(length);
        let mut weights = Vec::with_capacity(length);

        for item in list {
            let dataset = &item.borrow(py).0;
            data.extend(dataset.data.iter().cloned());
            weights.extend(dataset.weights.iter().cloned());
        }

        DatasetWrapper(Dataset::new_with_weights(data, weights))
    }
}

#[pyclass]
#[pyo3(name = "DatasetChunks", module = "ppca_rs")]
struct DatasetChunks {
    stride: usize,
    length: usize,
    position: usize,
    dataset: Py<DatasetWrapper>,
}

#[pymethods]
impl DatasetChunks {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> Option<DatasetWrapper> {
        if self.position < self.length {
            let range = || self.position..usize::min(self.length, self.position + self.stride);
            let dataset = &self.dataset.borrow(py).0;
            let data = &dataset.data[range()];
            let weights = &dataset.weights[range()];
            let slice = Dataset::new_with_weights(data.to_owned(), weights.to_owned());

            self.position += self.stride;

            Some(DatasetWrapper(slice))
        } else {
            None
        }
    }
}

#[pyclass]
#[pyo3(name = "Prior", module = "ppca_rs")]
struct PriorWrapper(Prior);

#[pymethods]
impl PriorWrapper {
    #[new]
    pub fn new() -> PriorWrapper {
        PriorWrapper(Prior::default())
    }

    pub fn with_mean_prior(
        &self,
        py: Python,
        mean: Py<PyArray2<f64>>,
        mean_covariance: Py<PyArray2<f64>>,
    ) -> PyResult<Self> {
        let new = self.0.clone().with_mean_prior(
            to_nalgebra_vector(py, mean),
            to_nalgebra(py, mean_covariance),
        );
        Ok(PriorWrapper(new))
    }

    pub fn with_isotropic_noise_prior(&self, alpha: f64, beta: f64) -> Self {
        let new = self.0.clone().with_isotropic_noise_prior(alpha, beta);
        PriorWrapper(new)
    }

    pub fn with_transformation_precision(&self, precision: f64) -> Self {
        let new = self.0.clone().with_transformation_precision(precision);
        PriorWrapper(new)
    }
}

#[pyclass]
#[pyo3(name = "InferredMasked", module = "ppca_rs")]
struct InferredMaskedBatch {
    data: Vec<InferredMasked>,
}

#[pymethods]
impl InferredMaskedBatch {
    fn states(&self, py: Python) -> Py<PyArray2<f64>> {
        if self.data.len() == 0 {
            return DMatrix::<f64>::zeros(0, 0).to_pyarray(py).to_owned();
        }

        let rows = py.allow_threads(|| {
            self.data
                .par_iter()
                .map(|inferred| inferred.state().data.as_vec())
                .flatten()
                .copied()
                .collect::<Vec<_>>()
        });
        let matrix = DMatrix::from_row_slice(self.data.len(), self.data[0].state().len(), &rows);
        matrix.to_pyarray(py).to_owned()
    }

    fn covariances(&self, py: Python) -> Vec<Py<PyArray2<f64>>> {
        // No par iter for you because Python is not Sync.
        self.data
            .iter()
            .map(|inferred| inferred.covariance().to_pyarray(py).to_owned())
            .collect()
    }

    fn smoothed(&self, py: Python, ppca: &PPCAModelWrapper) -> DatasetWrapper {
        let outputs: Dataset = py.allow_threads(|| {
            self.data
                .par_iter()
                .map(|inferred| inferred.smoothed(&ppca.0))
                .map(MaskedSample::unmasked)
                .collect::<Vec<_>>()
                .into()
        });

        DatasetWrapper(outputs)
    }

    fn extrapolated(
        &self,
        py: Python,
        ppca: &PPCAModelWrapper,
        dataset: &DatasetWrapper,
    ) -> DatasetWrapper {
        let outputs: Dataset = py.allow_threads(|| {
            self.data
                .par_iter()
                .zip(&*dataset.0.data)
                .map(|(inferred, sample)| {
                    MaskedSample::unmasked(inferred.extrapolated(&ppca.0, sample))
                })
                .collect::<Vec<_>>()
                .into()
        });

        DatasetWrapper(outputs)
    }

    fn smoothed_covariances(&self, py: Python, ppca: &PPCAModelWrapper) -> Vec<Py<PyArray2<f64>>> {
        // No par iter for you because Python is not Sync.
        self.data
            .iter()
            .map(|inferred| {
                inferred
                    .smoothed_covariance(&ppca.0)
                    .to_pyarray(py)
                    .to_owned()
            })
            .collect()
    }

    fn smoothed_covariances_diagonal(&self, py: Python, ppca: &PPCAModelWrapper) -> DatasetWrapper {
        let output_covariances_diagonal: Dataset = py.allow_threads(|| {
            self.data
                .par_iter()
                .map(|inferred| inferred.smoothed_covariance_diagonal(&ppca.0))
                .map(MaskedSample::unmasked)
                .collect::<Vec<_>>()
                .into()
        });

        DatasetWrapper(output_covariances_diagonal)
    }

    fn extrapolated_covariances(
        &self,
        py: Python,
        ppca: &PPCAModelWrapper,
        dataset: &DatasetWrapper,
    ) -> Vec<Py<PyArray2<f64>>> {
        // No par iter for you because Python is not Sync.
        self.data
            .iter()
            .zip(&*dataset.0.data)
            .map(|(inferred, sample)| {
                inferred
                    .extrapolated_covariance(&ppca.0, sample)
                    .to_pyarray(py)
                    .to_owned()
            })
            .collect()
    }

    fn extrapolated_covariances_diagonal(
        &self,
        py: Python,
        ppca: &PPCAModelWrapper,
        dataset: &DatasetWrapper,
    ) -> DatasetWrapper {
        let output_covariances_diagonal: Dataset = py.allow_threads(|| {
            self.data
                .par_iter()
                .zip(&*dataset.0.data)
                .map(|(inferred, sample)| {
                    inferred.extrapolated_covariance_diagonal(&ppca.0, sample)
                })
                .map(MaskedSample::unmasked)
                .collect::<Vec<_>>()
                .into()
        });

        DatasetWrapper(output_covariances_diagonal)
    }

    fn posterior_sampler(&self, py: Python) -> PosteriorSamplerBatch {
        let posteriors = py.allow_threads(|| {
            self.data
                .par_iter()
                .map(|sample| sample.posterior_sampler())
                .collect::<Vec<_>>()
        });

        PosteriorSamplerBatch { posteriors }
    }
}

#[pyclass]
#[pyo3(name = "PosteriorSampler", module = "ppca_rs")]
struct PosteriorSamplerBatch {
    posteriors: Vec<PosteriorSampler>,
}

#[pymethods]
impl PosteriorSamplerBatch {
    fn sample(&self, py: Python) -> DatasetWrapper {
        let samples = py.allow_threads(|| {
            self.posteriors
                .par_iter()
                .map(|sample| MaskedSample::unmasked(sample.sample(&mut rand::thread_rng())))
                .collect::<Dataset>()
        });

        DatasetWrapper(samples)
    }
}

#[pyclass]
#[pyo3(name = "PPCAModel", module = "ppca_rs")]
#[derive(Debug, Clone)]
struct PPCAModelWrapper(PPCAModel);

#[pymethods]
impl PPCAModelWrapper {
    #[new]
    fn new(
        py: Python<'_>,
        isotropic_noise: f64,
        transform: Py<PyArray2<f64>>,
        mean: Py<PyArray2<f64>>,
    ) -> PyResult<PPCAModelWrapper> {
        Ok(PPCAModelWrapper(PPCAModel::new(
            isotropic_noise,
            to_nalgebra(py, transform),
            to_nalgebra_vector(py, mean),
        )))
    }

    #[staticmethod]
    fn load(bytes: &PyBytes) -> PyResult<PPCAModelWrapper> {
        Ok(PPCAModelWrapper(
            bincode::deserialize(bytes.as_bytes())
                .map_err(|err| pyo3::exceptions::PyException::new_err(err.to_string()))?,
        ))
    }

    fn dump<'a>(&self, py: Python<'a>) -> &'a PyBytes {
        PyBytes::new(
            py,
            &bincode::serialize(&self.0).expect("can always serialize PPCA model"),
        )
    }

    #[getter]
    fn output_size(&self) -> usize {
        self.0.output_size()
    }

    #[getter]
    fn state_size(&self) -> usize {
        self.0.state_size()
    }

    #[getter]
    fn n_parameters(&self) -> usize {
        self.0.n_parameters()
    }

    #[getter]
    fn singular_values(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.0
            .singular_values()
            .to_pyarray(py)
            .reshape((self.0.state_size(),))
            .expect("resizing is valid")
            .to_owned()
    }

    #[getter]
    fn transform(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.0.transform().to_pyarray(py).to_owned()
    }

    #[getter]
    fn isotropic_noise(&self) -> f64 {
        self.0.isotropic_noise()
    }

    #[getter]
    fn mean(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.0
            .mean()
            .transpose()
            .to_pyarray(py)
            .reshape((self.0.mean().len(),))
            .expect("resizing is valid")
            .to_owned()
    }

    #[staticmethod]
    fn init(state_size: usize, dataset: &DatasetWrapper) -> PPCAModelWrapper {
        PPCAModelWrapper(PPCAModel::init(state_size, &dataset.0))
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "PPCAModel(\
                isotropic_noise={}, \
                transform=array({}, dtype=\"float32\"), \
                mean=narray({}, dtype=\"float32\"))",
            self.isotropic_noise(),
            self.transform(py),
            self.mean(py),
        )
    }

    fn llk(&self, py: Python<'_>, dataset: &DatasetWrapper) -> f64 {
        py.allow_threads(|| self.0.llk(&dataset.0))
    }

    fn llks(&self, py: Python<'_>, dataset: &DatasetWrapper) -> Py<PyArray1<f64>> {
        let llks = py.allow_threads(|| self.0.llks(&dataset.0));
        llks.to_pyarray(py)
            .reshape(llks.len())
            .expect("can reshape")
            .to_owned()
    }

    fn sample(&self, py: Python<'_>, dataset_size: usize, mask_prob: f64) -> DatasetWrapper {
        py.allow_threads(|| DatasetWrapper(self.0.sample(dataset_size, mask_prob)))
    }

    fn infer(&self, py: Python<'_>, dataset: &DatasetWrapper) -> InferredMaskedBatch {
        InferredMaskedBatch {
            data: py.allow_threads(|| self.0.infer(&dataset.0)),
        }
    }

    fn smooth(&self, py: Python<'_>, dataset: &DatasetWrapper) -> DatasetWrapper {
        py.allow_threads(|| DatasetWrapper(self.0.smooth(&dataset.0)))
    }

    fn extrapolate(&self, py: Python<'_>, dataset: &DatasetWrapper) -> DatasetWrapper {
        py.allow_threads(|| DatasetWrapper(self.0.extrapolate(&dataset.0)))
    }

    fn iterate_with_prior(
        &self,
        py: Python<'_>,
        dataset: &DatasetWrapper,
        prior: &PriorWrapper,
    ) -> PPCAModelWrapper {
        py.allow_threads(|| PPCAModelWrapper(self.0.iterate_with_prior(&dataset.0, &prior.0)))
    }

    fn iterate(&self, py: Python<'_>, dataset: &DatasetWrapper) -> PPCAModelWrapper {
        py.allow_threads(|| PPCAModelWrapper(self.0.iterate(&dataset.0)))
    }

    fn to_canonical(&self, py: Python<'_>) -> PPCAModelWrapper {
        py.allow_threads(|| PPCAModelWrapper(self.0.to_canonical()))
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.0 = PPCAModelWrapper::load(s)?.0;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.dump(py).to_object(py))
    }

    pub fn __getnewargs__(
        &self,
        py: Python<'_>,
    ) -> PyResult<(f64, Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
        Ok((self.isotropic_noise(), self.transform(py), self.mean(py)))
    }
}

#[pyclass]
#[pyo3(name = "PPCAMix", module = "ppca_rs")]
#[derive(Debug, Clone)]
struct PPCAMixWrapper(PPCAMix);

#[pymethods]
impl PPCAMixWrapper {
    #[new]
    pub fn new(
        models: Vec<PPCAModelWrapper>,
        log_weights: PyReadonlyArray1<f64>,
    ) -> PPCAMixWrapper {
        PPCAMixWrapper(PPCAMix::new(
            models
                .into_iter()
                .map(|PPCAModelWrapper(model)| model)
                .collect(),
            log_weights
                .as_array()
                .into_iter()
                .copied()
                .collect::<Vec<_>>()
                .into(),
        ))
    }

    #[staticmethod]
    fn init(
        py: Python,
        n_models: usize,
        state_size: usize,
        dataset: &DatasetWrapper,
    ) -> PPCAMixWrapper {
        py.allow_threads(|| PPCAMixWrapper(PPCAMix::init(n_models, state_size, &dataset.0)))
    }

    #[staticmethod]
    fn load(bytes: &PyBytes) -> PyResult<PPCAMixWrapper> {
        Ok(PPCAMixWrapper(
            bincode::deserialize(bytes.as_bytes())
                .map_err(|err| pyo3::exceptions::PyException::new_err(err.to_string()))?,
        ))
    }

    fn dump<'a>(&self, py: Python<'a>) -> &'a PyBytes {
        PyBytes::new(
            py,
            &bincode::serialize(&self.0).expect("can always serialize PPCA model"),
        )
    }

    #[getter]
    fn output_size(&self) -> usize {
        self.0.output_size()
    }

    #[getter]
    fn state_sizes(&self) -> Vec<usize> {
        self.0.state_sizes()
    }

    #[getter]
    fn n_parameters(&self) -> usize {
        self.0.n_parameters()
    }

    #[getter]
    fn models(&self) -> Vec<PPCAModelWrapper> {
        self.0
            .models()
            .iter()
            .cloned()
            .map(PPCAModelWrapper)
            .collect()
    }

    #[getter]
    fn log_weights(&self, py: Python) -> Py<PyArray1<f64>> {
        self.0
            .log_weights()
            .clone()
            .to_pyarray(py)
            .reshape(self.0.log_weights().len())
            .expect("can reshape")
            .to_owned()
    }

    #[getter]
    fn weights(&self, py: Python) -> Py<PyArray1<f64>> {
        self.0
            .weights()
            .clone()
            .to_pyarray(py)
            .reshape(self.0.log_weights().len())
            .expect("can reshape")
            .to_owned()
    }

    pub fn llks(&self, py: Python, dataset: &DatasetWrapper) -> Py<PyArray1<f64>> {
        let llks = py.allow_threads(|| self.0.llks(&dataset.0));
        llks.to_pyarray(py)
            .reshape(llks.len())
            .expect("can reshape")
            .to_owned()
    }

    pub fn llk(&self, py: Python, dataset: &DatasetWrapper) -> f64 {
        py.allow_threads(|| self.0.llk(&dataset.0))
    }

    pub fn sample(
        &self,
        py: Python<'_>,
        dataset_size: usize,
        mask_probability: f64,
    ) -> DatasetWrapper {
        DatasetWrapper(py.allow_threads(|| self.0.sample(dataset_size, mask_probability)))
    }

    pub fn infer_cluster(&self, py: Python, dataset: &DatasetWrapper) -> Py<PyArray2<f64>> {
        py.allow_threads(|| self.0.infer_cluster(&dataset.0))
            .to_pyarray(py)
            .to_owned()
    }

    pub fn infer(&self, py: Python<'_>, dataset: &DatasetWrapper) -> InferredMaskedMixBatch {
        InferredMaskedMixBatch {
            data: py.allow_threads(|| self.0.infer(&dataset.0)),
        }
    }

    pub fn smooth(&self, py: Python, dataset: &DatasetWrapper) -> DatasetWrapper {
        DatasetWrapper(py.allow_threads(|| self.0.smooth(&dataset.0)))
    }

    pub fn extrapolate(&self, py: Python, dataset: &DatasetWrapper) -> DatasetWrapper {
        DatasetWrapper(py.allow_threads(|| self.0.extrapolate(&dataset.0)))
    }

    fn iterate_with_prior(
        &self,
        py: Python<'_>,
        dataset: &DatasetWrapper,
        prior: &PriorWrapper,
    ) -> PPCAMixWrapper {
        py.allow_threads(|| PPCAMixWrapper(self.0.iterate_with_prior(&dataset.0, &prior.0)))
    }

    pub fn iterate(&self, py: Python, dataset: &DatasetWrapper) -> PPCAMixWrapper {
        PPCAMixWrapper(py.allow_threads(|| self.0.iterate(&dataset.0)))
    }

    pub fn to_canonical(&self, py: Python) -> PPCAMixWrapper {
        PPCAMixWrapper(py.allow_threads(|| self.0.to_canonical()))
    }

    pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.0 = PPCAMixWrapper::load(s)?.0;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.dump(py).to_object(py))
    }

    pub fn __getnewargs__(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Vec<PPCAModelWrapper>, Py<PyArray1<f64>>)> {
        Ok((self.models(), self.log_weights(py)))
    }
}

#[pyclass]
#[pyo3(name = "InferredMaskedMix", module = "ppca_rs")]
struct InferredMaskedMixBatch {
    data: Vec<InferredMaskedMix>,
}

#[pymethods]
impl InferredMaskedMixBatch {
    fn log_posteriors(&self, py: Python) -> Py<PyArray2<f64>> {
        if self.data.len() == 0 {
            return DMatrix::<f64>::zeros(0, 0).to_pyarray(py).to_owned();
        }

        let rows = py.allow_threads(|| {
            self.data
                .par_iter()
                .flat_map(|inferred| inferred.log_posterior().as_slice().to_vec())
                .collect::<Vec<_>>()
        });
        let matrix =
            DMatrix::from_row_slice(self.data.len(), self.data[0].log_posterior().len(), &rows);
        matrix.to_pyarray(py).to_owned()
    }

    fn posteriors(&self, py: Python) -> Py<PyArray2<f64>> {
        if self.data.len() == 0 {
            return DMatrix::<f64>::zeros(0, 0).to_pyarray(py).to_owned();
        }

        let rows = py.allow_threads(|| {
            self.data
                .par_iter()
                .flat_map(|inferred| inferred.posterior().as_slice().to_vec())
                .collect::<Vec<_>>()
        });
        let matrix =
            DMatrix::from_row_slice(self.data.len(), self.data[0].log_posterior().len(), &rows);
        matrix.to_pyarray(py).to_owned()
    }

    fn states(&self, py: Python) -> Py<PyArray2<f64>> {
        if self.data.len() == 0 {
            return DMatrix::<f64>::zeros(0, 0).to_pyarray(py).to_owned();
        }

        let rows = py.allow_threads(|| {
            self.data
                .par_iter()
                .flat_map(|inferred| inferred.state().as_slice().to_vec())
                .collect::<Vec<_>>()
        });
        let matrix = DMatrix::from_row_slice(self.data.len(), self.data[0].state().len(), &rows);
        matrix.to_pyarray(py).to_owned()
    }

    fn covariances(&self, py: Python) -> Vec<Py<PyArray2<f64>>> {
        // No par iter for you because Python is not Sync.
        self.data
            .iter()
            .map(|inferred| inferred.covariance().to_pyarray(py).to_owned())
            .collect()
    }

    fn smoothed(&self, py: Python, ppca: &PPCAMixWrapper) -> DatasetWrapper {
        let outputs: Dataset = py.allow_threads(|| {
            self.data
                .par_iter()
                .map(|inferred| inferred.smoothed(&ppca.0))
                .map(MaskedSample::unmasked)
                .collect::<Vec<_>>()
                .into()
        });

        DatasetWrapper(outputs)
    }

    fn extrapolated(
        &self,
        py: Python,
        ppca: &PPCAMixWrapper,
        dataset: &DatasetWrapper,
    ) -> DatasetWrapper {
        let outputs: Dataset = py.allow_threads(|| {
            self.data
                .par_iter()
                .zip(&*dataset.0.data)
                .map(|(inferred, sample)| {
                    MaskedSample::unmasked(inferred.extrapolated(&ppca.0, sample))
                })
                .collect::<Vec<_>>()
                .into()
        });

        DatasetWrapper(outputs)
    }

    fn smoothed_covariances(&self, py: Python, ppca: &PPCAMixWrapper) -> Vec<Py<PyArray2<f64>>> {
        // No par iter for you because Python is not Sync.
        self.data
            .iter()
            .map(|inferred| {
                inferred
                    .smoothed_covariance(&ppca.0)
                    .to_pyarray(py)
                    .to_owned()
            })
            .collect()
    }

    fn smoothed_covariances_diagonal(&self, py: Python, ppca: &PPCAMixWrapper) -> DatasetWrapper {
        let output_covariances_diagonal: Dataset = py.allow_threads(|| {
            self.data
                .par_iter()
                .map(|inferred| inferred.smoothed_covariance_diagonal(&ppca.0))
                .map(MaskedSample::unmasked)
                .collect::<Vec<_>>()
                .into()
        });

        DatasetWrapper(output_covariances_diagonal)
    }

    fn extrapolated_covariances(
        &self,
        py: Python,
        ppca: &PPCAMixWrapper,
        dataset: &DatasetWrapper,
    ) -> Vec<Py<PyArray2<f64>>> {
        // No par iter for you because Python is not Sync.
        self.data
            .iter()
            .zip(&*dataset.0.data)
            .map(|(inferred, sample)| {
                inferred
                    .extrapolated_covariance(&ppca.0, sample)
                    .to_pyarray(py)
                    .to_owned()
            })
            .collect()
    }

    fn extrapolated_covariances_diagonal(
        &self,
        py: Python,
        ppca: &PPCAMixWrapper,
        dataset: &DatasetWrapper,
    ) -> DatasetWrapper {
        let output_covariances_diagonal: Dataset = py.allow_threads(|| {
            self.data
                .par_iter()
                .zip(&*dataset.0.data)
                .map(|(inferred, sample)| {
                    inferred.extrapolated_covariance_diagonal(&ppca.0, sample)
                })
                .map(MaskedSample::unmasked)
                .collect::<Vec<_>>()
                .into()
        });

        DatasetWrapper(output_covariances_diagonal)
    }

    fn posterior_sampler(&self, py: Python) -> PosteriorSamplerMixBatch {
        let posteriors = py.allow_threads(|| {
            self.data
                .par_iter()
                .map(|sample| sample.posterior_sampler())
                .collect::<Vec<_>>()
        });

        PosteriorSamplerMixBatch { posteriors }
    }
}

#[pyclass]
#[pyo3(name = "PosteriorSamplerMix", module = "ppca_rs")]
struct PosteriorSamplerMixBatch {
    posteriors: Vec<PosteriorSamplerMix>,
}

#[pymethods]
impl PosteriorSamplerMixBatch {
    fn sample(&self, py: Python) -> DatasetWrapper {
        let samples = py.allow_threads(|| {
            self.posteriors
                .par_iter()
                .map(|sample| MaskedSample::unmasked(sample.sample(&mut rand::thread_rng())))
                .collect::<Dataset>()
        });

        DatasetWrapper(samples)
    }
}
