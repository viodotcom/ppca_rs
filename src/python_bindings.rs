use bit_vec::BitVec;
use nalgebra::{DMatrix, DMatrixSlice, DVectorSlice};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::{prelude::*, types::PyBytes};
use rayon::prelude::*;

use crate::{
    ppca_model::{Dataset, InferredMasked, MaskedSample, PPCAModel},
    utils::Mask,
};

/// This module is implemented in Rust.
#[pymodule]
pub fn ppca_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PPCAModelWrapper>()?;
    m.add_class::<DatasetWrapper>()?;
    m.add_class::<InferredMaskedBatch>()?;
    Ok(())
}

#[pyclass]
#[pyo3(name = "Dataset")]
struct DatasetWrapper(Dataset);

#[pymethods]
impl DatasetWrapper {
    #[new]
    fn new(py: Python, ndarray: PyReadonlyArray2<f64>) -> PyResult<DatasetWrapper> {
        let n_samples = ndarray.shape()[0];
        let output_size = ndarray.shape()[1];
        let array_view = ndarray.as_array();
        let iter_sample =
            |sample_id| (0..output_size).map(move |dimension| array_view[(sample_id, dimension)]);

        let data = py.allow_threads(|| {
            (0..n_samples)
                .into_par_iter()
                .map(|sample_id| {
                    let data = iter_sample(sample_id).collect::<Vec<_>>().into();
                    let mask = iter_sample(sample_id)
                        .map(f64::is_finite)
                        .collect::<BitVec>();
                    MaskedSample::new(data, Mask(mask))
                })
                .collect()
        });

        Ok(DatasetWrapper(Dataset::new(data)))
    }

    fn numpy(&self, py: Python) -> Py<PyArray2<f64>> {
        let rows = py.allow_threads(|| {
            self.0
                .data
                .iter()
                .par_bridge()
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
}

#[pyclass]
#[pyo3(name = "InferredMasked")]
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
            .map(|inferred| inferred.covariace().to_pyarray(py).to_owned())
            .collect()
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
        mean: Py<PyArray1<f64>>,
    ) -> PyResult<PPCAModelWrapper> {
        Ok(PPCAModelWrapper(PPCAModel::new(
            isotropic_noise,
            (transform
                .as_ref(py)
                .try_readonly()?
                .try_as_matrix()
                .unwrap() as DMatrixSlice<f64>)
                .into_owned(),
            (mean.as_ref(py).try_readonly()?.try_as_matrix().unwrap() as DVectorSlice<f64>)
                .into_owned(),
        )))
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
    fn transform(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.0
            .output_covariance()
            .transform
            .to_pyarray(py)
            .to_owned()
    }

    #[getter]
    fn isotropic_noise(&self) -> f64 {
        self.0.output_covariance().isotropic_noise
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

    fn sample(&self, py: Python<'_>, dataset_size: usize, mask_prob: f64) -> DatasetWrapper {
        py.allow_threads(|| DatasetWrapper(self.0.sample(dataset_size, mask_prob)))
    }

    fn infer(&self, py: Python<'_>, dataset: &DatasetWrapper) -> InferredMaskedBatch {
        InferredMaskedBatch {
            data: py.allow_threads(|| self.0.infer(&dataset.0)),
        }
    }

    fn filter_extrapolate(&self, py: Python<'_>, dataset: &DatasetWrapper) -> DatasetWrapper {
        py.allow_threads(|| DatasetWrapper(self.0.filter_extrapolate(&dataset.0)))
    }

    fn extrapolate(&self, py: Python<'_>, dataset: &DatasetWrapper) -> DatasetWrapper {
        py.allow_threads(|| DatasetWrapper(self.0.extrapolate(&dataset.0)))
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
                self.0 = bincode::deserialize(s.as_bytes()).unwrap();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &bincode::serialize(&self.0).unwrap()).to_object(py))
    }

    pub fn __getnewargs__(
        &self,
        py: Python<'_>,
    ) -> PyResult<(f64, Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
        Ok((self.isotropic_noise(), self.transform(py), self.mean(py)))
    }
}
