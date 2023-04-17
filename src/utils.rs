use nalgebra::{DMatrix, DVector};
use numpy::PyArray2;
use pyo3::{Py, Python};

pub(crate) fn to_nalgebra(py: Python, x: Py<PyArray2<f64>>) -> DMatrix<f64> {
    let array = x.as_ref(py).to_owned_array();
    DMatrix::from_fn(array.nrows(), array.ncols(), |i, j| array[[i, j]])
}

pub(crate) fn to_nalgebra_vector(py: Python, x: Py<PyArray2<f64>>) -> DVector<f64> {
    let array = x.as_ref(py).to_owned_array();
    if array.nrows() == 1 {
        DVector::from_fn(array.ncols(), |i, _| array[[0, i]])
    } else if array.ncols() == 1 {
        DVector::from_fn(array.nrows(), |j, _| array[[j, 0]])
    } else {
        panic!(
            "Expected column- or row- vector; got {}x{} matrix",
            array.nrows(),
            array.ncols()
        );
    }
}
