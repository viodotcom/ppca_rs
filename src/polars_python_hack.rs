//! This is to make polars work. Extracted directly from:
//! https://github.com/pola-rs/polars/tree/master/examples/python_rust_compiled_function
//! (commit 4e35d9c798201069ec556379750c87574eb7c12c)

use arrow::ffi;
use polars::prelude::*;
use polars_arrow::export::arrow;
use pyo3::ffi::Py_uintptr_t;
use pyo3::{PyAny, PyResult};

/// Take an arrow array from python and convert it to a rust arrow array.
/// This operation does not copy data.
pub(crate) fn array_to_rust(arrow_array: &PyAny) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let array = Box::new(ffi::ArrowArray::empty());
    let schema = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = &*array as *const ffi::ArrowArray;
    let schema_ptr = &*schema as *const ffi::ArrowSchema;

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular, `_export_to_c` can go out of bounds
    arrow_array.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        let field = ffi::import_field_from_c(schema.as_ref()).unwrap();
        let array = ffi::import_array_from_c(*array, field.data_type).unwrap();
        Ok(array.into())
    }
}
