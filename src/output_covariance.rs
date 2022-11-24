use nalgebra::{DMatrix, DVector};
use serde_derive::{Deserialize, Serialize};
use std::borrow::Cow;

use crate::utils::Mask;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OutputCovariance<'a> {
    pub(crate) isotropic_noise: f64,
    pub(crate) transform: Cow<'a, DMatrix<f64>>,
}

impl<'a> OutputCovariance<'a> {
    pub(crate) fn new_owned(
        isotropic_noise: f64,
        transform: DMatrix<f64>,
    ) -> OutputCovariance<'static> {
        OutputCovariance {
            isotropic_noise,
            transform: Cow::Owned(transform),
        }
    }

    pub(crate) fn output_size(&self) -> usize {
        self.transform.nrows()
    }

    pub(crate) fn state_size(&self) -> usize {
        self.transform.ncols()
    }

    // pub(crate) fn owned(&self) -> OutputCovariance<'static> {
    //     OutputCovariance {
    //         isotropic_noise: self.isotropic_noise,
    //         transform: Cow::Owned(self.transform.as_ref().clone()),
    //     }
    // }

    // pub(crate) fn matrix(&self) -> DMatrix<f64> {
    //     DMatrix::identity(self.output_size(), self.output_size()) * self.isotropic_noise.powi(2)
    //         + &*self.transform * self.transform.transpose()
    // }

    pub(crate) fn inner_product(&self) -> DMatrix<f64> {
        self.transform.transpose() * &*self.transform
    }

    pub(crate) fn inner_matrix(&self) -> DMatrix<f64> {
        DMatrix::identity(self.state_size(), self.state_size()) * self.isotropic_noise.powi(2)
            + self.inner_product()
    }

    pub(crate) fn inner_inverse(&self) -> DMatrix<f64> {
        self.inner_matrix()
            .try_inverse()
            .expect("inner matrix is always invertible")
    }

    pub(crate) fn estimator_transform(&self) -> DMatrix<f64> {
        (self.transform.transpose()
            - self.inner_product() * self.inner_inverse() * self.transform.transpose())
            / self.isotropic_noise.powi(2)
    }

    pub(crate) fn estimator_covariance(&self) -> DMatrix<f64> {
        DMatrix::identity(self.state_size(), self.state_size())
            - self.estimator_transform() * &*self.transform
    }

    pub(crate) fn covariance_log_det(&self) -> f64 {
        // NOTE: not always `output_size > state_size`.
        self.inner_matrix().determinant().ln()
            + self.isotropic_noise.ln()
                * 2.0
                * (self.output_size() as f64 - self.state_size() as f64)
    }

    pub(crate) fn masked(&self, mask: &Mask) -> OutputCovariance<'static> {
        assert_eq!(mask.0.len(), self.output_size());
        OutputCovariance {
            isotropic_noise: self.isotropic_noise,
            transform: Cow::Owned(DMatrix::from_rows(
                &mask.filter(self.transform.row_iter()).collect::<Vec<_>>(),
            )),
        }
    }

    pub(crate) fn quadratic_form(&self, x: &DVector<f64>) -> f64 {
        let norm_squared = x.norm_squared();
        let transpose_transformed = self.transform.transpose() * x;

        (norm_squared
            // this is a scalar.
            - (transpose_transformed.transpose() * self.inner_inverse() * transpose_transformed)
                [(0, 0)])
            / self.isotropic_noise.powi(2)
    }
}
