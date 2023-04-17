use nalgebra::{DMatrix, DVector};
use serde_derive::{Deserialize, Serialize};
use std::borrow::Cow;

use crate::utils::Mask;

/// An utilities class with optmized functions to work on the _output covariance_ matrix.
///
/// The main motivation for this class is that the output covariance matrix has a too
/// high dimensionality for we to be able to work with it directly. On the other hand, it
/// has a simplified form:
/// ```
/// C_yy = I * sigma^2 + C * C^T
/// ```
/// where `C` is the transformation matrix of the PPCA model (see the `PPCAModel` class
/// for more information). With enough linear algebra shenanigans, we can reduce de
/// complexity of some operations involving this kind of matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OutputCovariance<'a> {
    /// The isotropic noise of the PPCA Model, denoted as `sigma`.
    pub(crate) isotropic_noise: f64,
    /// The matrix mapping hidden state to output state, denoted as `C`.
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

    /// Calculates the linear transformation that estimates the hidden state from the
    /// observation.
    ///
    /// # Warning: linear algebra shenanigans afoot!!
    ///
    /// Being the output covariance `sigma^2*I + C*C^T` and the estimator transform
    /// `C^T * (sigma^2*I + C*C^T)^-1`, we have a pesky inverse to calculate! Thankfully,
    /// we can use the Woodbury identity to the rescue!
    /// ```
    /// (sigma^2*I + C*C^T)^-1 = I/sigma^2 - C/sigma^2*(I + C^T*C/sigma^2)^-1*C^T/sigma^2
    /// ```
    /// The trick is that the new inverse that we have to calculate has only the
    /// dimension of the _hidden_ state and therefore, goes much faster. The full
    /// estimator is given by:
    /// ```
    /// C^T/sigma^2 - C^T*C/sigma^2*(I + C^T*C/sigma^2)^-1*C^T/sigma^2
    /// ```
    /// Which can be calculated in `O(output_length * state_length^3)`.
    pub(crate) fn estimator_transform(&self) -> DMatrix<f64> {
        (self.transform.transpose()
            - self.inner_product() * self.inner_inverse() * self.transform.transpose())
            / self.isotropic_noise.powi(2)
    }

    /// The covariance of the estimator that estimates hidden state from the observation.
    /// See `OutputCovariance.estimator_transform` for the explanation on the derivation.
    pub(crate) fn estimator_covariance(&self) -> DMatrix<f64> {
        DMatrix::identity(self.state_size(), self.state_size())
            - self.estimator_transform() * &*self.transform
    }

    /// Calculates the log of the determinant of the output covariance matrix form masked
    /// data. This uses the _Matrix Determinant Lemma_ shenanigan to speed up computation:
    /// ```
    /// det(I * sigma^2 + C * C^T) = det(I + C^T * C / sigma^2) * det(I * sigma^2)
    /// ```
    /// This can be simplified to
    /// ```
    /// det(I * sigma^2 + C * C^T) = det(I * sigma^2 + C^T * C)
    ///     * sigma^(2 * (output_size - state_size))
    /// ```
    /// The first `det` on the right side is the determinant of
    /// `OutputCovariance.inner_matrix`.
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
