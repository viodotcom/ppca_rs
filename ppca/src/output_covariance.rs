use nalgebra::{DMatrix, DVector};
use serde_derive::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::OnceLock;

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
    #[serde(default)]
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    inner_product: OnceLock<DMatrix<f64>>,
    #[serde(default)]
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    inner_matrix: OnceLock<DMatrix<f64>>,
    #[serde(default)]
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    inner_inverse: OnceLock<DMatrix<f64>>,
}

impl<'a> OutputCovariance<'a> {
    pub(crate) fn new_owned(
        isotropic_noise: f64,
        transform: DMatrix<f64>,
    ) -> OutputCovariance<'static> {
        OutputCovariance {
            isotropic_noise,
            transform: Cow::Owned(transform),
            inner_product: OnceLock::new(),
            inner_matrix: OnceLock::new(),
            inner_inverse: OnceLock::new(),
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

    fn do_inner_product(&self) -> DMatrix<f64> {
        self.transform.transpose() * &*self.transform
    }

    fn inner_product(&self) -> &DMatrix<f64> {
        self.inner_product.get_or_init(|| self.do_inner_product())
    }

    fn do_inner_matrix(&self) -> DMatrix<f64> {
        DMatrix::identity(self.state_size(), self.state_size())
            + self.inner_product() / self.isotropic_noise.powi(2)
    }

    fn inner_matrix(&self) -> &DMatrix<f64> {
        self.inner_matrix.get_or_init(|| self.do_inner_matrix())
    }

    fn do_inner_inverse(&self) -> DMatrix<f64> {
        self.inner_matrix()
            .clone()
            .try_inverse()
            .expect("inner matrix is always invertible")
    }

    fn inner_inverse(&self) -> &DMatrix<f64> {
        self.inner_inverse.get_or_init(|| self.do_inner_inverse())
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
    /// Which can be calculated in `O(output_length * state_length^3)`. This can be futher simplified to
    /// ```
    /// (I - C^T*C/sigma^2*(I + C^T*C/sigma^2)^-1) * C^T/sigma^2
    ///     = ((I + C^T*C/sigma^2) - C^T*C/sigma^2) * (I + C^T*C/sigma^2)^-1 * C^T/sigma^2
    ///     = (I + C^T*C/sigma^2)^-1 * C^T/sigma^2
    /// ```
    /// Which retains the same complexity, but uses fewer operations.
    pub(crate) fn estimator_transform(&self) -> DMatrix<f64> {
        self.inner_inverse() * self.transform.transpose() / self.isotropic_noise.powi(2)
    }

    /// The covariance of the estimator that estimates hidden state from the observation.
    /// See `OutputCovariance.estimator_transform` for the explanation on the derivation.
    pub(crate) fn estimator_covariance(&self) -> DMatrix<f64> {
        self.inner_inverse().clone()
    }

    /// Calculates the log of the determinant of the output covariance matrix form masked
    /// data. This uses the _Matrix Determinant Lemma_ shenanigan to speed up computation:
    /// ```
    /// det(I * sigma^2 + C * C^T) = det(I + C^T * C / sigma^2) * det(I * sigma^2)
    /// ```
    /// The first `det` on the right side is the determinant of
    /// `OutputCovariance.inner_matrix`.
    pub(crate) fn covariance_log_det(&self) -> f64 {
        self.inner_matrix().determinant().ln()
            + self.isotropic_noise.ln() * 2.0 * (self.output_size() as f64)
    }

    pub(crate) fn masked(&self, mask: &Mask) -> OutputCovariance<'static> {
        assert_eq!(mask.0.len(), self.output_size());
        OutputCovariance::new_owned(
            self.isotropic_noise,
            DMatrix::from_rows(&mask.filter(self.transform.row_iter()).collect::<Vec<_>>()),
        )
    }

    pub(crate) fn quadratic_form(&self, x: &DVector<f64>) -> f64 {
        let norm_squared = x.norm_squared();
        let transpose_transformed = self.transform.transpose() * x;

        (norm_squared
            - (transpose_transformed.transpose() * self.inner_inverse() * transpose_transformed)
                [(0, 0)]
                / self.isotropic_noise.powi(2))
            / self.isotropic_noise.powi(2)
    }
}
