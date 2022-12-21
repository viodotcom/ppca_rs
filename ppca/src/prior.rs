use nalgebra::{DMatrix, DVector};
use serde_derive::{Deserialize, Serialize};

/// A prior for the PPCA model. Use this class to mitigate overfit on training (especially on
/// frequently masked dimensions) and to input _a priori_ knowledge on what the PPCA should look
/// like.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prior {
    mean: Option<DVector<f64>>,
    mean_covariance: Option<DMatrix<f64>>,
    mean_precision: Option<DMatrix<f64>>,
    isotropic_noise_alpha: Option<f64>,
    isotropic_noise_beta: Option<f64>,
    transformation_precision: f64,
}

impl Default for Prior {
    fn default() -> Prior {
        Prior {
            mean: None,
            mean_covariance: None,
            mean_precision: None,
            isotropic_noise_alpha: None,
            isotropic_noise_beta: None,
            transformation_precision: 0.0,
        }
    }
}

impl Prior {
    /// Add a prior to the mean of the PPCA. The prior is a normal multivariate distribution.
    pub fn with_mean_prior(mut self, mean: DVector<f64>, mean_covariance: DMatrix<f64>) -> Self {
        assert_eq!(mean.len(), mean_covariance.nrows());
        assert_eq!(mean.len(), mean_covariance.ncols());
        self.mean = Some(mean);
        self.mean_precision = Some(
            mean_covariance
                .clone()
                .try_inverse()
                .expect("mean covariance should be invertible"),
        );
        self.mean_covariance = Some(mean_covariance);

        self
    }

    /// Add an isotropic noise prior. The prior is an Inverse Gamma distribution with shape `alpha`
    /// and rate `beta`.
    pub fn with_isotropic_noise_prior(mut self, alpha: f64, beta: f64) -> Self {
        assert!(alpha >= 0.0);
        assert!(beta >= 0.0);
        self.isotropic_noise_alpha = Some(alpha);
        self.isotropic_noise_beta = Some(beta);

        self
    }

    /// Impose an independent Normal prior to each dimension of the transformation matrix. The
    /// precision is the inverse of the variance of the Normal distribution (`1 / sigma ^ 2`).
    pub fn with_transformation_precision(mut self, precision: f64) -> Self {
        assert!(precision >= 0.0);
        self.transformation_precision = precision;

        self
    }

    pub fn mean(&self) -> Option<&DVector<f64>> {
        self.mean.as_ref()
    }

    pub fn mean_covariance(&self) -> Option<&DMatrix<f64>> {
        self.mean_covariance.as_ref()
    }

    pub fn has_isotropic_noise_prior(&self) -> bool {
        self.isotropic_noise_alpha.is_some()
    }

    pub fn isotropic_noise_alpha(&self) -> f64 {
        self.isotropic_noise_alpha
            .expect("isotropic noise prior not set")
    }

    pub fn isotropic_noise_beta(&self) -> f64 {
        self.isotropic_noise_beta
            .expect("isotropic noise prior not set")
    }

    pub fn transformation_precision(&self) -> f64 {
        self.transformation_precision
    }

    pub fn has_mean_prior(&self) -> bool {
        self.mean.is_some()
    }

    pub(crate) fn smooth_mean(&self, mean: DVector<f64>, precision: DMatrix<f64>) -> DVector<f64> {
        let (prior_mean, prior_precision) = self
            .mean
            .as_ref()
            .zip(self.mean_precision.as_ref())
            .expect("mean prior not set");
        let total_precision = prior_precision + &precision;
        let numerator = prior_precision * prior_mean + &precision * mean;

        total_precision
            .qr()
            .solve(&numerator)
            .expect("total precision matrix is always invertible")
    }
}
