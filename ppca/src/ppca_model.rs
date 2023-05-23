use bit_vec::BitVec;
use nalgebra::{DMatrix, DVector};
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::Bernoulli;
use rayon::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::Arc;

use crate::dataset::{Dataset, MaskedSample};
use crate::output_covariance::OutputCovariance;
use crate::prior::Prior;
use crate::utils::{standard_noise, standard_noise_matrix, Mask};

const LN_2PI: f64 = 1.8378770664093453;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPCAModelInner {
    output_covariance: OutputCovariance<'static>,
    mean: DVector<f64>,
}

/// A PPCA model which can infer missing values.
///
/// Each sample for this model behaves according to the following
/// statistical latent variable model.
/// ```
/// x ~ N(0; I(nxn))
/// y = C * x + y0 + noise
/// noise ~ N(0; sigma ^ 2 * I(mxm))
/// ```
/// Here, `x` is the latent state, y is the observed sample, that is an affine
/// transformation of the hidden state contaminated by isotropic noise.
///
/// ## Note
///
/// All arrays involved have to be of data type `float64`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPCAModel(Arc<PPCAModelInner>);

impl PPCAModel {
    pub fn new(isotropic_noise: f64, transform: DMatrix<f64>, mean: DVector<f64>) -> PPCAModel {
        PPCAModel(Arc::new(PPCAModelInner {
            output_covariance: OutputCovariance::new_owned(isotropic_noise, transform),
            mean,
        }))
    }

    /// Creates a new random __untrained__ model from a given latent state size and a dataset.
    pub fn init(state_size: usize, dataset: &Dataset) -> PPCAModel {
        assert!(!dataset.is_empty());
        let output_size = dataset.output_size().expect("dataset is not empty");
        let empty_dimensions = dataset.empty_dimensions();
        let mut rand_transform = standard_noise_matrix(output_size, state_size);

        for (dimension, mut row) in rand_transform.row_iter_mut().enumerate() {
            if empty_dimensions.contains(&dimension) {
                row.fill(0.0);
            }
        }

        PPCAModel(Arc::new(PPCAModelInner {
            output_covariance: OutputCovariance {
                isotropic_noise: 1.0,
                transform: Cow::Owned(rand_transform),
            },
            mean: DVector::zeros(output_size),
        }))
    }

    /// Then center of mass of the distribution in the output space.
    pub fn mean(&self) -> &DVector<f64> {
        &self.0.mean
    }

    /// The standard deviation of the noise in the output space.
    pub fn isotropic_noise(&self) -> f64 {
        self.0.output_covariance.isotropic_noise
    }

    /// The linear transformation from hidden state space to output space.
    pub fn transform(&self) -> &DMatrix<f64> {
        &self.0.output_covariance.transform
    }

    /// The number of features for this model.
    pub fn output_size(&self) -> usize {
        self.0.output_covariance.output_size()
    }

    /// The number of hidden values for this model.
    pub fn state_size(&self) -> usize {
        self.0.output_covariance.state_size()
    }

    /// Creates a zeroed `InferredMasked` struct that is compatible with this model.
    pub fn uninferred(&self) -> InferredMasked {
        InferredMasked {
            model: self.clone(),
            state: DVector::zeros(self.state_size()),
            covariance: DMatrix::identity(self.state_size(), self.state_size()),
        }
    }

    /// The total number of parameters involved in training (used for information criteria).
    pub fn n_parameters(&self) -> usize {
        1 + self.state_size() * self.output_size() + self.0.mean.nrows()
    }

    /// The relative strength of each hidden variable on the output. This is equivalent to the
    /// eigenvalues in the standard PCA.
    pub fn singular_values(&self) -> DVector<f64> {
        self.0
            .output_covariance
            .transform
            .column_iter()
            .map(|column| column.norm().sqrt())
            .collect::<Vec<_>>()
            .into()
    }

    /// Compute the log-likelihood of a single sample.
    pub fn llk_one(&self, sample: &MaskedSample) -> f64 {
        let sample = if !sample.is_empty() {
            sample
        } else {
            return 0.0;
        };

        let sub_sample = sample.mask.mask(&(sample.data_vector() - &self.0.mean));
        let sub_covariance = self.0.output_covariance.masked(&sample.mask);

        let llk = -sub_covariance.quadratic_form(&sub_sample) / 2.0
            - sub_covariance.covariance_log_det() / 2.0
            - LN_2PI / 2.0 * sub_covariance.output_size() as f64;

        llk
    }

    /// Compute the log-likelihood of a given dataset.
    pub fn llk(&self, dataset: &Dataset) -> f64 {
        dataset
            .data
            .par_iter()
            .zip(&dataset.weights)
            .map(|(sample, weight)| self.llk_one(sample) * weight)
            .sum()
    }

    /// Compute the log-likelihood for each sample in a given dataset.
    pub fn llks(&self, dataset: &Dataset) -> DVector<f64> {
        dataset
            .data
            .par_iter()
            .map(|sample| self.llk_one(sample))
            .collect::<Vec<_>>()
            .into()
    }

    /// Sample a single sample from the PPCA model and masks each entry according to a
    /// Bernoulli (coin-toss) distribution of probability `mask_prob` of erasing the
    /// generated value.
    pub fn sample_one(&self, mask_prob: f64) -> MaskedSample {
        let sampled_state: DVector<f64> =
            &*self.0.output_covariance.transform * standard_noise(self.state_size()) + &self.0.mean;
        let noise: DVector<f64> =
            self.0.output_covariance.isotropic_noise * standard_noise(self.output_size());
        let mask = Mask(
            Bernoulli::new(1.0 - mask_prob as f64)
                .expect("invalid mask probability")
                .sample_iter(rand::thread_rng())
                .take(self.output_size())
                .collect::<BitVec>(),
        );

        MaskedSample {
            data: mask.fillna(&(sampled_state + noise)),
            mask,
        }
    }

    /// Sample a full dataset from the PPCA model and masks each entry according to a
    /// Bernoulli (coin-toss) distribution of probability `mask_prob` of erasing the
    /// generated value.
    pub fn sample(&self, dataset_size: usize, mask_prob: f64) -> Dataset {
        (0..dataset_size)
            .into_par_iter()
            .map(|_| self.sample_one(mask_prob))
            .collect()
    }

    /// Infers the hidden components for one single sample. Use this method for
    /// fine-grain control on the properties you want to extract from the model.
    pub fn infer_one(&self, sample: &MaskedSample) -> InferredMasked {
        if sample.is_empty() {
            return self.uninferred();
        }

        let sub_sample = sample.mask.mask(&(sample.data_vector() - &self.0.mean));
        let sub_covariance = self.0.output_covariance.masked(&sample.mask);

        InferredMasked {
            model: self.clone(),
            state: sub_covariance.estimator_transform() * sub_sample,
            covariance: sub_covariance.estimator_covariance(),
        }
    }

    /// Infers the hidden components for each sample in the dataset. Use this method for
    /// fine-grain control on the properties you want to extract from the model.
    pub fn infer(&self, dataset: &Dataset) -> Vec<InferredMasked> {
        dataset
            .data
            .par_iter()
            .map(|sample| self.infer_one(sample))
            .collect()
    }

    /// Filters a single samples, removing noise from the extant samples and
    /// inferring the missing samples.
    pub fn smooth_one(&self, sample: &MaskedSample) -> MaskedSample {
        MaskedSample::unmasked(self.infer_one(sample).smoothed(&self))
    }

    /// Filters each sample of a given dataset, removing noise from the extant samples and
    /// inferring the missing samples.
    pub fn smooth(&self, dataset: &Dataset) -> Dataset {
        dataset
            .data
            .par_iter()
            .zip(&dataset.weights)
            .map(|(sample, &weight)| (self.smooth_one(sample), weight))
            .collect()
    }

    /// Extrapolates the missing values with the most probable values for a single sample. This
    /// method does not alter the extant values.
    pub fn extrapolate_one(&self, sample: &MaskedSample) -> MaskedSample {
        MaskedSample::unmasked(self.infer_one(sample).extrapolated(self, sample))
    }

    /// Extrapolates the missing values with the most probable values for a full dataset. This
    /// method does not alter the extant values.
    pub fn extrapolate(&self, dataset: &Dataset) -> Dataset {
        dataset
            .data
            .par_iter()
            .zip(&dataset.weights)
            .map(|(sample, &weight)| (self.extrapolate_one(sample), weight))
            .collect()
    }

    /// Makes one iteration of the EM algorithm for the PPCA over an observed dataset,
    /// returning the improved model. The log-likelihood will **always increase** for the
    /// returned model.
    #[must_use]
    pub fn iterate(&self, dataset: &Dataset) -> PPCAModel {
        self.iterate_with_prior(dataset, &Prior::default())
    }

    /// Makes one iteration of the EM algorithm for the PPCA over an observed dataset,
    /// using a supplied PPCA prior and returning the improved model. This method will
    /// not necessarily increase the log-likelihood of the returned model, but it will
    /// return an improved _maximum a posteriori_ (MAP) estimate of the PPCA model
    ///  according to the supplied prior.
    #[must_use]
    pub fn iterate_with_prior(&self, dataset: &Dataset, prior: &Prior) -> PPCAModel {
        let inferred = self.infer(dataset);

        // Updated transform:
        let total_cross_moment = dataset
            .data
            .par_iter()
            .zip(&dataset.weights)
            .zip(&inferred)
            .map(|((sample, &weight), inferred)| {
                let centered_filled = sample.mask.fillna(&(sample.data_vector() - &self.0.mean));
                weight * centered_filled * inferred.state.transpose()
            })
            .reduce(
                || DMatrix::zeros(self.output_size(), self.state_size()),
                |this, other| this + other,
            ); // sum() no work...
        let new_transform_rows = (0..self.output_size())
            .into_par_iter()
            .map(|idx| {
                let total_second_moment = dataset
                    .data
                    .iter()
                    .zip(&dataset.weights)
                    .zip(&inferred)
                    .filter(|((sample, _), _)| sample.mask.0[idx])
                    .map(|((_, &weight), inferred)| weight * inferred.second_moment())
                    // In case we get an empty dimension...
                    .chain([DMatrix::zeros(self.state_size(), self.state_size())])
                    .sum::<DMatrix<f64>>()
                    + prior.transformation_precision()
                        * DMatrix::<f64>::identity(self.state_size(), self.state_size());
                let cross_moment_row = total_cross_moment.row(idx).transpose();
                total_second_moment
                    .qr()
                    .solve(&cross_moment_row)
                    .unwrap_or_else(|| {
                        // Keep old row if you can't solve the linear system.
                        self.0
                            .output_covariance
                            .transform
                            .row(idx)
                            .transpose()
                            .clone_owned()
                    })
                    .transpose()
            })
            .collect::<Vec<_>>();
        let new_transform = DMatrix::from_rows(&new_transform_rows);

        // Updated isotropic noise:
        let (square_error, deviations_square_sum, total_deviation, totals) = dataset
            .data
            .par_iter()
            .zip(&dataset.weights)
            .zip(&inferred)
            .filter(|((sample, _), _)| !sample.is_empty())
            .map(
                |((sample, &weight), inferred)| {
                let sub_covariance = self.0.output_covariance.masked(&sample.mask);
                let sub_transform = &*sub_covariance.transform;
                let deviation = sample.mask.fillna(
                    &(sample.data_vector()
                        - &*self.0.output_covariance.transform * &inferred.state
                        - &self.0.mean),
                );

                (
                    weight * (sub_transform * &inferred.covariance).dot(&sub_transform),
                    weight * deviation.norm_squared(),
                    weight * deviation,
                    weight * sample.mask.as_vector(),
                )
            }).reduce_with(|
                (square_error, deviation_square_sum, total_deviation, totals),
                (square_error_, deviation_square_sum_, total_deviation_, totals_)| (
                    square_error + square_error_,
                    deviation_square_sum + deviation_square_sum_,
                    total_deviation + total_deviation_,
                    totals + totals_
                )
            ).expect("non-empty dataset");

        let isotropic_noise_sq = if prior.has_isotropic_noise_prior() {
            // Note: Inverse gamma inference here...
            // Recall _mode_ for inverse gamma:
            //     beta_post / (alpha_post + 1)
            // And...
            //     beta_post = [sum of all squarey stuff] / 2.0 + beta_prior
            //     alpha_post = [number of samples] / 2.0 + alpha_prior
            ((square_error + deviations_square_sum) / 2.0 + prior.isotropic_noise_beta())
                / (totals.sum() / 2.0 + prior.isotropic_noise_alpha() + 1.0)
        } else {
            (square_error + deviations_square_sum) / totals.sum()
        };

        let mut new_mean =
            total_deviation.zip_map(
                &totals,
                |sum, count| if count > 0.0 { sum / count } else { 0.0 },
            ) + &self.0.mean;

        if prior.has_mean_prior() {
            new_mean = prior.smooth_mean(
                new_mean,
                DMatrix::from_diagonal(&totals) / isotropic_noise_sq,
            );
        }

        PPCAModel(Arc::new(PPCAModelInner {
            output_covariance: OutputCovariance {
                transform: Cow::Owned(new_transform),
                isotropic_noise: isotropic_noise_sq.sqrt(),
            },
            mean: new_mean,
        }))
    }

    /// Returns a canonical version of this model. This does not alter the log-probability
    /// function nor the quality of the training. All it does is to transform the hidden
    /// variables.
    pub fn to_canonical(&self) -> PPCAModel {
        // Yes, we can have an empty state! In these case, there is nothing to be done.
        if self.state_size() == 0 {
            return self.clone();
        }

        let mut svd = self
            .0
            .output_covariance
            .transform
            .clone_owned()
            .svd(true, false);
        svd.v_t = Some(DMatrix::identity(self.state_size(), self.state_size()));

        let mut new_transform = svd.recompose().expect("all matrices were calculated");
        // Flip new transform
        for mut column in new_transform.column_iter_mut() {
            column *= column.sum().signum();
        }

        PPCAModel(Arc::new(PPCAModelInner {
            output_covariance: OutputCovariance::new_owned(
                self.0.output_covariance.isotropic_noise,
                new_transform,
            ),
            mean: self.0.mean.clone(),
        }))
    }
}

/// The inferred probability distribution in the state space of a given sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredMasked {
    model: PPCAModel,
    state: DVector<f64>,
    covariance: DMatrix<f64>,
}

impl InferredMasked {
    pub(crate) fn second_moment(&self) -> DMatrix<f64> {
        &self.state * self.state.transpose() + &self.covariance
    }

    /// The mean of the probability distribution in the state space.
    pub fn state(&self) -> &DVector<f64> {
        &self.state
    }

    /// The covariance matrix of the probability distribution in the state space. The covariances
    /// here can change from sample to sample, depending on the mask. If there is lots of masking
    /// in a sample, the covariance will be overall bigger.
    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.covariance
    }

    /// The smoothed output values for a given output model.
    pub fn smoothed(&self, ppca: &PPCAModel) -> DVector<f64> {
        &*ppca.0.output_covariance.transform * self.state() + &ppca.0.mean
    }

    /// The extrapolated output values for a given output model and extant values in a given
    /// sample.
    pub fn extrapolated(&self, ppca: &PPCAModel, sample: &MaskedSample) -> DVector<f64> {
        let filtered = self.smoothed(&ppca);
        sample.mask.choose(&sample.data_vector(), &filtered)
    }

    /// The covariance for the smoothed output values.
    ///
    /// # Note:
    ///
    /// Afraid of the big, fat matrix? The method `output_covariance_diagonal` might just
    /// save your life.
    pub fn smoothed_covariance(&self, ppca: &PPCAModel) -> DMatrix<f64> {
        let covariance = &ppca.0.output_covariance;

        DMatrix::identity(covariance.output_size(), covariance.output_size())
            * covariance.isotropic_noise.powi(2)
            + &*covariance.transform * &self.covariance * covariance.transform.transpose()
    }

    /// Returns an _approximation_ of the smoothed output covariance matrix, treating each masked
    /// output as an independent normal distribution.
    ///
    /// # Note:
    ///
    /// Use this not to get lost with big matrices in the output, losing CPU, memory and hair.
    pub fn smoothed_covariance_diagonal(&self, ppca: &PPCAModel) -> DVector<f64> {
        // Here, we will calculate `I sigma^2 + C Sxx C^T` for the unobserved samples in a
        // clever way...

        let covariance = &ppca.0.output_covariance;

        // The `inner_inverse` part.
        let transformed_state_covariance = &*covariance.transform * &self.covariance;

        // Now comes the trick! Calculate only the diagonal elements of the
        // `transformed_state_covariance * C^T` part.
        let noiseless_output_diagonal = transformed_state_covariance
            .row_iter()
            .zip(covariance.transform.row_iter())
            .map(|(row_left, row_right)| row_left.dot(&row_right));

        // Finally, add the isotropic noise term...
        noiseless_output_diagonal
            .map(|noiseless_output_variance| {
                noiseless_output_variance + covariance.isotropic_noise.powi(2)
            })
            .collect::<Vec<_>>()
            .into()
    }

    /// The covariance for the extrapolated values for a given output model and extant values in a given
    /// sample.
    ///
    /// # Note:
    ///
    /// Afraid of the big, fat matrix? The method `output_covariance_diagonal` might just
    /// save your life.
    pub fn extrapolated_covariance(&self, ppca: &PPCAModel, sample: &MaskedSample) -> DMatrix<f64> {
        let negative = sample.mask().negate();

        if !negative.0.any() {
            return DMatrix::zeros(ppca.output_size(), ppca.output_size());
        }

        let sub_covariance = ppca.0.output_covariance.masked(&negative);

        let output_covariance =
            DMatrix::identity(sub_covariance.output_size(), sub_covariance.output_size())
                * sub_covariance.isotropic_noise.powi(2)
                + &*sub_covariance.transform
                    * &self.covariance
                    * sub_covariance.transform.transpose();

        negative.expand_matrix(output_covariance)
    }

    /// Returns an _approximation_ of the extrapolated output covariance matrix, treating each masked
    /// output as an independent normal distribution.
    ///
    /// # Note
    ///
    /// Use this not to get lost with big matrices in the output, losing CPU, memory and hair.
    pub fn extrapolated_covariance_diagonal(
        &self,
        ppca: &PPCAModel,
        sample: &MaskedSample,
    ) -> DVector<f64> {
        // Here, we will calculate `I sigma^2 + C Sxx C^T` for the unobserved samples in a
        // clever way...

        let negative = sample.mask().negate();

        if !negative.0.any() {
            return DVector::zeros(ppca.output_size());
        }

        let sub_covariance = ppca.0.output_covariance.masked(&negative);

        // The `inner_inverse` part.
        let transformed_state_covariance = &*sub_covariance.transform * &self.covariance;

        // Now comes the trick! Calculate only the diagonal elements of the
        // `transformed_state_covariance * C^T` part.
        let noiseless_output_diagonal = transformed_state_covariance
            .row_iter()
            .zip(sub_covariance.transform.row_iter())
            .map(|(row_left, row_right)| row_left.dot(&row_right));

        // Finally, add the isotropic noise term...
        let diagonal_reduced = noiseless_output_diagonal
            .map(|noiseless_output_variance| {
                noiseless_output_variance + sub_covariance.isotropic_noise.powi(2)
            })
            .collect::<Vec<_>>()
            .into();

        negative.expand(&diagonal_reduced)
    }

    /// Samples from the posterior distribution of an inferred sample. The sample is smoothed, that
    /// is, it does not include the model isotropic noise.
    pub fn posterior_sampler(&self) -> PosteriorSampler {
        let cholesky = self
            .covariance
            .clone()
            .cholesky()
            .expect("Cholesky decomposition failed");
        PosteriorSampler {
            model: self.model.clone(),
            state: self.state.clone(),
            cholesky_l: cholesky.l(),
        }
    }
}

/// Samples from the posterior distribution of an inferred sample. The sample is smoothed, that
/// is, it does not include the model isotropic noise.
pub struct PosteriorSampler {
    model: PPCAModel,
    state: DVector<f64>,
    cholesky_l: DMatrix<f64>,
}

impl Distribution<DVector<f64>> for PosteriorSampler {
    fn sample<R>(&self, rng: &mut R) -> DVector<f64>
    where
        R: Rng + ?Sized,
    {
        // State noise:
        let standard: DVector<f64> = (0..self.state.len())
            .map(|_| rand_distr::StandardNormal.sample(rng))
            .collect::<Vec<_>>()
            .into();
        // Output noise:
        let noise: DVector<f64> = (0..self.model.output_size())
            .map(|_| {
                let standard: f64 = rand_distr::StandardNormal.sample(rng);
                self.model.0.output_covariance.isotropic_noise * standard
            })
            .collect::<Vec<_>>()
            .into();

        noise
            + self.model.mean()
            + self.model.transform() * (&self.state + &self.cholesky_l * standard)
    }
}

#[cfg(test)]
mod test {
    use bit_vec::BitVec;
    use nalgebra::{dmatrix, dvector};

    use super::*;

    fn toy_model() -> PPCAModel {
        PPCAModel::new(
            0.1,
            dmatrix![
                1.0, 1.0, 0.0;
                1.0, 0.0, 1.0;
            ]
            .transpose(),
            dvector![0.0, 1.0, 0.0],
        )
    }

    fn output_covariance() -> OutputCovariance<'static> {
        OutputCovariance::new_owned(
            0.1,
            dmatrix![
                1.0, 1.0, 0.0;
                1.0, 0.0, 1.0;
            ]
            .transpose(),
        )
    }

    #[test]
    fn test_quadratic_form() {
        let output_covariance = output_covariance();
        approx::assert_relative_eq!(
            output_covariance.quadratic_form(&dvector![1.0, 1.0, 1.0]),
            34.219288
        );
    }

    #[test]
    fn test_covariance_log_det() {
        let output_covariance = output_covariance();
        approx::assert_relative_eq!(output_covariance.covariance_log_det(), -3.49328);
    }

    #[test]
    fn test_llk() {
        let model = toy_model();
        dbg!(model.llk(&Dataset::new(vec![MaskedSample {
            data: dvector![1.0, 2.0, 3.0],
            mask: Mask(BitVec::from_elem(3, true)),
        }])));
    }
}
