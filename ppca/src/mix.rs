use std::sync::Arc;

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, WeightedIndex};
use rayon::prelude::*;
use serde_derive::{Deserialize, Serialize};

use crate::dataset::{Dataset, MaskedSample};
use crate::ppca_model::{self, InferredMasked, PPCAModel};
use crate::Prior;

/// Performs Bayesian inference in the log domain.
fn robust_log_softmax(data: DVector<f64>) -> DVector<f64> {
    let max = data.max();
    let log_norm = data.iter().map(|&xi| (xi - max).exp()).sum::<f64>().ln();
    data.map(|xi| xi - max - log_norm)
}

/// Performs Bayesian inference in the log domain.
fn robust_log_softnorm(data: DVector<f64>) -> f64 {
    let max = data.max();
    let log_norm = data.iter().map(|&xi| (xi - max).exp()).sum::<f64>().ln();
    max + log_norm
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PPCAMixInner {
    output_size: usize,
    models: Vec<PPCAModel>,
    log_weights: DVector<f64>,
}

/// A mixture of PPCA models. Each PPCA model is associated with a prior probability
/// expressed in log-scale. This models allows for modelling of data clustering and
/// non-linear learning of data. However, it will use significantly more memory and is
/// not guaranteed to converge to a global maximum.
///
/// # Notes
///
/// * The list of log-weights does not need to be normalized. Normalization is carried out
/// internally.
/// * Each PPCA model in the mixture might have its own state size. However, all PPCA
/// models must have the same output space. Additionally, the set of PPCA models must be
/// non-empty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPCAMix(Arc<PPCAMixInner>);

impl PPCAMix {
    pub fn new(models: Vec<PPCAModel>, log_weights: DVector<f64>) -> PPCAMix {
        assert!(models.len() > 0);
        assert_eq!(models.len(), log_weights.len());

        let output_sizes = models
            .iter()
            .map(PPCAModel::output_size)
            .collect::<Vec<_>>();
        let mut unique_sizes = output_sizes.clone();
        unique_sizes.dedup();
        assert_eq!(
            unique_sizes.len(),
            1,
            "Model output sizes are not the same: {output_sizes:?}"
        );

        PPCAMix(Arc::new(PPCAMixInner {
            output_size: unique_sizes[0],
            models,
            log_weights: robust_log_softmax(log_weights),
        }))
    }

    /// Creates a new random __untrained__ model from a given number of PPCA modes, a latent state
    /// size, a dataset and a smoothing factor. The smoothing factor helps with overfit of rarely
    /// occurring dimensions. If you don't care about that, set it to `0.0`.
    pub fn init(n_models: usize, state_size: usize, dataset: &Dataset) -> PPCAMix {
        PPCAMix::new(
            (0..n_models)
                .map(|_| PPCAModel::init(state_size, dataset))
                .collect(),
            vec![0.0; n_models].into(),
        )
    }

    /// The number of features for this model.
    pub fn output_size(&self) -> usize {
        self.0.output_size
    }

    /// The number of hidden values for each PPCA model.
    pub fn state_sizes(&self) -> Vec<usize> {
        self.0.models.iter().map(PPCAModel::state_size).collect()
    }

    /// The total number of parameters involved in training (used for information criteria).
    pub fn n_parameters(&self) -> usize {
        self.0
            .models
            .iter()
            .map(PPCAModel::n_parameters)
            .sum::<usize>()
            + self.0.models.len()
            - 1
    }

    /// The list of constituent PPCA models.
    pub fn models(&self) -> &[PPCAModel] {
        &self.0.models
    }

    /// The log-strength (or log-_a priori_ probability) for each PPCA model.
    pub fn log_weights(&self) -> &DVector<f64> {
        &self.0.log_weights
    }

    /// The strength (or _a priori_ probability) for each PPCA model.
    pub fn weights(&self) -> DVector<f64> {
        self.0.log_weights.map(f64::exp)
    }

    /// Sample a full dataset from the PPCA model and masks each entry according to a
    /// Bernoulli (coin-toss) distribution of probability `mask_prob` of erasing the
    /// generated value.
    pub fn sample(&self, dataset_size: usize, mask_probability: f64) -> Dataset {
        let index = WeightedIndex::new(self.0.log_weights.iter().copied().map(f64::exp))
            .expect("can create WeighedIndex from distribution");
        (0..dataset_size)
            .into_par_iter()
            .map(|_| {
                let model_idx = index.sample(&mut rand::thread_rng());
                self.0.models[model_idx].sample_one(mask_probability)
            })
            .collect()
    }

    /// Computes the log-likelihood for each constituent PPCA model.
    pub(crate) fn llks_one(&self, sample: &MaskedSample) -> DVector<f64> {
        self.0
            .models
            .iter()
            .map(|model| model.llk_one(sample))
            .collect::<Vec<_>>()
            .into()
    }

    /// Computes the log-likelihood for a single sample.
    pub fn llk_one(&self, sample: &MaskedSample) -> f64 {
        robust_log_softnorm(self.llks_one(sample) + &self.0.log_weights)
    }

    /// Computes the log-likelihood for each sample in a dataset.
    pub fn llks(&self, dataset: &Dataset) -> DVector<f64> {
        dataset
            .data
            .par_iter()
            .map(|sample| self.llk_one(sample))
            .collect::<Vec<_>>()
            .into()
    }

    /// Computes the total log-likelihood for a given dataset.
    pub fn llk(&self, dataset: &Dataset) -> f64 {
        // Rayon doesn't like to sum empty stuff...
        if dataset.is_empty() {
            return 0.0;
        }

        dataset
            .data
            .par_iter()
            .zip(&dataset.weights)
            .map(|(sample, &weight)| weight * self.llk_one(sample))
            .sum::<f64>()
    }

    /// Returns the _posterior_ distribution (i.e., with Bayes' rule applied) for each sample in
    /// the given dataset. Each row of the matrix corresponds to a categorical distribution on the
    /// probability of a sample belonging to a particular PPCA model.
    pub fn infer_cluster(&self, dataset: &Dataset) -> DMatrix<f64> {
        let rows: Vec<_> = dataset
            .data
            .par_iter()
            .map(|sample| {
                robust_log_softmax(self.llks_one(sample) + &self.0.log_weights).transpose()
            })
            .collect();

        DMatrix::from_rows(&*rows)
    }

    /// Creates a zeroed `InferredMasked` struct that is compatible with this model. This
    /// returns the _prior_ associated with this model.
    pub fn uninferred(&self) -> InferredMaskedMix {
        InferredMaskedMix {
            log_posterior: self.log_weights().clone(),
            inferred: self
                .models()
                .iter()
                .map(|model| model.uninferred())
                .collect::<Vec<_>>(),
        }
    }

    /// Infers the probability distribution of a single sample.
    pub fn infer_one(&self, sample: &MaskedSample) -> InferredMaskedMix {
        InferredMaskedMix {
            log_posterior: robust_log_softmax(self.llks_one(sample) + &self.0.log_weights),
            inferred: self
                .0
                .models
                .iter()
                .map(|model| model.infer_one(sample))
                .collect::<Vec<_>>(),
        }
    }

    /// Infers the probability distribution of a given dataset.
    pub fn infer(&self, dataset: &Dataset) -> Vec<InferredMaskedMix> {
        dataset
            .data
            .par_iter()
            .map(|sample| self.infer_one(sample))
            .collect()
    }

    /// Filters a single samples, removing noise from it and inferring the missing dimensions.
    pub fn smooth_one(&self, sample: &MaskedSample) -> MaskedSample {
        MaskedSample::unmasked(self.infer_one(sample).smoothed(self))
    }

    /// Filters a dataset of samples, removing noise from the extant samples and
    /// inferring the missing samples.
    pub fn smooth(&self, dataset: &Dataset) -> Dataset {
        dataset
            .data
            .par_iter()
            .map(|sample| self.smooth_one(sample))
            .collect()
    }

    /// Extrapolates the missing values from a sample with the most probable values.
    pub fn extrapolate_one(&self, sample: &MaskedSample) -> MaskedSample {
        MaskedSample::unmasked(self.infer_one(sample).extrapolated(self, sample))
    }

    /// Extrapolates the missing values from a dataset with the most probable values.
    pub fn extrapolate(&self, dataset: &Dataset) -> Dataset {
        dataset
            .data
            .par_iter()
            .map(|sample| self.extrapolate_one(sample))
            .collect()
    }

    /// Makes one iteration of the EM algorithm for the PPCA mixture model over an
    /// observed dataset, returning a improved model. The log-likelihood will **always increase**
    /// for the returned model.
    #[must_use]
    pub fn iterate(&self, dataset: &Dataset) -> PPCAMix {
        self.iterate_with_prior(dataset, &Prior::default())
    }

    /// Makes one iteration of the EM algorithm for the PPCA mixture over an observe
    /// dataset, using a supplied PPCA prior (same for all constituent PPCA models) and
    /// returning the improved model. This method will not necessarily increase the
    /// log-likelihood of the returned model, but it will return an improved _maximum a
    /// posteriori_ (MAP) estimate of the PPCA model according to the supplied prior.
    #[must_use]
    pub fn iterate_with_prior(&self, dataset: &Dataset, prior: &Prior) -> PPCAMix {
        // This is already parallelized internally; no need to further parallelize.
        let llks = self
            .0
            .models
            .iter()
            .map(|model| model.llks(dataset))
            .collect::<Vec<_>>();
        let log_posteriors = (0..dataset.len())
            .into_par_iter()
            .map(|idx| {
                let llk: DVector<f64> = llks.iter().map(|llk| llk[idx]).collect::<Vec<_>>().into();
                robust_log_softmax(llk + &self.0.log_weights)
            })
            .collect::<Vec<_>>();

        let (iterated_models, log_weights): (Vec<_>, Vec<f64>) = self
            .0
            .models
            .iter()
            .enumerate()
            .map(|(i, model)| {
                // Log-posteriors for this particular model.
                let log_posteriors: Vec<_> = log_posteriors
                    .par_iter()
                    .zip(&dataset.weights)
                    .filter(|&(_, &wi)| wi > 0.0)
                    .map(|(lp, &wi)| wi.ln() + lp[i])
                    .collect();
                // Let the NaN silently propagate... everything will blow up before this
                // is all over.
                let max_posterior: f64 = log_posteriors
                    .par_iter()
                    .filter_map(|&xi| ordered_float::NotNan::new(xi).ok())
                    .max()
                    .expect("dataset not empty")
                    .into();
                // Use un-normalized posteriors as weights for numerical stability. One of
                // the entries is guaranteed to be 1.0.
                let unnorm_posteriors: Vec<_> = log_posteriors
                    .par_iter()
                    .map(|&p| f64::exp(p - max_posterior))
                    .collect();
                let logsum_posteriors =
                    unnorm_posteriors.iter().copied().sum::<f64>().ln() + max_posterior;
                let dataset = dataset.with_weights(unnorm_posteriors);

                (model.iterate_with_prior(&dataset, prior), logsum_posteriors)
            })
            .unzip();

        PPCAMix(Arc::new(PPCAMixInner {
            output_size: self.0.output_size,
            models: iterated_models,
            log_weights: robust_log_softmax(log_weights.into()),
        }))
    }

    /// Maps [`PPCAModel::to_canonical`] for each constituent model.
    pub fn to_canonical(&self) -> PPCAMix {
        PPCAMix(Arc::new(PPCAMixInner {
            output_size: self.0.output_size,
            models: self.0.models.iter().map(PPCAModel::to_canonical).collect(),
            log_weights: self.0.log_weights.clone(),
        }))
    }
}

/// The inferred probability distribution in the state space of a given sample of a PPCA Mixture
/// Model. This class is the analogous of [`InferredMasked`] for the [`PPCAMix`] model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredMaskedMix {
    log_posterior: DVector<f64>,
    inferred: Vec<InferredMasked>,
}

impl InferredMaskedMix {
    /// The logarithm of the posterior distribution over the PPCA model indices.
    pub fn log_posterior(&self) -> &DVector<f64> {
        &self.log_posterior
    }

    /// The posterior distribution over the PPCA model indices.
    pub fn posterior(&self) -> DVector<f64> {
        self.log_posterior.map(f64::exp)
    }

    /// The mean of the posterior distribution in the state space.
    pub fn state(&self) -> DVector<f64> {
        self.log_posterior
            .iter()
            .zip(&self.inferred)
            .map(|(&pi, inferred)| pi * inferred.state())
            .sum()
    }

    /// The covariance matrices of the posterior distribution in the state space.
    pub fn covariance(&self) -> DMatrix<f64> {
        let mean = self.state();
        self.inferred
            .iter()
            .zip(&self.posterior())
            .map(|(inferred, &weight)| {
                weight
                    * (inferred.covariance()
                        + (inferred.state() - &mean) * (inferred.state() - &mean).transpose())
            })
            .sum::<DMatrix<f64>>()
    }

    /// The smoothed output values for a given output model.
    pub fn smoothed(&self, mix: &PPCAMix) -> DVector<f64> {
        self.inferred
            .iter()
            .zip(&self.posterior())
            .zip(&mix.0.models)
            .map(|((inferred, &weight), ppca)| weight * inferred.smoothed(ppca))
            .sum::<DVector<f64>>()
    }

    /// The extrapolated output values for a given output model and the corresponding sample.
    pub fn extrapolated(&self, mix: &PPCAMix, sample: &MaskedSample) -> DVector<f64> {
        self.inferred
            .iter()
            .zip(&self.posterior())
            .zip(&mix.0.models)
            .map(|((inferred, &weight), ppca)| weight * inferred.extrapolated(ppca, sample))
            .sum::<DVector<f64>>()
    }

    /// The covariance for the smoothed output values.
    ///
    /// # Note:
    ///
    /// Afraid of the big, fat matrix? The method `output_covariance_diagonal` might just
    /// save your life.
    pub fn smoothed_covariance(&self, mix: &PPCAMix) -> DMatrix<f64> {
        let mean = self.smoothed(mix);
        self.inferred
            .iter()
            .zip(&self.posterior())
            .zip(&mix.0.models)
            .map(|((inferred, &weight), ppca)| {
                weight
                    * (inferred.smoothed_covariance(ppca)
                        + (inferred.smoothed(ppca) - &mean)
                            * (inferred.smoothed(ppca) - &mean).transpose())
            })
            .sum::<DMatrix<f64>>()
    }

    /// Returns an _approximation_ of the smoothed output covariance matrix, treating each masked
    /// output as an independent normal distribution.
    ///
    /// # Note:
    ///
    /// Use this not to get lost with big matrices in the output, losing CPU, memory and hair.
    pub fn smoothed_covariance_diagonal(&self, mix: &PPCAMix) -> DVector<f64> {
        let mean = self.smoothed(mix);
        self.inferred
            .iter()
            .zip(&self.posterior())
            .zip(&mix.0.models)
            .map(|((inferred, &weight), ppca)| {
                weight
                    * (inferred.smoothed_covariance_diagonal(ppca)
                        + (inferred.smoothed(ppca) - &mean).map(|v| v.powi(2)))
            })
            .sum()
    }

    /// The covariance for the extraplated values for a given output model and extant values in a given
    /// sample.
    ///
    /// # Note:
    ///
    /// Afraid of the big, fat matrix? The method `output_covariance_diagonal` might just
    /// save your life.
    pub fn extrapolated_covariance(&self, mix: &PPCAMix, sample: &MaskedSample) -> DMatrix<f64> {
        let mean = self.extrapolated(mix, sample).clone();
        self.inferred
            .iter()
            .zip(&self.posterior())
            .zip(&mix.0.models)
            .map(|((inferred, &weight), ppca)| {
                weight
                    * (inferred.smoothed_covariance(ppca)
                        + (inferred.extrapolated(ppca, sample) - &mean)
                            * (inferred.extrapolated(ppca, sample) - &mean).transpose())
            })
            .sum::<DMatrix<f64>>()
    }

    /// Returns an _approximation_ of the extrapolated output covariance matrix, treating each masked
    /// output as an independent normal distribution.
    ///
    /// # Note
    ///
    /// Use this not to get lost with big matrices in the output, losing CPU, memory and hair.
    pub fn extrapolated_covariance_diagonal(
        &self,
        mix: &PPCAMix,
        sample: &MaskedSample,
    ) -> DVector<f64> {
        let mean = self.extrapolated(mix, sample);
        self.inferred
            .iter()
            .zip(&self.posterior())
            .zip(&mix.0.models)
            .map(|((inferred, &weight), ppca)| {
                weight
                    * (inferred.extrapolated_covariance_diagonal(ppca, sample)
                        + (inferred.extrapolated(ppca, sample) - &mean).map(|v| v.powi(2)))
            })
            .sum()
    }

    /// Samples from the posterior distribution of an inferred sample. The sample is smoothed, that
    /// is, it does not include the model isotropic noise.
    pub fn posterior_sampler(&self) -> PosteriorSamplerMix {
        let index = WeightedIndex::new(self.posterior().iter().copied())
            .expect("failed to create WeightedIndex for posterior");
        let posteriors = self
            .inferred
            .iter()
            .map(InferredMasked::posterior_sampler)
            .collect::<Vec<_>>();
        PosteriorSamplerMix { index, posteriors }
    }
}

/// Samples from the posterior distribution of an inferred sample. The sample is smoothed, that
/// is, it does not include the model isotropic noise.
pub struct PosteriorSamplerMix {
    index: WeightedIndex<f64>,
    posteriors: Vec<ppca_model::PosteriorSampler>,
}

impl Distribution<DVector<f64>> for PosteriorSamplerMix {
    fn sample<R>(&self, rng: &mut R) -> DVector<f64>
    where
        R: Rng + ?Sized,
    {
        let posterior = self.index.sample(rng);
        self.posteriors[posterior].sample(rng)
    }
}
