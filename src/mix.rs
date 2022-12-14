use nalgebra::{DMatrix, DVector};
use rand_distr::{Distribution, WeightedIndex};
use rayon::prelude::*;
use serde_derive::{Deserialize, Serialize};

use crate::ppca_model::{Dataset, InferredMasked, MaskedSample, PPCAModel};

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
pub struct PPCAMix {
    output_size: usize,
    models: Vec<PPCAModel>,
    log_weights: DVector<f64>,
}

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

        PPCAMix {
            output_size: unique_sizes[0],
            models,
            log_weights: robust_log_softmax(log_weights),
        }
    }

    pub fn init(n_models: usize, state_size: usize, dataset: &Dataset, smoothing_factor: f64) -> PPCAMix {
        PPCAMix::new(
            (0..n_models)
                .map(|_| PPCAModel::init(state_size, dataset, smoothing_factor))
                .collect(),
            vec![0.0; n_models].into(),
        )
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn state_sizes(&self) -> Vec<usize> {
        self.models.iter().map(PPCAModel::state_size).collect()
    }

    pub fn n_parameters(&self) -> usize {
        self.models
            .iter()
            .map(PPCAModel::n_parameters)
            .sum::<usize>()
            + self.models.len()
            - 1
    }

    pub fn models(&self) -> &[PPCAModel] {
        &self.models
    }

    pub fn log_weights(&self) -> &DVector<f64> {
        &self.log_weights
    }

    pub fn sample(&self, dataset_size: usize, mask_probability: f64) -> Dataset {
        let index = WeightedIndex::new(self.log_weights.iter().copied().map(f64::exp))
            .expect("can create WeigtedIndex from distribution");
        (0..dataset_size)
            .into_par_iter()
            .map(|_| {
                let model_idx = index.sample(&mut rand::thread_rng());
                self.models[model_idx].sample_one(mask_probability)
            })
            .collect()
    }

    pub(crate) fn llks_one(&self, sample: &MaskedSample) -> DVector<f64> {
        self.models
            .iter()
            .map(|model| model.llk_one(sample))
            .collect::<Vec<_>>()
            .into()
    }

    pub fn llks(&self, dataset: &Dataset) -> DVector<f64> {
        dataset
            .data
            .par_iter()
            .map(|sample| robust_log_softnorm(self.llks_one(sample) + &self.log_weights))
            .collect::<Vec<_>>()
            .into()
    }

    pub fn llk(&self, dataset: &Dataset) -> f64 {
        // Rayon doesn't like to sum empty stuff...
        if dataset.is_empty() {
            return 0.0;
        }

        dataset
            .data
            .par_iter()
            .zip(&dataset.weights)
            .map(|(sample, &weight)| {
                weight * robust_log_softnorm(self.llks_one(sample) + &self.log_weights)
            })
            .sum::<f64>()
    }

    pub fn infer_cluster(&self, dataset: &Dataset) -> DMatrix<f64> {
        let rows: Vec<_> = dataset
            .data
            .par_iter()
            .map(|sample| robust_log_softmax(self.llks_one(sample) + &self.log_weights).transpose())
            .collect();

        DMatrix::from_rows(&*rows)
    }

    pub(crate) fn infer_one(&self, sample: &MaskedSample) -> InferredMaskedMix {
        InferredMaskedMix {
            log_posterior: robust_log_softmax(self.llks_one(sample) + &self.log_weights),
            inferred: self
                .models
                .iter()
                .map(|model| model.infer_one(sample))
                .collect::<Vec<_>>(),
        }
    }

    pub fn infer(&self, dataset: &Dataset) -> Vec<InferredMaskedMix> {
        dataset
            .data
            .par_iter()
            .map(|sample| self.infer_one(sample))
            .collect()
    }

    pub fn smooth(&self, dataset: &Dataset) -> Dataset {
        let smooths = self
            .models
            .iter()
            .map(|model| model.smooth(dataset))
            .collect::<Vec<_>>();
        let clusters = self.infer_cluster(dataset);

        (0..dataset.len())
            .into_par_iter()
            .map(|i| {
                let posterior = clusters.row(i).map(f64::exp);
                let smoothed = posterior
                    .iter()
                    .zip(&*smooths[i].data)
                    .map(|(&pi, smooth_i)| pi * smooth_i.masked_vector())
                    .sum();
                MaskedSample::unmasked(smoothed)
            })
            .collect()
    }

    pub fn extrapolate(&self, dataset: &Dataset) -> Dataset {
        let exrapolated = self
            .models
            .iter()
            .map(|model| model.extrapolate(dataset))
            .collect::<Vec<_>>();
        let clusters = self.infer_cluster(dataset);

        (0..dataset.len())
            .into_par_iter()
            .map(|i| {
                let posterior = clusters.row(i).map(f64::exp);
                let smoothed = posterior
                    .iter()
                    .zip(&*exrapolated[i].data)
                    .map(|(&pi, extrap_i)| pi * extrap_i.masked_vector())
                    .sum();
                MaskedSample::unmasked(smoothed)
            })
            .collect()
    }

    pub fn iterate(&self, dataset: &Dataset) -> PPCAMix {
        // This is already parallelized internally; no need to further parallelize.
        let llks = self
            .models
            .iter()
            .map(|model| model.llks(dataset))
            .collect::<Vec<_>>();
        let log_posteriors = (0..dataset.len())
            .into_par_iter()
            .map(|idx| {
                let llk: DVector<f64> = llks.iter().map(|llk| llk[idx]).collect::<Vec<_>>().into();
                robust_log_softmax(llk + &self.log_weights)
            })
            .collect::<Vec<_>>();

        let (iterated_models, log_weights): (Vec<_>, Vec<f64>) = self
            .models
            .iter()
            .enumerate()
            .map(|(i, model)| {
                // Log-posteriors for this particulat model.
                let log_posteriors: Vec<_> = log_posteriors.par_iter().map(|lp| lp[i]).collect();
                // Let the NaN silently propagate... everything will blow up before this
                // is all over.
                let max_posterior: f64 = log_posteriors
                    .par_iter()
                    .filter_map(|&xi| ordered_float::NotNan::new(xi).ok())
                    .max()
                    .expect("dataset not empty")
                    .into();
                // Use unnormalized posteriors as weights for numerical stability. One of
                // the entries is guaranteed to be 1.0.
                let unnorm_posteriors: Vec<_> = log_posteriors
                    .par_iter()
                    .map(|&p| f64::exp(p - max_posterior))
                    .collect();
                let logsum_posteriors =
                    unnorm_posteriors.iter().copied().sum::<f64>().ln() + max_posterior;
                let dataset = dataset.with_weights(unnorm_posteriors);

                (model.iterate(&dataset), logsum_posteriors)
            })
            .unzip();

        PPCAMix {
            output_size: self.output_size,
            models: iterated_models,
            log_weights: robust_log_softmax(log_weights.into()),
        }
    }

    pub fn to_canonical(&self) -> PPCAMix {
        PPCAMix {
            output_size: self.output_size,
            models: self.models.iter().map(PPCAModel::to_canonical).collect(),
            log_weights: self.log_weights.clone(),
        }
    }
}

pub struct InferredMaskedMix {
    log_posterior: DVector<f64>,
    inferred: Vec<InferredMasked>,
}

impl InferredMaskedMix {
    pub fn log_posterior(&self) -> &DVector<f64> {
        &self.log_posterior
    }

    pub fn posterior(&self) -> DVector<f64> {
        self.log_posterior.map(f64::exp)
    }

    pub fn state(&self) -> DVector<f64> {
        self.log_posterior
            .iter()
            .zip(&self.inferred)
            .map(|(&pi, inferred)| pi * inferred.state())
            .sum()
    }

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

    pub fn smoothed(&self, mix: &PPCAMix) -> DVector<f64> {
        self.inferred
            .iter()
            .zip(&self.posterior())
            .zip(&mix.models)
            .map(|((infered, &weight), ppca)| weight * infered.smoothed(ppca))
            .sum::<DVector<f64>>()
    }

    pub fn extrapolated(&self, mix: &PPCAMix, sample: &MaskedSample) -> DVector<f64> {
        self.inferred
            .iter()
            .zip(&self.posterior())
            .zip(&mix.models)
            .map(|((infered, &weight), ppca)| weight * infered.extrapolated(ppca, sample))
            .sum::<DVector<f64>>()
    }

    pub fn smoothed_covariance(&self, mix: &PPCAMix) -> DMatrix<f64> {
        let mean = self.smoothed(mix);
        self.inferred
            .iter()
            .zip(&self.posterior())
            .zip(&mix.models)
            .map(|((inferred, &weight), ppca)| {
                weight
                    * (inferred.smoothed_covariance(ppca)
                        + (inferred.smoothed(ppca) - &mean)
                            * (inferred.smoothed(ppca) - &mean).transpose())
            })
            .sum::<DMatrix<f64>>()
    }

    pub fn smoothed_covariance_diagonal(&self, mix: &PPCAMix) -> DVector<f64> {
        let mean = self.smoothed(mix);
        self.inferred
            .iter()
            .zip(&self.posterior())
            .zip(&mix.models)
            .map(|((infered, &weight), ppca)| {
                weight
                    * (infered.smoothed_covariance_diagonal(ppca)
                        + (infered.smoothed(ppca) - &mean).map(|v| v.powi(2)))
            })
            .sum()
    }

    pub fn extrapolated_covariance(&self, mix: &PPCAMix, sample: &MaskedSample) -> DMatrix<f64> {
        let mean = self.extrapolated(mix, sample).clone();
        self.inferred
            .iter()
            .zip(&self.posterior())
            .zip(&mix.models)
            .map(|((inferred, &weight), ppca)| {
                weight
                    * (inferred.smoothed_covariance(ppca)
                        + (inferred.extrapolated(ppca, sample) - &mean)
                            * (inferred.extrapolated(ppca, sample) - &mean).transpose())
            })
            .sum::<DMatrix<f64>>()
    }

    pub fn extrapolated_covariance_diagonal(
        &self,
        mix: &PPCAMix,
        sample: &MaskedSample,
    ) -> DVector<f64> {
        let mean = self.extrapolated(mix, sample);
        self.inferred
            .iter()
            .zip(&self.posterior())
            .zip(&mix.models)
            .map(|((infered, &weight), ppca)| {
                weight
                    * (infered.extrapolated_covariance_diagonal(ppca, sample)
                        + (infered.extrapolated(ppca, sample) - &mean).map(|v| v.powi(2)))
            })
            .sum()
    }
}
