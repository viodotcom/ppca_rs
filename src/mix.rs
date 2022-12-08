use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use crate::ppca_model::{Dataset, MaskedSample, PPCAModel};

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

#[derive(Debug, Clone)]
pub struct PPCAMix {
    models: Vec<PPCAModel>,
    log_weights: DVector<f64>,
}

impl PPCAMix {
    pub fn new(models: Vec<PPCAModel>, log_weights: DVector<f64>) -> PPCAMix {
        PPCAMix {
            models,
            log_weights: robust_log_softmax(log_weights),
        }
    }

    pub fn llks(&self, dataset: &Dataset) -> DVector<f64> {
        let llks = self
            .models
            .iter()
            .map(|model| model.llks(dataset))
            .collect::<Vec<_>>();

        (0..dataset.len())
            .into_par_iter()
            .map(|i| {
                let llks: DVector<f64> = llks.iter().map(|llk| llk[i]).collect::<Vec<_>>().into();
                robust_log_softnorm(llks + &self.log_weights)
            })
            .collect::<Vec<_>>()
            .into()
    }

    pub fn llk(&self, dataset: &Dataset) -> f64 {
        self.llks(dataset).sum()
    }

    pub fn infer_cluster(&self, dataset: &Dataset) -> DMatrix<f64> {
        let llks = self
            .models
            .iter()
            .map(|model| model.llks(dataset))
            .collect::<Vec<_>>();

        let rows = (0..dataset.len())
            .into_par_iter()
            .map(|i| {
                let llks: DVector<f64> = llks.iter().map(|llk| llk[i]).collect::<Vec<_>>().into();
                robust_log_softmax(llks + &self.log_weights).transpose()
            })
            .collect::<Vec<_>>();

        DMatrix::from_rows(&*rows)
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
            models: iterated_models,
            log_weights: robust_log_softmax(log_weights.into()),
        }
    }

    pub fn to_canonical(&self) -> PPCAMix {
        PPCAMix {
            models: self.models.iter().map(PPCAModel::to_canonical).collect(),
            log_weights: self.log_weights.clone(),
        }
    }
}
