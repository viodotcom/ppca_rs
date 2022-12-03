use bit_vec::BitVec;
use nalgebra::{DMatrix, DVector};
use rand::distributions::Distribution;
use rand_distr::Bernoulli;
use rayon::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::borrow::Cow;

use crate::output_covariance::OutputCovariance;
use crate::utils::{standard_noise, standard_noise_matrix, Mask};

const LN_2PI: f64 = 1.8378770664093453;

#[derive(Debug, Clone)]
pub struct MaskedSample {
    data: DVector<f64>,
    mask: Mask,
}

impl MaskedSample {
    pub fn new(data: DVector<f64>, mask: Mask) -> MaskedSample {
        MaskedSample { data, mask }
    }

    pub fn unmasked(data: DVector<f64>) -> MaskedSample {
        MaskedSample {
            mask: Mask::unmasked(data.len()),
            data,
        }
    }

    pub fn data_vector(&self) -> DVector<f64> {
        DVector::from(self.data.clone())
    }

    pub fn is_empty(&self) -> bool {
        !self.mask.0.any()
    }

    pub fn mask(&self) -> &Mask {
        &self.mask
    }

    pub fn masked_vector(&self) -> DVector<f64> {
        self.data
            .iter()
            .copied()
            .zip(&self.mask.0)
            .map(|(value, selected)| if selected { value } else { f64::NAN })
            .collect::<Vec<_>>()
            .into()
    }
}

pub struct Dataset {
    pub(crate) data: Vec<MaskedSample>,
}

impl From<Vec<MaskedSample>> for Dataset {
    fn from(value: Vec<MaskedSample>) -> Self {
        Dataset { data: value }
    }
}

impl Dataset {
    pub fn new(data: Vec<MaskedSample>) -> Dataset {
        Dataset { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn output_size(&self) -> Option<usize> {
        self.data.first().map(|sample| sample.mask().0.len())
    }

    pub fn empty_dimensions(&self) -> Vec<usize> {
        let Some(n_dimensions) = self.data.first().map(|sample| sample.mask().0.len()) else {
            return vec![]
        };
        let new_mask = || BitVec::from_elem(n_dimensions, false);
        let poormans_or = |mut this: BitVec, other: &BitVec| {
            for (position, is_selected) in other.iter().enumerate() {
                if is_selected {
                    this.set(position, true);
                }
            }
            this
        };

        let is_not_empty_dimension = self
            .data
            .par_iter()
            .fold(&new_mask, |buffer, sample| {
                poormans_or(buffer, &sample.mask().0)
            })
            .reduce(&new_mask, |this, other| poormans_or(this, &other));

        is_not_empty_dimension
            .into_iter()
            .enumerate()
            .filter(|(_, is_not_empty)| !is_not_empty)
            .map(|(dimension, _)| dimension)
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPCAModel {
    output_covariance: OutputCovariance<'static>,
    mean: DVector<f64>,
}

impl PPCAModel {
    pub fn new(isotropic_noise: f64, transform: DMatrix<f64>, mean: DVector<f64>) -> PPCAModel {
        PPCAModel {
            output_covariance: OutputCovariance::new_owned(isotropic_noise, transform),
            mean,
        }
    }

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

        PPCAModel {
            output_covariance: OutputCovariance {
                isotropic_noise: 1.0,
                transform: Cow::Owned(rand_transform),
            },
            mean: DVector::zeros(output_size),
        }
    }

    pub(crate) fn output_covariance(&self) -> &OutputCovariance<'static> {
        &self.output_covariance
    }

    pub(crate) fn mean(&self) -> &DVector<f64> {
        &self.mean
    }

    pub fn output_size(&self) -> usize {
        self.output_covariance.output_size()
    }

    pub fn state_size(&self) -> usize {
        self.output_covariance.state_size()
    }

    fn uninferred(&self) -> InferredMasked {
        InferredMasked {
            state: DVector::zeros(self.state_size()),
            covariance: DMatrix::identity(self.state_size(), self.state_size()),
        }
    }

    pub fn n_parameters(&self) -> usize {
        1 + self.state_size() * self.output_size() + self.mean.nrows()
    }

    pub fn singular_values(&self) -> DVector<f64> {
        self.output_covariance
            .transform
            .column_iter()
            .map(|column| column.norm().sqrt())
            .collect::<Vec<_>>()
            .into()
    }

    pub fn llk(&self, dataset: &Dataset) -> f64 {
        dataset
            .data
            .par_iter()
            .filter(|sample| !sample.is_empty())
            .map(|sample| {
                let sub_sample = sample.mask.mask(&(sample.data_vector() - &self.mean));
                let sub_covariance = self.output_covariance.masked(&sample.mask);

                -sub_covariance.quadratic_form(&sub_sample) / 2.0
                    - sub_covariance.covariance_log_det() / 2.0
                    - LN_2PI / 2.0 * sub_covariance.output_size() as f64
            })
            .sum()
    }

    pub fn sample(&self, dataset_size: usize, mask_prob: f64) -> Dataset {
        (0..dataset_size)
            .into_par_iter()
            .map(|_| {
                let sampled_state: DVector<f64> = &*self.output_covariance.transform
                    * standard_noise(self.state_size())
                    + &self.mean;
                let noise: DVector<f64> =
                    self.output_covariance.isotropic_noise * standard_noise(self.output_size());
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
            })
            .collect::<Vec<_>>()
            .into()
    }

    pub fn infer(&self, dataset: &Dataset) -> Vec<InferredMasked> {
        dataset
            .data
            .par_iter()
            .map(|sample| {
                if sample.is_empty() {
                    return self.uninferred();
                }

                let sub_sample = sample.mask.mask(&(sample.data_vector() - &self.mean));
                let sub_covariance = self.output_covariance.masked(&sample.mask);

                InferredMasked {
                    state: sub_covariance.estimator_transform() * sub_sample,
                    covariance: sub_covariance.estimator_covariance(),
                }
            })
            .collect()
    }

    pub fn smooth(&self, dataset: &Dataset) -> Dataset {
        self.infer(dataset)
            .into_par_iter()
            .map(|inferred| inferred.smoothed(&self))
            .map(MaskedSample::unmasked)
            .collect::<Vec<_>>()
            .into()
    }

    pub fn extrapolate(&self, dataset: &Dataset) -> Dataset {
        self.infer(dataset)
            .into_par_iter()
            .zip(&dataset.data)
            .map(|(inferred, sample)| inferred.extrapolated(self, sample))
            .collect::<Vec<_>>()
            .into()
    }

    pub fn iterate(&self, dataset: &Dataset) -> PPCAModel {
        let inferred = self.infer(dataset);

        // Updated transform:
        let total_cross_moment = dataset
            .data
            .par_iter()
            .zip(&inferred)
            .map(|(sample, inferred)| {
                let centered_filled = sample.mask.fillna(&(sample.data_vector() - &self.mean));
                centered_filled * inferred.state.transpose()
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
                    .zip(&inferred)
                    .filter(|(sample, _)| sample.mask.0[idx])
                    .map(|(_, inferred)| inferred.second_moment())
                    // In case we get an empty dimension...
                    .chain([DMatrix::zeros(self.state_size(), self.state_size())])
                    .sum::<DMatrix<f64>>()
                    + 1.0 * DMatrix::<f64>::identity(self.state_size(), self.state_size());
                let cross_moment_row = total_cross_moment.row(idx).transpose();
                total_second_moment
                    .qr()
                    .solve(&cross_moment_row)
                    .unwrap_or_else(|| {
                        // Keep old row if you can't solve the linear system.
                        self.output_covariance
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
        let (square_error, deviations_square_sum, total_deviation, totals) = dataset.data.par_iter().zip(&inferred)
        .filter(|(sample, _)| !sample.is_empty())
        .map(
            |(sample, inferred)| {
            let sub_covariance = self.output_covariance.masked(&sample.mask);
            let sub_transform = &*sub_covariance.transform;
            let deviation = sample.mask.fillna(
                &(sample.data_vector()
                    - &*self.output_covariance.transform * &inferred.state
                    - &self.mean),
            );

            (
                (sub_transform * &inferred.covariance).dot(&sub_transform),
                deviation.norm_squared(),
                deviation,
                sample.mask.as_vector()
            )
        }).reduce_with(|
            (square_error, deviation_square_sum, total_deviation, totals),
            (square_error_, deviation_square_sum_, total_deviation_, totals_)| (
                square_error + square_error_,
                deviation_square_sum + deviation_square_sum_,
                total_deviation + total_deviation_,
                totals + totals_
            )
        ).expect("nonempty dataset");

        let average_square_error = (square_error + deviations_square_sum) / totals.sum();
        let new_mean =
            total_deviation.zip_map(
                &totals,
                |sum, count| if count > 0.0 { sum / count } else { 0.0 },
            ) + &self.mean;

        PPCAModel {
            output_covariance: OutputCovariance {
                transform: Cow::Owned(new_transform),
                isotropic_noise: average_square_error.sqrt(),
            },
            mean: new_mean,
        }
    }

    pub fn to_canonical(&self) -> PPCAModel {
        let mut svd = self
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

        PPCAModel {
            output_covariance: OutputCovariance::new_owned(
                self.output_covariance.isotropic_noise,
                new_transform,
            ),
            mean: self.mean.clone(),
        }
    }
}

#[derive(Debug)]
pub struct InferredMasked {
    state: DVector<f64>,
    covariance: DMatrix<f64>,
}

impl InferredMasked {
    fn second_moment(&self) -> DMatrix<f64> {
        &self.state * self.state.transpose() + &self.covariance
    }

    pub fn state(&self) -> &DVector<f64> {
        &self.state
    }

    pub fn covariance(&self) -> &DMatrix<f64> {
        &self.covariance
    }

    pub fn smoothed(&self, ppca: &PPCAModel) -> DVector<f64> {
        &*ppca.output_covariance.transform * self.state() + &ppca.mean
    }

    pub fn extrapolated(&self, ppca: &PPCAModel, sample: &MaskedSample) -> MaskedSample {
        let filtered = self.smoothed(&ppca);
        MaskedSample::unmasked(sample.mask.choose(&sample.data_vector(), &filtered))
    }

    /// Afraid of the big, fat matrix? The method `output_covariance_diagonal` might just
    /// save your life.
    pub fn smoothed_covariance(&self, ppca: &PPCAModel) -> DMatrix<f64> {
        let covariance = &ppca.output_covariance;

        DMatrix::identity(covariance.output_size(), covariance.output_size())
            * covariance.isotropic_noise.powi(2)
            + &*covariance.transform * &self.covariance * covariance.transform.transpose()
    }

    /// Use this not to get lost with big matrices in the output, losing CPU, memory and hair.
    pub fn smoothed_covariance_diagonal(&self, ppca: &PPCAModel) -> DVector<f64> {
        // Here, we will calculate `I sigma^2 + C Sxx C^T` for the unobserved samples in a
        // clever way...

        let covariance = &ppca.output_covariance;

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

    /// Afraid of the big, fat matrix? The method `output_covariance_diagonal` might just
    /// save your life.
    pub fn extrapolated_covariance(&self, ppca: &PPCAModel, sample: &MaskedSample) -> DMatrix<f64> {
        let negative = sample.mask().negate();

        if !negative.0.any() {
            return DMatrix::zeros(ppca.output_size(), ppca.output_size());
        }

        let sub_covariance = ppca.output_covariance.masked(&negative);

        let output_covariance =
            DMatrix::identity(sub_covariance.output_size(), sub_covariance.output_size())
                * sub_covariance.isotropic_noise.powi(2)
                + &*sub_covariance.transform
                    * &self.covariance
                    * sub_covariance.transform.transpose();

        negative.expand_matrix(output_covariance)
    }

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

        let sub_covariance = ppca.output_covariance.masked(&negative);

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
