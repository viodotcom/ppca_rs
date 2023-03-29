use bit_vec::BitVec;
use nalgebra::DVector;
use rayon::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::{ops::Index, sync::Arc};

use crate::utils::Mask;

/// A data sample with potentially missing values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskedSample {
    pub(crate) data: DVector<f64>,
    pub(crate) mask: Mask,
}

impl MaskedSample {
    /// Creates a masked sample from a vector, masking all elements which are not finite (e.g.,
    /// `NaN` and `inf`).
    pub fn mask_non_finite(data: DVector<f64>) -> MaskedSample {
        let mask = data.iter().copied().map(f64::is_finite).collect::<BitVec>();
        MaskedSample::new(data, Mask(mask))
    }

    /// Creates a masked sample from data and a mask. The value is considered missing if its index
    /// in the masked is set to `false`.
    pub fn new(data: DVector<f64>, mask: Mask) -> MaskedSample {
        MaskedSample { data, mask }
    }

    /// Creates a sample without any masked values.
    pub fn unmasked(data: DVector<f64>) -> MaskedSample {
        MaskedSample {
            mask: Mask::unmasked(data.len()),
            data,
        }
    }

    /// Returns the data vector associated with this sample.
    pub fn data_vector(&self) -> DVector<f64> {
        DVector::from(self.data.clone())
    }

    /// Returns `true` if all values are masked.
    pub fn is_empty(&self) -> bool {
        !self.mask.0.any()
    }

    /// Returns the mask of this sample. The value is considered missing if its index
    /// in the masked is set to `false`.
    pub fn mask(&self) -> &Mask {
        &self.mask
    }

    /// Returns whether the `idx` dimension in this sample is set.
    ///
    /// # Panics
    ///
    /// This function panics if `idx` is out of bounds.
    pub fn is_set(&self, idx: usize) -> bool {
        self.mask.is_set(idx)
    }

    /// Returns the data vector associated with this sample, substituting all masked values by `NaN`.
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

impl Index<usize> for MaskedSample {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        if self.is_set(index) {
            &self.data[index]
        } else {
            panic!("Index out of bounds: index {index} is masked in sample")
        }
    }
}

/// Represents a dataset. This is a wrapper over a 2D array of dimensions
/// `(n_samples, n_features)`.
///
/// ## Note
///
/// All arrays involved have to be of data type `float64`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// The data rows of this dataset.
    pub data: Arc<Vec<MaskedSample>>,
    /// The weights associated with each sample. Use this only if you are using the PPCA as a
    /// component of a greater EM scheme (or otherwise know what you are doing). Else, let the
    /// package set it automatically to 1.
    pub weights: Vec<f64>,
}

impl From<Vec<MaskedSample>> for Dataset {
    fn from(value: Vec<MaskedSample>) -> Self {
        Dataset {
            weights: vec![1.0; value.len()],
            data: Arc::new(value),
        }
    }
}

impl FromIterator<MaskedSample> for Dataset {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = MaskedSample>,
    {
        let data: Vec<_> = iter.into_iter().collect();
        Self::new(data)
    }
}

impl FromIterator<(MaskedSample, f64)> for Dataset {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (MaskedSample, f64)>,
    {
        let (data, weights): (Vec<_>, Vec<_>) = iter.into_iter().unzip();
        Self::new_with_weights(data, weights)
    }
}

impl FromParallelIterator<MaskedSample> for Dataset {
    fn from_par_iter<T>(iter: T) -> Self
    where
        T: IntoParallelIterator<Item = MaskedSample>,
    {
        let data: Vec<_> = iter.into_par_iter().collect();
        Self::new(data)
    }
}

impl FromParallelIterator<(MaskedSample, f64)> for Dataset {
    fn from_par_iter<T>(iter: T) -> Self
    where
        T: IntoParallelIterator<Item = (MaskedSample, f64)>,
    {
        let (data, weights): (Vec<_>, Vec<_>) = iter.into_par_iter().unzip();
        Self::new_with_weights(data, weights)
    }
}

impl Dataset {
    /// Creates a new dataset from a set of masked samples.
    pub fn new(data: Vec<MaskedSample>) -> Dataset {
        Dataset {
            weights: vec![1.0; data.len()],
            data: Arc::new(data),
        }
    }

    /// Creates a new dataset from a set of weighted masked samples.
    pub fn new_with_weights(data: Vec<MaskedSample>, weights: Vec<f64>) -> Dataset {
        assert_eq!(data.len(), weights.len());
        Dataset {
            data: Arc::new(data),
            weights,
        }
    }

    /// Creates a new dataset with the same sample, but with different weights. This operation is
    /// cheap, since it does not clone the dataset (it's protected by an `Arc`).
    pub fn with_weights(&self, weights: Vec<f64>) -> Dataset {
        Dataset {
            data: self.data.clone(),
            weights,
        }
    }

    /// The length of this dataset.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether this dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The number of dimensions in each sample. Returns `None` if dataset is empty.
    pub fn output_size(&self) -> Option<usize> {
        self.data.first().map(|sample| sample.mask().0.len())
    }

    /// Lists the dimensions which as masked in __all__ samples in this dataset.
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
