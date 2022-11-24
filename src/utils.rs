use bit_vec::BitVec;
use nalgebra::{DMatrix, DVector};
use rand::distributions::Distribution;
use rand_distr::StandardNormal;

pub(crate) fn standard_noise(size: usize) -> DVector<f64> {
    DVector::from(
        StandardNormal
            .sample_iter(rand::thread_rng())
            .take(size)
            .collect::<Vec<f64>>(),
    )
}

pub(crate) fn standard_noise_matrix(rows: usize, cols: usize) -> DMatrix<f64> {
    DMatrix::from_vec(
        rows,
        cols,
        StandardNormal
            .sample_iter(rand::thread_rng())
            .take(rows * cols)
            .collect(),
    )
}

#[derive(Debug, Clone)]
pub struct Mask(pub BitVec);

impl Mask {
    pub(crate) fn unmasked(size: usize) -> Mask {
        Mask(BitVec::from_elem(size, true))
    }

    pub(crate) fn filter<'a, I: IntoIterator>(&'a self, it: I) -> impl 'a + Iterator<Item = I::Item>
    where
        I::IntoIter: 'a,
    {
        self.0
            .iter()
            .zip(it)
            .filter(|(selected, _)| *selected)
            .map(|(_, element)| element)
    }

    pub(crate) fn mask(&self, vector: &DVector<f64>) -> DVector<f64> {
        self.filter(vector.data.as_vec())
            .copied()
            .collect::<Vec<_>>()
            .into()
    }

    pub(crate) fn fillna(&self, vector: &DVector<f64>) -> DVector<f64> {
        vector
            .data
            .as_vec()
            .iter()
            .zip(&self.0)
            .map(|(xi, selected)| if selected { *xi } else { 0.0 })
            .collect::<Vec<f64>>()
            .into()
    }

    pub(crate) fn as_vector(&self) -> DVector<f64> {
        self.0
            .iter()
            .map(|selected| selected as i8 as f64)
            .collect::<Vec<f64>>()
            .into()
    }

    pub(crate) fn choose(&self, selected: &DVector<f64>, excluded: &DVector<f64>) -> DVector<f64> {
        self.0
            .iter()
            .zip(selected)
            .zip(excluded)
            .map(
                |((is_selected, selected), excluded)| {
                    if is_selected {
                        *selected
                    } else {
                        *excluded
                    }
                },
            )
            .collect::<Vec<_>>()
            .into()
    }
}
