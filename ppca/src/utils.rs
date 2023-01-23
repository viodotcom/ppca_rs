use bit_vec::BitVec;
use nalgebra::{DMatrix, DVector};
use rand::distributions::Distribution;
use rand_distr::StandardNormal;
use serde_derive::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mask(pub BitVec);

impl Mask {
    pub(crate) fn unmasked(size: usize) -> Mask {
        Mask(BitVec::from_elem(size, true))
    }

    pub(crate) fn negate(&self) -> Mask {
        let mut neg = self.0.clone();
        neg.negate();
        Mask(neg)
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

    pub fn is_set(&self, idx: usize) -> bool {
        self.0[idx]
    }

    pub(crate) fn mask(&self, vector: &DVector<f64>) -> DVector<f64> {
        self.filter(vector.data.as_vec())
            .copied()
            .collect::<Vec<_>>()
            .into()
    }

    pub(crate) fn expand(&self, vector: &DVector<f64>) -> DVector<f64> {
        let mut it = vector.iter();
        let expanded = self
            .0
            .iter()
            .map(|selected| {
                if selected {
                    *it.next().expect("input vector too short for mask")
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>()
            .into();

        assert!(
            it.next().is_none(),
            "input vector has more entries than mask"
        );

        expanded
    }

    pub(crate) fn expand_matrix(&self, matrix: DMatrix<f64>) -> DMatrix<f64> {
        assert!(matrix.is_square(), "cannot expand rectangular matrices");
        let ncols = matrix.ncols();
        let mut it = matrix.row_iter();

        let expanded_rows = self
            .0
            .iter()
            .map(|selected| {
                if selected {
                    self.expand(
                        &it.next()
                            .expect("input matrix too short in row size for mask")
                            .transpose(),
                    )
                    .clone()
                } else {
                    DVector::zeros(ncols)
                }
            })
            .map(|row_in_column_form| row_in_column_form.transpose())
            .collect::<Vec<_>>();
        let expanded = DMatrix::from_rows(&expanded_rows);

        assert!(
            it.next().is_none(),
            "input matrix has more entries than mask"
        );

        expanded
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
