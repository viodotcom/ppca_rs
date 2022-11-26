mod output_covariance;
mod ppca_model;
mod python_bindings;
mod utils;

#[cfg(test)]
mod test {
    use super::*;

    use nalgebra::{dmatrix, dvector, DMatrix, DVector};
    use ppca_model::PPCAModel;
    use rand_distr::{Bernoulli, Distribution};

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

    #[test]
    fn test_toy_model() {
        let real_model = toy_model();
        let sample = real_model.sample(1_000, 0.2);
        let mut model = PPCAModel::init(2, &sample);

        for iter in 0..1600 {
            println!(
                "At iteration {} model aic is {}",
                iter + 1,
                2.0 * (model.n_parameters() as f64 - model.llk(&sample))
                    / sample.len() as f64
            );
            model = model.iterate(&sample);
        }

        dbg!(model);
    }

    fn big_toy_model() -> PPCAModel {
        fn standard_noise_matrix(rows: usize, cols: usize) -> DMatrix<f64> {
            DMatrix::from_vec(
                rows,
                cols,
                Bernoulli::new(0.1)
                    .unwrap()
                    .sample_iter(rand::thread_rng())
                    .take(rows * cols)
                    .map(|selected| selected as i8 as f64)
                    .collect(),
            )
        }

        PPCAModel::new(0.1, standard_noise_matrix(200, 16), DVector::zeros(200))
    }

    #[test]
    fn test_big_toy_model() {
        let real_model = big_toy_model();
        let sample = real_model.sample(100_000, 0.2);
        let mut model = PPCAModel::init(16, &sample);

        for iter in 0..24 {
            println!(
                "At iteration {} model aic is {}",
                iter + 1,
                2.0 * (model.n_parameters() as f64 - model.llk(&sample))
                    / sample.len() as f64
            );
            model = model.iterate(&sample);
        }

        model.to_canonical();
        // dbg!(model);
    }
}
