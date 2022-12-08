use crate::ppca_model::Dataset;
use nalgebra::dimension;
use polars::prelude::*;
use polars_lazy::{dsl::Expr, prelude::*};

pub struct DataFrameAdapter {
    keys: Vec<String>,
    dimensions: Vec<String>,
    metric: String,
    dimension_idx: DataFrame,
    sample_idx: DataFrame,
    dataset: Dataset,
}

impl DataFrameAdapter {
    pub fn build(
        df: &DataFrame,
        keys: Vec<String>,
        dimensions: Vec<String>,
        metric: String,
    ) -> Result<DataFrameAdapter, PolarsError> {
        let key_columns = keys
            .iter()
            .map(|key| Expr::Column(key.to_owned().into()))
            .collect::<Vec<_>>();
        let dimension_columns = dimensions
            .iter()
            .map(|dim| Expr::Column(dim.to_owned().into()))
            .collect::<Vec<_>>();

        let dimension_idx = df
            .clone()
            .lazy()
            .select(&dimension_columns)
            .unique(Some(dimensions.clone()), UniqueKeepStrategy::First)
            .sort_by_exprs(&dimension_columns, &vec![false; dimensions.len()], true)
            .with_row_count("__dim_idx", None)
            .collect()?;

        let samples = df
            .clone()
            .lazy()
            .join(
                dimension_idx.clone().lazy(),
                &dimension_columns,
                &dimension_columns,
                JoinType::Inner,
            )
            .groupby(&key_columns)
            .agg([
                Expr::Column("__dim_idx".into()),
                Expr::Column(metric.clone().into()),
            ])
            .with_row_count("__sample_idx", None)
            .collect()?;

        let sample_idx = samples.select(
            keys.iter()
                .map(|key| key.to_owned().into())
                .chain(["__sample_idx".into()])
                .collect::<Vec<Arc<str>>>(),
        )?;

        for (dimension, values) in samples
            .column("__dim_idx")?
            .iter()
            .zip(samples.column(&metric)?.iter())
        {
            // let dimension = dimension
        }

        Ok(DataFrameAdapter {
            keys,
            dimensions,
            metric,
            dimension_idx,
            sample_idx,
            dataset: todo!(),
        })
    }
}
