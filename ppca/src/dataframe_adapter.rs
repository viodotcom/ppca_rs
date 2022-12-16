use crate::{ppca_model::{Dataset, MaskedSample}, utils::Mask};
use bit_vec::BitVec;
use polars::prelude::*;
use polars_lazy::{dsl::Expr, prelude::*};

#[derive(Debug, Clone)]
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

        let dataset = samples
            .column("__dim_idx")?
            .iter()
            .zip(samples.column(&metric)?.iter())
            .map(|(dimension, values)| {
                let AnyValue::List(dimension) = dimension else {
                    panic!("`dimension` should be a series")
                };
                    let AnyValue::List(values) = values else {
                    panic!("`values` should be a series")
                };

                let dimension = dimension.u32().expect("dimension should be an u32");
                let values = values.f64().expect("value should be an f64");
                let output_size = dimension_idx.height();

                let mut data = vec![0.0; output_size];
                let mut mask = BitVec::from_elem(output_size, false);

                for (dim, val) in dimension
                    .into_iter()
                    .zip(values)
                    .filter_map(|(dim, val)| dim.zip(val))
                {
                    data[dim as usize] = val;
                    mask.set(dim as usize, true);
                }

                MaskedSample::new(data.into(), Mask(mask))
            })
            .collect::<Dataset>();

        Ok(DataFrameAdapter {
            keys,
            dimensions,
            metric,
            dimension_idx,
            sample_idx,
            dataset,
        })
    }
}
