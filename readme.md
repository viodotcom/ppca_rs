# _Probabilistic_ Principal Component Analysis (PPCA) model

[![PyPI version](https://badge.fury.io/py/ppca-rs.svg)](https://badge.fury.io/py/ppca-rs)
[![Crates.io version](https://img.shields.io/crates/v/ppca)](https://crates.io/crates/ppca)
[![Docs.rs version](https://img.shields.io/docsrs/ppca)](https://docs.rs/ppca)

This project implements a PPCA model implemented in Rust for Python using `pyO3` and `maturin`.

## Installing

This package is available in PyPI!
```bash
pip install ppca-rs
```

And you can also use it natively in Rust:
```bash
cargo add ppca
```

## Why use PPCA?

Glad you asked!

* The PPCA is a simples extension of the PCA (principal component analysis), but can be overall more robust to train.
* The PPCA is a _proper statistical model_. It doesn't spit out only the mean. You get standard deviations, covariances, and all the goodies that come from thre realm of probability and statistics.
* The PPCA model can handle _missing values_. If there is data missing from your dataset, it can extrapolate it with reasonable values and even give you a confidence interval.
* The training converges quickly and will always tend to a global maxima. No metaparameters to dabble with and no local maxima.

## Why use `ppca-rs`?

That's an easy one!

* It's written in Rust, with only a bit of Python glue on top. You can expect a performance in the same leage as of C code.
* It uses `rayon` to paralellize computations evenly across as many CPUs as you have.
* It also uses fancy Linear Algebra Trickery Technology to reduce computational complexity in key bottlenecks. 
* Battle-tested at Vio.com with some ridiculously huge datasets.


## Quick example

```python
import numpy as np
from ppca_rs import Dataset, PPCATrainer, PPCA

samples: np.ndarray

# Create your dataset from a rank 2 np.ndarray, where each line is a sample.
# Use non-finite values (`inf`s and `nan`) to signal masked values
dataset = Dataset(samples)

# Train the model (convenient edition!):
model: PPCAModel = PPCATrainer(dataset).train(state_size=10, n_iters=10)


# And now, here is a free sample of what you can do:

# Extrapolates the missing values with the most probable values:
extrapolated: Dataset = model.extrapolate(dataset)

# Smooths (removes noise from) samples and fills in missing values:
extrapolated: Dataset = model.filter_extrapolate(dataset)

# ... go back to numpy:
eextrapolated_np = extrapolated.numpy()

```

## Juicy extras!

* Tired of the linear? We have support for PPCA mixture models. Make the most of your data with clustering and dimensionality reduction in a single tool!
* Support for adaptation of DataFrames using either `pandas` or `polars`. Never juggle those `df`s in your code again.


## Building from soure

### Prerequisites

You will need [Rust](https://rust-lang.org/), which can be installed locally (i.e., without `sudo`) and you will also need `maturin`, which can be installed by 
```bash
pip install maturin
```
`pipenv` is also a good idea if you are going to mess around with it locally. At least, you need a `venv` set, otherwise, `maturin` will complain with you.

### Installing it locally

Check the `Makefile` for the available commands (or just type `make`). To install it locally, do
```bash
make install    # optional: i=python.version (e.g, `i=3.9`)
```

### Messing around and testing

To mess around, _inside a virtual environment_ (a `Pipfile` is provided for the `pipenv` lovers), do
```bash
maturin develop  # use the flag --release to unlock superspeed!
```
This will install the package locally _as is_ from source.

## How do I use this stuff?

See the examples in the `examples` folder. Also, all functions are type hinted and commented. If you are using `pylance` or `mypy`, it should be easy to navigate.

## Is it faster than the pure Python implemetation you made?

You bet!
