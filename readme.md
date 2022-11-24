# `Python+Rust` implementation of the _Probabilistic_ Principal Component Analysis model

This project implements a PPCA model for Python using `pyO3` and `maturin`.


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

To mess around, _inside a virtual environment_ (a `Pipfile ` is provided for the `pipenv` lovers), do
```bash
maturin develop  # use the flag --release to unlock superspeed!
```
This will install the package locally _as is_ from source.

## How do I use this stuff?

See the examples in the `examples` folder. Also, all functions are type hinted and commented. If you are using `pylance` or `mypy`, it should be easy to navigate.

## Is it faster than the pure Python implemetation you made?

You bet!
