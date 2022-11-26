
v?=3.9

help: ## Help for commands in Makefile
	@awk -F ':|##' '/^[^\t].+?:.*?##/ {\
		printf "\033[36m%-30s\033[0m %s\n", $$1, $$NF \
	}' $(MAKEFILE_LIST)

clean: ## Cleans the target/wheels folder
	rm -rf target/wheels

build: clean ## Builds the wheel using maturin.
	python -m maturin build --release --interpreter ${v}

install: build ## Builds and installs the wheel in the local environment.
	pip${v} install --force-reinstall --no-deps ./target/wheels/*.whl

publish:
	docker run --rm -v $(shell pwd):/io ghcr.io/pyo3/maturin publish \
		--compatibility manylinux2014 \
		--interpreter ${v} \
		--username ${user} \
		--password ${password}
