# To install make, use the following command:
# sudo apt-get install build-essential
# Install uv:
# curl -LsSf https://astral.sh/uv/install.sh | sh

help:
	bash run.sh help

clean:
	bash run.sh clean

initialize:
	bash run.sh create:venv
	bash run.sh install:dev
	bash run.sh install:package

install-package:
	bash run.sh install:package

install-dev:
	bash run.sh install:dev
	bash run.sh install:package

install-docs:
	bash run.sh install:docs
	bash run.sh install:package

model:
	bash run.sh model

pre-commit-install:
	bash run.sh pre-commit:install

pre-commit-update:
	bash run.sh pre-commit:update

pre-commit-run:
	bash run.sh pre-commit:run


test-model:
	bash run.sh test:model

test:
	bash run.sh test

test-cov:
	bash run.sh test:cov