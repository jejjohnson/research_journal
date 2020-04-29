.PHONY: conda format style types black test link check notebooks docs
.DEFAULT_GOAL = help

PYTHON = python
VERSION = 3.8
NAME = py_name
ROOT = ./
PIP = pip
CONDA = conda
SHELL = bash
ENV = src
HOST = 127.0.0.1
PORT = 3002

help:	## Display this help
		@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

# JUPYTER NOTEBOOKS
notebooks_to_docs: ## Move notebooks to docs notebooks directory
		@printf "\033[1;34mCreating notebook directory...\033[0m\n"
		mkdir -p docs/notebooks
		@printf "\033[1;34mRemoving old notebooks...\033[0m\n"
		rm -rf docs/notebooks/*.ipynb
		@printf "\033[1;34mCopying Notebooks to directory...\033[0m\n"
		rsync -zarv --progress notebooks/ docs/notebooks/ --include="*.ipynb" --exclude="*.csv" --exclude=".ipynb_checkpoints/" 
		@printf "\033[1;34mDone!\033[0m\n"
jlab_html:
		mkdir -p docs/notebooks
		jupyter nbconvert notebooks/*.ipynb --to html --output-dir docs/notebooks/

docs: ## Build site documentation with mkdocs
		@printf "\033[1;34mCreating full documentation with mkdocs...\033[0m\n"
		mkdocs build --config-file mkdocs.yml --clean --theme material --site-dir site/
		@printf "\033[1;34mmkdocs completed!\033[0m\n\n"

docs-deploy: docs ## Build site documentation with mkdocs
		@printf "\033[1;34mDeploying mkdocs...\033[0m\n"
		mkdocs gh-deploy
		@printf "\033[1;34mDeployment completed!\033[0m\n\n"

docs-live: ## Build mkdocs documentation live
		@printf "\033[1;34mStarting live docs with mkdocs...\033[0m\n"
		mkdocs serve --dev-addr $(HOST):$(PORT) --theme material

docs-live-d: ## Build mkdocs documentation live (quicker reload)
		@printf "\033[1;34mStarting live docs with mkdocs...\033[0m\n"
		mkdocs serve --dev-addr $(HOST):$(PORT) --dirtyreload --theme material
