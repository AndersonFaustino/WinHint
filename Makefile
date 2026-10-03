# WinHint — top-level shortcuts. Every target delegates to tooling/winhint.sh, which runs
# inside the conda env `winhint` created by tooling/create_conda_env.sh (environment.yml,
# requirements*.txt).
#
#   make venv            create/update the conda env (tooling/bootstrap_venv.sh; run
#                        automatically by validate and install-hooks)
#   make validate        commit gate: env check, every host test suite under the coverage
#                        gate (Python, C++ and C each > 90 % of lines), docstrings, strict docs
#   make install-hooks   install the git pre-commit hook: every commit runs make validate
#                        and is refused when it fails
#   make coverage        the coverage gate alone (reports in build/coverage/)
#   make docs            strict documentation build → build/site/index.html
#   make docs-serve      live preview at http://127.0.0.1:8000
#   make docs-check      docstring coverage + Google-style syntax
#   make help            this list

WINHINT := tooling/winhint.sh
MM      ?= $(shell command -v micromamba || echo $(HOME)/.local/share/mamba/bin/micromamba)

.PHONY: help venv validate install-hooks coverage docs docs-serve docs-check

help:
	@sed -n '5,16p' $(firstword $(MAKEFILE_LIST)) | sed 's/^# \{0,3\}//'

# Idempotent: creates the env on first use, updates it when its files change, else a no-op.
venv:
	@tooling/bootstrap_venv.sh

validate: venv
	$(WINHINT) validate

# pre-commit comes from the winhint env (requirements-dev.txt); the hook it writes
# calls that env's interpreter, so commits work from any shell.
install-hooks: venv
	$(MM) run -n winhint pre-commit install --install-hooks
	@echo "  Every commit now runs \`make validate\` and is refused if it fails."
	@echo "  (git commit --no-verify skips it; only for work-in-progress branches.)"

coverage:
	$(WINHINT) coverage

docs:
	$(WINHINT) docs:build

docs-serve:
	$(WINHINT) docs:serve

docs-check:
	$(WINHINT) docs:check
