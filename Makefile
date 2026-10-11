# Reproduction of "Adaptive SCHC ACK-on-Error" -- every target is idempotent.
#   make venv         create .venv and install the pinned dependencies
#   make test         fast checks (seconds): code, seeds, shipped results
#   make verify       re-run the WHOLE campaign into .verify/ and compare it
#                     with the shipped results/ (does not touch results/)
#   make experiments  re-run the campaign and OVERWRITE results/
#   make figures      regenerate figs/*.pdf|png from results/
#   make audit        re-check every numeric table cell of the manuscript
#   make paper        build sn-article.pdf (needs a LaTeX installation)
#   make all          experiments + figures + audit + paper
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: venv test verify experiments figures audit paper all clean

venv:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements-lock.txt

test:
	$(PY) -m unittest discover -s tests -v

verify:
	rm -rf .verify && mkdir -p .verify
	$(PY) -m simulator.experiments --all --out .verify
	$(PY) scripts/compare_results.py results .verify

experiments:
	$(PY) -m simulator.experiments --all

figures:
	$(PY) -m simulator.figures

audit:
	$(PY) audit_tables.py

paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error sn-article.tex

all: experiments figures audit paper

clean:
	latexmk -c sn-article.tex >/dev/null 2>&1 || true
	rm -rf .verify
