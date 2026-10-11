# Adaptive SCHC ACK-on-Error — simulator and reproduction package

Companion artefact of the paper **“Adaptive SCHC ACK-on-Error for Reducing Downlink
Traffic while Preserving Reliability in Constrained IoT Environments based on
LPWANs”** (A. F. Atta, L. Toutain, P. Maillé).

It contains the manuscript, a Monte-Carlo simulator of the SCHC fragmentation /
reassembly procedure over LoRaWAN (pessimistic AoE, optimistic AoE, and the adaptive
AASS / AASS-H controllers), the closed-form evaluation of the paper's model, the raw
results of every experiment, and the scripts that regenerate every figure and check
every numeric table cell of the manuscript.

> \*\*TL;DR\*\* — `make venv \&\& make test \&\& make verify` reproduces the full campaign
> from scratch and compares it, number by number, with the results shipped in
> `results/`

## Contents

```
simulator/                       the simulator (Python package)
  lora\_phy.py                      LoRa Time-on-Air, EU868 duty cycle, energy model
  channel.py                       i.i.d., Gilbert-Elliott, trace-driven and jammed channels
  schc\_aoe.py                      SCHC F/R state machine: pessimistic and optimistic AoE
  analytic.py                      closed-form evaluation of the model, Eq. (1)-(14)
  aass.py                          AASS and AASS-H controllers, cost-optimal threshold
  stats.py                         confidence intervals, paired significance tests
  experiments.py                   the experiment campaign (entry point)
  figures.py                       figure generation, reads results/ only
  example\_trace.py, traces/        trace-driven example (synthetic placeholder trace)
results/\*.json                   raw results of the 8 experiments (the reference)
figs/                            figures: generated \*.pdf/\*.png + static protocol diagrams
audit\_tables.py                  re-checks every numeric table cell of the manuscript
scripts/compare\_results.py       number-by-number comparison of two results directories
tests/test\_repro.py              fast self-checks (seconds)
Makefile, requirements\*.txt      one-command entry points and dependency pins
GITHUB\_SETUP.md                  how this repository was meant to be published
```

## Requirements

* Python **3.11 or newer** (verified with 3.12.3; the authors ran 3.11.15).
* `numpy`, `scipy`, `matplotlib` — nothing else. The exact versions the results were
verified with are in `requirements-lock.txt` (NumPy 2.4.4, SciPy 1.17.1,
Matplotlib 3.10.8).
* Only for rebuilding the PDF: a TeX Live installation with `pdflatex`, `bibtex`,
`latexmk` and the packages `algorithmicx`, `manyfoot`, `mathrsfs`, `enumitem`,
`listings`, `booktabs`, `multirow`, `appendix` (all in a standard full TeX Live).

## Quick start

```bash
git clone <URL-OF-THIS-REPOSITORY> schc-optimistic-ack
cd schc-optimistic-ack

make venv          # creates .venv and installs requirements-lock.txt
make test          # a few seconds: determinism, shipped results, table audit
make verify        # re-runs the WHOLE campaign into .verify/ and compares with results/
```

`make verify` ends with `IDENTICAL` when the re-run reproduces `results/` to within
1e-9 relative tolerance. It never overwrites `results/`. (On the verified setup every
simulated quantity is bit-identical to the shipped results; a few closed-form values of
`simulator/analytic.py` differ in the last digits, below 1e-14 relative, between Python
3.11 and 3.12 because of the platform maths library.)

No `make`? (Windows) The same steps by hand, from the repository root:

```bash
python -m venv .venv \&\& . .venv/bin/activate         # Windows: .venv\\Scripts\\activate
pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python -m simulator.experiments --all --out .verify
python scripts/compare\_results.py results .verify
```

## Regenerating everything

```bash
make experiments   # python -m simulator.experiments --all     -> overwrites results/\*.json
make figures       # python -m simulator.figures               -> writes figs/\*.pdf and \*.png
make audit         # python audit\_tables.py                    -> checks every table cell
make paper         # latexmk ... sn-article.tex                -> sn-article.pdf
make all           # the four steps above, in order
```

A single experiment can be run on its own, e.g.
`python -m simulator.experiments validation comparison` (do **not** combine names
with `--all`). Figures are built from `results/\*.json` only, never from a live
simulation, so the text and the figures cannot drift apart. Measured run times on one
core (Python 3.12.3, 1 vCPU): `validation` 58 s, `comparison` 86 s, `gilbert\_elliott`
89 s, `alpha` 15 s, `energy\_budget` 39 s, `aass` 316 s, `adversary` 14 s, `probing`
287 s — about **15 minutes** in total; figures take a few seconds. Times scale with CPU speed.

## Where each result of the paper comes from

|Paper|Produced by|Source of the numbers|
|-|-|-|
|Fig. 1 – 4 (SCHC messages, F/R timelines, optimistic AoE, false ACKs)|static diagrams `figs/fig2.png` … `fig7.png`|not generated by the simulator|
|Fig. 5, Table 9 — model vs simulation|experiment `validation` → `fig9\_validation`|`results/validation.json`, `simulator/analytic.py`|
|Fig. 6a — correlated losses|experiment `gilbert\_elliott` → `fig11\_robustness`|`results/gilbert\_elliott.json`|
|Fig. 6b — sensitivity to α|experiment `alpha` → `fig11\_robustness`|`results/alpha.json`|
|Fig. 7, Table 10 — comparison of the four schemes|experiment `comparison` → `fig10\_comparison`|`results/comparison.json`|
|Fig. 8, Table 11 — AASS / AASS-H, optimal threshold|experiment `aass` → `fig13\_aass`|`results/aass.json`|
|Fig. 9 — active adversary|experiment `adversary` → `fig14\_adversary`|`results/adversary.json`|
|Tables 5, 6 — Time-on-Air, duty cycle, energy, All-1 design space|experiment `energy\_budget`|`results/energy\_budget.json`, `simulator/lora\_phy.py`|
|Table 4 — impact of K\_max|closed form|`simulator/analytic.py`|
|Table 12 — probing period of AASS-H|experiment `probing`|`results/probing.json`|
|*(not in the manuscript)* `figs/fig12\_energy.\*`|experiment `energy\_budget`|generated for completeness|

`audit\_tables.py` parses `sn-article.tex` and re-checks every numeric cell of
Tables 4, 5, 6, 9, 10, 11 and 12 against the analytical module, the PHY model or the
stored results, at the precision at which it is printed.

## Reproducibility guarantees

* **Deterministic seeds.** Each configuration's seed is a BLAKE2b digest of the
experiment name and the configuration index (not Python's salted `hash()`), so
re-running gives the same numbers in every process and on every machine.
* **Common random numbers** drive the approaches being compared.
* **Fig. 8a shows one representative stream**, not stream 0: among the 200 streams,
the one whose number of mode changes is within a small tolerance of the mean and
whose delay in leaving the optimistic mode is closest to the mean (rule in
`simulator/experiments.py::\_representative\_rep`). Its index and statistics are
stored in `results/aass.json` (`trace\_rep`, `trace\_switches`, `trace\_delay`) and
quoted in the figure caption; `make test` checks that the caption and the data agree.
* Differences you might see on other software versions are expected to be confined to
the last digits of floating-point values; `scripts/compare\_results.py` tolerates 1e-9
(relative) by default and reports anything larger.

## Running against a measured trace

The loss process is confined to `simulator/channel.py`:

```bash
python -m simulator.example\_trace                       # bundled placeholder trace
python -m simulator.example\_trace path/to/my\_trace.txt  # your own capture
```

A trace file holds one 0/1 flag per transmitted frame (1 = frame not received), one
value per line or space/comma separated; `#` starts a comment. Any experiment can be
driven the same way by passing `channel="trace:<path>"` to `simulate\_batch`; the
empirical process then supersedes `p\_u`, `p\_d` and α. `simulator/traces/example\_trace.txt`
is a **synthetic placeholder** provided only so that the example runs; no result of the
paper derives from it.

## Notes on the model

`F` counts *regular* fragments only; the All-1 fragment is accounted for separately.
The simulator keeps the receiver's true state (`received`) and the sender's *belief*
(`known\_missing`) in two separate vectors, so a lost ACK leaves the two views
desynchronised — this is what produces both the deferred repairs of pessimistic AoE
and the false ACKs of optimistic AoE.

## Troubleshooting

|Symptom|Cause / fix|
|-|-|
|`ModuleNotFoundError: No module named 'simulator'`|run commands from the repository root, with `python -m simulator.<module>`|
|`ModuleNotFoundError: numpy` (or scipy, matplotlib)|activate the environment: `. .venv/bin/activate`, or `make venv`|
|`error: --all runs every experiment and cannot be combined…`|drop `--all` when naming experiments|
|`make verify` reports `\[FAIL]` with differences around 1e-12|different NumPy/SciPy build: rerun with `--rtol 1e-6`; larger gaps are real, please open an issue|
|`make paper` fails on a missing `.sty`|install the TeX Live package named in the error (full TeX Live recommended)|
|Figures differ slightly in appearance|different Matplotlib version; the data are unchanged|

## Licences

* **Code** (`simulator/`, `scripts/`, `tests/`, `audit\_tables.py`, `Makefile`): MIT, see `LICENSE`.

## Citation

See `CITATION.cff`. Once the paper is published, replace the placeholder by the final
reference and the archived DOI (Zenodo) of the tagged release.

