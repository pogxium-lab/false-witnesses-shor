# Changelog

## v1.2 — 2026-11-01

- Corrected the uncertainty of the offset-corrected ratio (X_s - X_e)/(X_u - X_e) of the sine-window run: the edge-pair
  term X_e enters numerator and denominator and its error is now propagated once (delta method). The ratio is
  1.0632 ± 0.0094 (6.7 sigma from 1, 1.1 sigma below the ideal 1.074), instead of ± 0.0247 (2.6 sigma). Fixed in
  `src/run_ibm_sine_window.py` and `analysis/make_figures_arith.py`; `figures/numbers_arith.json` regenerated. The
  archived run outputs in `data/` are unchanged (they keep the value printed at run time).
- Preprint v1.2: same correction in the text, and clarifications on what the experimental design uses (inputs of the
  generator are N, a and t; the choice t = 5 for the factoring runs and its dependence on r; the testability rule of
  the blind run is equivalent to r(a) < 2^t(a); the classical check C2 is not poly(log N) in general). No data,
  pre-registered statistic or verdict changes.

## v1.1 — 2026-10-28

- Release files renamed (`false-witnesses-shor-v1.1.zip`); citations now use the concept DOIs, which always
  resolve to the latest version (code and data: doi:10.5281/zenodo.23731575; preprint: doi:10.5281/zenodo.23731577).
- Preprint v1.1: acknowledgment of the use of IBM Quantum services added; no change to results or data.

## v1.0 — 2026-10-28

Initial public release accompanying the preprint *False witnesses for small Shor experiments: a coherent modular
multiplication modulo 21* (doi:10.5281/zenodo.23731578). Archived as doi:10.5281/zenodo.23731576.

- Generic Shor circuit generator on a line of heavy-hex qubits, exact arithmetic model, IBM-calibrated noise model.
- False-witness controls (classical exponent, decohered exponent bits, scrambled constants, multiplier 1,
  multiplier outside the subgroup) built by the same generator.
- Scripts of the ten IBM jobs on `ibm_kingston` (27 and 28 October 2026), pre-registration and external
  time-stamping tools (RFC 3161, OpenTimestamps), blind draw of the base from a Bitcoin block hash.
- Raw data of the ten jobs and the scripts that recompute every hardware number, figure and table of the preprint.
