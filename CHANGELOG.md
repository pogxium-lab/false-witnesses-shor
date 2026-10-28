# Changelog

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
