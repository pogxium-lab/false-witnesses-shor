# False witnesses for small Shor experiments — code and data

Code and raw data for the preprint *False witnesses for small Shor experiments: a coherent modular multiplication
modulo 21* (Clément Chauffert, 2026). The repository contains a generic Shor circuit generator that takes only
`N`, `a` and the number of exponent bits `t` and lays the whole circuit out on a line of heavy-hex qubits (no
routing). It also contains the false-witness controls built by the same generator with the same two-qubit
structure, an exact arithmetic model of the generated circuits, and an IBM-calibrated noise model. The scripts of
the ten IBM Quantum jobs run on `ibm_kingston` on 27 and 28 October 2026 are included, together with their
pre-registration and external time-stamping tools (RFC 3161, OpenTimestamps, a blind draw of the base from a Bitcoin
block hash). Finally, there are the raw data of these jobs and the analysis scripts that recompute every hardware
number, figure and table of the preprint.

| | |
|---|---|
| Preprint (all versions) | [doi:10.5281/zenodo.23731577](https://doi.org/10.5281/zenodo.23731577) |
| Archived code and data (all versions) | [doi:10.5281/zenodo.23731575](https://doi.org/10.5281/zenodo.23731575) |
| Source repository | <https://github.com/pogxium-lab/false-witnesses-shor> |
| Website | <https://pogxium.com> |

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Reproducing the numbers, figures and tables of the paper](#reproducing-the-numbers-figures-and-tables-of-the-paper)
4. [Hardware experiments](#hardware-experiments)
5. [Verifying the time-stamps and the blind draw](#verifying-the-time-stamps-and-the-blind-draw)
6. [IBM jobs](#ibm-jobs)
7. [Data dictionary](#data-dictionary)
8. [Repository layout](#repository-layout)
9. [Citation](#citation)
10. [Licence](#licence)
11. [Contact](#contact)

## Installation

Python 3.10 or later (tested with Python 3.12.10).

```
git clone https://github.com/pogxium-lab/false-witnesses-shor.git
cd false-witnesses-shor
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip install -r requirements.txt
```

Tested versions: qiskit 2.6.1, qiskit-aer 0.18.2, qiskit-ibm-runtime 0.51.0, numpy 2.5.4, scipy 1.18.1,
matplotlib 3.11.2. All commands below are run from the repository root. No IBM account is needed except to submit
new jobs.

## Quick start

Build a circuit, place it on a device line and compute its exact output distribution:

```python
import sys; sys.path.insert(0, "src")
from dataclasses import replace
import generator_core as GC, shor_line_generator as LG, exact_model as EM
from hardware import HW

g = replace(GC.BASE_GENOME, m0=1)                    # window of 2 indices, no guard bit
qc = LG.build(15, 7, 3, g, LG.Opt(semi=1))           # N=15, a=7, t=3, semiclassical measurements
tq, path = HW("FakeKingston").place(qc)              # line chosen from the calibration, no routing
print(GC.count_2q(tq), path)                         # (724, 0): CZ outside / inside conditional blocks
P = EM.shor_P(15, 7, 3, g)                           # exact output distribution of the exponent register
print(P[EM.crit(15, 7, 3, "C2")].sum())              # ideal success under criterion C2: 0.5
```

False witnesses come from the same generator: `LG.Opt(semi=1, decohere=(2,))` (exponent bit e2 decohered),
`LG.Opt(semi=1, decohere=(0, 1, 2))` (classical exponent), `LG.Opt(semi=1, consts=EM.fake_consts(15, 7, 3, 1))`
(scrambled constants).

Self-test of the package:

```
python src/verify_package.py            # about 1 min
python src/verify_package.py --quick    # about 15 s (FakeKingston only, fewer shots)
```

It checks that (i) the noiseless circuits agree with the exact model (total variation distance below 2e-9 for
N = 15, a = 7, t = 3 and a = 2, t = 4, true circuit and controls; the semiclassical circuit sampled with 20 000 shots
agrees within statistical noise); (ii) after placement on FakeKingston and FakeFez the circuits have **724 CZ**
(N = 15, a = 7, t = 3; 734 for the scrambled-constants control) and **1538 CZ** (N = 21, a = 2, t = 5; 1540 for the
scrambled-constants control); (iii) the ideal C2 success values are 0.5000 / 0.2500 / 0.2500 / 0.1250 (N = 15: true,
classical exponent, decohered e2, scrambled constants) and 0.2008 for the N = 21 true circuit.

## Reproducing the numbers, figures and tables of the paper

| Command | Output | Runtime |
|---|---|---|
| `python analysis/make_figures.py` | every hardware number, figure and table (runs the two scripts below) | ~10 s |
| `python analysis/make_figures_arith.py` | modular multiplication N = 21 (three runs), N = 21 t = 5 with DD, sine-window test | ~5 s |
| `python analysis/make_figures_blind.py` | blind test and Ramsey / echo job of 28 October | ~5 s |
| `python analysis/measurement_trap.py` | measurement trap, Sec. II (stdout; archived as `data/measurement_trap.txt`) | ~15 s |
| `python analysis/poisson_criterion.py` | Poisson criterion for per-round checks, effective noise scale | < 1 s |

Runtimes are for a recent laptop. `make_figures.py` reads only `data/` and writes into `figures/`:

* figures `fig_shor15`, `fig_shor21`, `fig_an`, `fig_wall`, `fig_scaling`, `fig_arith`, `fig_sine`, `fig_blind`
  (PDF and PNG; `fig_blind` is not used in the paper, whose text gives the same numbers);
* LaTeX tables `tab_shor15.tex`, `tab_shor21.tex`, `tab_an.tex`, `tab_calib.tex`, `tab_arith.tex`;
* all numbers in `numbers.json`, `numbers_arith.json`, `numbers_blind.json`.

Every success rate, statistic and fit is recomputed from the raw counts and **asserted** against the analysis
printed at run time (`runtime_analysis_*.json`, `analysis_*.json`); the scripts stop on any mismatch. They also check
`SHA256SUMS.txt`, the SHA-256 of each pre-registration file against the IBM job tags, that the submitted circuits
are byte-identical to the pre-registered ones, the Bitcoin headers and the rule of the blind draw, and (if
`openssl` is on the PATH) the RFC 3161 signatures. Excerpt of the expected output:

```
N=15: true 0.2894±0.0019 {'classical_exponent': (0.2506, 14.2), 'decohered_e2': (0.2443, 16.6), 'scrambled_constants': (0.2436, 16.9)} rand z +21.3
N=21: true 0.0621±0.0010 {'classical_exponent': (0.0633, -0.7), 'decohered_e2': (0.062, 0.0), 'decohered_e3': (0.0614, 0.4), 'decohered_e4': (0.0619, 0.1), 'scrambled_constants': (0.0599, 1.2)} rand z -0.4
...
  z job 1: {'mult_by_1': -3.5, 'wrong_constant': -2.0, 'decohered_e2': -2.6, 'classical_exponent': -4.5} NO SIGNAL
  z job 2: {'mult_by_1_DD': 3.2, 'wrong_constant_DD': 3.7, 'classical_exponent_DD': 4.0} SIGNAL | DD - noDD +4.6 | noDD - job1 +2.8
...
Replication (job 3): SHA256SUMS True | circuits identical to job 2: True | tags ['prereg_sha256:3b2ad7bc133e8e1c3bbd6a8deec12dcc17c6c286d8e8a94b973cf35520404538']
  z job 3: {'mult_by_1_DD': 5.9, 'wrong_constant_DD': 4.9, 'classical_exponent_DD': 5.7} SIGNAL REPLICATED | job 3 - job 2 +0.7 | mean of DD runs +0.0202±0.0019 (10.5 sigma from 0)
...
  D=+0.0319±0.0046 (+7.0 sigma; ideal +0.0647, pred +0.0361, 8.06 sigma) ratio=1.0632±0.0247 (theory 1.0739, +2.6 sigma from 1) eta=0.5789±0.0084 delta=+0.0263±0.0071 chi2=4.5
Blind test N=21 (28 Oct): SHA256SUMS True | stamped files {...} | chain True | RFC 3161 True
    block 973529 00000000000000000001... header time 2026-10-28T09:14:00+00:00 k = 9 a = 19 NOT TESTABLE -> redraw
    block 973530 00000000000000000001... header time 2026-10-28T09:20:50+00:00 k = 7 a = 16 testable
  true                V = +0.1384 +- 0.0215 (68800 shots)  P = [0.2592 0.2497 0.2419 0.2492]  pred +0.1668
  z: {'classical_exponent': 3.19, 'decohered_control': 2.62, 'mult_by_1': 2.1, 'wrong_constant': 3.81} | V_true/se +6.43 | NO SIGNAL
    true                T = +0.0168 +- 0.0038  (+4.42 sigma from 0)  y1 marginal +0.0178 +- 0.0038
  ZZ shift of 44 by 45: -0.70 +- 0.60 kHz (calibrated 8.02 kHz)
```

The readout-robust parity contrast `T` of the blind test is a **post hoc** analysis (not pre-registered); the
scripts label it as such.

## Hardware experiments

Every hardware script runs a local **dry run** by default (IBM fake backend or an archived calibration snapshot);
nothing is submitted without `--ibm` and an interactive confirmation (or `--yes`), and the QPU time of each job is
capped. `--plan` prints the circuits, CZ counts, ESP, shot plan and estimated QPU time, then exits.

| Script | Experiment |
|---|---|
| `src/run_ibm.py` | Shor N = 15 (a = 7, t = 3) or N = 21 (a = 2, t = 5): true circuit and false witnesses on the same line, criterion C2 |
| `src/an_code_demo.py` | AN code (x3) and carry-uncomputation check on an addition echo; `--archived JOB META` re-analyses archived counts |
| `src/run_ibm_arith.py` | one controlled multiplication by 16 mod 21 (t = 3) against four witnesses, statistic S = P(y in {0,4}) - P(y in {2,6}) |
| `src/run_ibm_arith_dd.py` | the same with X-X dynamical decoupling (DD) in every idle window, coherent-error model |
| `src/run_ibm_replication.py` | exact replication of the DD circuits, shot allocation maximising the smallest expected margin |
| `src/run_ibm_shor_dd.py` | Shor N = 21, t = 5 with DD; identity check against the circuits of the run without DD |
| `src/run_ibm_sine_window.py` | Hadamard test of one modular overflow for uniform and sine coset windows |
| `src/run_ibm_blind.py` | blind test N = 21: base drawn after pre-registration from a future Bitcoin block hash |
| `src/run_ibm_ramsey.py` | Ramsey and Hahn-echo characterisation of the line qubits |
| `src/noise_model.py` | noisy simulation (Aer trajectories) of the true circuit and of all controls |

The success criterion C2 ("exact order": a continued-fraction denominator d <= N of y/2^t is the order of a, and
Miller's method then splits N) and every verdict rule were fixed before the runs.

### Pre-registration workflow

The pre-registered runs follow the same pattern (`DIR` = directory of the run):

```
python src/snapshot_calibration.py ibm_kingston DIR prereg            # read-only: calibration_prereg.json, context_prereg.json
python src/run_ibm_replication.py --source data/2026-10-27_ibm_kingston_arith/prereg_arith_dd.json \
       --calibration DIR/calibration_prereg.json --prereg DIR/prereg_replication.json --live-check
python src/run_ibm_shor_dd.py --calibration DIR/calibration_prereg.json --prereg DIR/prereg_shor21_t5_dd.json \
       --reference-qpy data/2026-10-27_ibm_kingston/job_dbgb58djm0ks73eg9aig_circuits.qpy --live-check
python src/prereg_tools.py sums DIR                                   # SHA256SUMS.txt of the pre-registration files
python src/timestamp_tools.py stamp DIR/prereg_*.json DIR/prereg_*_circuits.qpy DIR/SHA256SUMS.txt   # RFC 3161 + OpenTimestamps
python src/run_ibm_replication.py --ibm --prereg DIR/prereg_replication.json --outdir DIR --tag-prereg
python src/run_ibm_shor_dd.py --ibm --prereg DIR/prereg_shor21_t5_dd.json --outdir DIR --tag-prereg
```

1. The dry run writes the pre-registration file (hypothesis, statistic, witnesses, verdict rule, physical qubits,
   CZ counts, ESP, shot plan, QPU estimate, exactness checks, noisy predictions) and the transpiled circuits
   (`*_circuits.qpy`).
2. `SHA256SUMS.txt` lists the pre-registration files; all of them are time-stamped externally (only hashes leave
   the machine).
3. At submission the script checks the circuit file against the hash stored in the pre-registration and both files
   against `SHA256SUMS.txt`, re-checks the circuits (CZ counts, exact simulation, ISA validity on the live target),
   refuses to run if the estimated QPU time exceeds the pre-registered cap, and writes
   `prereg_sha256:<SHA-256 of the pre-registration file>` into the IBM job tags (`--tag-prereg`).
4. The job is analysed with the pre-registered rule and archived (`job_<id>.json`, `job_<id>_circuits.qpy`,
   `meta_<id>.json`, `analysis_<id>.json`).

The blind test adds a draw phase between pre-registration and submission:

```
python src/run_ibm_blind.py --prereg-write --calibration DIR/calibration_prereg.json --outdir DIR
#   writes prereg_blind.json (block height H = tip + 2), prereg_blind_circuits_a<a>.qpy, SHA256SUMS.txt,
#   RFC 3161 (*.tsq, *.tsr, DIR/tsa/) and OpenTimestamps (*.ots) proofs, timestamps_prereg.json
python src/run_ibm_blind.py --draw --outdir DIR                 # once block H has 2 confirmations
python src/run_ibm_blind.py --ibm --outdir DIR --plan           # live checks only
python src/run_ibm_blind.py --ibm --outdir DIR                  # submission (asks for confirmation)
python src/run_ibm_ramsey.py --calibration DIR/calibration_prereg.json --prereg DIR/prereg_ramsey.json --live-check
python src/run_ibm_ramsey.py --ibm --prereg DIR/prereg_ramsey.json --outdir DIR
```

The earlier pre-registered runs use `src/run_ibm_arith.py`, `src/run_ibm_arith_dd.py` and
`src/run_ibm_sine_window.py` with the options `--calibration`, `--prereg`, `--ibm`, `--outdir` (see `--help`).

### Running on IBM Quantum with your own account

Store your IBM Quantum Platform credentials once (the API key is not echoed; the instance may be left empty):

```
python src/run_ibm.py --configure
```

or set the environment variables `QISKIT_IBM_TOKEN` and, optionally, `QISKIT_IBM_INSTANCE`. Scripts that accept
`--account NAME` use a named saved Qiskit Runtime account. Then, for example:

```
python src/run_ibm.py --plan                                     # dry: circuits, CZ counts, ESP, shot plan, QPU time
python src/run_ibm.py                                            # dry run on FakeKingston (noisy dynamic circuits)
python src/run_ibm.py --ibm --backend ibm_kingston --outdir results    # real device, asks for confirmation
python src/run_ibm.py --ibm --job <JOB_ID> --outdir results            # read back and analyse a job
python src/archive_job.py <JOB_ID> results                       # archive raw counts, metrics and circuits
```

Outputs go to `results/` by default (ignored by git). Calibrations change daily, so the physical line chosen by the
package rule and the predicted success differ from those archived here.

## Verifying the time-stamps and the blind draw

Every file listed in a `SHA256SUMS.txt` or carrying a `.ots` / `.tsr` proof is byte-identical to the file that was
hashed or time-stamped (`.gitattributes` disables line-ending conversion so that this holds on every platform).

**SHA-256 lists.** The lists were written on Windows with CRLF line endings; strip them before checking:

```
cd data/2026-10-27_ibm_kingston_replication       # likewise data/2026-10-28_ibm_kingston_blind
tr -d '\r' < SHA256SUMS.txt | sha256sum -c
```

**RFC 3161 time-stamps (FreeTSA, OpenSSL 3).** A response is the authority's signature over the SHA-256 of the
file and the time; it proves that the file existed at that time, independently of the author's clock:

```
cd data/2026-10-28_ibm_kingston_blind
for f in prereg_blind.json SHA256SUMS.txt prereg_ramsey.json prereg_ramsey_circuits.qpy \
         prereg_blind_circuits_a*.qpy blind_draw.json blind_draw_circuits.qpy; do
  openssl ts -verify -in $f.tsr -data $f -CAfile tsa/cacert.pem -untrusted tsa/tsa.crt   # "Verification: OK"
  openssl ts -reply -in $f.tsr -text | grep "Time stamp"                                  # signed time (UTC)
done
```

Signed times: pre-registration files 2026-10-28 08:51:32-37 UTC; draw record and drawn circuits 09:31:55-56 UTC.
`python src/timestamp_tools.py verify FILE ...` runs the same checks.

**OpenTimestamps proofs** (`pip install opentimestamps-client`). `ots verify` checks the Bitcoin attestation
against a local Bitcoin node; without one, a file and its `.ots` proof can be checked at
<https://opentimestamps.org>:

```
ots verify data/2026-10-27_ibm_kingston_replication/SHA256SUMS.txt.ots
ots verify data/2026-10-28_ibm_kingston_blind/prereg_blind.json.ots
```

The proofs of 27 October were created at 21:57:18 UTC, before both submissions (22:05:34 and 22:19:29 UTC), and are
anchored in Bitcoin block 973466. The pre-registration files of 28 October are anchored in block 973528, before
the deciding block 973529; the draw record and the drawn circuits in blocks 973532-973533.

**Hash chain.** The SHA-256 of each time-stamped pre-registration file appears as `prereg_sha256:<hash>` in the
tags of its IBM job (`meta_<id>.json`, field `job_tags`), the blind job also carries `draw_sha256:<hash>` of
`blind_draw.json`, and the submitted circuits `job_<id>_circuits.qpy` are byte-identical to the pre-registered or
drawn circuit files.

**Blind draw.** The rule fixed in `prereg_blind.json` is `k = int(blockhash(H), 16) mod 10`,
`a = [2, 4, 5, 8, 10, 11, 13, 16, 17, 19][k]`, with a redraw on the next block if the drawn base is not testable.
The block hashes can be checked on any Bitcoin block explorer:

| Height | Block hash | k | a |
|---|---|---|---|
| 973529 | `00000000000000000001c4f0428bfab7afa65cf3fb636ed92ed5b328a24bfa15` | 9 | 19 (not testable, redraw) |
| 973530 | `0000000000000000000180afce86c1acd1fea37fcf1f38be4eb6817ed991333f` | 7 | 16 |

```
python -c "print(int('0000000000000000000180afce86c1acd1fea37fcf1f38be4eb6817ed991333f', 16) % 10)"   # 7
```

`analysis/make_figures_blind.py` re-checks offline the 80-byte headers stored in `blind_draw.json` (double
SHA-256, proof of work, consecutive blocks) and the draw rule.

## IBM jobs

All jobs ran on `ibm_kingston` (IBM Heron r2) with the Sampler primitive. Times are job creation times (UTC); QPU
time is the time charged by IBM. Total: 10 jobs, 1 636 121 shots, 515 s of QPU time.

| Job ID | Created (UTC) | Purpose | Shots | QPU (s) |
|---|---|---|---|---|
| `dbg9q25jm0ks73eg6hc0` | 2026-10-27 12:46 | AN code (x3) + carry check, addition echo, 4 inputs, 9 qubits | 4 x 4 000 | 7 |
| `dbg9qfgiudkc73e424jg` | 2026-10-27 12:46 | Shor N = 15, a = 7, t = 3: true circuit (724 CZ, 14 qubits) + 3 false witnesses | 60 000 + 3 x 46 666 | 62 |
| `dbgb58djm0ks73eg9aig` | 2026-10-27 14:18 | Shor N = 21, a = 2, t = 5: true circuit (1538 CZ, 19 qubits) + 5 false witnesses | 60 000 + 5 x 27 999 | 69 |
| `dbgeesljm0ks73egfmc0` | 2026-10-27 18:03 | Hadamard test of one modular overflow, uniform vs sine coset window (12 qubits, 133 CZ) | 2 x 66 000 + 3 x 16 000 | 51 |
| `dbgehntrd2mc73cuiqsg` | 2026-10-27 18:09 | controlled x16 mod 21 (t = 3, 585 CZ, 17 qubits), no DD: true circuit + 4 witnesses | 91 000 + 4 x 45 000 | 81 |
| `dbgfu89j4t8s73ac9l00` | 2026-10-27 19:44 | same with X-X DD: true circuit + 3 witnesses, plus the true circuit without DD | 70 000 + 3 x 30 000 + 40 000 | 61 |
| `dbgi0bkdgr2s73cri510` | 2026-10-27 22:05 | replication of the 4 DD circuits (bit-identical), time-stamped pre-registration | 65 300 + 41 900 + 36 700 + 36 500 | 55 |
| `dbgi6s8te5oc738i7rr0` | 2026-10-27 22:19 | Shor N = 21, a = 2, t = 5 with DD, time-stamped pre-registration | 50 900 + 5 x 23 700 | 61 |
| `dbgs2e8te5oc738irid0` | 2026-10-28 09:32 | blind test N = 21: a = 16 drawn from Bitcoin block 973530, t = 2 (545 CZ, 15 qubits) + 4 witnesses | 68 800 + 4 x 34 400 | 62 |
| `dbgs3a9t9uhs73c8s9j0` | 2026-10-28 09:34 | Ramsey and Hahn echo on qubits 44, 54, 45 (101 circuits x 2 parameter values) | 202 x 64 | 6 |

Main results (details and uncertainties in the paper and in `figures/numbers*.json`):

* **N = 15:** true circuit 0.2894 ± 0.0019 under C2, above every false witness by 14.2-16.9 sigma and above random
  (0.25) by 21.3 sigma.
* **N = 21, t = 5:** no signal with or without DD (true 0.0621 ± 0.0010 and 0.0644 ± 0.0011; random 0.0625), as
  predicted before submission (ESP about 0.3 % and 0.16 %).
* **x16 mod 21:** the run without DD shows a negative S (coherent idle phase); with DD, S = +0.0189 ± 0.0027 above
  every witness by 3.2-4.0 sigma, replicated at +0.0217 ± 0.0028 (4.9-5.9 sigma).
* **Sine window:** D = +0.0319 ± 0.0046 (+7.0 sigma, pre-registered).
* **Blind test:** visibility of the true circuit +0.138 ± 0.022 (6.4 sigma from 0) but margins of 2.1-3.8 sigma
  against the witnesses: NO SIGNAL under the pre-registered rule (>= 3 sigma against every witness). The post hoc
  parity contrast gives +0.017 ± 0.004 for the true circuit and values within 1 sigma of 0 for every witness.

## Data dictionary

### Common files

| File | Content |
|---|---|
| `job_<id>.json` | Raw IBM result: `job_id`, `backend`, `status`, `created`, `metrics` (IBM timestamps `created` / `running` / `finished`, `usage.qpu_charge_time_seconds`), `estimated_usage`, and `circuits[i].registers`: counts of every classical register of circuit `i` (one count dictionary per parameter value for parameter sweeps). |
| `job_<id>_circuits.qpy` | Exact transpiled circuits submitted (Qiskit QPY), in the order of `meta_<id>.json` → `order`. |
| `meta_<id>.json` | Submission metadata: N, a, t, circuit `order`, shot `plan`, backend, physical line, CZ counts, `qpu_estimate_s`; for pre-registered runs `prereg`, `prereg_sha256`, `job_tags`. |
| `analysis_<id>.json`, `runtime_analysis_<id>.json` | Analysis printed at run time (re-computed and asserted by the analysis scripts). |
| `calibration_<tag>.json` | Full backend properties snapshot (T1, T2, readout and gate errors, durations). |
| `context_<tag>.json` | `timestamp` of the snapshot, `last_update_date` of the calibration, `pending_jobs`. |
| `dryrun_*_log.txt`, `run_*_log.txt` | Console output of the dry runs (pre-registration) and of the submissions and analyses. |

In the Shor registers `y` is the exponent register (`y0` = first measured bit), `k1` the window outcome and `dec`
the decohered bits.

### `data/2026-10-27_ibm_kingston/` (AN code, Shor N = 15 and N = 21)

| File | Content |
|---|---|
| `job_*`, `meta_*`, `runtime_analysis_*` | Jobs `dbg9q25jm0ks73eg6hc0` (AN code), `dbg9qfgiudkc73e424jg` (N = 15), `dbgb58djm0ks73eg9aig` (N = 21). The AN metadata lists the input value `x0` of each circuit, the CZ counts, the layout and `duration_s`; the runtime analyses give successes `k` out of `n` per circuit and the `random_baseline`. |
| `calibration_before_N15.json`, `calibration_after_N15.json`, `calibration_before_N21.json`, `context_*.json` | Snapshots before and after the runs. |
| `run_log_an_code.txt`, `run_log_shor15.txt`, `run_log_shor21.txt` | English rendering of the console output recorded during the runs (numerical content unchanged): circuits, CZ counts, calibrated ESP, ideal success, shot plan, job ID, analysis. `make_figures.py` reads the ideal success, CZ count and ESP printed before submission from them. |

Circuit labels: `true` (true circuit), `classical_exponent` (all exponent bits measured right after their
preparation), `decohered_eK` (exponent bit e_K measured right after its preparation), `scrambled_constants`
(multiplication constants replaced, same gates). For the AN job, the register `meas` holds x (5 low bits) and the
carries (4 high bits); filters `raw`, `AN`, `carries`, `AN+carries`.

### `data/2026-10-27_ibm_kingston_arith/` (x16 mod 21 without and with DD, sine window)

| File | Content |
|---|---|
| `job_*`, `meta_*`, `analysis_*` | Jobs `dbgeesljm0ks73egfmc0` (sine window; register `meas` holds c in bit 0 and J, one count dictionary per parameter set, see `pubs` in the metadata), `dbgehntrd2mc73cuiqsg` (no DD), `dbgfu89j4t8s73ac9l00` (DD). |
| `prereg_arith.json`, `prereg_arith_dd.json`, `prereg_sine_window.json` (+ `*_circuits.qpy`) | Pre-registration files written before submission, archived unmodified: statistic, verdict rule, witnesses, physical qubits, shot plan, CZ counts, ESP, exactness checks, noisy predictions; for the DD run also the hypothesis, the coherent-error model (static detuning of qubit 44, calibrated ZZ) and the SHA-256 of the circuit files. |
| `calibration_*`, `context_*` | Snapshots at pre-registration (`prereg`, `prereg_dd`), before and after each job. |

Labels: `true`, `mult_by_1` (multiplier 16 replaced by 1), `wrong_constant` (multiplier 10, outside <2>),
`decohered_e2`, `classical_exponent`; suffix `_DD` / `_noDD` in the DD run. Sine window: `uniform`, `sine`,
`middle_pair`, `edge_pair`, `sine_no_translation`.

### `data/2026-10-27_ibm_kingston_replication/` (replication of the DD run, N = 21 t = 5 with DD)

| File | Content |
|---|---|
| `job_*`, `meta_*`, `analysis_*` | Jobs `dbgi0bkdgr2s73cri510` (replication) and `dbgi6s8te5oc738i7rr0` (N = 21, t = 5 with DD), with the IBM job tags. |
| `prereg_replication.json`, `prereg_shor21_t5_dd.json` (+ `*_circuits.qpy`) | Pre-registration files: statistic, verdict rule, circuits with their SHA-256, physical qubits, shot allocation, exactness and live-device checks, standard-model predictions and, for the replication, a data-based prediction with the probability of meeting the rule. |
| `SHA256SUMS.txt` | SHA-256 of the two pre-registration files, their circuit files, `calibration_prereg.json`, `context_prereg.json` and the two dry-run logs. |
| `*.ots` | OpenTimestamps proofs of the pre-registration files, circuit files and `SHA256SUMS.txt`. |

Labels: as in the DD run; for t = 5, `true_DD`, `classical_exponent_DD`, `decohered_e2_DD` ... `decohered_e4_DD`,
`scrambled_constants_DD`.

### `data/2026-10-28_ibm_kingston_blind/` (blind test, Ramsey / echo)

| File | Content |
|---|---|
| `prereg_blind.json`, `prereg_blind_circuits_a<a>.qpy` (a = 2, 4, 8, 13, 16) | Pre-registration of the blind test, written before the deciding block existed: draw rule (H = 973529, chain tip 973527), candidates, redraw rule, testable bases, rules from N and a, witnesses, statistic, verdict rule, per-candidate circuits with SHA-256, exactness checks, noisy predictions. |
| `prereg_ramsey.json`, `prereg_ramsey_circuits.qpy` | Pre-registration of the Ramsey / echo job: circuits, delays, fit models, validation on an exact simulation with known detunings. |
| `calibration_prereg.json`, `context_prereg.json` | Snapshot used to write both pre-registrations. |
| `SHA256SUMS.txt` | SHA-256 of the two pre-registration files, their circuit files and the snapshot. |
| `blind_draw.json`, `blind_draw_circuits.qpy` | Draw record: block hashes and 80-byte headers from two explorers, header checks, k, a and testability of each draw, parameters of the drawn base, shot plan, rebuilt circuits (identical to `prereg_blind_circuits_a16.qpy`), predictions. |
| `*.tsq`, `*.tsr`, `tsa/cacert.pem`, `tsa/tsa.crt` | RFC 3161 time-stamp queries, signed responses (FreeTSA) and the authority's certificates. |
| `*.ots` | OpenTimestamps proofs of the same files. |
| `timestamps_prereg.json`, `timestamps_draw.json` | Records of the time-stamping (SHA-256, signed time, verification status; not themselves time-stamped). |
| `job_*`, `meta_*`, `analysis_*` | Jobs `dbgs2e8te5oc738irid0` (blind test; register `y`) and `dbgs3a9t9uhs73c8s9j0` (Ramsey / echo; register `meas`, one count dictionary per parameter value, bit j = j-th measured qubit of the family). |

Labels: `true`, `classical_exponent`, `decohered_control` (control bit of the multiplication decohered),
`mult_by_1`, `wrong_constant`.

### Other files

| File | Content |
|---|---|
| `data/scaling_results.csv` | Routed CZ counts (FakeFez) of three successive generator versions for random moduli (input of the scaling figure; only the third version, the line generator, is part of this package). |
| `data/measurement_trap.txt` | Output of `analysis/measurement_trap.py`. |

### Notes on the archived files

* Paths quoted inside the time-stamped pre-registration files refer to the directory layout used at run time;
  `experiences/<date>_kingston*/` corresponds to `data/<date>_ibm_kingston*/` in this repository, and the package
  was then named `shor_generique`. In the files that are not hashed or time-stamped (logs, metadata), these paths
  are given in the repository layout.
* In `prereg_shor21_t5_dd.json` the circuits of the earlier run without DD are referred to by their labels at run
  time: `vrai` = `true`, `faux_cls` = `classical_exponent`, `faux_eK` = `decohered_eK`, `faux_cst` =
  `scrambled_constants`.
* The key names of `job_*.json`, `meta_*.json`, `runtime_analysis_*.json` and `context_*.json` of the first folder,
  and the console logs of the first three jobs, are given in English; no value was changed. In each `run_*_log.txt`
  of the last two folders one line unrelated to the experiment was removed; these logs are not covered by any hash
  or time-stamp. The `command` field of `timestamps_*.json` gives the OpenTimestamps client call in its generic form.

## Repository layout

```
src/
  shor_line_generator.py   line generator build(N, a, t, genome, Opt(...)): true circuit and controls
  line_layout.py           linear-nearest-neighbour primitives (labelled line, fused swaps, QFT on a line)
  generator_core.py        register sizes, lookup-table rule, multiplication constants, gate counter
  coset_arithmetic.py      coset windows, bit sums, continued fractions, Miller's method
  exact_model.py           exact output distribution P(y) and success criteria C1, C2
  hardware.py              line choice from the calibration, routing-free transpilation, ESP
  noise_model.py           calibrated noisy simulation (Aer trajectories)
  dynamic_sim.py           exact simulation of transpiled dynamic circuits (deferred measurement)
  run_ibm*.py, an_code_demo.py   hardware experiments (see "Hardware experiments")
  prereg_tools.py          SHA-256 lists, fingerprint and live checks, submission with tags
  timestamp_tools.py       RFC 3161 and OpenTimestamps: stamp / verify
  snapshot_calibration.py  read-only calibration and context snapshot of a device
  archive_job.py           archive a job (raw counts, metrics, submitted circuits)
  verify_package.py        self-test
analysis/
  make_figures.py          all hardware numbers, figures and tables (calls the two scripts below)
  make_figures_arith.py    modular multiplication runs, t = 5 with DD, sine-window test
  make_figures_blind.py    blind test, Ramsey / echo, hash chain and draw checks
  measurement_trap.py      measurement trap (Sec. II)
  poisson_criterion.py     Poisson criterion for per-round checks
data/                      raw data of the ten IBM jobs (see "Data dictionary")
figures/                   figures, LaTeX tables and numbers*.json produced by analysis/make_figures.py
CITATION.cff, CHANGELOG.md, LICENSE, requirements.txt
```

## Citation

Please cite the preprint and, if you use the code or data, this archive (see also `CITATION.cff`):

```bibtex
@misc{chauffert2026falsewitnesses,
  author    = {Chauffert, Cl{\'e}ment},
  title     = {False witnesses for small {Shor} experiments: a coherent modular multiplication modulo 21},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23731577},
  note      = {Preprint}
}

@software{chauffert2026falsewitnesses_code,
  author    = {Chauffert, Cl{\'e}ment},
  title     = {False witnesses for small {Shor} experiments: code and data},
  version   = {1.1},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23731575},
  url       = {https://github.com/pogxium-lab/false-witnesses-shor}
}
```

## Licence

* Code (`src/`, `analysis/`): MIT, see `LICENSE`.
* Data, figures and documentation (`data/`, `figures/`, this README): Creative Commons Attribution 4.0
  International (CC BY 4.0), <https://creativecommons.org/licenses/by/4.0/>.

The package contains no third-party code; methods from the literature (coset representation, Draper adders,
semiclassical Fourier transform, AN codes) are re-implemented and cited in the docstrings and in the paper. The
FreeTSA certificates in `data/2026-10-28_ibm_kingston_blind/tsa/` are public certificates of that authority,
included so that the RFC 3161 responses can be verified.

## Contact

Clément Chauffert — contact@pogxium.com — ORCID [0009-0002-8230-0236](https://orcid.org/0009-0002-8230-0236) —
<https://pogxium.com>
