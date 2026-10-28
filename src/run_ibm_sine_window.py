"""Sine versus uniform coset window: shallow Hadamard test of the translation by N on an IBM device.

Default: DRY RUN (local noisy simulation); nothing is submitted without --ibm AND confirmation. QPU budget capped
at 60 s.

  Plan only:                    python src/run_ibm_sine_window.py --calibration CAL.json --plan
  Dry run + pre-registration:   python src/run_ibm_sine_window.py --calibration CAL.json --prereg PREREG.json
  Real device (submits the transpiled circuit and the shot plan stored with the pre-registration):
                                python src/run_ibm_sine_window.py --ibm --prereg PREREG.json --outdir DIR
  Read back a job:              python src/run_ibm_sine_window.py --ibm --job <ID> --outdir DIR

What is measured. The coset state of the work register of the generator is |C_a(v)> = sum_j a(j) |v + jN>, with a
window a(j) over M = 2^m indices. The register can only represent the coset approximately: adding N (one modular
"overflow") shifts the window by one index, and the overlap of the shifted state with the original one is the
lag-one autocorrelation of the window,
    rho_a(1) = <C_a(v)| T_N |C_a(v)> = sum_j a(j) a(j+1):  7/8 = 0.875 (uniform), cos(pi/9) = 0.9397 (sine).
This overlap is the per-overflow fidelity factor of coset arithmetic; the sine window used by the generator is the
optimum for this quantity. A Hadamard test measures it directly: control qubit c in |+>, controlled translation
c-T_N, readout of c in the X basis, <X_c> = Re <psi| T_N |psi> = rho_a(1).

Circuit (N = 21, m = 3, w = 8; 12 qubits). The register R is prepared directly in the Fourier basis with the value
v + T K = 1 + 2 x 13 = 27 = 6 + 21 (two additions of K = 13 to v = 1 with one real overflow; additions of constants
are single-qubit phases in the Fourier basis). An index register J holds the window, R += N J by controlled
phases, c-T_N by controlled phases, then J goes to the Fourier basis; the phase exp(2 i pi k j / M) carried by J
becomes a phase exp(-2 i pi k / M) on c after the translation and is removed by three controlled phases J -> c
(deferred measurement). ONE transpiled circuit is used for every setting; windows and controls differ only by
rotation angles (parameters).

Settings (ideal <X_c> = A):
    uniform               uniform window                       A = 0.875
    sine                  sine window                          A = cos(pi/9) = 0.9397
    middle_pair           (|3> + |4>)/sqrt 2                   A = 0.5
    edge_pair             (|0> + |7>)/sqrt 2                   A = 0 (null control: measures the readout offset)
    sine_no_translation   sine window, c-T_N and correction off A = 1 (attenuation of the test itself)

Statistic (fixed before the run): D = <X_c>(sine) - <X_c>(uniform), ideal +0.0647.
Verdict rule: WINDOW EFFECT OBSERVED iff D >= 3 standard errors, SE = sqrt[(1 - Xs^2)/ns + (1 - Xu^2)/nu]
(binomial, independent settings). Secondary, reported: ratio (Xs - Xe)/(Xu - Xe) corrected for the readout offset
measured by edge_pair (theory 1.0739, insensitive to a common attenuation), fit <X_c> = eta A + delta over the four
windows (common attenuation eta, offset delta, chi^2 with 2 degrees of freedom), and sine_no_translation.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import time
import warnings
from pathlib import Path

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile  # noqa: E402
from qiskit.circuit import Parameter, ParameterVector  # noqa: E402
from qiskit.synthesis import OneQubitEulerDecomposer  # noqa: E402
from qiskit.synthesis.qft import synth_qft_full  # noqa: E402

import generator_core as GC  # noqa: E402
from hardware import HW  # noqa: E402

TWO_PI = 2 * math.pi
N0, M0, V0, K0, T0 = 21, 3, 1, 13, 2
HARD_CAP = 60.0
Z_REQ = 3.0
REP_DELAY = 250e-6
OVERHEAD = 2.0                    # s per circuit / parameter set (conservative; IBM documentation)
MAIN, CONTROLS = ("uniform", "sine"), ("middle_pair", "edge_pair", "sine_no_translation")
CONTROL_SHARE = 0.25              # shots of each control relative to each main setting
_dec = OneQubitEulerDecomposer("U3")


# =============================================================================== circuit
def ang(K, j, w):
    """Phase of qubit j (Fourier register without swaps, weight 2^(w-1-j)) that adds K mod 2^w."""
    th = (TWO_PI * (K % (1 << w)) * (1 << (w - 1 - j)) / (1 << w)) % TWO_PI
    return th if min(th, TWO_PI - th) > 1e-12 else 0.0


def windows(M=8):
    j = np.arange(M)
    W = {"uniform": np.ones(M), "sine": np.sin(np.pi * (j + 1) / (M + 1))}
    p = np.zeros(M)
    p[M // 2 - 1] = p[M // 2] = 1
    W["middle_pair"] = p
    q = np.zeros(M)
    q[0] = q[M - 1] = 1
    W["edge_pair"] = q
    return {k: v / np.linalg.norm(v) for k, v in W.items()}


def autocorr(a):
    return float(np.dot(a[:-1], a[1:]))


def window_angles(a):
    """Angles of the fixed 3-qubit template for a symmetric window: Schmidt decomposition of the two low
    qubits, the top qubit makes the state symmetric."""
    M = len(a)
    assert M == 8 and np.allclose(a, a[::-1])
    B = (np.sqrt(2) * a[: M // 2]).reshape(2, 2)
    U, S, Vt = np.linalg.svd(B)
    return [2 * math.atan2(S[1], S[0]), *_dec.angles(U), *_dec.angles(Vt.T)]


def template(N=N0, m=M0, v=V0, K=K0, T=T0):
    """Registers c (control), J (m qubits, window index), R (w qubits, coset, Fourier basis)."""
    M = 1 << m
    w = (N * M - 1).bit_length()
    c = QuantumRegister(1, "c")
    J = QuantumRegister(m, "j")
    R = QuantumRegister(w, "r")
    cr = ClassicalRegister(1 + m, "meas")
    qc = QuantumCircuit(c, J, R, cr)
    th = ParameterVector("win", 7)
    s, g = Parameter("s"), Parameter("g")
    for j in range(w):                                   # |v + T K> in the Fourier basis (product state)
        qc.h(R[j])
        a0 = ang(v + T * K, j, w)
        if a0:
            qc.p(a0, R[j])
    qc.ry(th[0], J[1])                                   # window state on J
    qc.cx(J[1], J[0])
    qc.u(th[1], th[2], th[3], J[1])
    qc.u(th[4], th[5], th[6], J[0])
    qc.h(J[2])
    qc.cx(J[2], J[1])
    qc.cx(J[2], J[0])
    for i in range(m):                                   # R += N J
        for j in range(w):
            a_ = ang(N << i, j, w)
            if a_:
                qc.cp(a_, J[i], R[j])
    qc.h(c[0])
    for j in range(w):                                   # controlled translation c-T_N (s = 1)
        a_ = ang(N, j, w)
        if a_:
            qc.cp(s * a_, c[0], R[j])
    qc.compose(synth_qft_full(m, do_swaps=False), J, inplace=True)
    for i in range(m):                                   # phase correction (g = 1), k = sum_i J_i 2^(m-1-i)
        qc.cp(g * TWO_PI * (1 << (m - 1 - i)) / M, J[i], c[0])
    qc.h(c[0])
    qc.measure(c[0], cr[0])
    for i in range(m):
        qc.measure(J[i], cr[1 + i])
    return qc, dict(th=th, s=s, g=g, w=w, M=M)


def settings():
    """label -> (window, s, g, ideal <X_c>)."""
    W = windows()
    out = {k: (W[k], 1.0, 1.0, autocorr(W[k])) for k in ("uniform", "sine", "middle_pair", "edge_pair")}
    out["sine_no_translation"] = (W["sine"], 0.0, 0.0, 1.0)
    return out


def values(P, a, s, g):
    """Parameter values of one setting, by parameter name."""
    vals = {p.name: float(x) for p, x in zip(P["th"], window_angles(a))}
    vals[P["s"].name] = float(s)
    vals[P["g"].name] = float(g)
    return vals


def vec(circ, vals):
    """Values in the parameter order of `circ` (transpiled or not)."""
    return [vals[p.name] for p in circ.parameters]


def prepare(hw, seeds=40, layout=None):
    """Transpile once (parameterised); the layout with the best estimated success probability (gates, readout,
    idle periods, today's calibration) over `seeds` transpiler seeds, unless `layout` is given."""
    qc, P = template()
    st = settings()
    ref = qc.assign_parameters(vec(qc, values(P, *st["sine"][:3])))
    best = None
    for sd in range(seeds if layout is None else 1):
        t = transpile(ref, hw.be, optimization_level=3, seed_transpiler=sd, initial_layout=layout)
        sc = hw.esp(t)["total"]
        if best is None or sc > best[0]:
            best = (sc, sd, t.layout.initial_index_layout(filter_ancillas=True))
    _, sd, lay = best
    tq = transpile(qc, hw.be, optimization_level=3, seed_transpiler=sd, initial_layout=lay)
    binds = {k: values(P, a, s, g) for k, (a, s, g, _) in st.items()}
    bound = tq.assign_parameters(vec(tq, binds["sine"]))
    lg = hw.esp(bound)
    info = dict(tq=tq, binds=binds, ideal={k: v[3] for k, v in st.items()}, cz=GC.count_2q(bound)[0],
                esp=math.exp(lg["total"]), duration=lg["duration_us"], layout=[int(q) for q in lay], seed=sd,
                logical_qubits=qc.num_qubits, w=P["w"],
                roles={"c": int(lay[0]), "J": [int(q) for q in lay[1:1 + M0]], "R": [int(q) for q in lay[1 + M0:]]})
    return info


def qpu_time(plan, info):
    return OVERHEAD * len(plan) + sum(n * (REP_DELAY + info["duration"] * 1e-6 + 5e-6) for n in plan.values())


def shot_plan(info, budget):
    names = list(MAIN) + list(CONTROLS)
    w = {k: (1.0 if k in MAIN else CONTROL_SHARE) for k in names}
    per = sum(w.values()) * (REP_DELAY + info["duration"] * 1e-6 + 5e-6)
    unit = (0.98 * budget - OVERHEAD * len(names)) / per
    plan = {k: int(unit * w[k]) // 1000 * 1000 for k in names}
    return plan, qpu_time(plan, info)


def pubs(info, plan):
    """Two PUBs of the same circuit: the main windows, then the controls (one parameter set per setting)."""
    tq, b = info["tq"], info["binds"]
    return [(tq, [vec(tq, b[k]) for k in MAIN], plan[MAIN[0]]),
            (tq, [vec(tq, b[k]) for k in CONTROLS], plan[CONTROLS[0]])]


# =============================================================================== statistics
def x_from_counts(cnt):
    """<X_c> from counts of the 'meas' register (bit 0 = c), binomial standard error."""
    n = sum(cnt.values())
    k0 = sum(v for key, v in cnt.items() if (int(key, 2) if isinstance(key, str) else int(key)) & 1 == 0)
    x = (2 * k0 - n) / n
    return x, math.sqrt(max(1 - x * x, 1e-12) / n), n


def analyse(xs):
    """xs: label -> (x, se, n)."""
    print()
    print("=== Sine versus uniform window, coset N=21 (w=8, M=8), Hadamard test of T_N ===")
    ideal = {k: v[3] for k, v in settings().items()}
    for k, (x, e, n) in xs.items():
        print(f"  {k:20s} <X_c> = {x:+.4f} +- {e:.4f}   (ideal {ideal[k]:.4f}, {n} shots)")
    (xu, su, _), (xs_, ss, _) = xs["uniform"], xs["sine"]
    D, sD = xs_ - xu, math.hypot(su, ss)
    z = D / sD
    res = dict(X={k: dict(x=v[0], se=v[1], n=v[2]) for k, v in xs.items()}, D=D, se_D=sD, z=z)
    print(f"  D = sine - uniform = {D:+.4f} +- {sD:.4f}  ({z:+.1f} sigma; ideal +{ideal['sine'] - ideal['uniform']:.4f})")
    if "edge_pair" in xs:
        xe, se_, _ = xs["edge_pair"]
        num, den = xs_ - xe, xu - xe
        r = num / den
        sr = abs(r) * math.sqrt((ss ** 2 + se_ ** 2) / num ** 2 + (su ** 2 + se_ ** 2) / den ** 2) if den else float("nan")
        rth = ideal["sine"] / ideal["uniform"]
        print(f"  offset-corrected ratio (Xs - Xe)/(Xu - Xe) = {r:.4f} +- {sr:.4f}  (theory {rth:.4f}; no window "
              f"effect: 1; {(r - 1) / sr:+.1f} sigma from 1, {(r - rth) / sr:+.1f} sigma from theory)")
        labs = [k for k in ("uniform", "sine", "middle_pair", "edge_pair") if k in xs]
        A = np.array([ideal[k] for k in labs])
        X = np.array([xs[k][0] for k in labs])
        Wt = np.array([1 / xs[k][1] ** 2 for k in labs])
        Dm = np.vstack([A, np.ones_like(A)]).T
        cov = np.linalg.inv(Dm.T @ (Wt[:, None] * Dm))
        eta, delta = cov @ (Dm.T @ (Wt * X))
        chi2 = float((Wt * (X - Dm @ np.array([eta, delta])) ** 2).sum())
        print(f"  fit <X_c> = eta A + delta: eta = {eta:.4f} +- {math.sqrt(cov[0, 0]):.4f}, delta = {delta:+.4f} +- "
              f"{math.sqrt(cov[1, 1]):.4f}, chi2 = {chi2:.1f} ({len(labs) - 2} dof)")
        res.update(ratio=r, se_ratio=sr, ratio_theory=rth, eta=float(eta), delta=float(delta), chi2=chi2)
    if "sine_no_translation" in xs:
        print(f"  sine_no_translation (attenuation of the test without c-T_N): {xs['sine_no_translation'][0]:.4f}")
    verdict = (f"WINDOW EFFECT OBSERVED: sine - uniform >= {Z_REQ:.0f} sigma" if z >= Z_REQ else
               f"NO SIGNIFICANT WINDOW EFFECT ({z:.1f} sigma)")
    print("  VERDICT:", verdict)
    res["verdict"] = verdict
    return res


# =============================================================================== dry run
def noisy_probs(hw, info, label, scale):
    """Exact noisy distribution of c (density matrix): calibrated depolarising gate errors, readout errors,
    T1/T2 relaxation in idle periods (noise_model.noisy_copy)."""
    from qiskit_aer import AerSimulator
    import noise_model as NM
    bound = info["tq"].assign_parameters(vec(info["tq"], info["binds"][label]))
    tqn = NM.noisy_copy(hw, bound, scale, False, set(), {})
    small, pos = NM.compact(tqn, [])
    sim = AerSimulator(method="density_matrix", max_parallel_threads=4)
    cnt = sim.run(small, shots=1_000_000, seed_simulator=3).result().get_counts()
    return {k.replace(" ", ""): v for k, v in cnt.items()}


def dry_run(hw, info, plan, scales, seed=5):
    rng = np.random.default_rng(seed)
    out = {}
    for sc in scales:
        t0 = time.time()
        X = {}
        for k in plan:
            x, _, _ = x_from_counts(noisy_probs(hw, info, k, sc))
            X[k] = x
        z = (X["sine"] - X["uniform"]) / math.sqrt((1 - X["sine"] ** 2) / plan["sine"]
                                                   + (1 - X["uniform"] ** 2) / plan["uniform"])
        print(f"  s={sc}: noisy <X_c> " + ", ".join(f"{k} {v:+.4f}" for k, v in X.items())
              + f"; D = {X['sine'] - X['uniform']:+.4f}, expected significance {z:.1f} sigma [{time.time() - t0:.0f} s]")
        sample = {}
        for k, n in plan.items():
            k0 = int(rng.binomial(n, (1 + X[k]) / 2))
            x = (2 * k0 - n) / n
            sample[k] = (x, math.sqrt(max(1 - x * x, 1e-12) / n), n)
        print(f"-- analysis of one binomial sample of the plan, error scale {sc} --")
        out[sc] = dict(X=X, D=X["sine"] - X["uniform"], z_expected=z, analysis_of_sample=analyse(sample))
    return out


def exactness(info):
    """Noiseless check of the transpiled circuit: <X_c> of every setting against the ideal value."""
    from qiskit_aer import AerSimulator
    import noise_model as NM
    sim = AerSimulator(method="statevector")
    out = {}
    for k, b in info["binds"].items():
        bound = info["tq"].assign_parameters(vec(info["tq"], b))
        used = sorted({bound.find_bit(q).index for ins in bound.data for q in ins.qubits})
        small, _ = NM.compact(bound, used)
        cnt = sim.run(small, shots=200_000, seed_simulator=1).result().get_counts()
        x, e, _ = x_from_counts({kk.replace(" ", ""): v for kk, v in cnt.items()})
        out[k] = dict(x=x, se=e, ideal=info["ideal"][k])
        print(f"  {k:20s} noiseless <X_c> = {x:+.4f} +- {e:.4f} (ideal {info['ideal'][k]:.4f})")
    return out


# =============================================================================== main
def qpy_path(prereg):
    p = Path(prereg)
    return p.with_name(p.stem + "_circuits.qpy")


def summary(info, plan, tq_est, budget, name):
    print(f"Backend/calibration: {name}")
    print(f"N={N0}, m={M0}, w={info['w']}, v={V0}, +{K0} x {T0} (1 + 26 = 6 + 21: one overflow); "
          f"{info['logical_qubits']} logical qubits, {info['cz']} CZ, duration ~{info['duration']:.1f} us, "
          f"ESP {info['esp']:.3f}")
    print(f"  physical qubits {info['layout']}  (c = {info['roles']['c']}, J = {info['roles']['J']}, "
          f"R = {info['roles']['R']}), transpiler seed {info['seed']}")
    print(f"Plan: {plan} -> {sum(plan.values())} shots, estimated QPU time {tq_est:.1f} s (budget {budget:.0f} s)")


def write_prereg(path, args, info, plan, tq_est, budget, exact, proj):
    st = settings()
    doc = {
        "experiment": "Sine versus uniform coset window: Hadamard test of the translation by N (N=21, m=3, w=8)",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pre-registered before any submission",
        "calibration_snapshot": str(args.calibration),
        "settings": {k: dict(ideal_X=round(v[3], 6), s=v[1], g=v[2]) for k, v in st.items()},
        "statistic": "D = <X_c>(sine) - <X_c>(uniform), ideal cos(pi/9) - 7/8 = +0.0647",
        "standard_error": "SE = sqrt[(1 - Xs^2)/ns + (1 - Xu^2)/nu] (binomial, independent settings)",
        "verdict_rule": "WINDOW EFFECT OBSERVED iff D >= 3 SE; otherwise NO SIGNIFICANT WINDOW EFFECT",
        "secondary": ["offset-corrected ratio (Xs - Xe)/(Xu - Xe), theory 1.0739",
                      "weighted fit <X_c> = eta A + delta over uniform, sine, middle_pair, edge_pair (chi2, 2 dof)",
                      "sine_no_translation: attenuation of the test without c-T_N"],
        "backend": "ibm_kingston",
        "physical_qubits": info["layout"], "roles": info["roles"], "transpiler_seed": info["seed"],
        "cz": info["cz"], "duration_us": round(info["duration"], 2), "esp": round(info["esp"], 5),
        "dynamical_decoupling": "enabled on the device (static circuit); not modelled in the predictions",
        "shots": plan, "pubs": [list(MAIN), list(CONTROLS)], "qpu_estimate_s": round(tq_est, 1), "budget_s": budget,
        "noiseless_check": exact,
        "noisy_predictions": {str(sc): dict(X={k: round(v, 5) for k, v in p["X"].items()}, D=round(p["D"], 5),
                                            z_expected=round(p["z_expected"], 2)) for sc, p in proj.items()},
        "noise_model": "noise_model.noisy_copy (calibrated depolarising gate errors, symmetric readout errors, "
                       "T1/T2 relaxation in idle periods), exact density-matrix simulation",
    }
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"\nPre-registration written: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ibm", action="store_true", help="REAL device (otherwise local dry run)")
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--account", default=None, help="name of a saved Qiskit Runtime account")
    ap.add_argument("--calibration", default="FakeKingston",
                    help="dry run: calibration snapshot (.json, see snapshot_calibration.py) or FakeXxx model")
    ap.add_argument("--budget", type=float, default=HARD_CAP, help=f"QPU time cap (s), <= {HARD_CAP:.0f}")
    ap.add_argument("--seeds", type=int, default=40, help="transpiler seeds tried for the layout")
    ap.add_argument("--scales", default="1,1.5", help="dry run: error scale factors of the noise model")
    ap.add_argument("--plan", action="store_true", help="print circuit and shot plan, then exit")
    ap.add_argument("--prereg", default=None,
                    help="dry run: pre-registration file to WRITE; --ibm: pre-registration file to READ")
    ap.add_argument("--job", default=None, help="ID of a submitted job to read back and analyse")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()
    budget = min(args.budget, HARD_CAP)
    outdir = Path(args.outdir)

    if args.job:
        from archive_job import archive, service
        svc = service(args.account)
        meta = json.loads((outdir / f"meta_{args.job}.json").read_text(encoding="utf-8"))
        res = analyse(read_result(svc.job(args.job).result(), meta["pubs"]))
        (outdir / f"analysis_{args.job}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        archive(args.job, outdir, svc)
        return

    if not args.ibm:
        hw = HW(args.calibration)
        info = prepare(hw, args.seeds)
        plan, tq_est = shot_plan(info, budget)
        summary(info, plan, tq_est, budget, args.calibration)
        if args.prereg:
            from qiskit import qpy
            Path(args.prereg).parent.mkdir(parents=True, exist_ok=True)
            with open(qpy_path(args.prereg), "wb") as fh:
                qpy.dump([info["tq"]], fh)
            print(f"Transpiled circuit saved: {qpy_path(args.prereg)}")
        if args.plan:
            return
        print("\nNoiseless check of the transpiled circuit:")
        exact = exactness(info)
        scales = [float(x) for x in args.scales.split(",")]
        print(f"\nDRY RUN: noisy density-matrix simulation, scales {scales}")
        proj = dry_run(hw, info, plan, scales)
        if args.prereg:
            write_prereg(args.prereg, args, info, plan, tq_est, budget, exact, proj)
        return

    # ---------------------------------------------------------------- real device
    if not args.prereg:
        sys.exit("--ibm requires --prereg (the pre-registration file fixes the circuit, layout and shot plan)")
    pre = json.loads(Path(args.prereg).read_text(encoding="utf-8"))
    from qiskit import qpy
    from archive_job import archive, service
    from snapshot_calibration import snapshot
    svc = service(args.account)
    be = svc.backend(args.backend)
    hw = HW(be)
    info = prepare(hw, layout=pre["physical_qubits"])
    with open(qpy_path(args.prereg), "rb") as fh:
        info["tq"] = qpy.load(fh)[0]                     # the exact pre-registered circuit
    bound = info["tq"].assign_parameters(vec(info["tq"], info["binds"]["sine"]))
    info["cz"] = GC.count_2q(bound)[0]
    if info["cz"] != pre["cz"]:
        sys.exit(f"stored circuit has {info['cz']} CZ, pre-registered {pre['cz']}")
    lg = hw.esp(bound)
    info.update(esp=math.exp(lg["total"]), duration=lg["duration_us"], layout=pre["physical_qubits"],
                roles=pre["roles"], seed=pre["transpiler_seed"])
    plan = {k: int(v) for k, v in pre["shots"].items()}
    tq_est = qpu_time(plan, info)
    print("ESP below: today's live calibration, pre-registered circuit")
    summary(info, plan, tq_est, budget, be.name)
    if tq_est > budget:
        sys.exit(f"estimated QPU time {tq_est:.0f} s exceeds the budget {budget:.0f} s")
    if args.plan:
        return
    if not args.yes and input(f"Submit {sum(plan.values())} shots (~{tq_est:.0f} s QPU) to {be.name}? [y/N] "
                              ).strip().lower() != "y":
        print("Cancelled.")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    ctx = snapshot(be, svc, outdir, "before_sine_window")
    print(f"  QPU time available on the account before submission: {ctx['usage'].get('usage_remaining_seconds')} s")
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    sampler = Sampler(mode=be)
    sampler.options.max_execution_time = int(min(budget, max(30, 1.3 * tq_est)))
    sampler.options.dynamical_decoupling.enable = True
    job = sampler.run(pubs(info, plan))
    jid = job.job_id()
    meta = dict(experiment="sine_window_hadamard_test", pubs=[list(MAIN), list(CONTROLS)], plan=plan,
                backend=be.name, physical_qubits=info["layout"], roles=info["roles"], cz=info["cz"],
                prereg=str(args.prereg), date=datetime.datetime.now().isoformat(timespec="seconds"),
                qpu_estimate_s=tq_est)
    (outdir / f"meta_{jid}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Job submitted: {jid} (read back: --ibm --job {jid} --outdir {outdir})")
    res = analyse(read_result(job.result(), meta["pubs"]))
    (outdir / f"analysis_{jid}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    archive(jid, outdir, svc)
    snapshot(be, svc, outdir, "after_sine_window")


def read_result(result, pub_labels):
    xs = {}
    for labels, r in zip(pub_labels, result):
        arr = r.data.meas
        for i, lab in enumerate(labels):
            xs[lab] = x_from_counts(arr.get_counts(loc=i))
    return xs


if __name__ == "__main__":
    main()
