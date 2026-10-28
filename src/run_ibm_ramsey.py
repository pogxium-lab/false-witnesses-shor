"""Ramsey and Hahn-echo characterisation of qubit 44 of ibm_kingston (and, in the same job, qubits 45 and 54).

Purpose. The coherent-phase hypothesis of the 27 Oct arithmetic runs assumed a static frequency offset of
10-20 kHz of physical qubit 44 (holder of e1 during a long idle window). This job measures TODAY's idle detuning
of qubit 44 (and of the line qubits 45 and 54), its shift when the neighbour 45 is in |1> (static ZZ), T2* and
T2(echo). It measures today's detuning, after today's calibration of the drive frequency, NOT the one of 27 Oct:
it gives the order of magnitude of the idle detuning of this qubit, not a measurement of the past offset.

Circuits (explicit delays, native gates only, one parameter theta per circuit; sampler-level DD disabled):
    Ramsey      sx - delay(tau) - rz(2 pi f_art tau + theta) - sx - measure, f_art = 100 kHz artificial detuning,
                25 delays 0..30 us (multiples of 16 dt), theta in {0, pi/2} (two quadratures: signed detuning)
        ramsey_44_54      qubits 44 and 54 simultaneously, 45 idle in |0>
        ramsey_44_45one   same with 45 prepared in |1> (x at the start, measured at the end, post-selected on 1)
        ramsey_45         qubit 45 (44 and 54 idle)
    Hahn echo   sx - delay(tau/2) - x - delay(tau/2) - rz(theta) - sx - measure, 13 total delays 0..120 us,
                theta in {0, pi} (contrast)
        echo_44_54, echo_45
Analysis (fixed before the run): binomial maximum likelihood, standard errors from the Fisher information.
    Ramsey (per qubit and family): p1(tau, theta) = B + A exp(-tau / T2*) cos(2 pi (f_art + delta) tau + theta + phi0)
        delta = detuning in the convention of dynamic_sim (static offset = RZ(+2 pi delta tau), the convention of the
        27 Oct coherent-error model); starting points on a grid of delta in [-80, 80] kHz. Secondary: Gaussian decay
        exp(-(tau / T2*)^2). Derived: ZZ shift of 44 = delta(44 | 45 = 1) - delta(44 | 45 = 0), compared with the
        calibrated zz_4445.
    Echo: p1(tau, theta) = B - cos(theta) (C / 2) exp(-tau / T2e).
The sign convention and the estimators are validated before the run on an exact simulation of the circuits with
known detunings and ZZ (dynamic_sim with the calibration timeline).

  Plan:              python src/run_ibm_ramsey.py --calibration CAL.json --plan
  Pre-registration:  python src/run_ibm_ramsey.py --calibration CAL.json --prereg DIR/prereg_ramsey.json [--live-check]
  Real device:       python src/run_ibm_ramsey.py --ibm --prereg DIR/prereg_ramsey.json --outdir DIR [--yes] [--plan]
  Read back a job:   python src/run_ibm_ramsey.py --ibm --job ID --prereg DIR/prereg_ramsey.json --outdir DIR
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import warnings
from pathlib import Path

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import dynamic_sim as DS  # noqa: E402
import prereg_tools as PT  # noqa: E402
from hardware import HW  # noqa: E402

Q, NB, FAR = 44, 45, 54
F_ART = 100e3
RAMSEY_MAX, RAMSEY_POINTS = 30e-6, 25
ECHO_MAX, ECHO_POINTS = 120e-6, 13
ALIGN = 16                       # delays rounded to multiples of 16 dt
SHOTS = 64                       # per circuit and parameter value
HARD_CAP = 10.0                  # QPU seconds
FIXED_OVERHEAD = 2.0             # s per job (27 Oct jobs: charged time - shots x repetition ~ 2-4 s)
PER_CIRCUIT = 0.01               # s per circuit (assumed, not documented)
REP_DELAY = 250e-6
FAMILIES = {
    "ramsey_44_54": dict(kind="ramsey", probe=(Q, FAR), excite=(), measure=(Q, FAR)),
    "ramsey_44_45one": dict(kind="ramsey", probe=(Q, FAR), excite=(NB,), measure=(Q, FAR, NB)),
    "ramsey_45": dict(kind="ramsey", probe=(NB,), excite=(), measure=(NB,)),
    "echo_44_54": dict(kind="echo", probe=(Q, FAR), excite=(), measure=(Q, FAR)),
    "echo_45": dict(kind="echo", probe=(NB,), excite=(), measure=(NB,)),
}
THETAS = {"ramsey": [0.0, math.pi / 2], "echo": [0.0, math.pi]}


# =============================================================================== circuits
def delays_dt(kind, dt):
    if kind == "ramsey":
        taus = np.linspace(0, RAMSEY_MAX, RAMSEY_POINTS)
        return [int(round(t / dt / ALIGN)) * ALIGN for t in taus]
    taus = np.linspace(0, ECHO_MAX, ECHO_POINTS)
    return [2 * int(round(t / 2 / dt / ALIGN)) * ALIGN for t in taus]     # two equal halves


def circuit(fam, d_dt, dt, nq):
    spec = FAMILIES[fam]
    th = Parameter("theta")
    meas = ClassicalRegister(len(spec["measure"]), "meas")      # bit j = qubit spec["measure"][j]
    qc = QuantumCircuit(QuantumRegister(nq, "q"), meas, name=f"{fam}_{d_dt}")
    for q in spec["excite"]:
        qc.x(q)
    for q in spec["probe"]:
        qc.sx(q)
    if spec["kind"] == "ramsey":
        for q in spec["probe"]:
            if d_dt:
                qc.delay(d_dt, q, unit="dt")
            qc.rz(th + 2 * math.pi * F_ART * d_dt * dt, q)
            qc.sx(q)
    else:
        for q in spec["probe"]:
            if d_dt:
                qc.delay(d_dt // 2, q, unit="dt")
            qc.x(q)
            if d_dt:
                qc.delay(d_dt // 2, q, unit="dt")
            qc.rz(th, q)
            qc.sx(q)
    qc.barrier(*spec["measure"])
    for j, q in enumerate(spec["measure"]):
        qc.measure(q, meas[j])
    return qc


def build(hw):
    dt = hw.tg.dt
    out = {}
    for fam, spec in FAMILIES.items():
        for d in delays_dt(spec["kind"], dt):
            out[f"{fam}|{d}"] = circuit(fam, d, dt, hw.tg.num_qubits)
    return out


def duration(qc, hw):
    return DS.makespan(qc.assign_parameters([0.0]), hw)


def qpu_estimate(circs, hw, shots):
    t = FIXED_OVERHEAD + PER_CIRCUIT * len(circs)
    for k, qc in circs.items():
        npar = len(THETAS[FAMILIES[k.split("|")[0]]["kind"]])
        t += npar * shots * (REP_DELAY + duration(qc, hw) + 5e-6)
    return t


# =============================================================================== analysis
def p1_from_counts(counts, fam, q):
    """P(qubit q measured 1) and number of kept shots from the counts of the register `meas` (bit j = qubit
    FAMILIES[fam]["measure"][j]; bitstrings little-endian); families with 45 excited: shots with 45 = 1 only."""
    spec = FAMILIES[fam]
    m = list(spec["measure"])
    j = m.index(q)
    jp = m.index(NB) if NB in spec["excite"] else None
    n = k = 0
    for key, v in counts.items():
        b = key[::-1]
        if jp is not None and b[jp] != "1":
            continue
        n += v
        k += v * (b[j] == "1")
    return k / max(n, 1), n


def ramsey_model(x, tau, theta, gauss=False):
    B, A, T2, delta, phi0 = x
    dec = np.exp(-(tau / T2) ** 2) if gauss else np.exp(-tau / T2)
    return B + A * dec * np.cos(2 * math.pi * (F_ART + delta) * tau + theta + phi0)


def echo_model(x, tau, theta):
    B, C, T2 = x
    return B - np.cos(theta) * (C / 2) * np.exp(-tau / T2)


def ml_fit(model, starts, tau, theta, p, n, names, scales):
    """Binomial maximum likelihood; parameters internally rescaled by `scales`; covariance = inverse Fisher
    information (numerical Jacobian of the model)."""
    k1 = p * n

    def nll(z):
        m = np.clip(model(z * scales, tau, theta), 1e-6, 1 - 1e-6)
        return -float((k1 * np.log(m) + (n - k1) * np.log(1 - m)).sum())
    best = min((minimize(nll, np.array(s) / scales, method="Nelder-Mead",
                         options=dict(xatol=1e-8, fatol=1e-10, maxiter=20000)) for s in starts),
               key=lambda r: r.fun)
    x = best.x * scales
    m = np.clip(model(x, tau, theta), 1e-6, 1 - 1e-6)
    J = np.zeros((len(tau), len(x)))
    for i in range(len(x)):
        h = 1e-6 * max(abs(x[i]), scales[i])
        e = np.zeros(len(x))
        e[i] = h
        J[:, i] = (model(x + e, tau, theta) - model(x - e, tau, theta)) / (2 * h)
    I = J.T @ (J * (n / (m * (1 - m)))[:, None])
    try:
        cov = np.linalg.inv(I)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(x), np.nan)
    chi2 = float(((p - m) ** 2 * n / (m * (1 - m))).sum())
    return {**{nm: float(v) for nm, v in zip(names, x)}, **{nm + "_se": float(s) for nm, s in zip(names, se)},
            "chi2": chi2, "dof": int(len(tau) - len(x))}


def grid_start(tau, theta, p, n, gauss=False, T2=100e-6):
    """Starting point: for each delta of a 0.5 kHz grid in [-80, 80] kHz (T2* fixed), the model is linear in
    (B, A cos phi0, -A sin phi0): weighted linear least squares, best residual."""
    w = n / np.clip(p * (1 - p), 0.01, None)
    dec = np.exp(-(tau / T2) ** 2) if gauss else np.exp(-tau / T2)
    best = None
    for d in np.arange(-80e3, 80.01e3, 500.0):
        ph = 2 * math.pi * (F_ART + d) * tau + theta
        X = np.stack([np.ones_like(tau), dec * np.cos(ph), dec * np.sin(ph)], axis=1)
        sw = np.sqrt(w)
        c, *_ = np.linalg.lstsq(X * sw[:, None], p * sw, rcond=None)
        r = float((((X @ c) - p) ** 2 * w).sum())
        if best is None or r < best[0]:
            best = (r, d, c)
    _, d, c = best
    return [c[0], math.hypot(c[1], c[2]), T2, d, math.atan2(-c[2], c[1])]


def fit_ramsey(tau, theta, p, n, gauss=False):
    starts = [grid_start(tau, theta, p, n, gauss, T2) for T2 in (40e-6, 150e-6)]
    r = ml_fit(lambda x, t, th: ramsey_model(x, t, th, gauss), starts, tau, theta, p, n,
               ["B", "A", "T2star", "delta", "phi0"], np.array([1, 1, 1e-4, 1e4, 1]))
    if r["A"] < 0:                        # same curve: A -> -A, phi0 -> phi0 + pi
        r["A"], r["phi0"] = -r["A"], r["phi0"] + math.pi
    r["phi0"] = (r["phi0"] + math.pi) % (2 * math.pi) - math.pi
    return r


def fit_echo(tau, theta, p, n):
    starts = [[0.5, 0.9, T] for T in (50e-6, 150e-6, 400e-6)]
    return ml_fit(echo_model, starts, tau, theta, p, n, ["B", "C", "T2echo"], np.array([1, 1, 1e-4]))


def collect(data, dt):
    """data: {circuit key: [counts of `meas` for each theta]} -> per family and qubit arrays (tau, theta, p1, n)."""
    out = {}
    for key, per_theta in data.items():
        fam, d = key.split("|")
        spec = FAMILIES[fam]
        for th, cnt in zip(THETAS[spec["kind"]], per_theta):
            for q in spec["probe"]:
                p, n = p1_from_counts(cnt, fam, q)
                out.setdefault((fam, q), []).append((int(d) * dt, th, p, n))
    return {k: tuple(np.array(c) for c in zip(*v)) for k, v in out.items()}


def analyse(data, dt, zz_cal=None, verbose=True):
    res = {}
    for (fam, q), (tau, th, p, n) in collect(data, dt).items():
        if FAMILIES[fam]["kind"] == "ramsey":
            r = fit_ramsey(tau, th, p, n)
            g = fit_ramsey(tau, th, p, n, gauss=True)
            r["gaussian_decay"] = {k: g[k] for k in ("delta", "delta_se", "T2star", "T2star_se", "chi2")}
            if verbose:
                print(f"  {fam:16s} q{q}: delta = {r['delta'] / 1e3:+7.2f} +- {r['delta_se'] / 1e3:.2f} kHz  "
                      f"T2* = {r['T2star'] * 1e6:6.1f} +- {r['T2star_se'] * 1e6:.1f} us  A = {r['A']:.3f}  "
                      f"chi2/dof = {r['chi2']:.1f}/{r['dof']}  (Gaussian decay: delta = {g['delta'] / 1e3:+.2f} kHz)")
        else:
            r = fit_echo(tau, th, p, n)
            if verbose:
                print(f"  {fam:16s} q{q}: T2echo = {r['T2echo'] * 1e6:6.1f} +- {r['T2echo_se'] * 1e6:.1f} us  "
                      f"C = {r['C']:.3f}  chi2/dof = {r['chi2']:.1f}/{r['dof']}")
        r["n_shots_mean"] = float(np.mean(n))
        res[f"{fam}|q{q}"] = r
    a, b = res.get(f"ramsey_44_45one|q{Q}"), res.get(f"ramsey_44_54|q{Q}")
    if a and b:
        zz = a["delta"] - b["delta"]
        se = math.hypot(a["delta_se"], b["delta_se"])
        res["zz_shift_44_by_45"] = dict(value=zz, se=se, calibrated_zz_4445=zz_cal)
        if verbose:
            print(f"  ZZ shift of q44 when q45 = |1>: {zz / 1e3:+.2f} +- {se / 1e3:.2f} kHz (calibrated zz_4445 "
                  f"{(zz_cal or float('nan')) / 1e3:.2f} kHz)")
    return res


def counts_of(pub, fam):
    """Counts of the register `meas` for each parameter value."""
    return [pub.data.meas.get_counts(loc=i) for i in range(len(THETAS[FAMILIES[fam]["kind"]]))]


# =============================================================================== validation
_EXACT = {}


def simulate(circs, hw, detuning, zz, decay=None, rng=None, shots=None):
    """Exact distribution of `meas` for every circuit with static detunings and ZZ (dynamic_sim timeline);
    optionally an exponential loss of coherence of each probed qubit (decay {q: T2}: mixture with the same
    distribution where that qubit is uniformly random) and multinomial sampling."""
    data = {}
    for key, qc in circs.items():
        fam, d = key.split("|")
        spec = FAMILIES[fam]
        m = list(spec["measure"])
        tau = int(d) * hw.tg.dt
        per = []
        for th in THETAS[spec["kind"]]:
            ck = (key, th, tuple(sorted(detuning.items())), tuple(sorted(zz.items())))
            if ck not in _EXACT:
                _EXACT[ck] = np.clip(DS.register_probabilities(qc.assign_parameters([th]), creg="meas", hw=hw,
                                                               detuning=detuning, zz=zz), 0, None)
            P = _EXACT[ck].copy()
            if decay:
                for q in spec["probe"]:
                    if q in decay:
                        j = m.index(q)
                        T = P.reshape([2] * len(m))
                        ax = len(m) - 1 - j            # index = sum_j bit_j 2^j -> C-order axis of bit j
                        flat = T.mean(axis=ax, keepdims=True).repeat(2, axis=ax)
                        e = math.exp(-tau / decay[q])
                        P = (e * T + (1 - e) * flat).reshape(-1)
            P = P / P.sum()
            n = shots or 10 ** 9
            cnt = rng.multinomial(n, P) if rng is not None else np.round(P * n).astype(int)
            per.append({format(i, f"0{len(m)}b"): int(c) for i, c in enumerate(cnt) if c})
        data[key] = per
    return data


def validate(circs, hw, zz_cal, shots, n_mc=40, seed=3):
    """(1) Noiseless exact simulation with known offsets: the fits must return them (sign convention).
    (2) Monte Carlo with the planned shots and calibrated T2 (as T2* and T2echo): expected standard errors."""
    det = {Q: 15e3, NB: -4e3, FAR: 6e3}
    zz = {(Q, NB): zz_cal}
    print(f"  noiseless exact simulation with detunings {det} Hz and ZZ(44,45) = {zz_cal:.0f} Hz:")
    data = simulate(circs, hw, det, zz)
    res = analyse(data, hw.tg.dt, zz_cal)
    expect = {f"ramsey_44_54|q{Q}": det[Q], f"ramsey_44_54|q{FAR}": det[FAR], f"ramsey_45|q{NB}": det[NB],
              f"ramsey_44_45one|q{Q}": det[Q] + zz_cal}
    recov = {k: dict(true=v, fitted=res[k]["delta"]) for k, v in expect.items()}
    ok = all(abs(r["fitted"] - r["true"]) < 300 for r in recov.values())
    print(f"  recovered: {recov} -> {'OK' if ok else 'FAILED'}")
    rng = np.random.default_rng(seed)
    t2 = {q: hw.t1t2(q)[1] for q in (Q, NB, FAR)}
    est = {}
    for _ in range(n_mc):
        r = analyse(simulate(circs, hw, det, zz, decay=t2, rng=rng, shots=shots), hw.tg.dt, zz_cal, verbose=False)
        for k, v in r.items():
            if k.startswith("ramsey"):
                est.setdefault(k, []).append(v["delta"])
            elif k.startswith("echo"):
                est.setdefault(k, []).append(v["T2echo"])
    mc = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for k, v in est.items()}
    for k, v in mc.items():
        unit, s = ("kHz", 1e3) if k.startswith("ramsey") else ("us", 1e-6)
        print(f"  Monte Carlo ({n_mc} x {shots} shots, T2 = calibrated): {k:24s} mean {v['mean'] / s:8.2f} "
              f"sd {v['sd'] / s:.2f} {unit}")
    return dict(noiseless=dict(detuning_hz=det, zz_hz=zz_cal, recovered=recov, ok=ok),
                monte_carlo=dict(reps=n_mc, shots=shots, T2_used=t2, estimates=mc))


# =============================================================================== files
def circuits_file(prereg):
    p = Path(prereg)
    return p.with_name(p.stem + "_circuits.qpy")


def calibrated_zz(calibration):
    return DS.calibrated_zz(calibration, [Q, NB]).get((Q, NB))


def summary(circs, hw, shots, est):
    fams = {}
    for k in circs:
        fams.setdefault(k.split("|")[0], []).append(int(k.split("|")[1]) * hw.tg.dt * 1e6)
    for f, taus in fams.items():
        print(f"  {f:16s} {len(taus)} delays {taus[0]:.2f}..{taus[-1]:.2f} us x {len(THETAS[FAMILIES[f]['kind']])} "
              f"theta, probe {FAMILIES[f]['probe']}, excite {FAMILIES[f]['excite']}")
    npub = sum(len(THETAS[FAMILIES[k.split('|')[0]]['kind']]) for k in circs)
    print(f"  {len(circs)} circuits, {npub} parameter values x {shots} shots = {npub * shots} shots; estimated QPU "
          f"{est:.1f} s (cap {HARD_CAP:.0f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ibm", action="store_true", help="REAL device")
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--account", default=None)
    ap.add_argument("--calibration", default=None, help="calibration snapshot (.json) for the local phase")
    ap.add_argument("--prereg", default=None, help="local: file to WRITE; --ibm: file to READ")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--live-check", action="store_true", help="local: ISA check against the live target")
    ap.add_argument("--mc", type=int, default=40, help="local: Monte-Carlo repetitions of the validation")
    ap.add_argument("--job", default=None)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if not args.ibm:
        if not args.calibration:
            sys.exit("--calibration is required")
        hw = HW(args.calibration)
        circs = build(hw)
        est = qpu_estimate(circs, hw, args.shots)
        summary(circs, hw, args.shots, est)
        if est > HARD_CAP:
            sys.exit("estimated QPU time exceeds the cap")
        bad = PT.isa_errors(circs, hw.tg)
        print("ISA check (calibration target):", bad or "all circuits valid")
        live = "not run"
        if args.live_check:
            from archive_job import service
            be = service(args.account).backend(args.backend)
            live = PT.isa_errors(circs, be.target) or "valid"
            print("ISA check (live target):", live)
            if live != "valid":
                sys.exit("ISA check failed")
        if args.plan:
            return
        zz_cal = calibrated_zz(args.calibration)
        print("\nValidation of the sign convention and of the estimators:")
        val = validate(circs, hw, zz_cal, args.shots, n_mc=args.mc)
        if not val["noiseless"]["ok"]:
            sys.exit("validation failed")
        if args.prereg:
            qf = circuits_file(args.prereg)
            Path(args.prereg).parent.mkdir(parents=True, exist_ok=True)
            with open(qf, "wb") as fh:
                qpy.dump(list(circs.values()), fh)
            cal_q = {q: dict(T1_us=hw.t1t2(q)[0] * 1e6, T2_us=hw.t1t2(q)[1] * 1e6) for q in (Q, NB, FAR)}
            doc = dict(
                experiment="Ramsey / Hahn echo of qubit 44 (and 45, 54) on ibm_kingston: today's idle detuning",
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                status="pre-registered before submission",
                script="src/run_ibm_ramsey.py (package false-witnesses-shor)",
                caveat="measures TODAY's detuning relative to today's calibrated drive frequency, not the offset of "
                       "27 Oct; gives the order of magnitude of the idle detuning of qubit 44",
                backend=args.backend, qubits=dict(main=Q, neighbour_on_line=NB, other_line_end=FAR),
                calibration_snapshot=Path(args.calibration).name, calibration_sha256=PT.sha256(args.calibration),
                calibrated=cal_q, calibrated_zz_4445_hz=zz_cal,
                families={f: dict(kind=s["kind"], probe=list(s["probe"]), excite=list(s["excite"]),
                                  measure=list(s["measure"]), theta=THETAS[s["kind"]]) for f, s in FAMILIES.items()},
                artificial_detuning_hz=F_ART, delays_dt={f: delays_dt(s["kind"], hw.tg.dt) for f, s in FAMILIES.items()},
                dt_s=hw.tg.dt, circuits_order=list(circs), circuits_file=qf.name, circuits_sha256=PT.sha256(qf),
                shots_per_circuit_and_theta=args.shots, qpu_estimate_s=round(est, 2), cap_s=HARD_CAP,
                qpu_estimator=f"{FIXED_OVERHEAD} s + {PER_CIRCUIT} s per circuit + shots x (250 us + duration + 5 us)",
                analysis=dict(
                    ramsey="binomial ML of p1 = B + A exp(-tau/T2*) cos(2 pi (f_art + delta) tau + theta + phi0) per "
                           "qubit and family, both quadratures jointly; delta in the dynamic_sim convention (static "
                           "offset = RZ(+2 pi delta tau)); starts on a grid delta in [-80, 80] kHz; se from the "
                           "Fisher information; secondary: Gaussian decay",
                    post_selection="ramsey_44_45one: shots with c45 = 1 only",
                    zz="ZZ shift of 44 = delta(ramsey_44_45one, q44) - delta(ramsey_44_54, q44), compared with the "
                       "calibrated zz_4445",
                    echo="binomial ML of p1 = B - cos(theta) (C/2) exp(-tau/T2echo)",
                    reported="every fitted parameter with its se, chi2/dof, raw counts"),
                validation=val)
            Path(args.prereg).write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
            print(f"\nPre-registration written: {args.prereg}; circuits {qf}")
        return

    # ---------------------------------------------------------------- real device
    from archive_job import archive, service
    svc = service(args.account)
    outdir = Path(args.outdir)
    pre = json.loads(Path(args.prereg).read_text(encoding="utf-8"))
    if args.job:
        res_ = svc.job(args.job).result()
        data = {k: counts_of(p, k.split("|")[0]) for k, p in zip(pre["circuits_order"], res_)}
        res = analyse(data, pre["dt_s"], pre["calibrated_zz_4445_hz"])
        Path(outdir / f"analysis_{args.job}.json").write_text(json.dumps(res, indent=1, default=float),
                                                              encoding="utf-8")
        archive(args.job, outdir, svc)
        return
    qf = circuits_file(args.prereg)
    h_pre, h_q = PT.sha256(args.prereg), PT.sha256(qf)
    sums = PT.read_sums(Path(args.prereg).parent) or {}
    checks = {"circuits match prereg": h_q == pre["circuits_sha256"],
              "prereg in SHA256SUMS": sums.get(Path(args.prereg).name) == h_pre,
              "circuits in SHA256SUMS": sums.get(qf.name) == h_q,
              "time-stamp proofs present": all(Path(str(f) + s).exists() for f in (args.prereg, qf)
                                               for s in (".ots", ".tsr"))}
    print("Fingerprint checks:", checks)
    if not all(checks.values()):
        sys.exit("fingerprint check failed")
    with open(qf, "rb") as fh:
        circs = dict(zip(pre["circuits_order"], qpy.load(fh)))
    be = svc.backend(args.backend)
    hw = HW(be)
    if abs(hw.tg.dt - pre["dt_s"]) > 1e-15:
        sys.exit("dt of the live target differs from the pre-registration")
    bad = PT.isa_errors(circs, be.target)
    print("ISA check (live target):", bad or "all circuits valid")
    if bad:
        sys.exit("ISA check failed")
    shots = int(pre["shots_per_circuit_and_theta"])
    est = qpu_estimate(circs, hw, shots)
    summary(circs, hw, shots, est)
    if est > HARD_CAP:
        sys.exit("estimated QPU time exceeds the cap")
    if args.plan:
        return
    if not args.yes and input(f"Submit {len(circs)} circuits to {be.name}? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return
    from snapshot_calibration import snapshot
    ctx = snapshot(be, svc, outdir, "before_ramsey")
    rem = ctx["usage"].get("usage_remaining_seconds")
    print(f"  QPU time available on the account before submission: {rem} s")
    if rem is not None and float(rem) < 1.5 * est:
        sys.exit("not enough QPU time left in the usage window")
    pubs = [(qc, [[t] for t in THETAS[FAMILIES[k.split("|")[0]]["kind"]]], shots) for k, qc in circs.items()]
    job = PT.submit(be, pubs, HARD_CAP, est, tags=[PT.TAG_PREFIX + h_pre])
    # PT.submit caps the execution time at max(60, 1.3 est) within the budget: here min(HARD_CAP, ...) = HARD_CAP
    jid = job.job_id()
    meta = dict(experiment="ramsey_echo_q44", order=list(circs), shots=shots, backend=be.name,
                prereg=Path(args.prereg).name, prereg_sha256=h_pre, job_tags=[PT.TAG_PREFIX + h_pre],
                date=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), qpu_estimate_s=est)
    (outdir / f"meta_{jid}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Job submitted: {jid}")
    res_ = job.result()
    data = {k: counts_of(p, k.split("|")[0]) for k, p in zip(circs, res_)}
    res = analyse(data, pre["dt_s"], pre["calibrated_zz_4445_hz"])
    (outdir / f"analysis_{jid}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    archive(jid, outdir, svc)
    snapshot(be, svc, outdir, "after_ramsey")


if __name__ == "__main__":
    main()
