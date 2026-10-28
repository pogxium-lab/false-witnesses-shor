"""Non-trivial modular-arithmetic test on an IBM device: N = 21, a = 2, t = 3, with false witnesses.

Default: DRY RUN (local noisy simulation); nothing is submitted without --ibm AND confirmation. QPU budget capped
at 90 s.

  Plan only:                    python src/run_ibm_arith.py --calibration CAL.json --plan
  Dry run + pre-registration:   python src/run_ibm_arith.py --calibration CAL.json --prereg PREREG.json
  Real device (submits the transpiled circuits and the shot plan stored with the pre-registration):
                                python src/run_ibm_arith.py --ibm --prereg PREREG.json --outdir DIR
  Read back a job:              python src/run_ibm_arith.py --ibm --job <ID> --outdir DIR

Circuit. The line generator (shor_line_generator.build, genome m0 = 1, semiclassical measurements) for N = 21,
a = 2, t = 3 uses 17 qubits and about 585 CZ. The two low exponent bits e0, e1 drive a lookup table of unreduced
powers (1, 2, 4, 8); the high bit e2 controls ONE coset-encoded multiplication by a^4 = 16 mod 21, an element of
order 3. Exact output distribution of the exponent register (exact_model.shor_P), y = 0..7:
    true circuit           [3, 2, 1, 2, 3, 2, 1, 2] / 16
    multiplication by 1    [4, 0, 4, 0, 4, 0, 4, 0] / 16
    any other witness      uniform 1/8
Statistic (fixed before the run): S = P(y in {0, 4}) - P(y in {2, 6}); ideal 0.25 for the true circuit, 0 for
every witness. S depends only on the bits y0 (measured on e2, first) and y1 (measured on e1).

Witnesses, produced by the same generator with the same sequence of two-qubit blocks on the same physical line
(only rotation angles or added measurements differ):
    mult_by_1          multiplier 16 replaced by 1 (no arithmetic, zero angles kept as gates);
    wrong_constant     multiplier outside the subgroup <2> mod 21 (exact_model.fake_consts: 10);
    decohered_e2       e2 measured in the Z basis right after its preparation;
    classical_exponent all exponent bits measured right after their preparation.
Readout and relaxation bias on the bits y0, y1 shifts S for every circuit on the same qubits; the comparison is
therefore made only against witnesses run on the same line in the same job.

Verdict rule: SIGNAL if S(true) - S(witness) >= 3 standard errors for EACH witness. For one circuit with n shots,
S is a difference of two cells of one multinomial, Var(S_hat) = [pA + pB - (pA - pB)^2] / n; circuits are
independent, so the standard error of a difference is the root of the sum of the two variances. Secondary
(reported, not used for the verdict): conditional statistic S / P(y0 = 0) (first-order insensitive to the readout
asymmetry of y1), fitted visibility and log-likelihood of each ideal distribution, classical overlap.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from qiskit import transpile  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

import exact_model as EM  # noqa: E402
import generator_core as GC  # noqa: E402
import run_ibm as RI  # noqa: E402
import shor_line_generator as LG  # noqa: E402
from hardware import BASIS, HW  # noqa: E402

N, A_BASE, T_BITS = 21, 2, 3
GENOME = replace(GC.BASE_GENOME, m0=1)
HARD_CAP = 90.0
Z_REQ = 3.0
CELLS_A, CELLS_B = (0, 4), (2, 6)


def witnesses():
    L = min(GC.lookup_bits(N, A_BASE, T_BITS), 2)
    return {
        "true": LG.Opt(semi=1),
        "mult_by_1": LG.Opt(semi=1, consts=((2, 1),)),
        "wrong_constant": LG.Opt(semi=1, consts=EM.fake_consts(N, A_BASE, T_BITS, L)),
        "decohered_e2": LG.Opt(semi=1, decohere=(2,)),
        "classical_exponent": LG.Opt(semi=1, decohere=tuple(range(T_BITS))),
    }


def ideal_P(opt):
    return EM.shor_P(N, A_BASE, T_BITS, GENOME, order=opt.order, consts=opt.consts, table=opt.table,
                     decohere=opt.decohere)


# =============================================================================== statistics
def s_stat(P):
    """S, and the single-shot variance of its estimator, from a distribution (or normalised counts)."""
    pa = float(sum(P[y] for y in CELLS_A))
    pb = float(sum(P[y] for y in CELLS_B))
    return pa - pb, pa + pb - (pa - pb) ** 2


def as_array(counts):
    v = np.zeros(1 << T_BITS)
    for y, n in counts.items():
        v[int(y)] += n
    return v


def s_from_counts(counts):
    c = as_array(counts)
    n = c.sum()
    s, var1 = s_stat(c / n)
    return s, math.sqrt(var1 / n), int(n)


def z_expected(P1, n1, P2, n2):
    """Expected significance of S(P1) - S(P2) with n1, n2 shots."""
    s1, v1 = s_stat(P1)
    s2, v2 = s_stat(P2)
    return (s1 - s2) / math.sqrt(v1 / n1 + v2 / n2)


def visibility_fit(c, P):
    """Max-likelihood visibility v in [0, 1] of v P + (1 - v) U; returns (v, 2 * [lnL(v) - lnL(0)])."""
    U = np.full(len(P), 1 / len(P))

    def nll(v):
        return -float((c * np.log(np.clip(v * P + (1 - v) * U, 1e-300, None))).sum())
    r = minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded")
    return float(r.x), 2 * (nll(0.0) - float(r.fun))


def analyse(counts_by, ideals):
    print()
    print(f"=== Analysis: S = P(y in {set(CELLS_A)}) - P(y in {set(CELLS_B)}), N={N}, a={A_BASE}, t={T_BITS} ===")
    st = {k: s_from_counts(c) for k, c in counts_by.items()}
    sv, ev, nv = st["true"]
    res = {"S": {}, "z": {}}
    zs = []
    for name, (s, e, n) in st.items():
        c = as_array(counts_by[name])
        q = (c[0] + c[2] + c[4] + c[6]) / n
        line = (f"  {name:19s}: S = {s:+.4f} +- {e:.4f}  (ideal {s_stat(ideals[name])[0]:+.4f}, {n} shots; "
                f"S/P(y0=0) = {s / q:+.4f})")
        res["S"][name] = dict(S=s, se=e, n=n, S_cond=s / q, P_y0_0=q)
        if name != "true":
            z = (sv - s) / math.hypot(ev, e)
            zs.append(z)
            res["z"][name] = z
            line += f"   true - {name} = {sv - s:+.4f} ({z:+.1f} sigma)"
        print(line)
    zmin = min(zs)
    hyp = {"true": ideals["true"], "mult_by_1": ideals["mult_by_1"], "uniform": np.full(1 << T_BITS, 1 / 8)}
    print("  Secondary (not used for the verdict): visibility v and 2 ln LR (v P_ideal + (1-v) U against U); "
          "overlap sum sqrt(p_obs p_ideal)")
    res["secondary"] = {}
    for name, cnt in counts_by.items():
        c = as_array(cnt)
        p = c / c.sum()
        parts = []
        res["secondary"][name] = {}
        for h, P in hyp.items():
            v, llr = visibility_fit(c, P) if h != "uniform" else (0.0, 0.0)
            ov = float(np.sqrt(p * P).sum())
            res["secondary"][name][h] = dict(v=v, two_lnLR=llr, overlap=ov)
            parts.append(f"{h}: v={v:.3f} 2lnLR={llr:5.1f} ov={ov:.4f}")
        print(f"    {name:19s} " + " | ".join(parts))
    verdict = (f"SIGNAL: the true circuit exceeds every witness by >= {Z_REQ:.0f} sigma" if zmin >= Z_REQ else
               f"NO SIGNIFICANT SIGNAL (smallest difference at {zmin:.1f} sigma)")
    print("  VERDICT:", verdict)
    res.update(z_min=zmin, verdict=verdict)
    return res


# =============================================================================== circuits and plan
def prepare(hw, line=None):
    opts = witnesses()
    if line is None:
        _, line = hw.place(LG.build(N, A_BASE, T_BITS, GENOME, opts["true"], measure=True))
    info = {}
    for name, opt in opts.items():
        qc = LG.build(N, A_BASE, T_BITS, GENOME, opt, measure=True)
        qd = transpile(qc, basis_gates=BASIS, optimization_level=0)
        tq = transpile(qd, hw.be, optimization_level=2, seed_transpiler=1, initial_layout=list(line))
        lg = hw.esp(tq)
        P = ideal_P(opt)
        info[name] = dict(tq=tq, cz=GC.count_2q(tq)[0], cz_cond=GC.count_2q(tq)[1], duration=lg["duration_us"],
                          esp=math.exp(lg["total"]), opt=opt, P=P, S=s_stat(P)[0], blocks=len(two_qubit_blocks(qc)))
    return tuple(int(q) for q in line), info


def two_qubit_blocks(qc):
    """Sequence of (qubits) of the two-qubit operations of the generator output, before synthesis."""
    return [tuple(qc.find_bit(q).index for q in ins.qubits) for ins in qc.data
            if ins.operation.num_qubits == 2 and ins.operation.name != "if_else"]


def shot_plan(info, budget):
    """Largest plan within the budget; the true circuit gets sqrt(k) times the shots of each of the k witnesses
    (optimal for k comparisons against one reference with similar variances)."""
    names = list(info)
    k = len(names) - 1
    w = {n: (math.sqrt(k) if n == "true" else 1.0) for n in names}
    per = sum(w[n] * (RI.REP_DELAY + info[n]["duration"] * 1e-6 + 5e-6) for n in names)
    unit = (0.98 * budget - RI.OVERHEAD * len(names)) / per
    plan = {n: int(unit * w[n]) // 1000 * 1000 for n in names}
    return plan, RI.qpu_time(plan, info)


# =============================================================================== dry run
def projection(hw, line, info, plan, scales, ntraj, ntraj_true, seed=11):
    """Exact P(y) averaged over noisy trajectories (static circuit with semiclassical emulation, calibrated
    noise_model), then expected significance with the planned shots and one multinomial sample of the plan."""
    import noise_model as NM
    st = NM.Setup(N, T_BITS, GENOME, hw.name if isinstance(hw.name, str) and hw.name.endswith(".json") else hw.be,
                  a=A_BASE)
    st.hw, st.path = hw, tuple(line)
    rng = np.random.default_rng(seed)
    out = {}
    for sc in scales:
        Py, Sse = {}, {}
        for name in plan:
            t0 = time.time()
            r = st.run(replace(info[name]["opt"], semi=0), sc, ntraj_true if name == "true" else ntraj)
            P = np.clip(r["Py"], 0, None)
            Py[name] = P / P.sum()
            S = s_stat(Py[name])[0]
            seS = float(np.std([s_stat(p)[0] for p in r["per"]], ddof=1) / math.sqrt(len(r["per"])))
            Sse[name] = seS
            print(f"  s={sc:<4} [{name:19s}] S = {S:+.4f} +- {seS:.4f} (trajectories)  ESP={r['esp']:.3f}  "
                  f"P(y) = {np.round(Py[name], 4)}  [{time.time() - t0:.0f} s]", flush=True)
        z = {n: z_expected(Py["true"], plan["true"], Py[n], plan[n]) for n in plan if n != "true"}
        se_d = math.sqrt(sum(s_stat(Py[n])[1] / plan[n] for n in ("true", min(z, key=z.get))))
        zr = (Sse["true"] / se_d)
        print(f"  s={sc}: expected significance with the plan: " + ", ".join(f"{k} {v:.1f}" for k, v in z.items())
              + f"  -> min {min(z.values()):.1f} sigma (+- {zr:.1f} from the trajectory error of S(true))")
        sample = {n: {y: int(v) for y, v in enumerate(rng.multinomial(plan[n], Py[n]))} for n in plan}
        out[sc] = dict(P={n: [round(float(x), 5) for x in p] for n, p in Py.items()},
                       S={n: s_stat(p)[0] for n, p in Py.items()}, S_traj_se=Sse, z_expected=z,
                       z_min=min(z.values()), z_min_traj_se=zr,
                       sample=sample)
    return out


def exactness(seeds=(1, 2, 3, 4)):
    """Noiseless static circuit (Aer statevector, one branch of the mid-circuit measurements per seed) against
    the exact model: total variation distance (up to y -> -y)."""
    from verify_package import static_check
    return {name: static_check(N, A_BASE, T_BITS, GENOME, replace(opt, semi=0), seeds)
            for name, opt in witnesses().items()}


# =============================================================================== main
def qpy_path(prereg):
    p = Path(prereg)
    return p.with_name(p.stem + "_circuits.qpy")


def save_circuits(prereg, info):
    from qiskit import qpy
    path = qpy_path(prereg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        qpy.dump([info[k]["tq"] for k in info], fh)
    print(f"Transpiled circuits saved: {path} (order {list(info)})")


def load_circuits(prereg, order):
    from qiskit import qpy
    with open(qpy_path(prereg), "rb") as fh:
        circs = qpy.load(fh)
    return dict(zip(order, circs))


def summary(line, info, plan, tq_est, budget, hwname):
    print(f"Backend/calibration: {hwname}")
    print(f"N={N}, a={A_BASE}, t={T_BITS}, physical line ({len(line)} qubits): {list(line)}")
    for name, d in info.items():
        print(f"  {name:19s} CZ={d['cz']:4d} (+{d['cz_cond']} conditional)  2q blocks={d['blocks']}  "
              f"duration~{d['duration']:.0f} us  ESP={d['esp']:.4f}  ideal S={d['S']:+.4f}  "
              f"ideal P={np.round(d['P'], 4)}")
    print(f"Plan: {plan} -> {sum(plan.values())} shots, estimated QPU time {tq_est:.1f} s (budget {budget:.0f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ibm", action="store_true", help="REAL device (otherwise local dry run)")
    ap.add_argument("--backend", default="ibm_kingston", help="real device name (with --ibm)")
    ap.add_argument("--account", default=None, help="name of a saved Qiskit Runtime account")
    ap.add_argument("--calibration", default="FakeKingston",
                    help="dry run: calibration snapshot (.json, see snapshot_calibration.py) or FakeXxx model")
    ap.add_argument("--budget", type=float, default=HARD_CAP, help=f"QPU time cap (s), <= {HARD_CAP:.0f}")
    ap.add_argument("--plan", action="store_true", help="print circuits and shot plan, then exit")
    ap.add_argument("--prereg", default=None,
                    help="dry run: pre-registration file to WRITE; --ibm: pre-registration file to READ (line, plan)")
    ap.add_argument("--scales", default="1,1.5", help="dry run: error scale factors of the noise model")
    ap.add_argument("--traj", type=int, default=100, help="dry run: noisy trajectories per witness circuit")
    ap.add_argument("--traj-true", type=int, default=400, help="dry run: noisy trajectories for the true circuit")
    ap.add_argument("--shots-dry", type=int, default=0, help="dry run: shots of the noisy DYNAMIC circuits (slow)")
    ap.add_argument("--no-exact", action="store_true", help="dry run: skip the noiseless exactness check")
    ap.add_argument("--job", default=None, help="ID of a submitted job to read back and analyse")
    ap.add_argument("--outdir", default="results", help="directory for metadata, analysis and archives")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()
    budget = min(args.budget, HARD_CAP)
    outdir = Path(args.outdir)
    ideals = {k: ideal_P(o) for k, o in witnesses().items()}

    if args.job:
        from archive_job import archive, service
        svc = service(args.account)
        meta = json.loads((outdir / f"meta_{args.job}.json").read_text(encoding="utf-8"))
        result = svc.job(args.job).result()
        cb = {name: RI.counts_y(r) for name, r in zip(meta["order"], result)}
        res = analyse(cb, ideals)
        (outdir / f"analysis_{args.job}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        archive(args.job, outdir, svc)
        return

    if not args.ibm:
        hw = HW(args.calibration)
        line, info = prepare(hw)
        plan, tq_est = shot_plan(info, budget)
        summary(line, info, plan, tq_est, budget, args.calibration)
        if args.prereg:
            save_circuits(args.prereg, info)
        if args.plan:
            return
        out = {}
        if not args.no_exact:
            print("\nExactness (noiseless Aer statevector vs exact model, TVD):")
            out["exactness_tvd"] = exactness()
            for k, v in out["exactness_tvd"].items():
                print(f"  {k:19s} TVD = {v:.1e}")
        if args.shots_dry:
            print(f"\nDRY RUN (chain check): {args.shots_dry} shots per noisy DYNAMIC circuit")
            cb = RI.dry_run(info, {k: args.shots_dry for k in plan}, hw)
            analyse(cb, ideals)
        scales = [float(x) for x in args.scales.split(",")]
        print(f"\nDRY RUN (projection): {args.traj} noisy trajectories per witness, "
              f"{args.traj_true} for the true circuit, scales {scales}")
        proj = projection(hw, line, info, plan, scales, args.traj, args.traj_true)
        for sc, p in proj.items():
            print(f"\n-- projected analysis, error scale {sc} (one multinomial sample of the plan) --")
            p["analysis_of_sample"] = analyse(p["sample"], ideals)
        if args.prereg:
            write_prereg(args.prereg, args, line, info, plan, tq_est, budget, proj, out.get("exactness_tvd"))
        return

    # ---------------------------------------------------------------- real device
    if not args.prereg:
        sys.exit("--ibm requires --prereg (the pre-registration file fixes the line and the shot plan)")
    pre = json.loads(Path(args.prereg).read_text(encoding="utf-8"))
    from archive_job import archive, service
    from snapshot_calibration import snapshot
    svc = service(args.account)
    be = svc.backend(args.backend)
    hw = HW(be)
    line, info = prepare(hw, tuple(pre["physical_line"]))
    stored = load_circuits(args.prereg, list(pre["circuits"]))
    for k, d in info.items():                 # the exact pre-registered circuits are submitted
        d["tq"] = stored[k]
        d["cz"] = GC.count_2q(stored[k])[0]
        if d["cz"] != pre["circuits"][k]["cz"]:
            sys.exit(f"{k}: stored circuit has {d['cz']} CZ, pre-registered {pre['circuits'][k]['cz']}")
        lg = hw.esp(stored[k])
        d["esp"], d["duration"] = math.exp(lg["total"]), lg["duration_us"]
    plan = {k: int(v) for k, v in pre["shots"].items()}
    tq_est = RI.qpu_time(plan, info)
    print("ESP below: today's live calibration, pre-registered circuits")
    summary(line, info, plan, tq_est, budget, be.name)
    _, best = hw.place(LG.build(N, A_BASE, T_BITS, GENOME, witnesses()["true"], measure=True))
    if tuple(best) != tuple(line):
        print(f"  note: today's best line is {list(best)}; the pre-registered line is kept")
    if tq_est > budget:
        sys.exit(f"estimated QPU time {tq_est:.0f} s exceeds the budget {budget:.0f} s")
    if args.plan:
        return
    if not args.yes and input(f"Submit {len(plan)} circuits ({sum(plan.values())} shots, ~{tq_est:.0f} s QPU) to "
                              f"{be.name}? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    ctx = snapshot(be, svc, outdir, "before_arith")
    print(f"  QPU time available on the account before submission: {ctx['usage'].get('usage_remaining_seconds')} s")
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    sampler = Sampler(mode=be)
    sampler.options.max_execution_time = int(min(budget, max(60, 1.3 * tq_est)))
    sampler.options.dynamical_decoupling.enable = False      # dynamic circuits
    order = list(plan)
    job = sampler.run([(info[k]["tq"], None, plan[k]) for k in order])
    jid = job.job_id()
    meta = dict(experiment="arith_N21_a2_t3", N=N, a=A_BASE, t=T_BITS, order=order, plan=plan, backend=be.name,
                physical_line=list(line), cz={k: info[k]["cz"] for k in order}, prereg=str(args.prereg),
                date=datetime.datetime.now().isoformat(timespec="seconds"), qpu_estimate_s=tq_est)
    (outdir / f"meta_{jid}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Job submitted: {jid} (read back: --ibm --job {jid} --outdir {outdir})")
    result = job.result()
    cb = {name: RI.counts_y(r) for name, r in zip(order, result)}
    res = analyse(cb, ideals)
    (outdir / f"analysis_{jid}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    archive(jid, outdir, svc)
    snapshot(be, svc, outdir, "after_arith")


def write_prereg(path, args, line, info, plan, tq_est, budget, proj, tvd):
    first = proj[min(proj)]
    doc = {
        "experiment": "Non-trivial modular arithmetic test, N=21, a=2, t=3 (one controlled multiplication by 16 "
                      "mod 21, order 3)",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pre-registered before any submission",
        "generator": "shor_line_generator.build(21, 2, 3, replace(BASE_GENOME, m0=1), Opt(semi=1))",
        "calibration_snapshot": str(args.calibration),
        "statistic": "S = P(y in {0,4}) - P(y in {2,6}) (exponent register y, 3 bits)",
        "standard_error": "Var(S_hat) = [pA + pB - (pA - pB)^2] / n per circuit (multinomial); difference of two "
                          "independent circuits: sqrt(Var1 + Var2)",
        "verdict_rule": "SIGNAL iff S(true) - S(w) >= 3 standard errors for EACH witness w; otherwise NO SIGNAL",
        "secondary": ["S / P(y0=0) (conditional, first-order insensitive to y1 readout asymmetry)",
                      "visibility fit and 2 ln LR of the ideal true / mult_by_1 distributions against uniform",
                      "overlap sum_y sqrt(p_obs p_ideal)"],
        "witnesses": {
            "mult_by_1": "multiplier 16 -> 1 (consts=((2,1),)), zero angles kept as gates",
            "wrong_constant": f"multiplier {dict(info['wrong_constant']['opt'].consts)[2]} (outside <2> mod 21)",
            "decohered_e2": "e2 measured in Z right after its preparation",
            "classical_exponent": "e0, e1, e2 measured in Z right after their preparation"},
        "backend": "ibm_kingston",
        "physical_line": list(line),
        "shots": plan,
        "qpu_estimate_s": round(tq_est, 1),
        "budget_s": budget,
        "circuits": {k: dict(cz=d["cz"], cz_conditional=d["cz_cond"], two_qubit_blocks=d["blocks"],
                             duration_us=round(d["duration"], 1), esp=round(d["esp"], 5),
                             ideal_P=[round(float(x), 5) for x in d["P"]], ideal_S=round(d["S"], 5))
                     for k, d in info.items()},
        "exactness_tvd_statevector_vs_model": tvd,
        "noisy_predictions": {str(sc): dict(S={k: round(v, 5) for k, v in p["S"].items()},
                                            S_trajectory_se={k: round(v, 5) for k, v in p["S_traj_se"].items()},
                                            P=p["P"],
                                            z_expected={k: round(v, 2) for k, v in p["z_expected"].items()},
                                            z_min=round(p["z_min"], 2), z_min_trajectory_se=round(p["z_min_traj_se"], 2))
                              for sc, p in proj.items()},
        "predicted_verdict_scale_1": "SIGNAL" if first["z_min"] >= Z_REQ else "NO SIGNAL",
        "trajectories": {"true": args.traj_true, "witness": args.traj},
        "noise_model": "noise_model.py: calibrated depolarising CZ/1q errors, symmetric readout errors, T1/T2 "
                       "relaxation in idle periods, feed-forward latency; scale s multiplies every error rate",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"\nPre-registration written: {path}")


if __name__ == "__main__":
    main()
