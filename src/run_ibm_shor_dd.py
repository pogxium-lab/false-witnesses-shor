"""Generic Shor circuit N = 21, a = 2, t = 5 with dynamical decoupling: true circuit and its five controls, one job.

Circuits: those of run_ibm.py for N = 21 (same generator, same controls, line chosen by the package rule
hardware.HW.place on the calibration snapshot: beam search over simple paths of the coupling graph scored by the
calibrated CZ and readout errors weighted by the use of each edge / position, plus a small coherence term; the
three best paths x transpiler seeds 1, 2, smallest two-qubit count then best score; every circuit is then
transpiled on that line with seed 1), each with an X-X echo in every idle window longer than --min-window
(default 1 us) inserted as in run_ibm_arith_dd.py (IBM dynamic-circuit scheduler: ALAP with blocks delimited by
control flow, staggered spacings on neighbouring qubits, explicit delays; only single-qubit X gates and delays are
added, no two-qubit gate):
    true_DD                    true circuit (ideal C2 success 0.2008)
    classical_exponent_DD      all exponent bits measured right after their preparation (0.0625 = random)
    decohered_e2_DD, _e3_DD, _e4_DD   one exponent bit measured right after its preparation (0.1087, 0.1277, 0.1478)
    scrambled_constants_DD     multiplication constants replaced (0.0945)
Criterion (unchanged from the run without decoupling): C2 "exact order" (a continued-fraction denominator d <= N
of y / 2^t is the order of a and Miller's method splits N). Verdict rule: SIGNAL iff the C2 success of true_DD
exceeds that of EACH control and the uniform random value 1/16 by >= 3 standard errors (binomial; difference of
two independent circuits: root of the sum of the variances); otherwise NO SIGNAL.

Exactness: every stored circuit is simulated exactly with its mid-circuit measurements, resets and feed-forward
(dynamic_sim, deferred measurement, compact mode) and compared with the exact model and with the same circuit
without decoupling.

  Checks + pre-registration:
      python src/run_ibm_shor_dd.py --calibration CAL.json --prereg OUT/prereg_shor21_t5_dd.json
                                    [--reference-qpy JOB_circuits.qpy] [--account NAME --live-check]
  Real device:
      python src/run_ibm_shor_dd.py --ibm --prereg OUT/prereg_shor21_t5_dd.json --outdir OUT [--account NAME]
                                    [--tag-prereg] [--yes] [--plan]
  Read back a job:
      python src/run_ibm_shor_dd.py --ibm --job <ID> --outdir OUT [--account NAME]
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
from qiskit import qpy  # noqa: E402

import dynamic_sim as DS  # noqa: E402
import exact_model as EM  # noqa: E402
import generator_core as GC  # noqa: E402
import prereg_tools as PT  # noqa: E402
import run_ibm as RI  # noqa: E402
import run_ibm_arith_dd as RD  # noqa: E402
import shor_line_generator as LG  # noqa: E402
from hardware import HW  # noqa: E402

N, A_BASE, T_BITS = 21, 2, 5
BUDGET = 70.0
EXEC_CAP = 90
Z_REQ = 3.0
EXACT_TOL = 1e-6
SHOT_STEP = 100
# circuit labels of run_ibm.py -> labels of this run
LABELS = {"true": "true_DD", "classical_exponent": "classical_exponent_DD", "decohered_e2": "decohered_e2_DD",
          "decohered_e3": "decohered_e3_DD", "decohered_e4": "decohered_e4_DD",
          "scrambled_constants": "scrambled_constants_DD"}
REFERENCE_ORDER = ("true", "classical_exponent", "decohered_e2", "decohered_e3", "decohered_e4", "scrambled_constants")
LINE_RULE = ("hardware.HW.place(shor_line_generator.build(21, 2, 5, genome, Opt(semi=1), measure=True)) on the "
             "calibration snapshot: beam search (width 4000) over simple paths of the coupling graph, score = sum "
             "over edges of (two-qubit weight) x log(1 - e_CZ) + sum over positions of (measurements) x "
             "log(1 - e_readout) + coherence term -(15 us / T1 + 15 us / T2); the 3 best paths x transpiler seeds "
             "(1, 2), smallest two-qubit count then best score; circuits then transpiled on that line with "
             "seed_transpiler=1, optimization_level=2 (run_ibm.prepare)")


# =============================================================================== circuits
def build(hw, min_window):
    t, g, path, base = RI.prepare(N, hw, A_BASE, T_BITS)
    info = {}
    for k, d in base.items():
        tq = RD.add_dd(d["tq"], hw.tg, hw.be.coupling_map, min_window)
        info[LABELS[k]] = describe(hw, tq, d["tq"], d["opt"], g, k)
    return g, tuple(int(q) for q in path), info


def describe(hw, tq, tq0, opt, g, label):
    cz, czc = GC.count_2q(tq)
    lg = hw.esp(RD.strip_delays(tq), dd=True)
    lg0 = hw.esp(tq0)
    P = EM.shor_P(N, A_BASE, T_BITS, g, order=opt.order, consts=opt.consts, table=opt.table, decohere=opt.decohere)
    s = EM.crit(N, A_BASE, T_BITS, "C2")
    return dict(tq=tq, tq0=tq0, opt=opt, label=label, cz=cz, cz_cond=czc, cz_without_dd=GC.count_2q(tq0)[0],
                duration=DS.makespan(tq, hw) * 1e6, duration_without_dd=lg0["duration_us"],
                esp=math.exp(lg["total"]), esp_without_dd=math.exp(lg0["total"]), gates1q_log=lg["gates1q"],
                gates1q_log_without_dd=lg0["gates1q"], P=P, p_ideal=float(P[s].sum()), p_rand=float(s.mean()),
                added=RD.added_gates(tq, tq0), n_x=int(tq.count_ops().get("x", 0)))


def shot_plan(info, budget):
    """run_ibm.py rule: 30 % of the shots on the true circuit, the rest split equally among the controls; the
    largest such plan within the budget (run_ibm.qpu_time), rounded down."""
    names = list(info)
    share = {n: (RI.TRUE_SHARE if n == "true_DD" else (1 - RI.TRUE_SHARE) / (len(names) - 1)) for n in names}
    per = sum(share[n] * (RI.REP_DELAY + info[n]["duration"] * 1e-6 + 5e-6) for n in names)
    total = (0.98 * budget - RI.OVERHEAD * len(names)) / per
    plan = {n: int(total * share[n] // SHOT_STEP * SHOT_STEP) for n in names}
    while RI.qpu_time(plan, info) > budget:
        plan = {k: v - SHOT_STEP for k, v in plan.items()}
    return plan


def summary(line, info, plan, tq_est, budget, hwname):
    print(f"Backend/calibration: {hwname}")
    print(f"N={N}, a={A_BASE}, t={T_BITS}, physical line ({len(line)} qubits): {list(line)}")
    for name, d in info.items():
        add = ", ".join(f"+{v} {k}" for k, v in sorted(d["added"].items())) or "none"
        print(f"  {name:24s} CZ={d['cz']:4d} (+{d['cz_cond']} conditional)  X={d['n_x']:4d}  added: {add:26s} "
              f"duration~{d['duration']:.1f} us  ESP={d['esp']:.4f} (without DD {d['esp_without_dd']:.4f})  "
              f"ideal C2 success={d['p_ideal']:.4f} (random {d['p_rand']:.4f})")
    print(f"Plan: {plan} -> {sum(plan.values())} shots, estimated QPU time {tq_est:.1f} s (budget {budget:.0f} s)")


# =============================================================================== checks and predictions
def exactness(info):
    s = EM.crit(N, A_BASE, T_BITS, "C2")
    out = {}
    for name, d in info.items():
        t0 = time.time()
        p = DS.register_probabilities(d["tq"], compact=True)
        p0 = DS.register_probabilities(d["tq0"], compact=True)
        out[name] = dict(max_abs_dev_model=float(np.abs(p - d["P"]).max()),
                         max_abs_dev_without_dd=float(np.abs(p - p0).max()), C2_success=float(p[s].sum()))
        print(f"  {name:24s} max|P - P_model| = {out[name]['max_abs_dev_model']:.1e}  max|P - P_without_DD| = "
              f"{out[name]['max_abs_dev_without_dd']:.1e}  C2 success = {out[name]['C2_success']:.4f} "
              f"(model {d['p_ideal']:.4f})  [{time.time() - t0:.0f} s]", flush=True)
    return out


def expected_z(p, plan):
    pv, nv = p["true_DD"], plan["true_DD"]
    z = {}
    for k in plan:
        if k != "true_DD":
            z[k] = (pv - p[k]) / math.sqrt(pv * (1 - pv) / nv + p[k] * (1 - p[k]) / plan[k])
    z["random"] = (pv - 1 / 16) / math.sqrt(pv * (1 - pv) / nv)
    return z


def power(p, plan, n=200_000, seed=3):
    """Probability that the verdict rule is met (Gaussian sampling errors of the plan)."""
    rng = np.random.default_rng(seed)
    obs = {k: p[k] + rng.normal(0, math.sqrt(p[k] * (1 - p[k]) / plan[k]), n) for k in plan}
    ov = obs["true_DD"]
    sv = np.sqrt(np.clip(ov * (1 - ov), 1e-12, None) / plan["true_DD"])
    ok = (ov - 1 / 16) / sv >= Z_REQ
    for k in plan:
        if k != "true_DD":
            sk = np.sqrt(np.clip(obs[k] * (1 - obs[k]), 1e-12, None) / plan[k])
            ok &= (ov - obs[k]) / np.hypot(sv, sk) >= Z_REQ
    return float(ok.mean())


def noisy_prediction(hw, line, g, info, plan, scales, ntraj, ntraj_true, seed=11):
    """Standard noise model (noise_model.py, static circuit with semiclassical emulation, DD: half of the pure
    dephasing removed; no coherent errors, leakage or crosstalk) on the line and calibration of the run. The
    excess over random of each C2 success is multiplied by the error factor of the added X gates."""
    import noise_model as NM
    st = NM.Setup(N, T_BITS, g, hw.name, a=A_BASE)
    st.hw, st.path = hw, tuple(line)
    out = {}
    for sc in scales:
        p, se, Py = {}, {}, {}
        for name, d in info.items():
            t0 = time.time()
            r = st.run(replace(d["opt"], semi=0), sc, ntraj_true if name == "true_DD" else ntraj, dd=True,
                       seed=seed)
            f = math.exp(d["gates1q_log"] - d["gates1q_log_without_dd"])
            p[name] = d["p_rand"] + (r["C2"]["P"] - d["p_rand"]) * f
            se[name] = r["C2"]["se"]
            Py[name] = [round(float(x), 6) for x in r["Py"]]
            print(f"  s={sc:<4} [{name:24s}] C2 success = {p[name]:.5f} +- {se[name]:.5f} (trajectories; ideal "
                  f"{d['p_ideal']:.4f})  ESP(model) = {r['esp']:.4f}  [{time.time() - t0:.0f} s]", flush=True)
        z = expected_z(p, plan)
        pw = power(p, plan)
        print(f"  s={sc}: expected significance: " + ", ".join(f"{k} {v:+.2f}" for k, v in z.items())
              + f" -> min {min(z.values()):+.2f}; probability that the rule is met {pw:.3f}")
        out[str(sc)] = dict(C2_success=p, C2_success_traj_se=se, P_y=Py, z_expected=z, z_min=min(z.values()),
                            probability_rule_met=pw)
    return out


# =============================================================================== analysis
def analyse(counts_by):
    s2 = EM.crit(N, A_BASE, T_BITS, "C2")
    s1 = EM.crit(N, A_BASE, T_BITS, "C1")
    p0 = float(s2.mean())
    print()
    print(f"=== Analysis (criterion C2 'exact order', N={N}, a={A_BASE}, t={T_BITS}, dynamical decoupling) ===")
    print(f"  uniform random : {p0:.4f}")
    res = {"random": p0, "C2": {}, "z": {}}
    kv = sum(v for y, v in counts_by["true_DD"].items() if s2[y])
    nv = sum(counts_by["true_DD"].values())
    pv = kv / nv
    for name, c in counts_by.items():
        n = sum(c.values())
        k = sum(v for y, v in c.items() if s2[y])
        p = k / n
        res["C2"][name] = dict(k=k, n=n, p=p, C1=sum(v for y, v in c.items() if s1[y]) / n)
        line = f"  {name:24s}: {p:.4f} +- {math.sqrt(p * (1 - p) / n):.4f}  ({n} shots)"
        if name != "true_DD":
            res["z"][name] = RI.z_diff(kv, nv, k, n)
            line += f"   true - {name} = {pv - p:+.4f} ({res['z'][name]:+.1f} sigma)"
        print(line)
    res["z"]["random"] = (pv - p0) / math.sqrt(pv * (1 - pv) / nv)
    print(f"  true - random = {pv - p0:+.4f} ({res['z']['random']:+.1f} sigma)")
    print("  (criterion C1, for information only: " + ", ".join(f"{k} {v['C1']:.4f}" for k, v in res["C2"].items())
          + ")")
    zmin = min(res["z"].values())
    v = (f"SIGNAL: the true circuit beats every control and the random baseline at >= {Z_REQ:.0f} sigma"
         if zmin >= Z_REQ else f"NO SIGNIFICANT SIGNAL (worst difference at {zmin:+.1f} sigma)")
    print("  VERDICT:", v)
    res.update(z_min=zmin, verdict=v)
    return res


# =============================================================================== files
def save_circuits(prereg, info):
    path = PT.qpy_path(prereg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        qpy.dump([info[k]["tq"] for k in info], fh)
    print(f"Circuits saved: {path} (order {list(info)})")
    return path


def compare_reference(info, ref_qpy, order):
    """Are the circuits without decoupling identical to those of an earlier job (its archived QPY)?"""
    with open(ref_qpy, "rb") as fh:
        ref = dict(zip(order, qpy.load(fh)))
    return {LABELS[k]: bool(ref[k] == info[LABELS[k]]["tq0"]) for k in order}


def write_prereg(path, args, g, line, info, plan, tq_est, budget, exact, checks, pred, qfile):
    doc = {
        "experiment": "Generic Shor circuit N=21, a=2, t=5 with dynamical decoupling: true circuit and five "
                      "controls on one line, one job",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pre-registered before any submission",
        "generator": "run_ibm.prepare(21, hw, 2, 5): shor_line_generator.build(21, 2, 5, replace(BASE_GENOME, m0=1, "
                     "fill=0, G0=0), Opt(semi=1, ...)) ; controls as in run_ibm.circuits",
        "labels_of_the_run_without_decoupling": {v: k for k, v in LABELS.items()},
        "calibration_snapshot": str(args.calibration),
        "calibration_snapshot_sha256": PT.sha256(args.calibration),
        "line_rule": LINE_RULE,
        "statistic": "C2 success: fraction of shots whose y gives, through a continued-fraction denominator d <= 21 "
                     "of y/32, the exact order of 2 mod 21 (2^d = 1 and 2^(d/p) != 1 for every prime p | d) and "
                     "Miller's method splits 21 (exact_model.crit(21, 2, 5, 'C2'))",
        "standard_error": "binomial sqrt(p(1-p)/n) per circuit; difference of two independent circuits: root of the "
                          "sum of the variances (run_ibm.z_diff); against random: (p - 1/16) / sqrt(p(1-p)/n)",
        "verdict_rule": "SIGNAL iff C2(true_DD) - C2(w) >= 3 standard errors for EACH control w and C2(true_DD) - "
                        "1/16 >= 3 standard errors; otherwise NO SIGNAL. Criterion C1 reported for information "
                        "only.",
        "dynamical_decoupling": dict(
            sequence="X-X", min_window_us=args.min_window,
            scheduler="run_ibm_arith_dd.add_dd: qiskit_ibm_runtime ALAPScheduleAnalysis + PadDynamicalDecoupling "
                      "(gate lengths converted to dt), staggered spacings on neighbouring qubits",
            durations="calibration snapshot", sampler_dynamical_decoupling="disabled (echoes are explicit gates)",
            added_two_qubit_gates=0),
        "circuits_order": list(info),
        "backend": "ibm_kingston",
        "physical_line": list(line),
        "shots": plan,
        "shot_rule": "run_ibm.py: 30 % of the shots on the true circuit, the rest split equally, largest plan within "
                     f"the budget, rounded down to {SHOT_STEP}",
        "qpu_estimate_s": round(tq_est, 1),
        "qpu_estimator": "run_ibm.qpu_time: 2 s per circuit + shots x (250 us + circuit duration + 5 us); run "
                         "without decoupling: estimate 78 s, charged 69 s",
        "budget_s": budget,
        "max_execution_time_s": EXEC_CAP,
        "circuits_file": str(qfile), "circuits_sha256": PT.sha256(qfile),
        "circuits": {k: dict(label_without_dd=d["label"], cz=d["cz"], cz_conditional=d["cz_cond"],
                             cz_without_dd=d["cz_without_dd"], x_gates=d["n_x"], added_operations=d["added"],
                             duration_us=round(d["duration"], 2), duration_without_dd_us=round(d["duration_without_dd"], 2),
                             esp=round(d["esp"], 6), esp_without_dd=round(d["esp_without_dd"], 6),
                             ideal_C2_success=round(d["p_ideal"], 5), random_C2_success=round(d["p_rand"], 5))
                     for k, d in info.items()},
        "exactness_noiseless_dynamic_simulation": exact,
        "checks": checks,
        "noisy_predictions_standard_model": pred,
        "trajectories": {"true": args.traj_true, "control": args.traj},
        "expected_outcome": "no signal: the estimated success probability of every circuit is a few 1e-3, the "
                            "predicted excess over random is far below the statistical resolution of the plan",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    print(f"\nPre-registration written: {path}")


# =============================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ibm", action="store_true", help="REAL device (otherwise local checks)")
    ap.add_argument("--backend", default="ibm_kingston", help="real device name")
    ap.add_argument("--account", default=None, help="name of a saved Qiskit Runtime account")
    ap.add_argument("--calibration", default=None, help="calibration snapshot (.json) for the local checks")
    ap.add_argument("--reference-qpy", default=None,
                    help="local: archived circuits of the run without decoupling (identity check, optional)")
    ap.add_argument("--live-check", action="store_true",
                    help="local: also check ISA validity, gate durations and the line rule on the live target")
    ap.add_argument("--budget", type=float, default=BUDGET, help=f"QPU seconds for the shot plan (<= {BUDGET:.0f})")
    ap.add_argument("--min-window", type=float, default=1.0, help="shortest idle window (us) receiving an echo")
    ap.add_argument("--plan", action="store_true", help="print circuits and shot plan, then exit")
    ap.add_argument("--prereg", default=None, help="local: file to WRITE; --ibm: file to READ")
    ap.add_argument("--scales", default="1", help="local: error scale factors of the noise model")
    ap.add_argument("--traj", type=int, default=40, help="local: noisy trajectories per control")
    ap.add_argument("--traj-true", type=int, default=100, help="local: noisy trajectories for true_DD")
    ap.add_argument("--no-noise", action="store_true", help="local: skip the noisy prediction")
    ap.add_argument("--job", default=None, help="ID of a submitted job to read back and analyse")
    ap.add_argument("--outdir", default="results", help="directory for metadata, analysis and archives")
    ap.add_argument("--tag-prereg", action="store_true",
                    help="--ibm: put the sha256 of the pre-registration file into the job tags")
    ap.add_argument("--skip-exact", action="store_true",
                    help="--ibm: skip the exact re-simulation of the stored circuits (about 2 minutes)")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()
    budget = min(args.budget, BUDGET)
    outdir = Path(args.outdir)

    if args.job:
        from archive_job import archive, service
        svc = service(args.account)
        meta = json.loads((outdir / f"meta_{args.job}.json").read_text(encoding="utf-8"))
        result = svc.job(args.job).result()
        res = analyse({n: RI.counts_y(r) for n, r in zip(meta["order"], result)})
        (outdir / f"analysis_{args.job}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        archive(args.job, outdir, svc)
        return

    if not args.ibm:
        if not args.calibration:
            sys.exit("local mode requires --calibration")
        hw = HW(args.calibration)
        g, line, info = build(hw, args.min_window * 1e-6)
        plan = shot_plan(info, budget)
        tq_est = RI.qpu_time(plan, info)
        summary(line, info, plan, tq_est, budget, args.calibration)
        checks = {"line_rule_result": list(line)}
        bad2q = {k: d["added"] for k, d in info.items()
                 if set(d["added"]) - {"x", "delay", "barrier"} or d["cz"] != d["cz_without_dd"]}
        checks["only_x_delay_barrier_added"] = not bad2q
        if bad2q:
            sys.exit(f"decoupling added operations other than X / delay / barrier: {bad2q}")
        bad = PT.isa_errors({k: d["tq"] for k, d in info.items()}, hw.tg)
        checks["isa_calibration_target"] = bad or "all circuits valid"
        print("ISA check (calibration target):", checks["isa_calibration_target"])
        if bad:
            sys.exit("ISA check failed")
        if args.reference_qpy:
            checks["reference_circuits"] = str(args.reference_qpy)
            checks["reference_circuits_sha256"] = PT.sha256(args.reference_qpy)
            checks["circuits_without_dd_identical_to_reference"] = compare_reference(info, args.reference_qpy,
                                                                                      REFERENCE_ORDER)
            print("Circuits without DD identical to the reference job:",
                  checks["circuits_without_dd_identical_to_reference"])
        if args.live_check:
            from archive_job import service
            be = service(args.account).backend(args.backend)
            bad = PT.isa_errors({k: d["tq"] for k, d in info.items()}, be.target)
            checks["isa_live_target"] = bad or "all circuits valid"
            checks["live_check_time"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            checks["duration_changes_live_vs_calibration"] = {
                k: list(v) for k, v in PT.duration_changes(hw.tg, be.target, list(line)).items()}
            hw_live = HW(be)
            _, live_line = hw_live.place(LG.build(N, A_BASE, T_BITS, g, info["true_DD"]["opt"], measure=True))
            checks["line_rule_on_live_target"] = [int(q) for q in live_line]
            print("ISA check (live target):", checks["isa_live_target"])
            print("Gate durations changed (live vs calibration):",
                  checks["duration_changes_live_vs_calibration"] or "none")
            print("Line rule on the live target:", checks["line_rule_on_live_target"],
                  "(same)" if tuple(live_line) == line else "(DIFFERENT; the snapshot line is kept)")
            if bad:
                sys.exit("ISA check failed on the live target")
        if args.plan:
            return
        print("\nExactness (noiseless exact simulation of the dynamic circuits):")
        exact = exactness(info)
        if [k for k, v in exact.items() if v["max_abs_dev_model"] > EXACT_TOL or v["max_abs_dev_without_dd"] > 1e-12]:
            sys.exit("exactness check FAILED")
        pred = {}
        if not args.no_noise:
            scales = [float(x) for x in args.scales.split(",")]
            print(f"\nStandard noise model: {args.traj} trajectories per control, {args.traj_true} for the true "
                  f"circuit, scales {scales}")
            pred = noisy_prediction(hw, line, g, info, plan, scales, args.traj, args.traj_true)
        if args.prereg:
            qfile = save_circuits(args.prereg, info)
            write_prereg(args.prereg, args, g, line, info, plan, tq_est, budget, exact, checks, pred, qfile)
        return

    # ---------------------------------------------------------------- real device
    if not args.prereg:
        sys.exit("--ibm requires --prereg (the pre-registration file fixes the circuits and the shot plan)")
    pre = json.loads(Path(args.prereg).read_text(encoding="utf-8"))
    order = list(pre["circuits_order"])
    h_pre = PT.check_fingerprints(args.prereg, pre)
    with open(PT.qpy_path(args.prereg), "rb") as fh:
        stored = dict(zip(order, qpy.load(fh)))
    from archive_job import archive, service
    from snapshot_calibration import snapshot
    svc = service(args.account)
    be = svc.backend(args.backend)
    hw = HW(be)
    line = tuple(pre["physical_line"])
    s2 = EM.crit(N, A_BASE, T_BITS, "C2")
    info = {}
    for k in order:
        c = pre["circuits"][k]
        cz = GC.count_2q(stored[k])[0]
        if cz != c["cz"]:
            sys.exit(f"{k}: stored circuit has {cz} CZ, pre-registered {c['cz']}")
        lg = hw.esp(RD.strip_delays(stored[k]), dd=True)
        info[k] = dict(duration=DS.makespan(stored[k], hw) * 1e6, esp=math.exp(lg["total"]), cz=cz)
        print(f"  {k:24s} CZ={cz}  duration~{info[k]['duration']:.1f} us  ESP (live calibration) = "
              f"{info[k]['esp']:.4f}  ideal C2 success {c['ideal_C2_success']:.4f}")
    plan = {k: int(v) for k, v in pre["shots"].items()}
    tq_est = RI.qpu_time(plan, info)
    print(f"Line {list(line)}; plan {plan} -> {sum(plan.values())} shots, estimated QPU time {tq_est:.1f} s")
    if tq_est > pre["budget_s"] + 1e-6:
        sys.exit(f"estimated QPU time {tq_est:.1f} s exceeds the pre-registered budget {pre['budget_s']} s")
    bad = PT.isa_errors(stored, be.target)
    print("ISA check (live target):", bad or "all circuits valid")
    if bad:
        sys.exit("ISA check failed")
    if not args.skip_exact:
        for k in order:
            p = DS.register_probabilities(stored[k], compact=True)
            dev = abs(float(p[s2].sum()) - pre["circuits"][k]["ideal_C2_success"])
            print(f"  exactness {k:24s} C2 success {float(p[s2].sum()):.5f} (pre-registered ideal "
                  f"{pre['circuits'][k]['ideal_C2_success']:.5f})", flush=True)
            if dev > 1e-4:
                sys.exit(f"{k}: exactness check failed")
    tags = PT.job_tags(args.prereg) if args.tag_prereg else None
    print("Job tags:", tags)
    if args.plan:
        return
    if not args.yes and input(f"Submit {len(plan)} circuits ({sum(plan.values())} shots, ~{tq_est:.0f} s QPU) to "
                              f"{be.name}? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    ctx = snapshot(be, svc, outdir, "before_shor21_t5_dd")
    available = ctx["usage"].get("usage_remaining_seconds")
    print(f"  QPU time available on the account before submission: {available} s")
    if available is not None and float(available) < tq_est * 1.2:
        sys.exit("not enough QPU time left in the usage window")
    job = PT.submit(be, [(stored[k], None, plan[k]) for k in order], EXEC_CAP, tq_est, tags)
    jid = job.job_id()
    meta = dict(experiment="shor_N21_a2_t5_dd", N=N, a=A_BASE, t=T_BITS, order=order, plan=plan, backend=be.name,
                physical_line=list(line), cz={k: info[k]["cz"] for k in order}, prereg=str(args.prereg),
                prereg_sha256=h_pre, job_tags=tags, date=datetime.datetime.now().isoformat(timespec="seconds"),
                qpu_estimate_s=tq_est)
    (outdir / f"meta_{jid}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Job submitted: {jid} (read back: --ibm --job {jid} --outdir {outdir})")
    result = job.result()
    res = analyse({n: RI.counts_y(r) for n, r in zip(order, result)})
    (outdir / f"analysis_{jid}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    archive(jid, outdir, svc)
    snapshot(be, svc, outdir, "after_shor21_t5_dd")


if __name__ == "__main__":
    main()
