"""Replication of the dynamical-decoupling test of the modular multiplication (N = 21, a = 2, t = 3), in one job.

The second run of run_ibm_arith_dd.py gave S(true_DD) = +0.0189 +- 0.0027 against the three DD witnesses at
+3.2 / +3.7 / +4.0 standard errors (pre-registered primary rule met). This script repeats that primary test with
the IDENTICAL circuits, read from the pre-registered circuit file of that run (--source), on the same physical
line, without the true circuit without decoupling:
    true_DD, mult_by_1_DD, wrong_constant_DD, classical_exponent_DD
Statistic and primary rule (unchanged, fixed before the run): S = P(y in {0,4}) - P(y in {2,6});
SIGNAL iff S(true_DD) > 0 AND S(true_DD) - S(w) >= 3 standard errors for EACH witness w (multinomial errors as in
run_ibm_arith.py). Also reported (not used for the verdict): fitted visibility and phase of the e1 + e2 coherence
of true_DD, S(true_DD) against the first DD run, S / P(y0 = 0).

Shot allocation (fixed before the run): with witness expectation S(w) = 0 the smallest margin
min_w (S_t - S_w) / sqrt(V_t / n_t + V_w / n_w) is maximised under the QPU budget by n_w = V_w / x for every
witness and n_t = V_t / y with y / x = sqrt(c_t V_t / sum_w c_w V_w) (c = QPU seconds per shot); the single-shot
variances V are those observed in the first DD run. This allocation does not depend on the expected S_t.

  Checks + pre-registration (exact simulation, noisy and data-based predictions):
      python src/run_ibm_replication.py --source DIR/prereg_arith_dd.json --calibration CAL.json
                                        --prereg OUT/prereg_replication.json [--account NAME --live-check]
  Real device (submits the pre-registered circuits and shot plan):
      python src/run_ibm_replication.py --ibm --prereg OUT/prereg_replication.json --outdir OUT [--account NAME]
                                        [--tag-prereg] [--yes] [--plan]
  Read back a job:
      python src/run_ibm_replication.py --ibm --job <ID> --outdir OUT [--account NAME]
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
from qiskit import qpy  # noqa: E402

import dynamic_sim as DS  # noqa: E402
import prereg_tools as PT  # noqa: E402
import run_ibm as RI  # noqa: E402
import run_ibm_arith as RA  # noqa: E402
import run_ibm_arith_dd as RD  # noqa: E402
from hardware import HW  # noqa: E402

BUDGET = 60.0            # QPU seconds for the shot plan (package estimator run_ibm.qpu_time)
EXEC_CAP = 80            # max_execution_time of the job (safety cap above the plan)
Z_REQ = 3.0
EXACT_TOL = 1e-6
ORDER = ("true_DD", "mult_by_1_DD", "wrong_constant_DD", "classical_exponent_DD")
WITNESSES = ORDER[1:]
SHOT_STEP = 100


# =============================================================================== source run
def load_source(source):
    """Pre-registration of the first DD run, its circuits (checked against its sha256), the circuits of the run
    without decoupling they were derived from, its analysis and raw counts."""
    source = Path(source)
    pre = json.loads(source.read_text(encoding="utf-8"))
    qfile = PT.qpy_path(source)
    if PT.sha256(qfile) != pre["circuits_sha256"]:
        sys.exit(f"{qfile} differs from the circuits pre-registered in {source}")
    with open(qfile, "rb") as fh:
        circs = dict(zip(pre["circuits_order"], qpy.load(fh)))
    ref_pre = PT.resolve(pre["reference_preregistration"], source.parent)
    ref_qpy = PT.resolve(pre["reference_circuits"], source.parent)
    with open(ref_qpy, "rb") as fh:
        base = dict(zip(list(json.loads(ref_pre.read_text(encoding="utf-8"))["circuits"]), qpy.load(fh)))
    prev = previous_dd_run(source.parent)
    return pre, circs, base, qfile, prev


def previous_dd_run(directory):
    """Analysis of the first DD run (file analysis_<job>.json containing true_DD) and its raw counts."""
    for f in sorted(Path(directory).glob("analysis_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "true_DD" not in d.get("S", {}):
            continue
        jid = f.stem.replace("analysis_", "")
        meta = json.loads((f.parent / f"meta_{jid}.json").read_text(encoding="utf-8"))
        raw = json.loads((f.parent / f"job_{jid}.json").read_text(encoding="utf-8"))
        counts = {k: {int(y, 2): v for y, v in raw["circuits"][meta["order"].index(k)]["registers"]["y"].items()}
                  for k in ORDER}
        var1 = {k: d["S"][k]["se"] ** 2 * d["S"][k]["n"] for k in ORDER}
        return dict(job=jid, S={k: dict(S=d["S"][k]["S"], se=d["S"][k]["se"], n=d["S"][k]["n"]) for k in ORDER},
                    z_primary=d["z_primary"], single_shot_variance=var1,
                    phase_fit_true_DD=d["phase_fit"]["true_DD"], counts=counts)
    sys.exit(f"no analysis of the first DD run in {directory}")


# =============================================================================== circuits
def build(hw, circs, base):
    ideals = {k: RA.ideal_P(o) for k, o in RA.witnesses().items()}
    info = {}
    for name in ORDER:
        b, dd = RD.CIRCUITS[name]
        RD.describe(info, name, circs[name], hw, ideals[b], b, dd, base[b])
        info[name]["gates1q_log_base"] = hw.esp(RD.strip_delays(base[b]))["gates1q"]
    return info


def cost(info, name):
    return RI.REP_DELAY + info[name]["duration"] * 1e-6 + 5e-6


def allocate(info, var1, budget):
    """Shots maximising the smallest expected margin true_DD - witness (witness S = 0) within the budget."""
    a = cost(info, "true_DD") * var1["true_DD"]
    b = sum(cost(info, w) * var1[w] for w in WITNESSES)
    avail = budget - RI.OVERHEAD * len(ORDER)
    s = (math.sqrt(a) + math.sqrt(b)) / avail
    raw = {"true_DD": var1["true_DD"] / (s * math.sqrt(a))}
    raw.update({w: var1[w] / (s * math.sqrt(b)) for w in WITNESSES})
    plan = {k: int(v // SHOT_STEP * SHOT_STEP) for k, v in raw.items()}
    while RI.qpu_time(plan, info) > budget:
        plan = {k: v - SHOT_STEP for k, v in plan.items()}
    return plan


def summary(line, info, plan, tq_est, budget, hwname):
    print(f"Backend/calibration: {hwname}")
    print(f"N={RA.N}, a={RA.A_BASE}, t={RA.T_BITS}, physical line ({len(line)} qubits): {list(line)}")
    for name, d in info.items():
        add = ", ".join(f"+{v} {k}" for k, v in sorted(d["added"].items())) or "none"
        print(f"  {name:22s} CZ={d['cz']:4d} (+{d['cz_cond']} conditional)  X={d['n_x']:4d}  added: {add:24s} "
              f"duration~{d['duration']:.1f} us  ESP={d['esp']:.4f}  ideal S={d['S']:+.4f}")
    print(f"Plan: {plan} -> {sum(plan.values())} shots, estimated QPU time {tq_est:.1f} s (budget {budget:.0f} s)")


# =============================================================================== predictions
def z_margins(S, var1, plan):
    se = {k: math.sqrt(var1[k] / plan[k]) for k in plan}
    z = {w: (S["true_DD"] - S[w]) / math.hypot(se["true_DD"], se[w]) for w in WITNESSES}
    return z, min(z.values())


def power(S_mean, S_sd, var1, plan, n=200_000, seed=3):
    """Probability that the primary rule is met: true S drawn from N(S_mean, S_sd) (uncertainty of the input
    estimate; S_sd = 0 for a fixed value), witnesses at S = 0, Gaussian sampling errors of the plan."""
    rng = np.random.default_rng(seed)
    se = {k: math.sqrt(var1[k] / plan[k]) for k in plan}
    st = rng.normal(S_mean, S_sd, n) if S_sd > 0 else np.full(n, S_mean)
    obs_t = st + rng.normal(0, se["true_DD"], n)
    ok = obs_t > 0
    for w in WITNESSES:
        obs_w = rng.normal(0, se[w], n)
        ok &= (obs_t - obs_w) / math.hypot(se["true_DD"], se[w]) >= Z_REQ
    return float(ok.mean())


def data_prediction(prev, info, plan, var1):
    """From the first DD run itself: S(true_DD) = 0.25 v (fitted visibility, phase refocused; same circuits),
    witnesses 0. Expected margins, and probability of meeting the rule at the point estimate and averaged over
    the uncertainty of v."""
    f = prev["phase_fit_true_DD"]
    S = {k: 0.0 for k in plan}
    S["true_DD"] = 0.25 * f["v"]
    z, zmin = z_margins(S, var1, plan)
    S_obs = prev["S"]["true_DD"]
    z_obs, zmin_obs = z_margins({**S, "true_DD": S_obs["S"]}, var1, plan)
    out = dict(visibility=f["v"], visibility_se=f["v_se"], S=S, z=z, z_min=zmin,
               probability_rule_met=power(S["true_DD"], 0.0, var1, plan),
               probability_rule_met_v_uncertain=power(S["true_DD"], 0.25 * f["v_se"], var1, plan),
               observed_S_true_DD=S_obs["S"], z_min_at_observed_S=zmin_obs,
               probability_rule_met_observed_S_uncertain=power(S_obs["S"], S_obs["se"], var1, plan))
    print(f"  data-based (visibility {f['v']:.4f} +- {f['v_se']:.4f}): S(true_DD) = {S['true_DD']:+.4f}  "
          f"z = " + ", ".join(f"{k} {v:+.2f}" for k, v in z.items()) + f"  -> min {zmin:+.2f}; "
          f"P(rule met) = {out['probability_rule_met']:.2f} (fixed S), {out['probability_rule_met_v_uncertain']:.2f} "
          f"(v uncertain)")
    print(f"  data-based (observed S(true_DD) = {S_obs['S']:+.4f} +- {S_obs['se']:.4f}): z min {zmin_obs:+.2f}; "
          f"P(rule met) = {out['probability_rule_met_observed_S_uncertain']:.2f} (S uncertain)")
    return out


def standard_prediction(hw, line, info, plan, var1, ntraj, ntraj_true):
    """Standard noise model (run_ibm_arith_dd.noisy_projection: calibrated depolarising errors, readout, T1/T2
    with half of the pure dephasing removed by DD, no coherent errors) on today's calibration, times the error
    factor of the added X gates (absent from the static circuits of the model)."""
    nz = RD.noisy_projection(hw, line, info, plan, [1.0], ntraj, ntraj_true)[1.0]
    S = {k: nz["S"][k] * math.exp(info[k]["gates1q_log"] - info[k]["gates1q_log_base"]) for k in plan}
    v1 = {k: max(RA.s_stat(nz["P"][k])[1], 1e-12) for k in plan}
    z, zmin = z_margins(S, v1, plan)
    out = dict(S_model=nz["S"], S_model_traj_se=nz["S_traj_se"], S=S, z=z, z_min=zmin,
               P={k: [round(float(x), 5) for x in v] for k, v in nz["P"].items()},
               probability_rule_met=power(S["true_DD"], nz["S_traj_se"]["true_DD"], var1, plan))
    print(f"  standard model: S = " + ", ".join(f"{k} {v:+.4f}" for k, v in S.items())
          + f"  z min {zmin:+.2f}; P(rule met) {out['probability_rule_met']:.2f}")
    return out


# =============================================================================== analysis
def analyse(counts_by, pre):
    ideals = {k: RA.ideal_P(o) for k, o in RA.witnesses().items()}
    print()
    print(f"=== Analysis (replication): S = P(y in {set(RA.CELLS_A)}) - P(y in {set(RA.CELLS_B)}), N={RA.N}, "
          f"a={RA.A_BASE}, t={RA.T_BITS}, dynamical decoupling ===")
    st = {k: RA.s_from_counts(c) for k, c in counts_by.items()}
    res = {"S": {}, "z_primary": {}}
    for name, (s, e, n) in st.items():
        c = RA.as_array(counts_by[name])
        q = (c[0] + c[2] + c[4] + c[6]) / n
        res["S"][name] = dict(S=s, se=e, n=n, S_cond=s / q, P_y0_0=q)
        print(f"  {name:22s}: S = {s:+.4f} +- {e:.4f}  (ideal {RA.s_stat(ideals[RD.CIRCUITS[name][0]])[0]:+.4f}, "
              f"{n} shots; S/P(y0=0) = {s / q:+.4f}; P(y0=0) = {q:.4f})")
    sv, ev, _ = st["true_DD"]
    for w in WITNESSES:
        s, e, _ = st[w]
        res["z_primary"][w] = (sv - s) / math.hypot(ev, e)
        print(f"    true_DD - {w:22s} = {sv - s:+.4f} ({res['z_primary'][w]:+.1f} sigma)")
    zmin = min(res["z_primary"].values())
    primary = sv > 0 and zmin >= Z_REQ
    prev = (pre or {}).get("previous_dd_run", {}).get("S", {}).get("true_DD")
    if prev:
        d = sv - prev["S"]
        zr = d / math.hypot(ev, prev["se"])
        w1, w2 = 1 / prev["se"] ** 2, 1 / ev ** 2
        comb = (w1 * prev["S"] + w2 * sv) / (w1 + w2)
        res.update(z_vs_first_dd_run=zr, S_combined=comb, S_combined_se=1 / math.sqrt(w1 + w2))
        print(f"    true_DD - first DD run   = {d:+.4f} ({zr:+.1f} sigma); inverse-variance mean of both runs "
              f"{comb:+.4f} +- {1 / math.sqrt(w1 + w2):.4f} (reported, not used for the verdict)")
    res["phase_fit"] = {}
    for name, base, bit in [("true_DD", "true", 1), ("mult_by_1_DD", "mult_by_1", 2)]:
        f = RD.phase_fit(counts_by[name], base, bit)
        res["phase_fit"][name] = f
        if "phase_deg" in f:
            print(f"    fit {name:18s} v = {f['v']:.4f} +- {f['v_se']:.4f}  phase = {f['phase_deg']:+.1f} +- "
                  f"{f['phase_se_deg']:.1f} deg  2lnLR = {f['two_lnLR']:.1f}")
        else:
            print(f"    fit {name:18s} v cos(phase) = {f['x']:+.4f} +- {f['x_se']:.4f}  2lnLR = {f['two_lnLR']:.1f}")
    v = (f"SIGNAL REPLICATED: S(true_DD) > 0 and exceeds every DD witness by >= {Z_REQ:.0f} sigma" if primary else
         f"NOT REPLICATED (S(true_DD) = {sv:+.4f}, smallest difference {zmin:+.1f} sigma)")
    print("  PRIMARY VERDICT:", v)
    res.update(z_primary_min=zmin, verdict_primary=v)
    return res


# =============================================================================== files
def save_circuits(prereg, circs):
    path = PT.qpy_path(prereg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        qpy.dump([circs[k] for k in ORDER], fh)
    print(f"Circuits saved: {path} (order {list(ORDER)})")
    return path


def load_circuits(prereg):
    with open(PT.qpy_path(prereg), "rb") as fh:
        return dict(zip(ORDER, qpy.load(fh)))


def write_prereg(path, args, src_pre, src_qpy, line, info, plan, tq_est, budget, exact, checks, prev, pred_data,
                 pred_std, qfile, var1):
    def r(x, n=5):
        return {k: round(float(v), n) for k, v in x.items()}
    doc = {
        "experiment": "Replication of the dynamical-decoupling test of one controlled modular multiplication "
                      "(N=21, a=2, t=3), identical circuits, one job",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pre-registered before any submission",
        "source_preregistration": str(args.source), "source_preregistration_sha256": PT.sha256(args.source),
        "source_circuits": str(src_qpy), "source_circuits_sha256": PT.sha256(src_qpy),
        "circuits_identical_to_source": True,
        "previous_dd_run": {k: v for k, v in prev.items() if k != "counts"},
        "calibration_snapshot": str(args.calibration),
        "calibration_snapshot_sha256": PT.sha256(args.calibration),
        "statistic": "S = P(y in {0,4}) - P(y in {2,6}) (exponent register y, 3 bits)",
        "standard_error": "Var(S_hat) = [pA + pB - (pA - pB)^2] / n per circuit (multinomial); difference of two "
                          "independent circuits: sqrt(Var1 + Var2)",
        "verdict_rule_primary": "SIGNAL (replicated) iff S(true_DD) > 0 AND S(true_DD) - S(w) >= 3 standard errors "
                                "for EACH w in {mult_by_1_DD, wrong_constant_DD, classical_exponent_DD}; otherwise "
                                "NOT REPLICATED",
        "reported_not_for_verdict": ["S(true_DD) - S(true_DD, first DD run) and inverse-variance mean of both runs",
                                     "fitted visibility and phase of the e1+e2 coherence of true_DD; v cos(phase) "
                                     "of the e2 coherence of mult_by_1_DD",
                                     "S / P(y0=0) and P(y0=0) of every circuit"],
        "shot_allocation": "maximin of the expected margins true_DD - witness with witness S = 0: n_w = V_w / x, "
                           "n_t = V_t / y, y / x = sqrt(c_t V_t / sum_w c_w V_w), c = QPU s per shot "
                           "(run_ibm.qpu_time), V = single-shot variance of S observed in the first DD run; rounded "
                           f"down to {SHOT_STEP}",
        "single_shot_variance_used": r(var1, 6),
        "circuits_order": list(ORDER),
        "backend": "ibm_kingston",
        "physical_line": list(line),
        "shots": plan,
        "qpu_estimate_s": round(tq_est, 1),
        "qpu_estimator": "run_ibm.qpu_time: 2 s per circuit + shots x (250 us repetition delay + circuit duration + "
                         "5 us); first DD run: estimate 67.5 s, charged 61 s",
        "budget_s": budget,
        "max_execution_time_s": EXEC_CAP,
        "circuits_file": str(qfile), "circuits_sha256": PT.sha256(qfile),
        "circuits": {k: dict(logical=d["base"], dd=d["dd"], cz=d["cz"], cz_conditional=d["cz_cond"],
                             x_gates=d["n_x"], added_operations=d["added"], duration_us=round(d["duration"], 2),
                             esp=round(d["esp"], 5), ideal_P=[round(float(x), 5) for x in d["P"]],
                             ideal_S=round(d["S"], 5)) for k, d in info.items()},
        "exactness_noiseless_dynamic_simulation": exact,
        "checks": checks,
        "predictions": {
            "data_based": dict(description="S(true_DD) = 0.25 x fitted visibility of true_DD in the first DD run "
                                           "(phase refocused, same circuits); witnesses 0; probability of meeting "
                                           "the rule by Gaussian simulation of the plan",
                               **{k: (r(v) if isinstance(v, dict) else round(v, 5)) for k, v in pred_data.items()}),
            "standard_noise_model": (dict(description=RD.noisy_projection.__doc__.split("\n")[0] + " Today's "
                                          "calibration; S times the X-gate error factor.",
                                          trajectories={"true": args.traj_true, "witness": args.traj},
                                          **{k: (r(v) if isinstance(v, dict) and k != "P" else
                                                 (v if k == "P" else round(v, 5))) for k, v in pred_std.items()})
                                     if pred_std else None)},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"\nPre-registration written: {path}")


# =============================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ibm", action="store_true", help="REAL device (otherwise local checks)")
    ap.add_argument("--backend", default="ibm_kingston", help="real device name")
    ap.add_argument("--account", default=None, help="name of a saved Qiskit Runtime account")
    ap.add_argument("--source", default=None, help="pre-registration of the first DD run (its circuits are reused)")
    ap.add_argument("--calibration", default=None, help="calibration snapshot (.json) for the local checks")
    ap.add_argument("--live-check", action="store_true",
                    help="local: also check ISA validity and gate durations on the live target (read-only)")
    ap.add_argument("--budget", type=float, default=BUDGET, help=f"QPU seconds for the shot plan (<= {BUDGET:.0f})")
    ap.add_argument("--plan", action="store_true", help="print circuits and shot plan, then exit")
    ap.add_argument("--prereg", default=None, help="local: file to WRITE; --ibm: file to READ")
    ap.add_argument("--traj", type=int, default=100, help="local: noisy trajectories per witness")
    ap.add_argument("--traj-true", type=int, default=300, help="local: noisy trajectories for true_DD")
    ap.add_argument("--no-noise", action="store_true", help="local: skip the standard noisy prediction")
    ap.add_argument("--job", default=None, help="ID of a submitted job to read back and analyse")
    ap.add_argument("--outdir", default="results", help="directory for metadata, analysis and archives")
    ap.add_argument("--tag-prereg", action="store_true",
                    help="--ibm: put the sha256 of the pre-registration file into the job tags")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()
    budget = min(args.budget, BUDGET)
    outdir = Path(args.outdir)

    if args.job:
        from archive_job import archive, service
        svc = service(args.account)
        meta = json.loads((outdir / f"meta_{args.job}.json").read_text(encoding="utf-8"))
        pre = json.loads(Path(meta["prereg"]).read_text(encoding="utf-8"))
        result = svc.job(args.job).result()
        res = analyse({n: RI.counts_y(r) for n, r in zip(meta["order"], result)}, pre)
        (outdir / f"analysis_{args.job}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        archive(args.job, outdir, svc)
        return

    if not args.ibm:
        if not (args.source and args.calibration):
            sys.exit("local mode requires --source (first DD run pre-registration) and --calibration")
        src_pre, src_circs, base, src_qpy, prev = load_source(args.source)
        circs = {k: src_circs[k] for k in ORDER}
        line = tuple(src_pre["physical_line"])
        hw = HW(args.calibration)
        info = build(hw, circs, base)
        var1 = prev["single_shot_variance"]
        plan = allocate(info, var1, budget)
        tq_est = RI.qpu_time(plan, info)
        summary(line, info, plan, tq_est, budget, args.calibration)
        checks = {}
        src_cal = PT.resolve(src_pre["calibration_snapshot"], Path(args.source).parent)
        checks["duration_changes_vs_source_calibration"] = {
            k: list(v) for k, v in PT.duration_changes(HW(str(src_cal)).tg, hw.tg, list(line)).items()}
        bad = PT.isa_errors(circs, hw.tg)
        checks["isa_calibration_target"] = bad or "all circuits valid"
        print("ISA check (calibration target):", checks["isa_calibration_target"])
        print("Gate durations changed since the source calibration (line qubits):",
              checks["duration_changes_vs_source_calibration"] or "none")
        if bad:
            sys.exit("ISA check failed")
        if args.live_check:
            from archive_job import service
            be = service(args.account).backend(args.backend)
            bad = PT.isa_errors(circs, be.target)
            checks["isa_live_target"] = bad or "all circuits valid"
            checks["live_check_time"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            checks["duration_changes_live_vs_source_calibration"] = {
                k: list(v) for k, v in PT.duration_changes(HW(str(src_cal)).tg, be.target, list(line)).items()}
            print("ISA check (live target):", checks["isa_live_target"])
            print("Gate durations changed (live vs source calibration):",
                  checks["duration_changes_live_vs_source_calibration"] or "none")
            if bad:
                sys.exit("ISA check failed on the live target")
        if args.plan:
            return
        print("\nExactness (noiseless exact simulation of the dynamic circuits):")
        exact = RD.exactness(info, base)
        if [k for k, v in exact.items() if v["max_abs_dev_model"] > EXACT_TOL or v["max_abs_dev_without_dd"] > 1e-12]:
            sys.exit("exactness check FAILED")
        print("\nPredictions with the plan:")
        pred_data = data_prediction(prev, info, plan, var1)
        pred_std = None
        if not args.no_noise:
            pred_std = standard_prediction(hw, line, info, plan, var1, args.traj, args.traj_true)
        if args.prereg:
            qfile = save_circuits(args.prereg, circs)
            write_prereg(args.prereg, args, src_pre, src_qpy, line, info, plan, tq_est, budget, exact, checks, prev,
                         pred_data, pred_std, qfile, var1)
        return

    # ---------------------------------------------------------------- real device
    if not args.prereg:
        sys.exit("--ibm requires --prereg (the pre-registration file fixes the circuits and the shot plan)")
    pre = json.loads(Path(args.prereg).read_text(encoding="utf-8"))
    if list(pre["circuits_order"]) != list(ORDER):
        sys.exit("unexpected circuit order in the pre-registration")
    h_pre = PT.check_fingerprints(args.prereg, pre)
    stored = load_circuits(args.prereg)
    src = PT.resolve(pre["source_circuits"], Path(args.prereg).parent)
    if PT.sha256(src) != pre["source_circuits_sha256"]:
        sys.exit(f"{src} differs from the source circuits recorded in the pre-registration")
    with open(src, "rb") as fh:
        src_circs = dict(zip(json.loads(PT.resolve(pre["source_preregistration"], Path(args.prereg).parent)
                                        .read_text(encoding="utf-8"))["circuits_order"], qpy.load(fh)))
    if any(stored[k] != src_circs[k] for k in ORDER):
        sys.exit("stored circuits are not identical to the circuits of the first DD run")
    from archive_job import archive, service
    from snapshot_calibration import snapshot
    svc = service(args.account)
    be = svc.backend(args.backend)
    hw = HW(be)
    line = tuple(pre["physical_line"])
    ideals = {k: RA.ideal_P(o) for k, o in RA.witnesses().items()}
    info = {}
    for k in ORDER:
        b, dd = RD.CIRCUITS[k]
        RD.describe(info, k, stored[k], hw, ideals[b], b, dd)
        info[k]["added"] = pre["circuits"][k]["added_operations"]
        if info[k]["cz"] != pre["circuits"][k]["cz"]:
            sys.exit(f"{k}: stored circuit has {info[k]['cz']} CZ, pre-registered {pre['circuits'][k]['cz']}")
    plan = {k: int(v) for k, v in pre["shots"].items()}
    tq_est = RI.qpu_time(plan, info)
    print("ESP below: today's live calibration, pre-registered circuits")
    summary(line, info, plan, tq_est, pre["budget_s"], be.name)
    if tq_est > pre["budget_s"] + 1e-6:
        sys.exit(f"estimated QPU time {tq_est:.1f} s exceeds the pre-registered budget {pre['budget_s']} s")
    for k in ORDER:
        dev = float(np.abs(DS.register_probabilities(stored[k]) - info[k]["P"]).max())
        print(f"  exactness {k:22s} max|P - P_ideal| = {dev:.1e}")
        if dev > EXACT_TOL:
            sys.exit(f"{k}: exactness check failed")
    bad = PT.isa_errors(stored, be.target)
    print("ISA check (live target):", bad or "all circuits valid")
    if bad:
        sys.exit("ISA check failed")
    tags = PT.job_tags(args.prereg) if args.tag_prereg else None
    print("Job tags:", tags)
    if args.plan:
        return
    if not args.yes and input(f"Submit {len(plan)} circuits ({sum(plan.values())} shots, ~{tq_est:.0f} s QPU) to "
                              f"{be.name}? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    ctx = snapshot(be, svc, outdir, "before_replication")
    available = ctx["usage"].get("usage_remaining_seconds")
    print(f"  QPU time available on the account before submission: {available} s")
    if available is not None and float(available) < tq_est * 1.2:
        sys.exit("not enough QPU time left in the usage window")
    job = PT.submit(be, [(stored[k], None, plan[k]) for k in ORDER], EXEC_CAP, tq_est, tags)
    jid = job.job_id()
    meta = dict(experiment="arith_N21_a2_t3_dd_replication", N=RA.N, a=RA.A_BASE, t=RA.T_BITS, order=list(ORDER),
                plan=plan, backend=be.name, physical_line=list(line), cz={k: info[k]["cz"] for k in ORDER},
                prereg=str(args.prereg), prereg_sha256=h_pre, job_tags=tags,
                date=datetime.datetime.now().isoformat(timespec="seconds"), qpu_estimate_s=tq_est)
    (outdir / f"meta_{jid}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Job submitted: {jid} (read back: --ibm --job {jid} --outdir {outdir})")
    result = job.result()
    res = analyse({n: RI.counts_y(r) for n, r in zip(ORDER, result)}, pre)
    (outdir / f"analysis_{jid}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    archive(jid, outdir, svc)
    snapshot(be, svc, outdir, "after_replication")


if __name__ == "__main__":
    main()
