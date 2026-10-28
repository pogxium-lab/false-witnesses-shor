"""Generic Shor experiment for N = 15 (or 21) on an IBM device, with false witnesses and a random baseline.

Default: DRY RUN (local noisy simulation with an IBM noise model, FakeKingston by default); no account needed.
Nothing is submitted to IBM without --ibm AND confirmation; the QPU time of the job is capped (--budget, default
540 s, hard cap 590 s).

  Store the IBM account (once):   python src/run_ibm.py --configure
  Dry run (default):              python src/run_ibm.py [--N 21] [--backend FakeFez]
  Plan only (nothing is run):     python src/run_ibm.py --plan
  Real device:                    python src/run_ibm.py --ibm --backend ibm_kingston
  Read back a job:                python src/run_ibm.py --ibm --job <ID>

Circuits (ALL produced by the same generator, shor_line_generator.build, from N and a only; same physical line).
The labels below are those of the archived data:
  true                 true circuit: coset Shor on a line of the heavy-hex lattice (dropped tail, A prepared
                       directly, semiclassical order, exponent bits measured right after their multiplication with
                       feed-forward). N=15: a=7, t=3 (default): 14 qubits, 724 CZ; a=2, t=4: 16 qubits, 763 CZ.
                       N=21: a=2, t=5: 19 qubits, 1538 CZ.
  decohered_eK         same circuit, exponent bit e_K MEASURED right after its preparation (decohered). Such a
                       control is kept only if it DISCRIMINATES (ideal success < ideal success of the true circuit
                       - 0.02): decohering a bit already "read" by the work register (2^(K+1) divides r) changes
                       nothing, so it would not be a wrong circuit;
  classical_exponent   all exponent bits measured (classical exponent) = EXPERIMENTAL random baseline;
  scrambled_constants  scrambled multiplication constants (same gates, other angles).
Criterion (fixed BEFORE the experiment): C2 "exact order" = a convergent denominator d <= N of y/2^t IS the order
of a (a^d = 1 and a^(d/p) != 1 for every prime p | d) and Miller's method factors N. The weaker criterion C1
(any multiple of the order accepted) is shown for information only: at N=15, t=4, decohering e_2 raises the C1
success above that of the true circuit, because multiples of the order are produced by wrong circuits too.
A signal is declared only if the true circuit beats EACH control AND the uniform random baseline at >= 3 sigma.
"""
from __future__ import annotations

import argparse
import datetime
import getpass
import json
import math
import os
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

import exact_model as EM  # noqa: E402
import generator_core as GC  # noqa: E402
import shor_line_generator as LG  # noqa: E402
from hardware import BASIS, HW  # noqa: E402

REP_DELAY = 250e-6
OVERHEAD = 2.0                     # s per sub-job (IBM documentation)
HARD_CAP = 590.0
CRIT = "C2"                        # criterion of the verdict (see module docstring)
# Default settings (a, t): N=15 -> a=7, t=3 (724 CZ, 14 qubits, non-trivial multiplication by 4, best simulated
# contrast); variant a=2, t=4 (763 CZ, 16 qubits). N=21 -> a=2, t=5 (1538 CZ, 19 qubits).
SETTINGS = {15: dict(a=7, t=3, m0=1, fill=0, G0=0), 21: dict(a=2, t=5, m0=1, fill=0, G0=0)}
TRUE_SHARE = 0.30                  # share of shots on the true circuit; the rest is split among the controls


# =============================================================================== circuits
def circuits(N, a, t):
    r = SETTINGS[N]
    g = replace(GC.BASE_GENOME, m0=r["m0"], fill=r["fill"], G0=r["G0"])
    L = min(GC.lookup_bits(N, a, t), 2)
    ks = list(range(L, t))
    out = {"true": LG.Opt(semi=1)}
    out["classical_exponent"] = LG.Opt(semi=1, decohere=tuple(range(t)))
    for k in ks:
        out[f"decohered_e{k}"] = LG.Opt(semi=1, decohere=(k,))
    out["scrambled_constants"] = LG.Opt(semi=1, consts=EM.fake_consts(N, a, t, L))
    pv = ideal_success(N, a, t, g, out["true"])[0]
    for name in [k for k in out if k.startswith("decohered_e")]:
        if ideal_success(N, a, t, g, out[name])[0] > pv - 0.02:
            print(f"  ({name} dropped: not discriminating, same ideal success as the true circuit)")
            del out[name]
    return t, g, out


def ideal_success(N, a, t, g, opt):
    s = EM.crit(N, a, t, CRIT)
    P = EM.shor_P(N, a, t, g, order=opt.order, consts=opt.consts, table=opt.table, decohere=opt.decohere)
    return float(P[s].sum()), float(s.mean())


def prepare(N, hw, a, t):
    t, g, opts = circuits(N, a, t)
    ref = LG.build(N, a, t, g, opts["true"], measure=True)
    _, path = hw.place(ref)
    out = {}
    for name, opt in opts.items():
        qc = LG.build(N, a, t, g, opt, measure=True)
        qd = transpile(qc, basis_gates=BASIS, optimization_level=0)
        tq = transpile(qd, hw.be, optimization_level=2, seed_transpiler=1, initial_layout=list(path))
        cz = GC.count_2q(tq)[0]
        lg = hw.esp(tq)
        pid, prand = ideal_success(N, a, t, g, opt)
        out[name] = dict(tq=tq, cz=cz, duration=lg["duration_us"], esp=math.exp(lg["total"]), p_ideal=pid,
                         p_rand=prand, opt=opt)
    return t, g, path, out


# =============================================================================== statistics
def successes(counts, N, a, t, kind=CRIT):
    s = EM.crit(N, a, t, kind)
    n = sum(counts.values())
    k = sum(v for y, v in counts.items() if s[y])
    return k, n


def z_diff(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) / se if se > 0 else float("nan")


def analyse(N, a, t, counts_by):
    s = EM.crit(N, a, t, CRIT)
    p0 = float(s.mean())
    kv, nv = successes(counts_by["true"], N, a, t)
    pv = kv / nv
    print()
    print(f"=== Analysis (criterion {CRIT} 'exact order', N={N}, t={t}) ===")
    print(f"  uniform random : {p0:.4f}")
    res = {"random_baseline": p0}
    zs = []
    for name, c in counts_by.items():
        k, n = successes(c, N, a, t)
        p = k / n
        res[name] = dict(k=k, n=n, p=p)
        line = f"  {name:19s} : {p:.4f} +- {math.sqrt(p * (1 - p) / n):.4f}  ({n} shots)"
        if name != "true":
            z = z_diff(kv, nv, k, n)
            zs.append(z)
            line += f"   true - {name} = {pv - p:+.4f}  ({z:+.1f} sigma)"
        print(line)
    z0 = (pv - p0) / math.sqrt(pv * (1 - pv) / nv)
    zs.append(z0)
    print(f"  true - random = {pv - p0:+.4f} ({z0:+.1f} sigma)")
    zmin = min(zs)
    c1 = {k: successes(c, N, a, t, "C1") for k, c in counts_by.items()}
    print("  (for information, criterion C1: " + ", ".join(f"{k} {v[0] / v[1]:.4f}" for k, v in c1.items())
          + "; NOT used for the verdict)")
    verdict = ("SIGNAL: the true circuit beats every control and the random baseline at >= 3 sigma" if zmin >= 3 else
               "NO SIGNIFICANT SIGNAL (worst difference at %.1f sigma)" % zmin)
    print("  VERDICT:", verdict)
    res["z_min"] = zmin
    res["verdict"] = verdict
    return res


def shots_needed(pv, pf, z=3.0):
    """Shots PER circuit (same number for both) to resolve pv - pf at z sigma."""
    if pv <= pf:
        return float("inf")
    return z * z * (pv * (1 - pv) + pf * (1 - pf)) / (pv - pf) ** 2


def qpu_time(plan, info):
    t = OVERHEAD * len(plan)
    for name, n in plan.items():
        t += n * (REP_DELAY + info[name]["duration"] * 1e-6 + 5e-6)
    return t


# =============================================================================== IBM
def configure():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = getpass.getpass("IBM Cloud API key (not echoed): ").strip()
    inst = input("Instance CRN (Enter = automatic): ").strip() or None
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=token, instance=inst,
                                      overwrite=True, set_as_default=True)
    print("Saved. Devices:", [b.name for b in QiskitRuntimeService().backends(simulator=False, operational=True)])


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = os.environ.get("QISKIT_IBM_TOKEN")
    if tok:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=tok,
                                    instance=os.environ.get("QISKIT_IBM_INSTANCE"))
    return QiskitRuntimeService()


def counts_y(pub_result):
    return {int(k, 2): v for k, v in pub_result.data.y.get_counts().items()}


def dry_run(info, plan, hw, seed=5):
    """(1) Check of the whole chain: the REAL dynamic circuits (mid-circuit measurements, feed-forward) are
    simulated with calibrated noise (noise_model.noisy_copy) on plan[name] shots (slow: ~0.1-1 s per shot)."""
    from qiskit_aer import AerSimulator
    import noise_model as NM
    sim = AerSimulator(method="statevector", max_parallel_threads=4)
    out = {}
    for name, n in plan.items():
        tq = info[name]["tq"]
        tqn = NM.noisy_copy(hw, tq, 1.0, False, set(), {})
        used = sorted({tqn.find_bit(q).index for ins in tqn.data for q in ins.qubits})
        small, _ = NM.compact(tqn, used)
        t0 = time.time()
        res = sim.run(small, shots=n, seed_simulator=seed).result()
        cnt = res.get_counts()
        creg_y = [c for c in tq.cregs if c.name == "y"][0]
        idx = [tq.find_bit(b).index for b in creg_y]        # classical bits (register flattened by compact)
        cy = {}
        for key, v in cnt.items():
            bits = key.replace(" ", "")[::-1]
            y = sum(int(bits[i]) << j for j, i in enumerate(idx))
            cy[y] = cy.get(y, 0) + v
        out[name] = cy
        print(f"  [{name}] {n} shots simulated in {time.time() - t0:.0f} s")
    return out


def projection(N, a, t, g, info, plan, hwname, ntraj, seed=11):
    """(2) Projection to the full plan: exact P(y) averaged over `ntraj` noisy trajectories (static version with
    semiclassical emulation, noise_model.Setup), then multinomial sampling of the planned number of shots."""
    import noise_model as NM
    st = NM.Setup(N, t, g, hwname, a=a)
    rng = np.random.default_rng(seed)
    out = {}
    for name, n in plan.items():
        t0 = time.time()
        r = st.run(replace(info[name]["opt"], semi=0), 1.0, ntraj)
        P = np.clip(r["Py"], 0, None)
        c = rng.multinomial(n, P / P.sum())
        out[name] = {y: int(v) for y, v in enumerate(c) if v}
        print(f"  [{name}] {ntraj} trajectories in {time.time() - t0:.0f} s; P(y) = {np.round(P, 3)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configure", action="store_true", help="store the IBM account (once)")
    ap.add_argument("--ibm", action="store_true", help="REAL device (otherwise local dry run)")
    ap.add_argument("--backend", default=None, help="ibm_kingston (default with --ibm), ... or FakeXxx (dry run)")
    ap.add_argument("--N", type=int, default=15, choices=[15, 21])
    ap.add_argument("--a", type=int, default=None, help="base (default: 7 for 15, 2 for 21)")
    ap.add_argument("--t", type=int, default=None, help="exponent bits (default: 3 for 15/a=7, 4 for 15/a=2, 5 for 21)")
    ap.add_argument("--shots", type=int, default=200_000, help="TOTAL shots (30%% true circuit, rest split among controls)")
    ap.add_argument("--shots-dry", type=int, default=200, help="dry run: shots per simulated dynamic circuit (slow)")
    ap.add_argument("--traj", type=int, default=60, help="dry run: trajectories per circuit for the projection")
    ap.add_argument("--budget", type=float, default=540.0, help="QPU time cap (s), <= 590")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--job", default=None, help="ID of a submitted job to read back and analyse")
    ap.add_argument("--outdir", default="results", help="directory for job metadata and analyses")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()
    if args.configure:
        configure()
        return
    outdir = Path(args.outdir)
    a = args.a or SETTINGS[args.N]["a"]
    t_bits = args.t or (SETTINGS[args.N]["t"] if a == SETTINGS[args.N]["a"] else (4 if args.N == 15 else 5))
    budget = min(args.budget, HARD_CAP)

    if args.job:
        meta = json.loads((outdir / f"meta_{args.job}.json").read_text(encoding="utf-8"))
        job = service().job(args.job)
        print("Status:", job.status())
        result = job.result()
        cb = {name: counts_y(r) for name, r in zip(meta["order"], result)}
        res = analyse(meta["N"], meta.get("a", a), meta["t"], cb)
        (outdir / f"runtime_analysis_{args.job}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return

    if args.ibm:
        svc = service()
        be = svc.backend(args.backend or "ibm_kingston")
    else:
        be = args.backend or "FakeKingston"
    hw = HW(be)
    print(f"Backend: {hw.name} ({'REAL DEVICE' if args.ibm else 'dry run, local simulation'})")
    t, g, path, info = prepare(args.N, hw, a, t_bits)
    print(f"N={args.N}, a={a}, t={t}, physical line: {path}")
    for name, d in info.items():
        print(f"  {name:19s} CZ={d['cz']:4d}  duration~{d['duration']:.0f} us  ESP(calib.)={d['esp']:.3f}  "
              f"ideal success={d['p_ideal']:.4f} (random {d['p_rand']:.4f})")
    nf = len(info) - 1
    plan = {name: int(args.shots * (TRUE_SHARE if name == "true" else (1 - TRUE_SHARE) / nf)) for name in info}
    tq = qpu_time(plan, info)
    print(f"Plan: {plan}  -> estimated QPU time {tq:.0f} s (budget {budget:.0f} s)")
    if tq > budget:
        f = budget / tq * 0.98
        plan = {k: int(v * f) for k, v in plan.items()}
        tq = qpu_time(plan, info)
        print(f"  reduced to the budget: {plan} -> {tq:.0f} s")
    if args.plan:
        return
    if not args.ibm:
        n = args.shots_dry
        print()
        print(f"DRY RUN (1/2): chain check, {n} shots per noisy DYNAMIC circuit (few shots: slow).")
        cb = dry_run(info, {k: n for k in plan}, hw)
        analyse(args.N, a, t, cb)
        print()
        print(f"DRY RUN (2/2): projection to the full plan ({sum(plan.values())} shots), {args.traj} trajectories per circuit.")
        cb = projection(args.N, a, t, g, info, plan, hw.name, args.traj)
        analyse(args.N, a, t, cb)
        return
    if not args.yes:
        if input(f"Submit {len(plan)} circuits ({sum(plan.values())} shots, ~{tq:.0f} s QPU) to {hw.name}? [y/N] "
                 ).strip().lower() != "y":
            print("Cancelled.")
            return
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    sampler = Sampler(mode=be)
    o = sampler.options
    o.max_execution_time = int(min(budget, max(60, 1.3 * tq)))
    o.dynamical_decoupling.enable = False      # dynamic circuits: DD not guaranteed
    order = list(plan)
    pubs = [(info[k]["tq"], None, plan[k]) for k in order]
    job = sampler.run(pubs)
    jid = job.job_id()
    outdir.mkdir(parents=True, exist_ok=True)
    meta = dict(N=args.N, a=a, t=t, order=order, plan=plan, backend=hw.name, path=list(path),
                date=datetime.datetime.now().isoformat(timespec="seconds"), qpu_estimate_s=tq)
    (outdir / f"meta_{jid}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Job submitted: {jid} (read back: --ibm --job {jid})")
    result = job.result()
    cb = {name: counts_y(r) for name, r in zip(order, result)}
    res = analyse(args.N, a, t, cb)
    (outdir / f"runtime_analysis_{jid}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
