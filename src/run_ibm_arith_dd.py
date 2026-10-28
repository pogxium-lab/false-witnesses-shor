"""Rerun of the modular-arithmetic test (N = 21, a = 2, t = 3) with dynamical decoupling, in one job.

Hypothesis under test. In the first run (run_ibm_arith.py) the true circuit gave S = -0.0136 +- 0.0024 instead of
a positive value. An exact simulation of the submitted dynamic circuits reproduces this sign with a coherent phase
of ~180 degrees on the coherence e1 + e2 of the exponent register, most simply a static frequency offset of
10-20 kHz of the physical qubit holding e1, which idles ~19-22 us (until the feed-forward of the e2 measurement)
without dynamical decoupling. The witnesses are insensitive to that phase by construction. A static offset is
refocused by an echo: inserting X-X pairs in the idle windows must restore S > 0 for the true circuit.

Circuits (one job, same physical line, same logical circuits as the first run):
    true_DD, mult_by_1_DD, wrong_constant_DD, classical_exponent_DD
        the circuits submitted in the first run (pre-registered QPY of --reference) with an X-X echo in every
        idle window longer than --min-window (default 1 us): IBM dynamic-circuit scheduler (ALAP, blocks
        delimited by control flow), staggered spacings on neighbouring qubits (X at 1/4 and 3/4 of the window,
        or at 1/2 and at the end), explicit delays. Only single-qubit X gates and delays are added.
    true_noDD
        the true circuit of the first run, unchanged (reproducibility of the negative S within the same job).

Primary rule (fixed before the run): SIGNAL iff S(true_DD) > 0 AND S(true_DD) - S(w) >= 3 standard errors for
EACH DD witness w. Secondary (phase hypothesis): S(true_DD) - S(true_noDD) >= 3 standard errors. Also reported:
S(true_noDD) against the first run, fitted visibility and phase of the e1 + e2 coherence (true circuits) and of
the e2 coherence (mult_by_1_DD), S / P(y0 = 0). Standard errors as in run_ibm_arith.py (multinomial).

  Plan only:          python src/run_ibm_arith_dd.py --reference REF_PREREG.json --calibration CAL.json --plan
  Checks + pre-registration (exact simulation, coherent-error model, noisy prediction):
                      python src/run_ibm_arith_dd.py --reference REF_PREREG.json --calibration CAL.json
                                                     --prereg PREREG_DD.json
  Real device (submits the pre-registered circuits and shot plan):
                      python src/run_ibm_arith_dd.py --ibm --prereg PREREG_DD.json --outdir DIR [--account NAME]
                                                     [--yes] [--plan]
  Read back a job:    python src/run_ibm_arith_dd.py --ibm --job <ID> --outdir DIR [--account NAME]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
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
from qiskit.circuit.library import XGate  # noqa: E402
from qiskit.transpiler import PassManager  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import dynamic_sim as DS  # noqa: E402
import generator_core as GC  # noqa: E402
import run_ibm as RI  # noqa: E402
import run_ibm_arith as RA  # noqa: E402
from hardware import HW  # noqa: E402

HARD_CAP = 80.0
Z_REQ = 3.0
EXACT_TOL = 1e-6         # exact model vs transpiled circuit (single-qubit synthesis tolerance ~1e-9)
# circuit -> (logical circuit of the first run, dynamical decoupling)
CIRCUITS = {"true_DD": ("true", True), "mult_by_1_DD": ("mult_by_1", True),
            "wrong_constant_DD": ("wrong_constant", True), "classical_exponent_DD": ("classical_exponent", True),
            "true_noDD": ("true", False)}
SHOTS = {"true_DD": 70000, "mult_by_1_DD": 30000, "wrong_constant_DD": 30000, "classical_exponent_DD": 30000,
         "true_noDD": 40000}
WITNESSES_DD = ("mult_by_1_DD", "wrong_constant_DD", "classical_exponent_DD")
E1_QUBIT = 44            # physical qubit holding e1 during its long idle window (first run)
DETUNINGS = (15e3, 20e3)


# =============================================================================== dynamical decoupling
def _dd_pass_class():
    from qiskit_ibm_runtime.transpiler.passes.scheduling import PadDynamicalDecoupling

    class PadDynamicalDecouplingDt(PadDynamicalDecoupling):
        """PadDynamicalDecoupling with the DD gate lengths expressed in dt.

        The pass reads the gate lengths from the Target (seconds) but compares them with idle intervals
        counted in dt; without this conversion the minimum-window rule is never applied and every echo
        lengthens its window by the duration of its gates."""

        def _pre_runhook(self, dag):
            super()._pre_runhook(dag)
            dt = self._target.dt
            for q, seqs in self._dd_sequence_lengths.items():
                self._dd_sequence_lengths[q] = [[int(round(x / dt)) for x in seq] for seq in seqs]

        def _visit_control_flow_op(self, node):
            """After a control-flow operation, restore the state of the enclosing block (block index, idle
            times keyed by its qubits): the visit of the inner blocks leaves their own state behind, and when the
            inner block indices exceed that of the next outer node the next outer block is never terminated
            (KeyError on the idle time of an outer qubit). The control-flow operation itself terminates the
            enclosing block before its inner blocks are visited, so the restored state is that of a freshly
            terminated block. Circuits without this situation are padded exactly as before."""
            block_idx, _ = self._node_start_time[node]
            super()._visit_control_flow_op(node)
            self._current_block_idx = block_idx
            self._idle_after = dict.fromkeys(self._block_dag.qubits, 0)
            self._block_duration = 0

    return PadDynamicalDecouplingDt


def add_dd(tq, target, coupling_map, min_window=1e-6):
    """X-X echo in every idle window longer than min_window (dynamic-circuit scheduler of qiskit_ibm_runtime)."""
    from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis
    xlen = float(np.median([target["x"][(q,)].duration for q in range(target.num_qubits)
                            if (q,) in target["x"] and target["x"][(q,)].duration]))
    pad = _dd_pass_class()(target=target, dd_sequences=[XGate(), XGate()], coupling_map=coupling_map,
                           pulse_alignment=getattr(target, "pulse_alignment", 1) or 1,
                           sequence_min_length_ratios=[min_window / (2 * xlen)])
    return PassManager([ALAPScheduleAnalysis(target=target), pad]).run(tq)


def pad_delays(tq, target):
    """Same scheduler, delays only (timing model of the circuit without decoupling; used for simulation only)."""
    from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis, PadDelay
    return PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)]).run(tq)


def isa_errors(circuits, target):
    """Instruction-set check of qiskit_ibm_runtime (the one applied at submission); empty if valid."""
    from qiskit_ibm_runtime.utils.utils import is_isa_circuit
    return {k: m for k, m in ((k, is_isa_circuit(c, target)) for k, c in circuits.items()) if m}


def added_gates(tq_dd, tq):
    a, b = tq_dd.count_ops(), tq.count_ops()
    return {k: int(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b) if a.get(k, 0) != b.get(k, 0)}


# =============================================================================== circuits
def load_reference(ref_prereg):
    """Line, circuit order and transpiled circuits of the first run (its pre-registered QPY)."""
    ref = json.loads(Path(ref_prereg).read_text(encoding="utf-8"))
    path = RA.qpy_path(ref_prereg)
    with open(path, "rb") as fh:
        circs = dict(zip(list(ref["circuits"]), qpy.load(fh)))
    return ref, circs, path


def build(hw, ref_circs, min_window):
    ideals = {k: RA.ideal_P(o) for k, o in RA.witnesses().items()}
    cmap = hw.be.coupling_map
    info = {}
    for name, (base, dd) in CIRCUITS.items():
        tq0 = ref_circs[base]
        tq = add_dd(tq0, hw.tg, cmap, min_window) if dd else tq0
        describe(info, name, tq, hw, ideals[base], base, dd, tq0)
    return info


def strip_delays(tq):
    out = tq.copy_empty_like()
    for ins in tq.data:
        if ins.operation.name != "delay":
            out.append(ins)
    return out


def describe(info, name, tq, hw, P, base, dd, tq0=None):
    """CZ counts, duration (ASAP with the explicit delays), ESP (hardware.esp on the circuit without its delays,
    which would otherwise count the wait of a qubit in |0> before its first gate as idle; DD: crude model of
    hardware.dd_t2), and the 1q-gate factor exp(log-ESP of the 1q gates) used for the predictions."""
    cz, czc = GC.count_2q(tq)
    lg = hw.esp(strip_delays(tq), dd=dd)
    info[name] = dict(tq=tq, base=base, dd=dd, cz=cz, cz_cond=czc, duration=DS.makespan(tq, hw) * 1e6,
                      esp=math.exp(lg["total"]), gates1q_log=lg["gates1q"], P=P, S=RA.s_stat(P)[0],
                      added=added_gates(tq, tq0) if tq0 is not None else {},
                      n_x=int(tq.count_ops().get("x", 0)))


def summary(line, info, plan, tq_est, budget, hwname):
    print(f"Backend/calibration: {hwname}")
    print(f"N={RA.N}, a={RA.A_BASE}, t={RA.T_BITS}, physical line ({len(line)} qubits): {list(line)}")
    for name, d in info.items():
        add = ", ".join(f"+{v} {k}" for k, v in sorted(d["added"].items())) or "none"
        print(f"  {name:22s} CZ={d['cz']:4d} (+{d['cz_cond']} conditional)  X={d['n_x']:4d}  added: {add:24s} "
              f"duration~{d['duration']:.1f} us  ESP={d['esp']:.4f}  ideal S={d['S']:+.4f}")
    print(f"Plan: {plan} -> {sum(plan.values())} shots, estimated QPU time {tq_est:.1f} s (budget {budget:.0f} s)")


# =============================================================================== checks
def exactness(info, ref_circs):
    """Exact noiseless P(y) of every dynamic circuit (deferred measurement): deviation from the exact model
    (limited by the ~1e-9 tolerance of the single-qubit synthesis of the transpiler) and from the same circuit
    without decoupling (the echoes must be exact identities)."""
    out, base_P = {}, {}
    for name, d in info.items():
        p = DS.register_probabilities(d["tq"])
        if d["base"] not in base_P:
            base_P[d["base"]] = DS.register_probabilities(ref_circs[d["base"]])
        out[name] = dict(max_abs_dev_model=float(np.abs(p - d["P"]).max()),
                         max_abs_dev_without_dd=float(np.abs(p - base_P[d["base"]]).max()),
                         S=float(RA.s_stat(p)[0]), P=[round(float(x), 6) for x in p])
        print(f"  {name:22s} max|P - P_model| = {out[name]['max_abs_dev_model']:.1e}  "
              f"max|P - P_without_DD| = {out[name]['max_abs_dev_without_dd']:.1e}  S = {out[name]['S']:+.6f}  "
              f"P = {np.round(p, 4)}")
    return out


def coherent_model(info, hw, calibration, line):
    """Exact P(y) with static coherent errors on the timeline: calibrated ZZ of the line and/or a frequency
    offset of the e1 qubit. Without decoupling, two timing models: ASAP, and the IBM dynamic-circuit scheduler
    (ALAP with blocks, explicit delays); with decoupling the timing is fixed by the inserted delays."""
    zz = DS.calibrated_zz(calibration, line)
    scen = {"zz": dict(zz=zz)}
    for d in DETUNINGS:
        scen[f"det{E1_QUBIT}_{d / 1e3:+.0f}kHz"] = dict(detuning={E1_QUBIT: d})
        scen[f"det{E1_QUBIT}_{d / 1e3:+.0f}kHz+zz"] = dict(detuning={E1_QUBIT: d}, zz=zz)
    circs = {}
    for name, d in info.items():
        if d["dd"]:
            circs[name] = d["tq"]
        else:
            circs[name + "|asap"] = d["tq"]
            circs[name + "|ibm_schedule"] = pad_delays(d["tq"], hw.tg)
    out = {}
    for sname, kw in scen.items():
        out[sname] = {}
        parts = []
        for cname, tq in circs.items():
            p = DS.register_probabilities(tq, hw=hw, **kw)
            out[sname][cname] = dict(S=round(float(RA.s_stat(p)[0]), 5), P_y0_0=round(float(p[0::2].sum()), 5))
            parts.append(f"{cname} {out[sname][cname]['S']:+.4f}")
        print(f"  {sname:18s} S: " + " | ".join(parts), flush=True)
    return out


def noisy_projection(hw, line, info, plan, scales, ntraj, ntraj_true, seed=11):
    """Standard noise model (noise_model.py): calibrated depolarising gate errors, readout, T1/T2 in idle
    periods (DD: half of the pure dephasing removed), feed-forward latency. No coherent errors."""
    import noise_model as NM
    st = NM.Setup(RA.N, RA.T_BITS, RA.GENOME, hw.name, a=RA.A_BASE)
    st.hw, st.path = hw, tuple(line)
    opts = RA.witnesses()
    out = {}
    for sc in scales:
        S, Sse, Py = {}, {}, {}
        for name, d in info.items():
            t0 = time.time()
            n = ntraj_true if d["base"] == "true" else ntraj
            r = st.run(replace(opts[d["base"]], semi=0), sc, n, dd=d["dd"], seed=seed)
            P = np.clip(r["Py"], 0, None)
            Py[name] = P / P.sum()
            S[name] = RA.s_stat(Py[name])[0]
            Sse[name] = float(np.std([RA.s_stat(p)[0] for p in r["per"]], ddof=1) / math.sqrt(len(r["per"])))
            print(f"  s={sc:<4} [{name:22s}] S = {S[name]:+.4f} +- {Sse[name]:.4f} (trajectories)  "
                  f"P(y) = {np.round(Py[name], 4)}  [{time.time() - t0:.0f} s]", flush=True)
        out[sc] = dict(S=S, S_traj_se=Sse, P=Py)
    return out


def predictions(noisy, coh, plan, info):
    """Expected significances: standard-model S multiplied, for the true circuits, by the factor
    S_coherent / 0.25 of each coherent scenario (witnesses: standard-model S), and for the DD circuits by the
    error factor of the added X gates (not in the static circuits of the standard model)."""
    ref1q = info["true_noDD"]["gates1q_log"]
    out = {}
    for sc, nz in noisy.items():
        for sname, cs in [("no coherent error", None)] + list(coh.items()):
            for sched in ("asap", "ibm_schedule"):
                S = {}
                for name in plan:
                    s = nz["S"][name] * math.exp(info[name]["gates1q_log"] - ref1q)
                    if name.startswith("true") and cs is not None:
                        key = name if name in cs else f"{name}|{sched}"
                        s *= cs[key]["S"] / 0.25
                    S[name] = s
                var = {n: (1 / plan[n]) * max(RA.s_stat(nz["P"][n])[1], 1e-12) for n in plan}
                z = {w: (S["true_DD"] - S[w]) / math.sqrt(var["true_DD"] + var[w]) for w in WITNESSES_DD}
                z2 = (S["true_DD"] - S["true_noDD"]) / math.sqrt(var["true_DD"] + var["true_noDD"])
                out[f"s={sc} | {sname} | noDD timing {sched}"] = dict(
                    S={k: round(v, 5) for k, v in S.items()}, z_primary={k: round(v, 2) for k, v in z.items()},
                    z_primary_min=round(min(z.values()), 2), z_secondary=round(z2, 2))
                if cs is None:
                    break
    for k, v in out.items():
        print(f"  {k:62s} S(true_DD)={v['S']['true_DD']:+.4f} S(true_noDD)={v['S']['true_noDD']:+.4f}  "
              f"z primary min {v['z_primary_min']:+.1f}  z secondary {v['z_secondary']:+.1f}")
    return out


# =============================================================================== phase fit
_RHO = {}


def exponent_rho(base):
    """Reduced density matrix of the exponent register (index e0 + 2 e1 + 4 e2) of the noiseless static
    circuit, before the semiclassical Fourier measurement."""
    if base not in _RHO:
        import shor_line_generator as LG
        from qiskit import transpile
        qc, ln = LG.build(RA.N, RA.A_BASE, RA.T_BITS, RA.GENOME, replace(RA.witnesses()[base], semi=0),
                          measure=False, return_line=True)
        sim = DS.simulator()              # also registers the save instructions of qiskit_aer
        qc.save_density_matrix([ln.pos[f"e{k}"] for k in range(RA.T_BITS)], label="r")
        _RHO[base] = np.asarray(sim.run(transpile(qc, sim, optimization_level=0), shots=1).result().data(0)["r"])
    return _RHO[base]


def model_P(r, phases):
    """P(y) of the semiclassical Fourier measurement with Z phases phases[k] on e_k (y0 measured on e2)."""
    e = np.arange(8)
    D = np.exp(1j * sum(phases[k] * ((e >> k) & 1) for k in range(3)))
    rr = r * np.outer(D, D.conj())
    V = np.exp(2j * np.pi * np.outer(np.arange(8), e) / 8) / math.sqrt(8)
    return np.real(np.einsum("ye,ef,yf->y", V.conj(), rr, V))


def phase_fit(counts, base, bit):
    """Maximum likelihood of v * P(phase phi on e_bit) + (1 - v) / 8 in the plane (x, y) = (v cos phi, v sin phi);
    returns v, phi (deg), x = v cos phi and their standard errors (inverse Hessian). When P(y) depends on the
    phase only through v cos phi (mult_by_1: e2 coherence seen only through P(y0 = 0)), only x is fitted."""
    c = RA.as_array(counts)
    r = exponent_rho(base)
    probe = [model_P(r, [a if k == bit else 0.0 for k in range(3)]) for a in (0.5, -0.5)]
    two_d = np.abs(probe[0] - probe[1]).max() > 1e-9

    def nll(x):
        v, ph = math.hypot(*x), math.atan2(x[1], x[0])
        P = v * model_P(r, [ph if k == bit else 0.0 for k in range(3)]) + (1 - v) / 8
        return -float((c * np.log(np.clip(P, 1e-300, None))).sum())
    f = nll if two_d else (lambda x: nll([x[0], 0.0]))
    starts = ([[0.1 * math.cos(a), 0.1 * math.sin(a)] for a in np.linspace(-3, 3, 7)] if two_d
              else [[-0.1], [0.1]])
    best = min((minimize(f, s0, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))
                for s0 in starts), key=lambda o: o.fun)
    x0 = np.asarray(best.x, dtype=float)
    k, h = len(x0), 1e-4
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)) / (4 * h * h)
    cov = np.linalg.inv(H)
    out = dict(x=float(x0[0]), x_se=float(math.sqrt(cov[0, 0])), two_lnLR=2 * (f(np.zeros(k)) - float(best.fun)),
               identifiable="v and phase" if two_d else "v cos(phase) only")
    if two_d:
        x, y = x0
        v = math.hypot(x, y)
        g_v, g_ph = np.array([x, y]) / v, np.array([-y, x]) / v ** 2
        out.update(v=v, v_se=float(math.sqrt(g_v @ cov @ g_v)), phase_deg=math.degrees(math.atan2(y, x)),
                   phase_se_deg=math.degrees(math.sqrt(g_ph @ cov @ g_ph)))
    return out


# =============================================================================== analysis
def analyse(counts_by, pre):
    ideals = {k: RA.ideal_P(o) for k, o in RA.witnesses().items()}
    print()
    print(f"=== Analysis: S = P(y in {set(RA.CELLS_A)}) - P(y in {set(RA.CELLS_B)}), N={RA.N}, a={RA.A_BASE}, "
          f"t={RA.T_BITS}, with / without dynamical decoupling ===")
    st = {k: RA.s_from_counts(c) for k, c in counts_by.items()}
    res = {"S": {}, "z_primary": {}}
    for name, (s, e, n) in st.items():
        c = RA.as_array(counts_by[name])
        q = (c[0] + c[2] + c[4] + c[6]) / n
        res["S"][name] = dict(S=s, se=e, n=n, S_cond=s / q, P_y0_0=q)
        print(f"  {name:22s}: S = {s:+.4f} +- {e:.4f}  (ideal {RA.s_stat(ideals[CIRCUITS[name][0]])[0]:+.4f}, "
              f"{n} shots; S/P(y0=0) = {s / q:+.4f}; P(y0=0) = {q:.4f})")
    sv, ev, _ = st["true_DD"]
    for w in WITNESSES_DD:
        s, e, _ = st[w]
        res["z_primary"][w] = (sv - s) / math.hypot(ev, e)
        print(f"    true_DD - {w:22s} = {sv - s:+.4f} ({res['z_primary'][w]:+.1f} sigma)")
    zmin = min(res["z_primary"].values())
    primary = sv > 0 and zmin >= Z_REQ
    s2, e2, _ = st["true_noDD"]
    z2 = (sv - s2) / math.hypot(ev, e2)
    print(f"    true_DD - true_noDD{'':5s} = {sv - s2:+.4f} ({z2:+.1f} sigma)")
    prev = (pre or {}).get("previous_run", {}).get("S_true")
    if prev:
        zr = (s2 - prev["S"]) / math.hypot(e2, prev["se"])
        print(f"    true_noDD - first run    = {s2 - prev['S']:+.4f} ({zr:+.1f} sigma; first run "
              f"{prev['S']:+.4f} +- {prev['se']:.4f})")
        res["z_noDD_vs_first_run"] = zr
    print("  Fitted coherence (v P_phase + (1 - v) U): phase on e1 + e2 for the true circuits, on e2 for "
          "mult_by_1_DD")
    res["phase_fit"] = {}
    for name, base, bit in [("true_DD", "true", 1), ("true_noDD", "true", 1), ("mult_by_1_DD", "mult_by_1", 2)]:
        f = phase_fit(counts_by[name], base, bit)
        res["phase_fit"][name] = f
        if "phase_deg" in f:
            print(f"    {name:22s} v = {f['v']:.4f} +- {f['v_se']:.4f}  phase = {f['phase_deg']:+.1f} +- "
                  f"{f['phase_se_deg']:.1f} deg  (v cos phase = {f['x']:+.4f} +- {f['x_se']:.4f})  "
                  f"2lnLR = {f['two_lnLR']:.1f}")
        else:
            print(f"    {name:22s} v cos(phase) = {f['x']:+.4f} +- {f['x_se']:.4f} (phase not identifiable)  "
                  f"2lnLR = {f['two_lnLR']:.1f}")
    v1 = (f"SIGNAL: S(true_DD) > 0 and exceeds every DD witness by >= {Z_REQ:.0f} sigma" if primary else
          f"NO SIGNIFICANT SIGNAL (S(true_DD) = {sv:+.4f}, smallest difference {zmin:+.1f} sigma)")
    v2 = (f"SUPPORTED: S(true_DD) - S(true_noDD) = {z2:+.1f} sigma >= {Z_REQ:.0f}" if z2 >= Z_REQ else
          f"NOT SUPPORTED: S(true_DD) - S(true_noDD) = {z2:+.1f} sigma < {Z_REQ:.0f}")
    print("  PRIMARY VERDICT:", v1)
    print("  SECONDARY (phase hypothesis):", v2)
    res.update(z_primary_min=zmin, z_secondary=z2, verdict_primary=v1, verdict_secondary=v2)
    return res


# =============================================================================== files
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_circuits(prereg, info):
    path = RA.qpy_path(prereg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        qpy.dump([info[k]["tq"] for k in info], fh)
    print(f"Circuits saved: {path} (order {list(info)})")
    return path


def load_circuits(prereg, order):
    with open(RA.qpy_path(prereg), "rb") as fh:
        return dict(zip(order, qpy.load(fh)))


def previous_result(ref_prereg):
    """First run (analysis and raw counts next to the reference pre-registration): S and fitted e1 + e2
    coherence of the true circuit."""
    for f in sorted(Path(ref_prereg).parent.glob("analysis_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "decohered_e2" not in d.get("S", {}):
            continue
        jid = f.stem.replace("analysis_", "")
        out = dict(job=jid, S_true=dict(S=d["S"]["true"]["S"], se=d["S"]["true"]["se"], n=d["S"]["true"]["n"]))
        raw = f.parent / f"job_{jid}.json"
        meta = f.parent / f"meta_{jid}.json"
        if raw.exists() and meta.exists():
            order = json.loads(meta.read_text(encoding="utf-8"))["order"]
            circ = json.loads(raw.read_text(encoding="utf-8"))["circuits"][order.index("true")]
            counts = {int(k, 2): v for k, v in circ["registers"]["y"].items()}
            out["phase_fit_true"] = {k: (round(v, 5) if isinstance(v, float) else v)
                                     for k, v in phase_fit(counts, "true", 1).items()}
        return out
    return {}


def empirical_prediction(prev, plan, info):
    """Prediction from the first run itself: the fitted visibility v of the true circuit is kept and its phase
    is refocused by the echoes (S(true_DD) = 0.25 v times the X-gate factor), S(true_noDD) = first-run S,
    witnesses 0; single-shot variances of a nearly uniform distribution (pA + pB = 1/2)."""
    if "phase_fit_true" not in prev:
        return {}
    v = prev["phase_fit_true"]["v"]
    S = {n: 0.0 for n in plan}
    S["true_DD"] = 0.25 * v * math.exp(info["true_DD"]["gates1q_log"] - info["true_noDD"]["gates1q_log"])
    S["true_noDD"] = prev["S_true"]["S"]
    var = {n: (0.5 - S[n] ** 2) / plan[n] for n in plan}
    z = {w: (S["true_DD"] - S[w]) / math.sqrt(var["true_DD"] + var[w]) for w in WITNESSES_DD}
    z2 = (S["true_DD"] - S["true_noDD"]) / math.sqrt(var["true_DD"] + var["true_noDD"])
    out = dict(S={k: round(x, 5) for k, x in S.items()}, z_primary={k: round(x, 2) for k, x in z.items()},
               z_primary_min=round(min(z.values()), 2), z_secondary=round(z2, 2), visibility_first_run=v)
    print(f"  empirical (first-run visibility {v:.4f}, phase refocused): S(true_DD)={S['true_DD']:+.4f} "
          f"S(true_noDD)={S['true_noDD']:+.4f}  z primary min {out['z_primary_min']:+.1f}  "
          f"z secondary {out['z_secondary']:+.1f}")
    return out


def write_prereg(path, args, line, info, plan, tq_est, budget, exact, coh, noisy, pred, qpy_file, ref_qpy,
                 prev):
    doc = {
        "experiment": "Rerun of the N=21, a=2, t=3 modular-arithmetic test with dynamical decoupling (one job)",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pre-registered before any submission",
        "hypothesis": "The negative S of the true circuit in the first run comes from a coherent phase of ~180 deg "
                      "on the e1+e2 coherence of the exponent register, accumulated in idle windows without "
                      "dynamical decoupling, most simply a static frequency offset of 10-20 kHz of physical "
                      f"qubit {E1_QUBIT} (holds e1, idles ~19-22 us until the feed-forward of the e2 measurement). "
                      "An X-X echo in the idle windows refocuses a static offset and must restore S > 0.",
        "previous_run": prev,
        "reference_preregistration": str(args.reference),
        "reference_circuits": str(ref_qpy), "reference_circuits_sha256": sha256(ref_qpy),
        "calibration_snapshot": str(args.calibration),
        "statistic": "S = P(y in {0,4}) - P(y in {2,6}) (exponent register y, 3 bits)",
        "standard_error": "Var(S_hat) = [pA + pB - (pA - pB)^2] / n per circuit (multinomial); difference of two "
                          "independent circuits: sqrt(Var1 + Var2)",
        "verdict_rule_primary": "SIGNAL iff S(true_DD) > 0 AND S(true_DD) - S(w) >= 3 standard errors for EACH "
                                "w in {mult_by_1_DD, wrong_constant_DD, classical_exponent_DD}; otherwise NO SIGNAL",
        "verdict_rule_secondary": "phase hypothesis SUPPORTED iff S(true_DD) - S(true_noDD) >= 3 standard errors",
        "reported": ["S(true_noDD) against the first run (reproducibility)",
                     "fitted visibility and phase of the e1+e2 coherence (true_DD, true_noDD) and of the e2 "
                     "coherence (mult_by_1_DD): v P(phase) + (1 - v) U, maximum likelihood",
                     "S / P(y0=0) and P(y0=0) of every circuit"],
        "dynamical_decoupling": dict(
            sequence="X-X", min_window_us=args.min_window,
            scheduler="qiskit_ibm_runtime.transpiler.passes.scheduling: ALAPScheduleAnalysis + "
                      "PadDynamicalDecoupling (gate lengths converted to dt), staggered spacings on neighbouring "
                      "qubits (coupling-map colouring): [1/4, 1/2, 1/4] or [1/2, 1/2, 0]",
            durations="calibration snapshot", sampler_dynamical_decoupling="disabled (echoes are explicit gates)"),
        "circuits_order": list(info),
        "backend": "ibm_kingston",
        "physical_line": list(line),
        "shots": plan,
        "qpu_estimate_s": round(tq_est, 1),
        "budget_s": budget,
        "circuits_file": str(qpy_file), "circuits_sha256": sha256(qpy_file),
        "circuits": {k: dict(logical=d["base"], dd=d["dd"], cz=d["cz"], cz_conditional=d["cz_cond"],
                             x_gates=d["n_x"], added_operations=d["added"], duration_us=round(d["duration"], 2),
                             esp=round(d["esp"], 5), ideal_P=[round(float(x), 5) for x in d["P"]],
                             ideal_S=round(d["S"], 5)) for k, d in info.items()},
        "true_noDD_identical_to_first_run_true": True,
        "exactness_noiseless_dynamic_simulation": exact,
        "coherent_error_model": dict(
            description="exact simulation of the dynamic circuits with static errors on an ASAP timeline "
                        "(calibrated durations, explicit delays, feed-forward latency 0.7 us): calibrated ZZ of "
                        f"the line and/or a frequency offset of qubit {E1_QUBIT}; noDD with two timing models "
                        "(ASAP; IBM dynamic-circuit scheduler)", S=coh),
        "noisy_predictions_standard_model": {str(sc): dict(S={k: round(v, 5) for k, v in p["S"].items()},
                                                           S_trajectory_se={k: round(v, 5)
                                                                            for k, v in p["S_traj_se"].items()},
                                                           P={k: [round(float(x), 5) for x in v]
                                                              for k, v in p["P"].items()})
                                             for sc, p in noisy.items()},
        "combined_predictions": pred,
        "trajectories": {"true": args.traj_true, "witness": args.traj},
        "noise_model": "noise_model.py: calibrated depolarising CZ/1q errors, symmetric readout errors, T1/T2 "
                       "relaxation in idle periods (DD: half of the pure dephasing removed), feed-forward latency; "
                       "no coherent errors. Combined prediction: S(true_*) of the standard model times "
                       "S_coherent / 0.25 of each coherent scenario; DD circuits also times the error factor of "
                       "the added X gates.",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"\nPre-registration written: {path}")


# =============================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ibm", action="store_true", help="REAL device (otherwise local checks)")
    ap.add_argument("--backend", default="ibm_kingston", help="real device name (with --ibm)")
    ap.add_argument("--account", default=None, help="name of a saved Qiskit Runtime account")
    ap.add_argument("--reference", default=None, help="pre-registration of the first run (its QPY is next to it)")
    ap.add_argument("--calibration", default=None, help="calibration snapshot (.json) for the local checks")
    ap.add_argument("--budget", type=float, default=HARD_CAP, help=f"QPU time cap (s), <= {HARD_CAP:.0f}")
    ap.add_argument("--min-window", type=float, default=1.0, help="shortest idle window (us) receiving an echo")
    ap.add_argument("--plan", action="store_true", help="print circuits and shot plan, then exit")
    ap.add_argument("--shots", default=None,
                    help="local: shot plan other than the default, e.g. true_DD=70000,mult_by_1_DD=30000,...")
    ap.add_argument("--prereg", default=None,
                    help="local: pre-registration file to WRITE; --ibm: pre-registration file to READ")
    ap.add_argument("--scales", default="1", help="local: error scale factors of the standard noise model")
    ap.add_argument("--traj", type=int, default=100, help="local: noisy trajectories per witness circuit")
    ap.add_argument("--traj-true", type=int, default=300, help="local: noisy trajectories per true circuit")
    ap.add_argument("--no-noise", action="store_true", help="local: skip the standard noisy prediction")
    ap.add_argument("--job", default=None, help="ID of a submitted job to read back and analyse")
    ap.add_argument("--outdir", default="results", help="directory for metadata, analysis and archives")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()
    budget = min(args.budget, HARD_CAP)
    outdir = Path(args.outdir)

    if args.job:
        from archive_job import archive, service
        svc = service(args.account)
        meta = json.loads((outdir / f"meta_{args.job}.json").read_text(encoding="utf-8"))
        pre = json.loads(Path(meta["prereg"]).read_text(encoding="utf-8")) if meta.get("prereg") else None
        result = svc.job(args.job).result()
        cb = {name: RI.counts_y(r) for name, r in zip(meta["order"], result)}
        res = analyse(cb, pre)
        (outdir / f"analysis_{args.job}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        archive(args.job, outdir, svc)
        return

    if not args.ibm:
        if not (args.reference and args.calibration):
            sys.exit("local mode requires --reference (first-run pre-registration) and --calibration")
        ref, ref_circs, ref_qpy = load_reference(args.reference)
        line = tuple(ref["physical_line"])
        hw = HW(args.calibration)
        info = build(hw, ref_circs, args.min_window * 1e-6)
        plan = dict(SHOTS)
        if args.shots:
            plan.update({k: int(v) for k, v in (x.split("=") for x in args.shots.split(","))})
            if set(plan) != set(CIRCUITS):
                sys.exit(f"--shots: unknown circuit name (circuits: {list(CIRCUITS)})")
        tq_est = RI.qpu_time(plan, info)
        summary(line, info, plan, tq_est, budget, args.calibration)
        if tq_est > budget:
            sys.exit(f"estimated QPU time {tq_est:.0f} s exceeds the budget {budget:.0f} s")
        bad = isa_errors({k: d["tq"] for k, d in info.items()}, hw.tg)
        print("ISA check (calibration target):", bad or "all circuits valid")
        if bad:
            sys.exit("ISA check failed")
        if args.plan:
            return
        print("\nExactness (noiseless exact simulation of the dynamic circuits vs exact model):")
        exact = exactness(info, ref_circs)
        bad = [k for k, v in exact.items()
               if v["max_abs_dev_model"] > EXACT_TOL or v["max_abs_dev_without_dd"] > 1e-12]
        if bad:
            sys.exit(f"exactness check FAILED for {bad}")
        print(f"\nCoherent-error model (calibrated ZZ; qubit {E1_QUBIT} offset {[d / 1e3 for d in DETUNINGS]} kHz):")
        coh = coherent_model(info, hw, args.calibration, line)
        noisy = {}
        if not args.no_noise:
            scales = [float(x) for x in args.scales.split(",")]
            print(f"\nStandard noise model: {args.traj} trajectories per witness, {args.traj_true} per true "
                  f"circuit, scales {scales}")
            noisy = noisy_projection(hw, line, info, plan, scales, args.traj, args.traj_true)
            print("\nCombined predictions (expected significance with the plan):")
            pred = predictions(noisy, coh, plan, info)
        else:
            pred = {}
        prev = previous_result(args.reference)
        print("\nPrediction from the first run itself:")
        pred["empirical | first-run visibility, phase refocused"] = empirical_prediction(prev, plan, info)
        if args.prereg:
            qfile = save_circuits(args.prereg, info)
            write_prereg(args.prereg, args, line, info, plan, tq_est, budget, exact, coh, noisy, pred, qfile,
                         ref_qpy, prev)
        return

    # ---------------------------------------------------------------- real device
    if not args.prereg:
        sys.exit("--ibm requires --prereg (the pre-registration file fixes the circuits and the shot plan)")
    pre = json.loads(Path(args.prereg).read_text(encoding="utf-8"))
    order = list(pre["circuits_order"])
    qfile = RA.qpy_path(args.prereg)
    if sha256(qfile) != pre["circuits_sha256"]:
        sys.exit(f"{qfile} differs from the pre-registered circuits (sha256)")
    stored = load_circuits(args.prereg, order)
    with open(pre["reference_circuits"], "rb") as fh:
        ref_true = dict(zip(list(json.loads(Path(pre["reference_preregistration"]).read_text(
            encoding="utf-8"))["circuits"]), qpy.load(fh)))["true"]
    if stored["true_noDD"] != ref_true:
        sys.exit("true_noDD is not identical to the true circuit of the first run")
    from archive_job import archive, service
    from snapshot_calibration import snapshot
    svc = service(args.account)
    be = svc.backend(args.backend)
    hw = HW(be)
    line = tuple(pre["physical_line"])
    ideals = {k: RA.ideal_P(o) for k, o in RA.witnesses().items()}
    info = {}
    for k in order:
        base, dd = CIRCUITS[k]
        describe(info, k, stored[k], hw, ideals[base], base, dd)
        info[k]["added"] = pre["circuits"][k]["added_operations"]
        if info[k]["cz"] != pre["circuits"][k]["cz"]:
            sys.exit(f"{k}: stored circuit has {info[k]['cz']} CZ, pre-registered {pre['circuits'][k]['cz']}")
    plan = {k: int(v) for k, v in pre["shots"].items()}
    tq_est = RI.qpu_time(plan, info)
    print("ESP below: today's live calibration, pre-registered circuits")
    summary(line, info, plan, tq_est, budget, be.name)
    if tq_est > budget:
        sys.exit(f"estimated QPU time {tq_est:.0f} s exceeds the budget {budget:.0f} s")
    print("Exactness of the stored circuits (noiseless exact simulation):")
    for k in order:
        dev = float(np.abs(DS.register_probabilities(stored[k]) - info[k]["P"]).max())
        print(f"  {k:22s} max|P - P_ideal| = {dev:.1e}")
        if dev > EXACT_TOL:
            sys.exit(f"{k}: exactness check failed")
    bad = isa_errors(stored, be.target)
    print("ISA check (live target):", bad or "all circuits valid")
    if bad:
        sys.exit("ISA check failed")
    if args.plan:
        return
    if not args.yes and input(f"Submit {len(plan)} circuits ({sum(plan.values())} shots, ~{tq_est:.0f} s QPU) to "
                              f"{be.name}? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    ctx = snapshot(be, svc, outdir, "before_arith_dd")
    available = ctx["usage"].get("usage_remaining_seconds")
    print(f"  QPU time available on the account before submission: {available} s")
    if available is not None and float(available) < tq_est * 1.2:
        sys.exit("not enough QPU time left in the usage window")
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    sampler = Sampler(mode=be)
    sampler.options.max_execution_time = int(min(budget, max(60, 1.3 * tq_est)))
    sampler.options.dynamical_decoupling.enable = False      # echoes are explicit gates of the circuits
    job = sampler.run([(stored[k], None, plan[k]) for k in order])
    jid = job.job_id()
    meta = dict(experiment="arith_N21_a2_t3_dd", N=RA.N, a=RA.A_BASE, t=RA.T_BITS, order=order, plan=plan,
                backend=be.name, physical_line=list(line), cz={k: info[k]["cz"] for k in order},
                prereg=str(args.prereg), date=datetime.datetime.now().isoformat(timespec="seconds"),
                qpu_estimate_s=tq_est)
    (outdir / f"meta_{jid}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Job submitted: {jid} (read back: --ibm --job {jid} --outdir {outdir})")
    result = job.result()
    cb = {name: RI.counts_y(r) for name, r in zip(order, result)}
    res = analyse(cb, pre)
    (outdir / f"analysis_{jid}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    archive(jid, outdir, svc)
    snapshot(be, svc, outdir, "after_arith_dd")


if __name__ == "__main__":
    main()
