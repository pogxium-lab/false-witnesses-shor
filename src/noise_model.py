"""Success rate under noise (Aer, quantum trajectories) of the line circuit and of its false witnesses, on IBM
noise models (FakeKingston, FakeFez, FakeBoston, ...).

Method: static circuit (exponent qubits kept to the end), placed on the best line of the backend, then
  * relaxation (T1, T2 of the physical qubit, divided by s; mixture of reset to |0> and Z, see relax_mixture)
    inserted in EVERY idle period (ASAP schedule, backend durations, window measurement + feed-forward latency);
    optional dynamical decoupling (half of the pure dephasing removed);
  * depolarising channel after every CZ / sx / x (calibrated error of the pair / qubit, times s); readout error
    (times s) before mid-circuit measurements and on every bit of y (confusion matrix);
  * semiclassical emulation: an exponent bit e_k (k >= L) has no idle period after its last gate (it is
    measured); e_1 and e_0 wait 1 and 2 x (measurement + latency) at the end (chained feed-forward);
  * every trajectory gives an exact state; P(y) = diag(F^dagger rho_e F) is averaged -> success probability,
    standard error over trajectories.
Not modelled: leakage, ZZ / measurement crosstalk, coherent errors, drift (hence the scales s = 1.5, 2).

Usage: python src/noise_model.py N t m0 fill G0 backend s1[,s2,...] ntraj [--dd] [--controls] [--a A] [--out F]
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import replace

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import IfElseOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error, pauli_error, reset_error

import exact_model as EM
import generator_core as GC
import shor_line_generator as LG
from hardware import BASIS, HW, LAT, dd_t2


def relax_mixture(t1, t2, tau):
    """Relaxation as a MIXTURE of circuits (reset to |0> with p_r = 1 - exp(-tau/T1), exact population and bias
    towards |0>; then Z with p_z such that the coherence equals exp(-tau/T2)). If T2 > T1 the reset alone
    already dephases slightly too much (exp(-tau/T1) < exp(-tau/T2)): slightly PESSIMISTIC in that case."""
    pr = 1 - math.exp(-tau / t1)
    target = math.exp(-tau / t2)
    pz = max(0.0, (1 - target / (1 - pr)) / 2) if pr < 1 else 0.0
    err = reset_error(pr)
    if pz > 1e-9:
        err = err.compose(pauli_error([("Z", pz), ("I", 1 - pz)]))
    return err


def noisy_copy(hw, tq, scale, dd, semi_e, tail_wait, p_min=1e-5, parts=("relax", "gate", "ro")):
    """Insert relaxation (idle periods) + depolarising errors (gates) + readout errors (measurements)."""
    out = tq.copy_empty_like()
    free, ready = {}, 0.0

    def relax(q, tau):
        if tau <= 0 or "relax" not in parts:
            return
        t1, t2 = hw.t1t2(q)
        t1, t2 = t1 / scale, t2 / scale
        if dd:
            t2 = dd_t2(t1, t2)
        t2 = min(t2, 2 * t1)
        if 1 - math.exp(-tau / t2) > p_min:
            out.append(relax_mixture(t1, t2, tau), [out.qubits[q]])

    for ins in tq.data:
        op = ins.operation
        qs = [tq.find_bit(q).index for q in ins.qubits]
        if op.name == "barrier":
            continue
        cond = isinstance(op, IfElseOp)
        d = 0.0 if cond else hw.dur(op.name, qs)
        start = max([free.get(q, 0.0) for q in qs] + ([ready] if cond else [0.0]))
        for q in qs:
            if q in free:
                relax(q, start - free[q])
        if op.name == "measure":
            em = min(hw.emeas(qs[0]) * scale, 0.4)
            if em > 1e-5 and "ro" in parts:
                out.append(pauli_error([("X", em), ("I", 1 - em)]), [out.qubits[qs[0]]])
        out.append(ins)
        if op.name == "cz":
            e = min(hw.ecz(*qs) * scale * 4 / 3, 0.9)
            if e > 1e-5 and "gate" in parts:
                out.append(depolarizing_error(e, 2), [out.qubits[q] for q in qs])
        elif op.name in ("sx", "x"):
            e = min(hw.e1(op.name, qs[0]) * scale * 2, 0.9)
            if e > 1e-5 and "gate" in parts:
                out.append(depolarizing_error(e, 1), [out.qubits[qs[0]]])
        for q in qs:
            free[q] = start + d
        if op.name == "measure":
            ready = max(ready, start + d + LAT)
    end = max(free.values())
    for q, f in free.items():
        if q in semi_e:
            continue                      # measured right after its multiplication (semiclassical)
        relax(q, end - f + tail_wait.get(q, 0.0))
    return out


def compact(tqn, keep):
    """Copy of the circuit restricted to the qubits actually used (plus `keep`)."""
    used = sorted({tqn.find_bit(q).index for ins in tqn.data for q in ins.qubits} | set(keep))
    pos = {q: i for i, q in enumerate(used)}
    small = QuantumCircuit(len(used), tqn.num_clbits)
    for ins in tqn.data:
        qs = [pos[tqn.find_bit(q).index] for q in ins.qubits]
        cs = [tqn.find_bit(c).index for c in ins.clbits]
        op = ins.operation
        if isinstance(op, IfElseOp):
            cond = (small.clbits[tqn.find_bit(op.condition[0]).index], op.condition[1])
            blk = op.blocks[0]
            with small.if_test(cond):
                for bi in blk.data:
                    small.append(bi.operation, [small.qubits[qs[blk.find_bit(q).index]] for q in bi.qubits])
            continue
        small.append(op, [small.qubits[i] for i in qs], [small.clbits[c] for c in cs])
    return small, pos


def confusion(Py, eps):
    """Bits of y flipped independently (eps[j] for bit j)."""
    t = len(eps)
    P = Py.copy()
    for j in range(t):
        Q = P.reshape(-1, 2, 1 << j)
        P = ((1 - eps[j]) * Q + eps[j] * Q[:, ::-1, :]).reshape(-1)
    return P


class Setup:
    """Circuit placed once (same physical line for the true circuit and its controls)."""

    def __init__(self, N, t, g, hwname, a=2):
        self.N, self.t, self.g, self.a = N, t, g, a
        self.hw = HW(hwname)
        qc, ln = LG.build(N, a, t, g, LG.Opt(), measure=False, return_line=True)
        self.tq, self.path = self.hw.place(qc)
        self.L = min(GC.lookup_bits(N, a, t), 2)

    def transpiled(self, opt):
        qc, ln = LG.build(self.N, self.a, self.t, self.g, opt, measure=False, return_line=True)
        qd = transpile(qc, basis_gates=BASIS, optimization_level=0)
        tq = transpile(qd, self.hw.be, optimization_level=2, seed_transpiler=1, initial_layout=list(self.path))
        lay = tq.layout.final_index_layout()
        ephys = [lay[ln.pos[f"e{k}"]] for k in range(self.t)]
        return tq, ephys

    def run(self, opt, scale, ntraj, dd=False, seed=7, batch=40, threads=3):
        t, L = self.t, self.L
        tq, ephys = self.transpiled(opt)
        semi_e = set(ephys[L:])
        tw = self.hw.tmeas + LAT
        tail = {ephys[1]: tw, ephys[0]: 2 * tw} if L == 2 else {ephys[0]: tw}
        tqn = noisy_copy(self.hw, tq, scale, dd, semi_e, tail)
        small, pos = compact(tqn, ephys)
        epos = [pos[q] for q in ephys]
        small.save_statevector(pershot=True)
        sim = AerSimulator(method="statevector", max_parallel_threads=threads, max_parallel_shots=1)
        E = 1 << t
        nq = small.num_qubits
        rest = [q for q in range(nq) if q not in epos]
        Fm = np.exp(2j * np.pi * np.outer(np.arange(E), np.arange(E)) / E) / math.sqrt(E)
        Pid = EM.shor_P(self.N, self.a, t, self.g, order=opt.order, consts=opt.consts, table=opt.table,
                        decohere=opt.decohere)
        per = []
        done = 0
        while done < ntraj:
            nb = min(batch, ntraj - done)
            res = sim.run(small, shots=nb, seed_simulator=seed + done).result()
            for sv in res.data(0)["statevector"]:
                psi = np.asarray(sv).reshape([2] * nq)
                axes_e = [nq - 1 - q for q in reversed(epos)]
                axes_r = [nq - 1 - q for q in rest]
                Mx = np.transpose(psi, axes_e + axes_r).reshape(E, -1)
                rho = Mx @ Mx.conj().T
                per.append(np.real(np.einsum("ij,ik,kj->j", Fm.conj(), rho, Fm)))
            done += nb
        per = np.array(per)
        Pm = per.mean(axis=0)
        # sign convention of y: the one of the noiseless model of the same circuit
        refl = lambda P: np.roll(P[::-1], 1)  # noqa: E731
        if np.abs(Pm - Pid).sum() > np.abs(refl(Pm) - Pid).sum():
            per = np.array([refl(p) for p in per])
        # readout of the bits of y: y_j is measured on e_{t-1-j}
        eps = [min(self.hw.emeas(ephys[t - 1 - j]) * scale, 0.4) for j in range(t)]
        per = np.array([confusion(p, eps) for p in per])
        lg = self.hw.esp(tq, scale=scale, dd=dd)
        out = dict(Py=per.mean(axis=0), per=per, esp=math.exp(lg["total"]), dur=lg["duration_us"],
                   cz=GC.count_2q(tq)[0], lg=lg)
        for kind in ("C1", "C2"):
            sc = EM.crit(self.N, self.a, t, kind)
            Sv = per[:, sc].sum(axis=1)
            out[kind] = dict(P=float(Sv.mean()), se=float(Sv.std(ddof=1) / math.sqrt(len(Sv))),
                             Pid=float(Pid[sc].sum()), Prand=float(sc.mean()))
        return out


def controls(N, a, t, L):
    """Standard false witnesses: each multiplication bit decohered, all bits decohered, scrambled constants."""
    ks = list(range(L, t))
    out = [("true", LG.Opt())]
    for k in ks:
        out.append((f"control: e{k} decohered", LG.Opt(decohere=(k,))))
    out.append(("control: classical exponent", LG.Opt(decohere=tuple(range(t)))))
    cs = EM.fake_consts(N, a, t, L)
    out.append(("control: scrambled constants " + ",".join(f"{c}" for _, c in cs), LG.Opt(consts=cs)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("N", "t", "m0", "fill", "G0"):
        ap.add_argument(name, type=int)
    ap.add_argument("backend", help="FakeKingston, FakeFez, FakeBoston, ...")
    ap.add_argument("scales", help="comma-separated error scales, e.g. 1,2")
    ap.add_argument("ntraj", type=int)
    ap.add_argument("--dd", action="store_true", help="dynamical decoupling model")
    ap.add_argument("--controls", action="store_true", help="also simulate the false witnesses")
    ap.add_argument("--a", type=int, default=2)
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SHOR_SEED", "7")))
    ap.add_argument("--out", default=None, help="optional .npz file for the per-trajectory distributions")
    args = ap.parse_args()
    scales = [float(x) for x in args.scales.split(",")]
    g = replace(GC.BASE_GENOME, m0=args.m0, fill=args.fill, G0=args.G0)
    t0 = time.time()
    st = Setup(args.N, args.t, g, args.backend, a=args.a)
    print(f"# {args.backend} N={args.N} a={args.a} t={args.t} m0={args.m0} fill={args.fill} G0={args.G0} "
          f"DD={args.dd} line={st.path}", flush=True)
    cases = controls(args.N, args.a, args.t, st.L) if args.controls else [("true", LG.Opt())]
    save = {}
    for sc in scales:
        for name, opt in cases:
            t1 = time.time()
            r = st.run(opt, sc, args.ntraj, dd=args.dd, seed=args.seed)
            save[f"s{sc}|{name}"] = r["per"]
            print(f"s={sc:<4} {name:34s} CZ={r['cz']} duration~{r['dur']:.0f}us ESP={r['esp']:.3f} | "
                  + " | ".join(f"{k}: {r[k]['P']:.4f} +- {r[k]['se']:.4f} (ideal {r[k]['Pid']:.4f}, random "
                               f"{r[k]['Prand']:.4f})" for k in ("C1", "C2"))
                  + f" | P(y)={np.array2string(r['Py'], precision=3, max_line_width=400)} [{time.time() - t1:.0f}s]",
                  flush=True)
    if args.out:
        np.savez_compressed(args.out, **{k.replace(' ', '_'): v for k, v in save.items()})
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
