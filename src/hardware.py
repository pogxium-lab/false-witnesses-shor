"""Hardware layer: any IBM backend (simulated FakeXxx model or real device from qiskit_ibm_runtime).

  * choice of the physical line of qubits from the calibration (beam search over simple paths of the coupling
    graph; score = sum over edges of (two-qubit gates on that edge) * log(1 - e_CZ)
    + sum over qubits of (measurements) * log(1 - e_readout) + a small coherence term);
  * transpilation WITHOUT routing (every two-qubit gate is already between neighbours), best of a few seeds;
  * estimated success probability (ESP): gates, readout, idle periods (T1/T2) and feed-forward latency;
    an optional scale factor s multiplies every error rate (s = 2: errors x2, T1/T2 / 2).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from qiskit import transpile
from qiskit.circuit import IfElseOp

import generator_core as GC

BASIS = ["cz", "rz", "sx", "x", "measure", "reset", "if_else"]
LAT = 0.7e-6          # feed-forward latency (measurement -> conditional gate), order of magnitude for Heron


@lru_cache(maxsize=None)
def backend(name):
    """Simulated backend: a FakeXxx model, or the path of a calibration snapshot (calibrated_backend)."""
    if name.endswith(".json"):
        return calibrated_backend(name)
    import qiskit_ibm_runtime.fake_provider as fp
    return getattr(fp, name)()


FAKE_OF = {"ibm_kingston": "FakeKingston", "ibm_fez": "FakeFez", "ibm_marrakesh": "FakeMarrakesh",
           "ibm_torino": "FakeTorino", "ibm_aachen": "FakeAachen", "ibm_brisbane": "FakeBrisbane"}


def calibrated_backend(path, base=None):
    """Offline copy of a real device: the fake backend of the same device (coupling map, basis, durations) with
    the properties replaced by a calibration snapshot (snapshot_calibration.py), so that line choice, ESP and the
    noise model use that day's calibration."""
    import json
    import qiskit_ibm_runtime.fake_provider as fp
    with open(path, encoding="utf-8") as fh:
        props = json.load(fh)
    be = getattr(fp, base or FAKE_OF[props["backend_name"]])()
    be._props_dict = props
    be._target = None
    be.calibration_snapshot = str(path)
    return be


def line_usage(qc):
    """Two-qubit gate weight (~CZ count) on each edge of the line and number of measurements per position."""
    n = qc.num_qubits
    edge = [0.0] * max(n - 1, 1)
    meas = [0.0] * n

    def walk(circ, qmap, wgt):
        for ins in circ.data:
            op = ins.operation
            qs = [qmap[circ.find_bit(q).index] for q in ins.qubits]
            if op.name == "if_else":
                for blk in op.blocks:
                    if blk is not None:
                        walk(blk, {i: qs[i] for i in range(len(qs))}, wgt * 0.5)
                continue
            if op.name == "measure":
                meas[qs[0]] += 1
            elif op.num_qubits == 2:
                a, b = sorted(qs)
                c = 3.0 if op.name == "unitary" else (2.0 if op.name == "cp" else 1.0)
                if b == a + 1:
                    edge[a] += c * wgt
            elif op.num_qubits == 3:
                a = min(qs)
                edge[a] += 3 * wgt
                edge[a + 1] += 3 * wgt
    walk(qc, {i: i for i in range(n)}, 1.0)
    return edge, meas


class HW:
    def __init__(self, name):
        """name: name of a simulated backend ("FakeKingston"), path of a calibration snapshot (.json), OR a
        backend object (real device)."""
        self.be = backend(name) if isinstance(name, str) else name
        self.name = name if isinstance(name, str) else self.be.name
        self.tg = self.be.target
        self.adj = {}
        for a_, b_ in self.be.coupling_map.get_edges():
            self.adj.setdefault(a_, set()).add(b_)
            self.adj.setdefault(b_, set()).add(a_)
        self.tmeas = float(np.median([self.tg["measure"][(q,)].duration for q in range(self.be.num_qubits)
                                      if self.tg["measure"][(q,)] and self.tg["measure"][(q,)].duration]))

    def ecz(self, a, b):
        for k in ((a, b), (b, a)):
            try:
                e = self.tg["cz"][k].error
                if e is not None:
                    return e
            except KeyError:
                pass
        return 0.5

    def e1(self, name, q):
        try:
            return self.tg[name][(q,)].error or 0.0
        except KeyError:
            return 0.0

    def emeas(self, q):
        try:
            return self.tg["measure"][(q,)].error or 0.02
        except KeyError:
            return 0.02

    def t1t2(self, q):
        p = self.tg.qubit_properties[q]
        t1 = p.t1 or 100e-6
        t2 = min(p.t2 or 100e-6, 2 * t1)
        return t1, t2

    def dur(self, name, qs):
        if name in ("rz", "barrier", "delay", "p"):
            return 0.0
        if name == "reset":
            return self.tmeas + 0.3e-6
        try:
            return self.tg[name][tuple(qs)].duration or 0.0
        except Exception:
            try:
                return self.tg[name][tuple(qs[::-1])].duration or 0.0
            except Exception:
                return 0.0

    # ------------------------------------------------------------------ path
    def best_paths(self, edge_w, meas_w, beam=4000, top=3):
        n = len(meas_w)
        lm = {q: math.log1p(-min(self.emeas(q), 0.5)) for q in range(self.be.num_qubits)}
        # coherence cost: ~30 us of activity per qubit (order of magnitude), small weight
        lc = {}
        for q in range(self.be.num_qubits):
            t1, t2 = self.t1t2(q)
            lc[q] = -(15e-6 / t1 + 15e-6 / t2)
        states = [((q,), meas_w[0] * lm[q] + lc[q]) for q in self.adj]
        for i in range(1, n):
            nxt = []
            for path, sc in states:
                last = path[-1]
                for nb in self.adj[last]:
                    if nb in path:
                        continue
                    s2 = sc + edge_w[i - 1] * math.log1p(-min(self.ecz(last, nb), 0.75)) + meas_w[i] * lm[nb] + lc[nb]
                    nxt.append((path + (nb,), s2))
            nxt.sort(key=lambda x: -x[1])
            states = nxt[:beam]
        return states[:top]

    def place(self, qc, top=3, seeds=(1, 2)):
        """Choose the physical line and transpile. Returns (transpiled circuit, path)."""
        ew, mw = line_usage(qc)
        qd = transpile(qc, basis_gates=BASIS, optimization_level=0)
        best = None
        for path, sc in self.best_paths(ew, mw, top=top):
            for s in seeds:
                tq = transpile(qd, self.be, optimization_level=2, seed_transpiler=s, initial_layout=list(path))
                o, c = GC.count_2q(tq)
                key = (o, -sc)
                if best is None or key < best[0]:
                    best = (key, tq, path)
        return best[1], best[2]

    # ------------------------------------------------------------------ ESP
    def esp(self, tq, scale=1.0, dd=False):
        """log-ESP split into gates, readout and idle periods (ASAP schedule with backend durations)."""
        free, ready = {}, 0.0
        lg = {"gates2q": 0.0, "gates1q": 0.0, "readout": 0.0, "idle": 0.0}
        idle = {}
        for ins in tq.data:
            op = ins.operation
            qs = [tq.find_bit(q).index for q in ins.qubits]
            if op.name == "barrier":
                continue
            if isinstance(op, IfElseOp):
                start = max([free.get(q, 0.0) for q in qs] + [ready])
                for q in qs:
                    if q in free:
                        idle[q] = idle.get(q, 0.0) + start - free[q]
                    free[q] = start
                continue
            d = self.dur(op.name, qs)
            start = max([free.get(q, 0.0) for q in qs] + [0.0])
            for q in qs:
                if q in free:
                    idle[q] = idle.get(q, 0.0) + start - free[q]
                free[q] = start + d
            if op.name == "measure":
                lg["readout"] += math.log1p(-min(self.emeas(qs[0]) * scale, 0.5))
                ready = max(ready, start + d + LAT)
            elif op.num_qubits == 2:
                lg["gates2q"] += math.log1p(-min(self.ecz(*qs) * scale, 0.75))
            elif op.num_qubits == 1 and op.name in ("sx", "x"):
                lg["gates1q"] += math.log1p(-min(self.e1(op.name, qs[0]) * scale, 0.75))
        for q, tau in idle.items():
            t1, t2 = self.t1t2(q)
            t1, t2 = t1 / scale, t2 / scale
            if dd:
                t2 = dd_t2(t1, t2)
            p = (1 - math.exp(-tau / t1)) / 4 + (1 - math.exp(-tau / t2)) / 2
            lg["idle"] += math.log1p(-min(p, 0.75))
        lg["total"] = sum(lg.values())
        lg["duration_us"] = max(free.values()) * 1e6
        return lg


def dd_t2(t1, t2):
    """Dynamical decoupling (crude model): half of the pure dephasing is removed."""
    gphi = max(1 / t2 - 1 / (2 * t1), 0.0)
    return 1 / (1 / (2 * t1) + gphi / 2)
