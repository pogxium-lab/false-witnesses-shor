"""Exact simulation of transpiled dynamic circuits (mid-circuit measurements, resets, feed-forward), optionally
with static coherent errors placed on a timeline.

Deferred measurement: measure -> CX onto a fresh record qubit (or nothing if the qubit is never used again);
reset -> SWAP with a fresh qubit in |0>; if_else (c == 1) -> every gate of the block controlled by the record
qubit of c. The probabilities of a classical register are then exact (no sampling, every branch of every
mid-circuit measurement included). Delays and barriers are identities.

Compact mode (compact=True, for circuits with many mid-circuit measurements): a measurement whose classical bit
is never read again (neither a feed-forward condition nor a bit of the register asked for) on a qubit that is used
afterwards is a Z dephasing, 1/2 (rho + Z rho Z); it is replaced by an exact average over the 2^d patterns of Z
gates instead of d record qubits. A reset that directly follows the measurement of the same qubit is done with a
CX from the record qubit (the qubit is then exactly |0>) instead of a SWAP with a fresh qubit.

Timeline (only when coherent errors are requested): ASAP schedule with the durations of the backend
(hardware.HW), explicit delays, conditional blocks starting after their classical bit + feed-forward latency.
Coherent errors accumulated between operations, from the start to the end of the circuit:
  detuning {q: Hz}     static frequency offset of physical qubit q -> RZ(2 pi delta tau);
  zz {(a, b): Hz}      static ZZ between physical qubits a, b -> CP(2 pi zeta tau) (not during a CZ on (a, b),
                       whose calibration absorbs it).
Rz gates are virtual (zero duration) and commute with both errors.
"""
from __future__ import annotations

import json
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import IfElseOp
from qiskit.circuit.library import CPhaseGate, RZGate

from hardware import LAT

SKIP = ("barrier", "delay", "id")
_SIM = None


def simulator():
    global _SIM
    if _SIM is None:
        from qiskit_aer import AerSimulator
        _SIM = AerSimulator(method="statevector")
    return _SIM


def delay_seconds(op, dt):
    unit = getattr(op, "unit", "dt")
    d = float(op.duration)
    return d * dt if unit == "dt" else d * {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}[unit]


def op_duration(op, qs, hw):
    """Duration (s) of an operation of the transpiled circuit (conditional block: sum of its operations)."""
    if op.name == "delay":
        return delay_seconds(op, hw.tg.dt)
    if isinstance(op, IfElseOp):
        blk = op.blocks[0]
        return sum(op_duration(b.operation, [qs[blk.find_bit(q).index] for q in b.qubits], hw) for b in blk.data)
    return hw.dur(op.name, qs)


def timeline(tq, hw, lat=LAT):
    """ASAP schedule: list of (start, end, index, instruction, physical qubits), sorted by start time."""
    free, ready, ev = {}, {}, []
    for i, ins in enumerate(tq.data):
        op = ins.operation
        if op.name == "barrier":
            continue
        qs = [tq.find_bit(q).index for q in ins.qubits]
        start = max([free.get(q, 0.0) for q in qs] + [0.0])
        if isinstance(op, IfElseOp):
            start = max(start, ready.get(tq.find_bit(op.condition[0]).index, 0.0) + lat)
        end = start + op_duration(op, qs, hw)
        for q in qs:
            free[q] = end
        if op.name == "measure":
            ready[tq.find_bit(ins.clbits[0]).index] = end
        ev.append((start, end, i, ins, qs))
    ev.sort(key=lambda e: (e[0], e[2]))
    return ev


def makespan(tq, hw, lat=LAT):
    return max(e[1] for e in timeline(tq, hw, lat))


def calibrated_zz(calibration, line):
    """Static ZZ (Hz) of the consecutive pairs of `line` from a calibration snapshot (keys zz_<a><b>, GHz)."""
    with open(calibration, encoding="utf-8") as fh:
        gen = {x["name"]: x["value"] for x in json.load(fh).get("general", [])}
    out = {}
    for a, b in zip(line, line[1:]):
        v = gen.get(f"zz_{a}{b}", gen.get(f"zz_{b}{a}"))
        if v:
            out[(a, b)] = float(v) * 1e9
    return out


def dead_measurements(tq, creg="y"):
    """Indices of the measurements whose classical bit is never read (no condition, not in `creg`) on a qubit that
    is used again afterwards (each is an exact Z dephasing)."""
    live = {tq.find_bit(b).index for r in tq.cregs if r.name == creg for b in r}
    for ins in tq.data:
        if isinstance(ins.operation, IfElseOp):
            live.add(tq.find_bit(ins.operation.condition[0]).index)
    last = {}
    for i, ins in enumerate(tq.data):
        if ins.operation.name not in SKIP + ("barrier",):
            for q in ins.qubits:
                last[tq.find_bit(q).index] = i
    return [i for i, ins in enumerate(tq.data) if ins.operation.name == "measure"
            and tq.find_bit(ins.clbits[0]).index not in live and last[tq.find_bit(ins.qubits[0]).index] > i]


def deferred(tq, hw=None, detuning=None, zz=None, lat=LAT, zpattern=None, compact=False):
    """Static circuit equivalent to the dynamic circuit tq; returns (circuit, record qubit of each clbit).
    zpattern: {instruction index of a dead measurement: 0 or 1} (compact mode, one dephasing branch)."""
    coherent = bool(detuning) or bool(zz)
    if coherent:
        ev = timeline(tq, hw, lat)
    else:
        ev = [(0.0, 0.0, i, ins, [tq.find_bit(q).index for q in ins.qubits]) for i, ins in enumerate(tq.data)
              if ins.operation.name != "barrier"]
    ev = [e for e in ev if e[3].operation.name not in SKIP]
    last_use = {}
    for k, e in enumerate(ev):
        for q in e[4]:
            last_use[q] = k
    used = sorted(last_use)
    pos = {q: i for i, q in enumerate(used)}
    zpattern = zpattern or {}
    n_extra, jm = 0, set()                # same bookkeeping as the main loop below
    for k, e in enumerate(ev):
        name = e[3].operation.name
        if name == "rz":
            continue
        if name == "reset" and compact and e[4][0] in jm:
            jm.discard(e[4][0])
            continue
        jm.difference_update(e[4])
        if name == "measure" and e[2] in zpattern:
            continue
        if name == "measure" and last_use[e[4][0]] > k:
            n_extra += 1
            if compact:
                jm.add(e[4][0])
        elif name == "reset":
            n_extra += 1
    qc = QuantumCircuit(len(used) + n_extra)
    nxt = len(used)
    rec = {}
    detuning = {q: f for q, f in (detuning or {}).items() if q in pos}
    zz = {e: f for e, f in (zz or {}).items() if e[0] in pos and e[1] in pos}
    last_e = {e: 0.0 for e in zz}
    last_q = {q: 0.0 for q in detuning}
    touch = {}
    just_measured = {}                    # qubit -> record qubit, while nothing else touched it (compact mode)
    for e in zz:
        for q in e:
            touch.setdefault(q, []).append(e)

    def flush(qs, t):
        for q in qs:
            for e in touch.get(q, ()):
                tau = t - last_e[e]
                if tau > 0:
                    qc.append(CPhaseGate(2 * math.pi * zz[e] * tau), [pos[e[0]], pos[e[1]]])
                    last_e[e] = t
            if q in detuning:
                tau = t - last_q[q]
                if tau > 0:
                    qc.rz(2 * math.pi * detuning[q] * tau, pos[q])
                    last_q[q] = t

    for k, (start, end, i, ins, qs) in enumerate(ev):
        op = ins.operation
        if op.name == "rz":
            qc.rz(op.params[0], pos[qs[0]])
            continue
        if coherent:
            flush(qs, start)
        if op.name == "reset" and compact and qs[0] in just_measured:
            qc.cx(just_measured.pop(qs[0]), pos[qs[0]])
            continue
        for q in qs:
            just_measured.pop(q, None)
        if op.name == "measure" and i in zpattern:
            if zpattern[i]:
                qc.z(pos[qs[0]])
            continue
        if op.name == "measure":
            cb = tq.find_bit(ins.clbits[0]).index
            if last_use[qs[0]] > k:
                qc.cx(pos[qs[0]], nxt)
                rec[cb] = nxt
                if compact:
                    just_measured[qs[0]] = nxt
                nxt += 1
            else:
                rec[cb] = pos[qs[0]]
            continue
        if op.name == "reset":
            qc.swap(pos[qs[0]], nxt)
            nxt += 1
            continue
        if isinstance(op, IfElseOp):
            clbit, val = op.condition
            if val not in (1, True) or (len(op.blocks) > 1 and op.blocks[1] is not None):
                raise ValueError(f"unsupported condition {op.condition} / else block")
            blk = op.blocks[0]
            ctrl = rec[tq.find_bit(clbit).index]
            for b in blk.data:
                if b.operation.name in SKIP:
                    continue
                bq = [pos[qs[blk.find_bit(q).index]] for q in b.qubits]
                g = RZGate(b.operation.params[0]) if b.operation.name == "rz" else b.operation
                qc.append(g.control(1), [ctrl] + bq)
            continue
        if op.name == "cz" and coherent:
            e = (qs[0], qs[1]) if (qs[0], qs[1]) in last_e else (qs[1], qs[0])
            qc.cz(pos[qs[0]], pos[qs[1]])
            if e in last_e:
                last_e[e] = end
            continue
        qc.append(op, [pos[q] for q in qs])
    return qc, rec


def register_probabilities(tq, creg="y", compact=False, **kw):
    """Exact P(value of the classical register `creg`), index = sum_j bit_j << j. compact=True: dead
    measurements averaged over their Z patterns, resets after a measurement done with a CX (fewer qubits)."""
    sim = simulator()                      # also registers the save instructions of qiskit_aer
    reg = [c for c in tq.cregs if c.name == creg][0]
    dead = dead_measurements(tq, creg) if compact else []
    total = None
    for bits in range(1 << len(dead)):
        pat = {i: (bits >> j) & 1 for j, i in enumerate(dead)}
        qc, rec = deferred(tq, zpattern=pat, compact=compact, **kw)
        qc.save_probabilities([rec[tq.find_bit(b).index] for b in reg], label="p")
        p = np.asarray(sim.run(qc, shots=1).result().data(0)["p"])
        total = p if total is None else total + p
    return total / (1 << len(dead))
