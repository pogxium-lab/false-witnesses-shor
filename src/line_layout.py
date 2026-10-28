"""Linear-nearest-neighbour primitives: a line of labelled qubits on which every two-qubit gate acts between
neighbours.

Registers are never moved back into place: the line keeps track of the position of every label ("sliding
registers"). The basic move is a block crossing (bubble-sort network): every pair (b, c) of two crossing blocks
meets exactly once, and the useful diagonal gate of that pair is fused with the swap
(CP + SWAP = 3 CZ, CZ + SWAP = 2 CZ, CX + SWAP = 2 CZ).

Contents: Line (positions, fused swaps, sorting), qft_line (QFT on a contiguous register, either orientation,
CP + SWAP fused), coset-preparation helpers (window preparation, inverse QFT and measurement of the window
register, classically-controlled phase correction) and the final uncontrolled addition pass.
"""
from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

import generator_core as GC

TWO_PI = 2 * math.pi
SWAPM = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def theta(K, j, w):
    """Unreduced phase added to qubit j of a w-qubit Fourier register (QFT without swaps) by +K."""
    return TWO_PI * K * (1 << (w - 1 - j)) / (1 << w)


def reduce_angle(th):
    th = th % TWO_PI
    return 0.0 if min(th, TWO_PI - th) < 1e-12 else th


# =============================================================================== line
class Line:
    def __init__(self, labels, cregs):
        self.L = list(labels)
        self.pos = {l: i for i, l in enumerate(self.L)}
        self.qc = QuantumCircuit(len(self.L))
        for c in cregs:
            self.qc.add_register(c)
        self.zero = set()           # labels known to be in |0>
        self.nswap = 0
        self.nfused = 0

    def q(self, l):
        return self.pos[l]

    def adj(self, a, b):
        assert abs(self.pos[a] - self.pos[b]) == 1, (a, b, self.pos[a], self.pos[b])

    # --- gates
    def g2(self, name, a, b, *par):
        self.adj(a, b)
        getattr(self.qc, name)(*par, self.pos[a], self.pos[b])
        self.zero.discard(a)
        self.zero.discard(b)

    def g1(self, name, a, *par):
        getattr(self.qc, name)(*par, self.pos[a])
        if name not in ("p", "rz", "z", "s", "sdg", "t", "tdg"):
            self.zero.discard(a)

    def u2(self, a, b, build):
        """Two-qubit block on neighbours (a, b) emitted as ONE unitary gate (synthesised with <= 3 CZ);
        build(qc2) writes the block on a QuantumCircuit(2) where 0 = a and 1 = b."""
        from qiskit.quantum_info import Operator
        self.adj(a, b)
        q2 = QuantumCircuit(2)
        build(q2)
        self.qc.append(UnitaryGate(Operator(q2).data, label="b"), [self.pos[a], self.pos[b]])
        self.zero.discard(a)
        self.zero.discard(b)

    def _swap_pos(self, p):
        u, v = self.L[p], self.L[p + 1]
        self.L[p], self.L[p + 1] = v, u
        self.pos[u], self.pos[v] = p + 1, p

    def swap(self, p, gate=None, zero_ok=True):
        """Swap positions p, p+1, fused with `gate` (applied BEFORE the swap):
        None | ('cp', theta) between the two qubits | ('cx', control_label)."""
        u, v = self.L[p], self.L[p + 1]
        if gate is None or (gate[0] == "cp" and reduce_angle(gate[1]) == 0.0):
            if zero_ok and (u in self.zero or v in self.zero):
                if u in self.zero and v in self.zero:
                    pass                                    # two |0>: nothing to do
                elif v in self.zero:                        # |x,0> -> |0,x>
                    self.qc.cx(p, p + 1)
                    self.qc.cx(p + 1, p)
                else:
                    self.qc.cx(p + 1, p)
                    self.qc.cx(p, p + 1)
            else:
                self.qc.append(UnitaryGate(SWAPM, label="sw"), [p, p + 1])
                self.nswap += 1
        else:
            if gate[0] == "cp":
                G = np.diag([1, 1, 1, np.exp(1j * gate[1])])
            elif gate[0] == "cx":
                # Qiskit ordering: qubit p is the least significant bit
                if gate[1] == u:     # control on p, target p+1
                    G = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
                else:
                    G = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
            else:
                raise ValueError(gate)
            self.qc.append(UnitaryGate(SWAPM @ G, label="f"), [p, p + 1])
            self.zero.discard(u)
            self.zero.discard(v)
            self.nfused += 1
        self._swap_pos(p)

    def sort_to(self, order, meet=None):
        """Rearrange the line into `order` (complete list) by bubble sort (minimal number of adjacent swaps,
        every inverted pair meets exactly once); meet(u, v) -> fused gate or None."""
        tgt = {l: i for i, l in enumerate(order)}
        assert len(tgt) == len(self.L)
        changed = True
        while changed:
            changed = False
            for p in range(len(self.L) - 1):
                u, v = self.L[p], self.L[p + 1]
                if tgt[u] > tgt[v]:
                    self.swap(p, meet(u, v) if meet else None)
                    changed = True

    def cross(self, B, C, meet):
        """Contiguous block B crosses the adjacent contiguous block C; internal orders are preserved."""
        pb = sorted(self.pos[b] for b in B)
        pc = sorted(self.pos[c] for c in C)
        Bs = [self.L[p] for p in pb]
        Cs = [self.L[p] for p in pc]
        lo = min(pb[0], pc[0])
        seg = Cs + Bs if pb[0] < pc[0] else Bs + Cs
        if pb[0] < pc[0]:
            assert pb[-1] + 1 == pc[0]
        else:
            assert pc[-1] + 1 == pb[0]
        order = self.L[:lo] + seg + self.L[lo + len(seg):]
        self.sort_to(order, meet)

    def block(self, labels):
        ps = sorted(self.pos[l] for l in labels)
        assert ps[-1] - ps[0] == len(ps) - 1, ("not contiguous", labels, ps)
        return [self.L[p] for p in ps]


# =============================================================================== QFT on the line
def qft_line(ln: Line, reg, inverse=False):
    """reg = labels q_0..q_{w-1} (by weight); applies synth_qft_full(w, do_swaps=False) (or its inverse) to these
    qubits whatever their orientation on the line (contiguous, increasing or decreasing).
    The register leaves with the opposite orientation."""
    w = len(reg)
    blk = ln.block(reg)
    idx = {l: i for i, l in enumerate(reg)}
    order = [idx[l] for l in blk]
    asc = order == list(range(w))
    desc = order == list(range(w - 1, -1, -1))
    assert asc or desc, ("orientation", order)
    ops = []
    for j in range(w - 1, -1, -1):
        ops.append(("h", j))
        for k in range(j - 1, -1, -1):
            ops.append(("cps", j, k, math.pi * 2.0 ** (k - j)))

    def play(seq, inv):
        for op in seq:
            if op[0] == "h":
                ln.g1("h", reg[op[1]])
            else:
                _, j, k, lam = op
                pj, pk = ln.pos[reg[j]], ln.pos[reg[k]]
                assert abs(pj - pk) == 1, (j, k)
                ln.swap(min(pj, pk), ("cp", -lam if inv else lam))
    if not inverse:
        play(ops, False)
    else:
        # inverse of the forward network started from the opposite orientation: reversed order, opposite angles
        play(list(reversed(ops)), True)


# =============================================================================== coset preparation helpers
def prep_window(ln: Line, J, kind, S, mq):
    qs = [ln.pos[l] for l in J]
    ln.qc.append(GC.stateprep(kind, S, mq), qs)
    for l in J:
        ln.zero.discard(l)


def iqft_and_measure_window(ln: Line, J, cJ):
    """QFTGate(mq).inverse() (with swaps) then measurement. The inverse QFT with swaps equals the inverse QFT
    without swaps on the reversed order followed by the swaps: classical bit i is therefore read on J[mq-1-i]."""
    mq = len(J)
    if mq == 1:
        ln.g1("h", J[0])
    else:
        qft_line(ln, list(reversed(J)), inverse=True)
    for i in range(mq):
        ln.qc.measure(ln.pos[J[mq - 1 - i]], cJ[i])
    for l in J:
        ln.qc.reset(ln.pos[l])
        ln.zero.add(l)


def phase_corr(ln: Line, R, cJ, N, M, comp):
    """Phase correction conditioned on the measured window outcome (and compensation terms comp)."""
    for i in range(len(cJ)):
        with ln.qc.if_test((cJ[i], 1)):
            for b, l in enumerate(R):
                th = (TWO_PI * (1 << i) * (1 << b) / (N * M)) % TWO_PI
                ln.qc.p(th, ln.pos[l])
            for l, coef in comp:
                ln.qc.p((-TWO_PI * (1 << i) * coef / (N * M)) % TWO_PI, ln.pos[l])


# =============================================================================== passes and orderings
def pair_index(xi, aj, u, v):
    if u in xi and v in aj:
        return xi[u], aj[v]
    if v in xi and u in aj:
        return xi[v], aj[u]
    return None


def final_pass(ln: Line, N, w, X, A, pend):
    """Pending uncontrolled second pass: A += sum_i x_i pend_i (X crosses A)."""
    xi = {l: i for i, l in enumerate(X)}
    aj = {l: j for j, l in enumerate(A)}

    def m(u, v):
        ij = pair_index(xi, aj, u, v)
        return None if ij is None else ("cp", theta(pend[ij[0]], ij[1], w))
    ln.cross(X, A, m)


def segment(ln, labels):
    ps = sorted(ln.pos[l] for l in labels)
    assert ps[-1] - ps[0] == len(ps) - 1
    return ps[0], ps[-1]


def inversions(cur, tgt_seq):
    idx = {l: i for i, l in enumerate(tgt_seq)}
    s = [idx[l] for l in cur]
    n = 0
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            n += s[i] > s[j]
    return n


def deinterleave_order(ln: Line, X, A, c, next_side):
    """Target order of the working zone: [c][X][A] or [A][X][c] with A on side next_side ('L', 'R' or 'any');
    c stays at its end. The cheapest admissible order (fewest adjacent swaps) is returned."""
    lo, hi = segment(ln, list(X) + list(A) + [c])
    seg = ln.L[lo:hi + 1]
    c_left = seg[0] == c
    body = [l for l in seg if l != c]
    xs = [l for l in body if l in X]
    as_ = [l for l in body if l in A]
    cands = []
    sides = ("L", "R") if next_side == "any" else (next_side,)
    for arev in (False, True):
        aseq = sorted(as_, key=A.index, reverse=arev)
        for sd in sides:
            core = (xs + aseq) if sd == "R" else (aseq + xs)
            blk = ([c] + core) if c_left else (core + [c])
            cands.append((inversions(seg, blk), blk))
    best = min(cands, key=lambda x: x[0])[1]
    return ln.L[:lo] + best + ln.L[hi + 1:]
