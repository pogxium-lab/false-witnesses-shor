"""Parameters and classical rules of the generic coset-representation Shor generator.

Everything here is computed from (N, a, t, genome) in time poly(log N): no table of size ~N and no knowledge of
the order of a is ever used.

  * Genome         : register-size and window rules (window bits, guard bits, window shape and offset,
                     choice of constant representatives);
  * derive         : register width w, window size S, window offset;
  * lookup_bits    : number L of low exponent bits handled by a classical-input lookup table; the table is used
                     only when a^(2^L - 1) < N, i.e. when its entries are unreduced powers (no modular reduction,
                     hence no hidden knowledge of the order);
  * mult_consts    : constants K1 (first pass, A += sum_i x_i K1_i) and K2 (second pass, A -= sum_i x_i K2_i)
                     of a controlled multiplication by c in coset representation;
  * anf            : algebraic normal form of the lookup table (sum of products of exponent bits);
  * helper gates and a two-qubit gate counter for transpiled circuits.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

from coset_arithmetic import window

TWO_PI = 2 * math.pi
EPS = 1e-12


# ============================================================================ genome
@dataclass(frozen=True)
class Genome:
    m0: int = 2          # window bits: mJ = m0 + round(ms * log2 n)
    ms: float = 0.0
    G0: int = 0          # guard bits: G = G0 + round(gs * log2 n)
    gs: float = 0.0
    fill: int = 0        # 1: the window spans all J = floor(2^w / N) available indices
    kind: str = "sine"   # window shape of the coset state
    off: str = "center"  # window offset: center | low | high
    czpi: int = 0        # cost model of representatives: CP(pi) counted as one CZ
    R: int = 0           # representatives k + tN with |t| <= R chosen by cost (0: centred representative)
    lam: float = 0.0     # penalty lam * (K/N)^2 in the representative choice

    def short(self):
        return " ".join(f"{k}={v}" for k, v in asdict(self).items())


BASE_GENOME = Genome()


def derive(N: int, g: Genome) -> dict:
    """Register sizes: n = bit length of N - 1, mJ window bits, S window indices, w register width,
    J = floor(2^w / N) representable copies, off = window offset, G guard bits, M = 2^mJ."""
    n = (N - 1).bit_length()
    lg = math.log2(max(n, 2))
    mJ = int(np.clip(g.m0 + round(g.ms * lg), 1, 10))
    G = int(max(0, g.G0 + round(g.gs * lg)))
    w = (N * (1 << mJ) - 1).bit_length() + G
    J = (1 << w) // N
    if g.fill:
        S = (1 << (w - G)) // N
        mq = max(1, (S - 1).bit_length())
    else:
        S = 1 << mJ
        mq = mJ
    off = {"center": (J - S) // 2, "low": 0, "high": J - S}[g.off]
    return dict(n=n, mJ=mq, S=S, w=w, J=J, off=off, G=G, M=1 << mq)


def lookup_bits(N: int, a: int, t: int) -> int:
    """Largest L <= t with a^(2^L - 1) < N (unreduced table entries), at least 1."""
    L = 0
    while L + 1 <= t and pow(a, (1 << (L + 1)) - 1) < N:
        L += 1
    return max(L, 1)


# ============================================================================ constants
def ang(K: int, j: int, w: int) -> float:
    """Phase on qubit j of a w-qubit Fourier register (QFT without swaps) that adds K mod 2^w."""
    th = (TWO_PI * (K % (1 << w)) * (1 << (w - 1 - j)) / (1 << w)) % TWO_PI
    return th if min(th, TWO_PI - th) > EPS else 0.0


def _angle_cost(K, w, czpi):
    c = 0
    for j in range(w):
        th = ang(K, j, w)
        if th == 0.0:
            continue
        c += 1 if (czpi and abs(th - math.pi) < 1e-9) else 2
    return c


def centered(v: int, N: int) -> int:
    v %= N
    return v - N if v > N // 2 else v


def pick_rep(k: int, N: int, w: int, g: Genome) -> int:
    """Representative of k (mod N): centred, or k0 + tN (|t| <= R) minimising gate cost + lam * (K/N)^2."""
    k0 = centered(k, N)
    if g.R == 0:
        return k0
    best = None
    for t in range(-g.R, g.R + 1):
        K = k0 + t * N
        s = _angle_cost(K, w, g.czpi) + g.lam * (K / N) ** 2
        if best is None or s < best[0] - 1e-12:
            best = (s, K)
    return best[1]


def mult_consts(N: int, c: int, w: int, g: Genome):
    """K1 (pass 1: A += sum x_i K1_i) and K2 (pass 2: A -= sum x_i K2_i) for a multiplication by c mod N."""
    ci = pow(c, -1, N)
    K1 = [pick_rep((c << i) % N, N, w, g) for i in range(w)]
    K2 = [-pick_rep(-((ci << i) % N), N, w, g) for i in range(w)]
    return K1, K2


def anf(T):
    """Algebraic normal form (Moebius transform) of a table T indexed by L bits."""
    L = (len(T) - 1).bit_length()
    c = list(T)
    for i in range(L):
        for s in range(len(c)):
            if s >> i & 1:
                c[s] -= c[s ^ (1 << i)]
    return c


# ============================================================================ gates and counting
@lru_cache(None)
def rccx_dg():
    """Inverse of Qiskit's relative-phase Toffoli (RCCX), as a gate."""
    g = QuantumCircuit(3, name="rccx_dg")
    g.rccx(0, 1, 2)
    return g.inverse().to_gate()


@lru_cache(None)
def stateprep(kind: str, S: int, mq: int):
    """Exact preparation of the window state over S of the 2^mq basis states."""
    a = np.zeros(1 << mq)
    a[:S] = window(kind, S)
    return StatePreparation(a)


def count_2q(qc):
    """(two-qubit gates outside classically-controlled blocks, two-qubit gates inside them)."""
    out, cond = 0, 0
    for ins in qc.data:
        op = ins.operation
        if op.name == "barrier":
            continue
        if op.name == "if_else":
            for blk in op.blocks:
                if blk is None:
                    continue
                o, c = count_2q(blk)
                cond += o + c
        elif op.num_qubits == 2:
            out += 1
        elif op.num_qubits >= 3:
            raise ValueError(op.name)
    return out, cond
