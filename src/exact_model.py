"""Exact arithmetic model of the line generator (shor_line_generator.build), including the controls.

Every block of the generator is an exact permutation of basis states plus phases, so the full output
distribution P(y) of the exponent register can be computed classically for small N:
  * chain     : work-register states (X, A, amplitudes) for every exponent value, multiplication order,
                scrambled constants / table;
  * shor_P    : exact P(y) of the circuit without noise (uniform exponent superposition, final inverse QFT);
                a decohered exponent bit e_k is treated as a classical copy appended to the work-register key;
  * ideal_P   : P(y) of a perfect Shor circuit (exact modular exponentiation);
  * crit      : success criteria C1 / C2 (see below);
  * fake_consts : scrambled multipliers for the scrambled-constants control.
The quantum circuit is compared with this model in verify_package.py (Aer statevector).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix

import generator_core as GC
from coset_arithmetic import bitsum, convergent_dens, success_table, window


def chain(N, a, t, g, exps, order="desc", consts=(), table=()):
    """Work-register states W_e (arrays X, A, amp of shape (len(exps), S^2)) after all multiplications."""
    d = GC.derive(N, g)
    w, S, off = d["w"], d["S"], d["off"]
    mask = (1 << w) - 1
    L = min(GC.lookup_bits(N, a, t), 2)
    wa = window(g.kind, S)
    jj = np.arange(S)
    JX, JA = np.repeat(jj, S), np.tile(jj, S)
    exps = np.asarray(exps, dtype=np.int64)
    E = len(exps)
    T = np.array(list(table) if table else [pow(a, int(v), N) for v in range(1 << L)], dtype=np.int64)
    X = T[exps & ((1 << L) - 1)][:, None] + ((JX + off) * N)[None, :]
    A = np.tile(((JA + off) * N).astype(np.int64), (E, 1))
    amp = np.tile((wa[JX] * wa[JA]).astype(complex), (E, 1))
    cm = dict(consts)
    ks = list(range(L, t))
    if order == "desc":
        ks = ks[::-1]
    for k in ks:
        rows = ((exps >> k) & 1) == 1
        if not rows.any():
            continue
        c = cm.get(k, pow(a, 1 << k, N))
        K1, K2 = GC.mult_consts(N, c, w, g)
        s1 = bitsum(X[rows], K1)
        t1 = (A[rows] + s1) & mask
        a2 = (X[rows] - bitsum(t1, K2)) & mask
        X[rows], A[rows] = t1, a2
    return X, A, amp, w


def P_from_states(X, A, amp, w, b):
    """P(y) = sum over work-register basis states of |sum_e b_e omega^{e y} W_e|^2 / E."""
    E, C = X.shape
    keys = (X.ravel() << np.int64(w)) | A.ravel()
    nz = amp.ravel() != 0
    rows = np.repeat(np.arange(E), C)[nz]
    _, cols = np.unique(keys[nz], return_inverse=True)
    Sm = csr_matrix((amp.ravel()[nz] * b[rows], (rows, cols.ravel())), shape=(E, cols.max() + 1)).tocsc()
    P = np.zeros(E)
    ncol = Sm.shape[1]
    step = max(1, int(4e6 // E))
    for c0 in range(0, ncol, step):
        D = Sm[:, c0:c0 + step].toarray()
        F = np.fft.fft(D, axis=0)
        P += (np.abs(F) ** 2).sum(axis=1)
    return P / E


def shor_P(N, a, t, g, order="desc", consts=(), table=(), decohere=()):
    E = 1 << t
    b = window("uniform", E)
    exps = np.arange(E)
    X, A, amp, w = chain(N, a, t, g, exps, order, consts, table)
    if decohere:
        dk = np.zeros(E, dtype=np.int64)
        for i, k in enumerate(decohere):
            dk |= ((exps >> k) & 1) << i
        X = (X << np.int64(len(decohere))) | dk[:, None]          # decohered bits appended to the key
    return P_from_states(X, A, amp, w + len(decohere), b)


def ideal_P(N, a, t):
    """Output distribution of a perfect Shor circuit with a uniform exponent superposition."""
    E = 1 << t
    b = window("uniform", E)
    vals = np.array([pow(a, e, N) for e in range(E)])
    P = np.zeros(E)
    for v in np.unique(vals):
        col = np.where(vals == v, b, 0.0)
        P += np.abs(np.fft.fft(col)) ** 2
    return P / E


_SUCC = {}


def succ(N, a, t):
    """Permissive criterion C1 as a boolean table over y (see crit)."""
    key = (N, a, t)
    if key not in _SUCC:
        _SUCC[key] = success_table(N, a, t, B=0, kmax=1)
    return _SUCC[key]


def summary(N, a, t, g, **kw):
    s = succ(N, a, t)
    P = shor_P(N, a, t, g, **kw)
    return float(P[s].sum()), float(s.mean()), P


# ----------------------------------------------------------------------------- success criteria
def _primes(n):
    out, p = set(), 2
    while p * p <= n:
        while n % p == 0:
            out.add(p)
            n //= p
        p += 1
    if n > 1:
        out.add(n)
    return out


def _miller(N, a, M_):
    s = (M_ & -M_).bit_length() - 1
    b = pow(a, M_ >> s, N)
    for _ in range(s + 1):
        c = b * b % N
        if c == 1 and b not in (1, N - 1):
            return True
        b = c
    return False


_CR = {}


def crit(N, a, t, kind="C2"):
    """C1: a convergent denominator d <= N of y/2^t satisfies a^d = 1 and Miller's method factors N from d.
    C2 ("exact order"): a convergent denominator d <= N IS the order (a^d = 1 and a^(d/p) != 1 for every prime
       p | d) and Miller's method factors N. Classical check in time poly(log N). C2 is C1 without the
       MULTIPLES of the order, which wrong circuits readily produce at small N (e.g. y = 0 is always a multiple):
       with C1, decohering a single exponent bit can even raise the success rate. C2 was fixed before the
       hardware runs; C1 is reported for information only."""
    key = (N, a, t, kind)
    if key in _CR:
        return _CR[key]
    Q = 1 << t
    if kind == "C1":
        ok = succ(N, a, t).copy()
    else:
        ok = np.zeros(Q, bool)
        for y in range(Q):
            for d in convergent_dens(y, Q, N):
                if d >= 1 and pow(a, d, N) == 1 and all(pow(a, d // p, N) != 1 for p in _primes(d)) \
                        and _miller(N, a, d):
                    ok[y] = True
                    break
    _CR[key] = ok
    return ok


def fake_consts(N, a, t, L):
    """Scrambled multipliers for the control: units OUTSIDE the subgroup <a> (otherwise c' = a^j would still be
    a disguised Shor circuit). The control may use any classical information: it is not part of the algorithm."""
    sub, x = set(), 1
    for _ in range(N):
        sub.add(x)
        x = x * a % N
    units = [u for u in range(2, N) if math.gcd(u, N) == 1 and u not in sub]
    rng = np.random.default_rng(N * 1000 + t)
    return tuple((k, int(rng.choice(units))) for k in range(L, t))
