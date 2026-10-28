"""Small numerical helpers shared by the generator and the exact model.

  * window(kind, M)      : normalised amplitude profile of a coset window over M indices;
  * bitsum(x, K)         : sum_i bit_i(x) * K[i] for integer arrays (byte lookup tables);
  * order(a, N)          : multiplicative order of a modulo N (used only by tests and controls);
  * convergent_dens, factor_from_multiple, success_table : classical post-processing of Shor outputs.
"""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------------- windows
def window(kind: str, M: int, beta: float = 6.0) -> np.ndarray:
    """Unit-norm window over M indices: 'uniform', 'sine', 'hann' or 'kaiser'."""
    j = np.arange(M)
    if kind in ("uniform", "uniforme"):
        a = np.ones(M)
    elif kind in ("sine", "sinus"):
        a = np.sin(np.pi * (j + 0.5) / M)
    elif kind == "hann":
        a = np.sin(np.pi * (j + 0.5) / M) ** 2
    elif kind == "kaiser":
        a = np.i0(beta * np.sqrt(1 - ((2 * j + 1 - M) / M) ** 2)) / np.i0(beta)
    else:
        raise ValueError(kind)
    return a / np.linalg.norm(a)


# ----------------------------------------------------------------------------- arithmetic
_TAB = {}


def bitsum(x: np.ndarray, K) -> np.ndarray:
    """sum_i bit_i(x) * K[i], computed with one 256-entry table per byte of x."""
    key = tuple(int(k) for k in K)
    tabs = _TAB.get(key)
    if tabs is None:
        u = np.arange(256)
        tabs = []
        for b0 in range(0, len(key), 8):
            T = np.zeros(256, dtype=np.int64)
            for i, k in enumerate(key[b0:b0 + 8]):
                T += ((u >> i) & 1) * np.int64(k)
            tabs.append(T)
        if len(_TAB) > 4096:
            _TAB.clear()
        _TAB[key] = tabs
    s = np.zeros(x.shape, dtype=np.int64)
    for j, T in enumerate(tabs):
        s += T[(x >> (8 * j)) & 255]
    return s


def order(a: int, N: int) -> int:
    r, x = 1, a % N
    while x != 1:
        x = x * a % N
        r += 1
    return r


# ----------------------------------------------------------------------------- post-processing
def convergent_dens(num: int, den: int, qmax: int) -> list[int]:
    """Denominators (<= qmax) of the continued-fraction convergents of num/den."""
    out = []
    h0, h1, k0, k1 = 0, 1, 1, 0
    while den:
        q_ = num // den
        h0, h1 = h1, q_ * h1 + h0
        k0, k1 = k1, q_ * k1 + k0
        num, den = den, num - q_ * den
        if k1 > qmax:
            break
        out.append(k1)
    return out


def factor_from_multiple(N: int, rp: int, bases=range(2, 12)):
    """Classical factoring from a multiple rp of the order (Miller's method, fixed bases)."""
    d, s = rp, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for g in bases:
        if math.gcd(g, N) != 1:
            continue
        x = pow(g, d, N)
        for _ in range(s):
            y = x * x % N
            if y == 1 and x not in (1, N - 1):
                f = math.gcd(x - 1, N)
                return f if 1 < f < N else None
            x = y
    return None


def success_table(N: int, a: int, t: int, B: int = 0, kmax: int = 1, qmax=None, factor: bool = True) -> np.ndarray:
    """succ[y]: a candidate r' = k*q (q a convergent denominator of y'/2^t, |y' - y| <= B, q <= qmax, k <= kmax)
    satisfies a^r' = 1 (mod N) [and, if factor, Miller's method then splits N]."""
    E = 1 << t
    qmax = qmax or N
    out = np.zeros(E, bool)
    cache = {}
    for y in range(E):
        for yp in range(y - B, y + B + 1):
            ypm = yp % E
            for q in convergent_dens(ypm, E, qmax):
                for k in range(1, kmax + 1):
                    rp = q * k
                    if rp in cache:
                        ok = cache[rp]
                    else:
                        ok = pow(a, rp, N) == 1 and (not factor or factor_from_multiple(N, rp) is not None)
                        cache[rp] = ok
                    if ok:
                        out[y] = True
                        break
                if out[y]:
                    break
            if out[y]:
                break
    return out
