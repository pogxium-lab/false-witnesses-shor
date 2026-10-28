"""Measurement trap: ideal success of perfect Shor versus a deliberately wrong circuit under several criteria.

Wrong circuit: exponent bit j copied into an ancilla that is never uncomputed (bit j decohered). Bits j with
2^(j+1) | r are excluded: for them e = e' mod r already implies e_j = e'_j, so the output law is unchanged and the
circuit is not wrong.

Success criteria (y sampled from Q = 2^t outcomes):
  C1      : some convergent d <= N of y/Q has a^d = 1 and Miller's reduction from d factors N (the permissive
            criterion C1 of the paper)
  best    : d = denominator of the best approximation of y/Q with denominator <= N (d alone); d even, a^d = 1,
            gcd(a^(d/2) +- 1, N) nontrivial
  best+2d : as best, with candidates d and 2d
  all+2d  : as C1, with candidates d and 2d for every convergent d <= N
Exact (noiseless) computation, no sampling. Usage: python analysis/measurement_trap.py
"""
from fractions import Fraction
from math import gcd
import numpy as np


def order(a, N):
    r, x = 1, a % N
    while x != 1:
        x = x * a % N
        r += 1
    return r


def conv_dens(y, Q):
    out, k0, k1, a, b = [], 1, 0, y, Q
    while b:
        t = a // b
        a, b = b, a - t * b
        k0, k1 = k1, t * k1 + k0
        out.append(k1)
    return out


def miller(N, a, M):
    s = (M & -M).bit_length() - 1
    b = pow(a, M >> s, N)
    for _ in range(s + 1):
        c = b * b % N
        if c == 1 and b not in (1, N - 1):
            return True
        b = c
    return False


def half(N, a, rp):
    if rp % 2 or pow(a, rp, N) != 1:
        return False
    h = pow(a, rp // 2, N)
    return 1 < gcd(h - 1, N) < N or 1 < gcd(h + 1, N) < N


def success(N, a, Q, crit):
    ok = np.zeros(Q, bool)
    for y in range(1, Q):
        if crit in ("best", "best+2d"):
            d = Fraction(y, Q).limit_denominator(N).denominator
            ok[y] = any(half(N, a, m * d) for m in ((1,) if crit == "best" else (1, 2)))
        else:
            ms = (1,) if crit == "C1" else (1, 2)
            ok[y] = any(pow(a, m * d, N) == 1 and miller(N, a, m * d)
                        for d in conv_dens(y, Q) if d <= N for m in ms)
    return ok


def dist(N, a, Q, j=None):
    e = np.arange(Q)
    v = np.array([pow(a, int(x), N) for x in e]) * 2
    if j is not None:
        v = v + ((e >> j) & 1)
    P = np.zeros(Q)
    for u in np.unique(v):
        P += np.abs(np.fft.fft((v == u).astype(float))) ** 2
    return P / (Q * Q)


print(f"{'N':>4} {'a':>2} {'r':>3} {'t':>3} | " + " | ".join(f"{c:^30}" for c in ("C1", "best", "best+2d", "all+2d")))
print(f"{'':>17}| " + " | ".join(f"{'rand':>6} {'Shor':>6} {'wrong':>6} {'bit':>3}  " for _ in range(4)))
for N, a in ((21, 2), (39, 2), (55, 2), (91, 2), (91, 3), (143, 2)):
    r = order(a, N)
    n = (N - 1).bit_length()
    for t in (5, 6, 7, 8, 2 * n):
        Q = 1 << t
        cells = []
        Ps = dist(N, a, Q)
        wrong = [(j, dist(N, a, Q, j)) for j in range(t) if r % (1 << (j + 1))]
        for crit in ("C1", "best", "best+2d", "all+2d"):
            ok = success(N, a, Q, crit)
            p0 = Ps[ok].sum()
            pw, jb = max((P[ok].sum(), j) for j, P in wrong)
            mark = "*" if pw > p0 + 1e-9 else " "
            cells.append(f"{ok.mean():6.3f} {p0:6.3f} {pw:6.3f}{mark}{jb:>3}  ")
        print(f"{N:>4} {a:>2} {r:>3} {t:>3} | " + " | ".join(cells), flush=True)
print("* = a wrong circuit (genuinely decohered bit) beats perfect Shor under that criterion")
