"""Generic Shor circuit generator emitted directly on a LINE of qubits (a path of the heavy-hex lattice).

The circuit is built from (N, a, t, genome) alone, in time poly(log N); its two-qubit gate STRUCTURE does not
depend on the values of the constants (only rotation angles change). Every two-qubit gate acts between neighbours
on the line, so transpilation onto a hardware path adds no routing.

Arithmetic: coset representation of the work register (Zalka; Gidney 2019), exact preparation of the window
state, Draper (Fourier) additions with centred constants, controlled multiplication in two passes
(A += c * sum_i x_i K1_i, controlled swap X <-> A, A -= sum_i x_i K2_i).

Line-specific constructions:
  * Controlled pass without AND gates or ancillas, by the parity identity
        beta * c * x * a = (beta / 2) (x a + c a - (x XOR c) a):
    X crosses A (phases x.a), c crosses X with CX+SWAP (x ^= c), X crosses A again (phases (x XOR c).a).
  * Uncontrolled second pass: pass 1 adds sum_i x_i (K2_i + c (K1_i - K2_i)) instead of c sum_i x_i K1_i;
    pass 2 subtracts sum_i x'_i K2_i without control. For c = 0 this is the identity, for c = 1 the usual
    multiplication. The pending pass 2 is merged with pass 1 of the next multiplication.
  * Controlled swap: X and A are interleaved, then c travels through Fredkin cells built from a CCZ by parity
    (CS = CP(pi/2)); the cell also undoes the flip x ^= c.
  * Dropped tail: after the LAST operation controlled by an exponent bit, everything left acts only on the work
    register (second half of the cells, de-interleaving, QFT(A), uncontrolled final pass). A fixed unitary U on
    the work register, which is never read, does not change the reduced state of the exponent register:
    Tr_W[(I (x) U) rho (I (x) U)^dagger] = Tr_W[rho]. It is therefore not emitted.
  * When the window has 2 indices (m = 1), A = coset(0) is prepared directly: |u> and |u + N> differ only on the
    bits of u XOR (u + N), so one Ry on a pivot plus a fan-out of neighbour CX gates suffice (no window register,
    no mid-circuit measurement for A).
  * Multiplication order 'asc' or 'desc'; 'desc' is the semiclassical order (bit e_{t-1} measured first).
  * Semiclassical mode (semi=True): every exponent bit e_k (k >= L) is measured right after its multiplication,
    in the basis rotated by the bits already measured (classically-controlled phases + H); recycle=True uses a
    single control qubit, reset and reused. The table bits e_0..e_{L-1} are measured last.

False witnesses (controls) are built by the SAME generator, with the same two-qubit gate sequence:
    decohere={k, ...} : e_k measured in the Z basis right after its preparation (decohered exponent bit);
    consts={k: c'}    : multiplier c' instead of a^(2^k) mod N (scrambled constants);
    table=[T0..T3]    : scrambled lookup table.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from qiskit import ClassicalRegister

import generator_core as GC
import line_layout as LN
from coset_arithmetic import window

TWO_PI = 2 * math.pi
theta = LN.theta
reduce_angle = LN.reduce_angle


@dataclass(frozen=True)
class Opt:
    order: str = "desc"        # multiplication order: 'asc' | 'desc'
    drop_tail: int = 1         # 1: drop everything after the last controlled operation
    prepA: str = "direct"      # 'direct' (if S = 2) | 'teleport' (window register + phase correction)
    semi: int = 0              # 1: semiclassical mid-circuit measurements (executable circuit)
    recycle: int = 0           # 1 (with semi): single recycled control qubit
    decohere: tuple = ()       # controls: bits e_k measured in Z right after H
    consts: tuple = ()         # controls: ((k, c'), ...)
    table: tuple = ()          # controls: scrambled table (T(0), T(1), ...)


# =============================================================================== multiplication
def _cell_flipped(ln, c, u, v, post=True):
    """u = x XOR c (flipped), v = a: controlled swap (c; x, a) that also undoes the flip; c moves behind
    (c-u-v -> u-v-c). CCZ(c, D, v) by parity: pi c D v = (pi/2)(D v + c v - (D XOR c) v)."""
    def b1(q):
        q.cx(1, 0)
        q.h(1)
        q.cp(-math.pi / 2, 0, 1)
    ln.u2(u, v, b1)
    ln.swap(min(ln.pos[c], ln.pos[u]), ("cx", c))
    ln.swap(min(ln.pos[c], ln.pos[v]), ("cp", math.pi / 2))
    if post:
        def b2(q):
            q.cp(math.pi / 2, 0, 1)
            q.h(1)
            q.cx(1, 0)
        ln.u2(u, v, b2)


def _cswap(ln, X, A, c, post=True):
    w = len(X)
    lo, hi = LN.segment(ln, list(X) + list(A) + [c])
    seg = ln.L[lo:hi + 1]
    c_left = seg[0] == c
    assert c_left or seg[-1] == c
    body = [l for l in seg if l != c]
    cands = []
    orders = {"A": [A.index(l) for l in body if l in A], "X": [X.index(l) for l in body if l in X]}
    for key in ("A", "X"):
        for rev in (False, True):
            perm = orders[key][::-1] if rev else orders[key]
            cells = []
            for i in perm:
                cells += [X[i], A[i]] if c_left else [A[i], X[i]]
            cand = ([c] + cells) if c_left else (cells + [c])
            cands.append((LN.inversions(seg, cand), cand))
    best = min(cands, key=lambda x: x[0])[1]
    ln.sort_to(ln.L[:lo] + best + ln.L[hi + 1:])
    cells = [l for l in best if l != c]
    pairs = [cells[2 * k:2 * k + 2] for k in range(w)]
    if not c_left:
        pairs = [p[::-1] for p in reversed(pairs)]
    for u, v in pairs:
        assert u in X
        _cell_flipped(ln, c, u, v, post=post)


def mult(ln, N, cconst, w, g, X, A, c, pend, next_side, last):
    """Multiplication by cconst controlled by c (pending pass 2 of the previous multiplication in `pend`).
    Returns the constants of the new pending pass 2 (None if the tail is dropped)."""
    K1, K2 = GC.mult_consts(N, cconst, w, g)
    xi = {l: i for i, l in enumerate(X)}
    aj = {l: j for j, l in enumerate(A)}
    D = [K1[i] - K2[i] for i in range(w)]
    P = [K2[i] + (pend[i] if pend is not None else 0) for i in range(w)]
    psi = [sum(theta(D[i], j, w) for i in range(w)) / 2 for j in range(w)]

    def m_plain(u, v):
        ij = LN.pair_index(xi, aj, u, v)
        return None if ij is None else ("cp", theta(P[ij[0]], ij[1], w) + theta(D[ij[0]], ij[1], w) / 2)

    def m_flip(u, v):
        ij = LN.pair_index(xi, aj, u, v)
        return None if ij is None else ("cp", -theta(D[ij[0]], ij[1], w) / 2)

    def m_cA(u, v):
        o = v if u == c else u
        return ("cp", psi[aj[o]]) if o in aj else None

    # Structure independent of the constants: a zero angle is still emitted as a fused swap (3 CZ), so that it
    # never becomes a cheaper "swap with |0>" (identical gate sequence for the true circuit and its controls).
    def force(f):
        def g_(u, v):
            r = f(u, v)
            if r is not None and r[0] == "cp" and reduce_angle(r[1]) == 0.0:
                return ("cp", 1e-9)
            return r
        return g_
    ln.cross(X, A, force(m_plain))
    ln.cross([c], X, lambda u, v: ("cx", c))
    ln.cross([c], A, force(m_cA))
    ln.cross(X, A, force(m_flip))
    LN.qft_line(ln, A, inverse=True)
    _cswap(ln, X, A, c, post=not last)
    if last:
        return None
    ln.sort_to(LN.deinterleave_order(ln, X, A, c, next_side))
    LN.qft_line(ln, A, inverse=False)
    return [-k for k in K2]


# =============================================================================== direct preparation of A (S = 2)
def prep_two(ln, R, u, v, wts):
    """R (contiguous, |0>) -> wts[0]|u> + wts[1]|v> with neighbour gates (Ry on a pivot + CX fan-out)."""
    w = len(R)
    dl = [b for b in range(w) if ((u ^ v) >> b) & 1]
    pos = {b: ln.pos[R[b]] for b in range(w)}
    piv = dl[0]
    vb = (v >> piv) & 1
    a0, a1 = (wts[0], wts[1]) if vb == 1 else (wts[1], wts[0])
    ln.g1("ry", R[piv], 2 * math.atan2(a1, a0))
    # fan-out: the pivot holds the branch bit; propagate along the line, then clean the non-members
    by_pos = sorted(range(w), key=lambda b: pos[b])
    ip = by_pos.index(piv)
    Dset = set(dl)
    for direction in (1, -1):
        seq = by_pos[ip + direction::direction] if direction == 1 else by_pos[:ip][::-1]
        tgt = list(seq)
        lastD = max([i for i, b in enumerate(tgt) if b in Dset], default=-1)
        tgt = tgt[:lastD + 1]
        prev = piv
        for b in tgt:
            ln.g2("cx", R[prev], R[b])
            prev = b
        chain = [piv] + tgt
        for i in range(len(chain) - 1, 0, -1):
            if chain[i] not in Dset:
                ln.g2("cx", R[chain[i - 1]], R[chain[i]])
    base = u if vb == 1 else v
    for b in range(w):
        if (base >> b) & 1:
            ln.g1("x", R[b])


# =============================================================================== full circuit
def build(N, a, t, g, opt: Opt = Opt(), measure=True, return_line=False):
    """Shor circuit for (N, a) with t exponent bits (uniform exponent superposition) on a line.
    Qubit i of the returned circuit is position i of the line. Classical registers: k1 (window outcome),
    [k2], y (exponent register outcome), [dec] (decohered bits of the controls)."""
    d = GC.derive(N, g)
    w, mq, S, off, M = d["w"], d["mJ"], d["S"], d["off"], d["M"]
    L = min(GC.lookup_bits(N, a, t), 2)
    X = [f"x{i}" for i in range(w)]
    A = [f"a{j}" for j in range(w)]
    Jx = [f"jx{i}" for i in range(mq)]
    direct = opt.prepA == "direct" and S == 2
    Ja = [] if direct else [f"ja{i}" for i in range(mq)]
    ks = list(range(L, t))
    if opt.order == "desc":
        ks = ks[::-1]
    cmap = dict(opt.consts)
    E = {k: f"e{k}" for k in range(t)}
    if opt.semi and opt.recycle:
        ctl_labels = ["c"]
    else:
        ctl_labels = [E[k] for k in ks]
    T = [E[0], "an", E[1]] if L == 2 else [E[0]]
    first_c = ctl_labels[0] if ctl_labels else None
    labels = list(reversed(ctl_labels[1:])) + ([first_c] if first_c else []) + A + Ja + Jx + T + X
    cJ1 = ClassicalRegister(mq, "k1")
    cregs = [cJ1]
    if Ja:
        cJ2 = ClassicalRegister(mq, "k2")
        cregs.append(cJ2)
    cout = ClassicalRegister(t, "y")
    cregs.append(cout)
    cdec = None
    if opt.decohere:
        cdec = ClassicalRegister(len(opt.decohere), "dec")
        cregs.append(cdec)
    ln = LN.Line(labels, cregs)
    for l in Jx + Ja + ["an"] + A + X:
        if l in ln.pos:
            ln.zero.add(l)
    qc = ln.qc
    lab_of = {}                          # exponent bit -> qubit label
    for k in range(t):
        if k < L or not (opt.semi and opt.recycle):
            lab_of[k] = E[k]
        else:
            lab_of[k] = "c"
    # 1. uniform exponent (table bits and controls; with recycling the control is started later)
    started = set()

    def start_exp(k):
        l = lab_of[k]
        ln.g1("h", l)
        if k in opt.decohere:
            qc.measure(ln.pos[l], cdec[list(opt.decohere).index(k)])
        started.add(k)
    for k in range(L):
        start_exp(k)
    if not (opt.semi and opt.recycle):
        for k in ks:
            start_exp(k)
    # 2. A = coset(0)
    if direct:
        u0, v0 = off * N, (off + 1) * N
        prep_two(ln, A, u0, v0, window(g.kind, 2))
        LN.qft_line(ln, A, inverse=False)
    else:
        for l in A:
            ln.g1("h", l)
        for j, l in enumerate(A):
            th = reduce_angle(theta(off * N, j, w))
            if th:
                ln.g1("p", l, th)
        LN.prep_window(ln, Ja, g.kind, S, mq)
        jai = {l: i for i, l in enumerate(Ja)}
        ajj = {l: j for j, l in enumerate(A)}

        def m_JA(u, v):
            jl, al = (u, v) if u in jai else (v, u)
            return ("cp", theta(N << jai[jl], ajj[al], w))
        ln.cross(Ja, A, m_JA)
        LN.iqft_and_measure_window(ln, Ja, cJ2)
        LN.qft_line(ln, A, inverse=True)
        LN.phase_corr(ln, A, cJ2, N, M, [])
        LN.qft_line(ln, A, inverse=False)
    # 3. X = coset(T(e0, e1)): window register and lookup-table controls cross X
    for l in X:
        ln.g1("h", l)
    Tab = list(opt.table) if opt.table else [pow(a, v, N) for v in range(1 << L)]
    cf = GC.anf(Tab)
    for j, l in enumerate(X):
        th = reduce_angle(theta(off * N + cf[0], j, w))
        if th:
            ln.g1("p", l, th)
    LN.prep_window(ln, Jx, g.kind, S, mq)
    if L == 2:
        ln.adj(E[0], "an")
        ln.adj("an", E[1])
        qc.rccx(ln.pos[E[0]], ln.pos[E[1]], ln.pos["an"])
        ln.zero.discard("an")
    ctl = {l: N << i for i, l in enumerate(Jx)}
    ctl[E[0]] = cf[1]
    if L == 2:
        ctl[E[1]] = cf[2]
        ctl["an"] = cf[3]
    xii = {l: i for i, l in enumerate(X)}

    def m_TX(u, v):
        cl, xl = (u, v) if u in ctl else (v, u)
        th = theta(ctl[cl], xii[xl], w)
        return ("cp", th if reduce_angle(th) else 1e-9)
    ln.cross(X, Jx + T, m_TX)
    LN.iqft_and_measure_window(ln, Jx, cJ1)
    comp = [(E[0], cf[1])] + ([(E[1], cf[2])] if L == 2 else [])
    if L == 2:
        for i in range(mq):
            with qc.if_test((cJ1[i], 1)):
                qc.p((-TWO_PI * (1 << i) * cf[3] / (N * M)) % TWO_PI, ln.pos["an"])
        qc.append(GC.rccx_dg(), [ln.pos[E[0]], ln.pos[E[1]], ln.pos["an"]])
        ln.zero.add("an")
    LN.qft_line(ln, X, inverse=True)
    LN.phase_corr(ln, X, cJ1, N, M, comp)
    # 4. multiplications
    pend = None
    meas_order = []                         # semiclassical measurement order (bits k)

    def semi_measure(k):
        """Measure e_k in the basis rotated by the bits already measured (semiclassical order, e_{t-1} first).
        The j-th measurement (j = t-1-k) gives bit y_j."""
        l = lab_of[k]
        j = t - 1 - k
        for jj, _ in enumerate(meas_order):
            angle = -math.pi / (1 << (j - jj))
            with qc.if_test((cout[jj], 1)):
                qc.p(angle, ln.pos[l])
        ln.g1("h", l)
        qc.measure(ln.pos[l], cout[j])
        meas_order.append(k)
    for idx, k in enumerate(ks):
        c = lab_of[k]
        if opt.semi and opt.recycle and k not in started:
            if idx > 0:
                qc.reset(ln.pos[c])
            start_exp(k)
        cconst = cmap.get(k, pow(a, 1 << k, N))
        lo, hi = LN.segment(ln, X + A)
        seg = ln.L[lo:hi + 1]
        a_left = seg[0] in A
        rest = [l for l in ln.L if l != c]
        if a_left:
            p = rest.index(seg[0])
            order = rest[:p] + [c] + rest[p:]
        else:
            p = rest.index(seg[-1])
            order = rest[:p + 1] + [c] + rest[p + 1:]
        ln.sort_to(order)
        last = (idx == len(ks) - 1) and opt.drop_tail
        if idx + 1 < len(ks):
            nl = lab_of[ks[idx + 1]]
            if nl == c:      # recycled control: it ends on the side opposite to its start; A must wait there
                nxt = "R" if ln.pos[c] < ln.pos[X[0]] else "L"
            else:
                nxt = "L" if ln.pos[nl] < ln.pos[X[0]] else "R"
        else:
            nxt = "any"
        pend = mult(ln, N, cconst, w, g, X, A, c, pend, nxt, last)
        if opt.semi and opt.order == "desc" and measure:
            semi_measure(k)
    if pend is not None:
        LN.final_pass(ln, N, w, X, A, pend)
    if measure:
        if opt.semi and opt.order == "desc":
            for k in range(L - 1, -1, -1):
                semi_measure(k)
        else:
            for k in range(t):
                qc.measure(ln.pos[E[k]], cout[k])
    if return_line:
        return qc, ln
    return qc
