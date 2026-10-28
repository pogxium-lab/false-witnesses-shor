"""Blind modular-arithmetic test, N = 21: the base a is drawn from public randomness committed in advance.

Purpose. The arithmetic tests of 27 October (a = 2, t = 3) were designed knowing the order of a. Here no design
choice depends on the order r of a or on the factors of N: every rule below takes (N, a) only, the base is drawn
AFTER the pre-registration from a Bitcoin block hash, and the statistic is the visibility of the ideal output
pattern of the generated circuit itself (exact classical simulation of the circuit, no separate use of r).

Draw. The pre-registration fixes a future block height H (current tip + 2 when the file is written; the file is
time-stamped before block H exists). Candidates: the units of Z_21* other than 1 and 20, in increasing order,
    C = [2, 4, 5, 8, 10, 11, 13, 16, 17, 19];   k = int(blockhash(H), 16) mod 10;   a = C[k].
The hash is read from two public explorers (mempool.space, blockstream.info) which must agree; the block header is
fetched and its double SHA-256 must equal the hash (and satisfy its proof-of-work target); block H must have at
least CONFIRMATIONS confirmations. Testability rule (fixed in advance, computed for every candidate at
pre-registration time from the generated circuits by exact simulation): if the ideal output pattern of the drawn
base's true circuit is indistinguishable from uniform (total variation distance to uniform < TESTABLE_TVD), the
base is NOT TESTABLE, the draw is reported and repeated with block H + 1, H + 2, ... (same rule), up to
MAX_REDRAWS blocks. Every draw, including skipped ones, is reported.

Per-base rules (from N and a only):
    L(a) = the generator's lookup-table rule (generator_core.lookup_bits, capped at 2): number of low exponent
           bits handled by a table of UNREDUCED powers (a^(2^L - 1) < N);
    t(a) = L(a) + 1: exactly one controlled modular multiplication after the table, by c = a^(2^L) mod N
           (for N = 21: t = 3 for a = 2, t = 2 for every other candidate);
    line = physical line chosen by hardware.HW.place (package rule) from the calibration snapshot taken at
           pre-registration time; X-X echoes inserted by the IBM dynamic-circuit scheduler as in
           run_ibm_arith_dd.py (min window 1 us).
Circuits (same generator, same two-qubit structure, same line, all with dynamical decoupling):
    true                 the generated circuit;
    classical_exponent   every exponent bit measured in Z right after its preparation;
    decohered_control    the control bit e_L of the multiplication measured in Z right after its preparation;
    mult_by_1            multiplier c replaced by 1;
    wrong_constant       multiplier c' = smallest unit of Z_N* outside the subgroup <a> (if none exists: smallest
                         unit not in {1, c}).
Statistic. Template P_ideal = exact noiseless distribution of the exponent register of the true circuit
(dynamic_sim.register_probabilities on the submitted circuit, deferred measurement). For EVERY circuit the
visibility V is fitted on its counts n_y by maximum likelihood in
    P_y(V) = V * P_ideal_y + (1 - V) / 2^t,   V unconstrained except P_y(V) > 0;
standard error from the observed Fisher information, se(V) = [sum_y n_y d_y^2 / P_y(V)^2]^(-1/2), d = P_ideal - U.
Discrimination (fixed by the exact model of the circuits, not by r): witness w is DISCRIMINATING iff the
visibility of its own ideal pattern on the template, v_w = <P_w - U, d> / <d, d>, is <= DISCRIM_MAX (0.5).
Non-discriminating witnesses (e.g. mult_by_1 when c = 1) are run and reported but do not enter the verdict.
Verdict: SIGNAL iff V_true > 0 AND (V_true - V_w) / sqrt(se_true^2 + se_w^2) >= 3 for EVERY discriminating witness
(at least one is required); otherwise NO SIGNAL.

  Pre-registration (heavy: circuits of all candidates, exactness, noisy predictions; fetches the block tip at the
  END and writes prereg_blind.json, the candidate circuit files and SHA256SUMS.txt; then time-stamps them):
      python src/run_ibm_blind.py --prereg-write --calibration DIR/calibration_prereg.json --outdir DIR
  Plan only (offline, every candidate):   python src/run_ibm_blind.py --plan --calibration CAL.json
  Draw (after block H has its confirmations; writes blind_draw.json + blind_draw_circuits.qpy, time-stamps them):
      python src/run_ibm_blind.py --draw --outdir DIR
  Real device (submits the drawn circuits; job tags: SHA-256 of the pre-registration and of the draw record):
      python src/run_ibm_blind.py --ibm --outdir DIR [--plan] [--yes] [--account NAME]
  Read back a job:  python src/run_ibm_blind.py --ibm --job ID --outdir DIR
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import shutil
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from qiskit import qpy, transpile  # noqa: E402

import dynamic_sim as DS  # noqa: E402
import exact_model as EM  # noqa: E402
import generator_core as GC  # noqa: E402
import prereg_tools as PT  # noqa: E402
import run_ibm as RI  # noqa: E402
import shor_line_generator as LG  # noqa: E402
from hardware import BASIS, HW  # noqa: E402

N = 21
GENOME = replace(GC.BASE_GENOME, m0=1)
CANDIDATES = [u for u in range(2, N - 1) if math.gcd(u, N) == 1]       # Z_21* without 1 and 20
assert CANDIDATES == [2, 4, 5, 8, 10, 11, 13, 16, 17, 19]
BUDGET = 70.0
Z_REQ = 3.0
TESTABLE_TVD = 0.02          # true pattern closer than this to uniform (TVD): base not testable, redraw
DISCRIM_MAX = 0.5            # witness discriminating iff the template visibility of its ideal pattern <= 0.5
CONFIRMATIONS = 2            # block H must be followed by at least one block
HEIGHT_OFFSET = 2            # H = tip + 2 at the time the pre-registration is written
MAX_REDRAWS = 20
MIN_WINDOW = 1e-6
EXACT_TOL = 1e-6
EXPLORERS = {"mempool.space": "https://mempool.space/api", "blockstream.info": "https://blockstream.info/api"}
WITNESSES = ("classical_exponent", "decohered_control", "mult_by_1", "wrong_constant")
# 27 Oct DD runs (a = 2): measured S of the true circuit (mean of the two DD jobs) / standard-model prediction
KAPPA_27OCT = 0.0202 / 0.0345
PREREG, DRAW, DRAW_QPY = "prereg_blind.json", "blind_draw.json", "blind_draw_circuits.qpy"


# =============================================================================== rules (N, a only)
def table_bits(a):
    """L(a): the generator's lookup-table rule (unreduced powers), capped at 2 as in the generator."""
    return min(GC.lookup_bits(N, a, 2), 2)


def exponent_bits(a):
    """t(a) = L(a) + 1: one controlled multiplication after the table."""
    t = table_bits(a) + 1
    assert min(GC.lookup_bits(N, a, t), 2) == t - 1      # the generator uses the same table at this t
    return t


def multiplier(a):
    return pow(a, 1 << table_bits(a), N)


def wrong_constant(a):
    """Smallest unit outside <a>; if none, smallest unit not in {1, c}."""
    units = [u for u in range(2, N) if math.gcd(u, N) == 1]
    sub, x = set(), 1
    for _ in range(N):
        sub.add(x)
        x = x * a % N
    out = [u for u in units if u not in sub]
    if out:
        return out[0], "smallest unit outside <a>"
    return [u for u in units if u != multiplier(a)][0], "no unit outside <a>: smallest unit not in {1, c}"


def options(a):
    L = table_bits(a)
    t = exponent_bits(a)
    return {"true": LG.Opt(semi=1),
            "classical_exponent": LG.Opt(semi=1, decohere=tuple(range(t))),
            "decohered_control": LG.Opt(semi=1, decohere=(L,)),
            "mult_by_1": LG.Opt(semi=1, consts=((L, 1),)),
            "wrong_constant": LG.Opt(semi=1, consts=((L, wrong_constant(a)[0]),))}


def model_P(a, opt):
    return EM.shor_P(N, a, exponent_bits(a), GENOME, order=opt.order, consts=opt.consts, table=opt.table,
                     decohere=opt.decohere)


# =============================================================================== statistic
def template_proj(P, T):
    """Visibility of the pattern P on the template T: <P - U, T - U> / <T - U, T - U> (nan if T = U)."""
    U = np.full(len(T), 1 / len(T))
    d = T - U
    dd = float(d @ d)
    return float((P - U) @ d / dd) if dd > 1e-15 else float("nan")


def tvd_uniform(P):
    return 0.5 * float(np.abs(P - 1 / len(P)).sum())


def fit_visibility(counts, T):
    """Maximum-likelihood V of V T + (1 - V) U on counts (array over y); returns (V, se, n).
    The log-likelihood is concave in V; Newton iterations from V = 0 inside the positivity interval."""
    c = np.asarray(counts, dtype=float)
    n = c.sum()
    U = np.full(len(T), 1 / len(T))
    d = np.asarray(T) - U
    pos, neg = d > 1e-15, d < -1e-15
    lo = max([-U[i] / d[i] for i in np.where(pos)[0]] + [-np.inf])
    hi = min([-U[i] / d[i] for i in np.where(neg)[0]] + [np.inf])
    V = 0.0
    for _ in range(200):
        p = U + V * d
        g = float((c * d / p).sum())
        h = -float((c * d * d / p ** 2).sum())
        step = -g / h
        Vn = V + step
        while not (lo < Vn < hi):                      # stay inside the positivity interval
            step /= 2
            Vn = V + step
        if abs(Vn - V) < 1e-13:
            V = Vn
            break
        V = Vn
    p = U + V * d
    info = float((c * d * d / p ** 2).sum())
    return float(V), float(1 / math.sqrt(info)) if info > 0 else float("inf"), int(n)


def expected_fit(P, T, n):
    """Large-sample V and its standard error for a circuit with true distribution P and n shots."""
    V, _, _ = fit_visibility(np.asarray(P) * 1e6, T)
    U = np.full(len(T), 1 / len(T))
    d = np.asarray(T) - U
    p = U + V * d
    info = n * float((np.asarray(P) * d * d / p ** 2).sum())
    return V, 1 / math.sqrt(info)


def verdict(fits, discriminating):
    vt, st, _ = fits["true"]
    z = {w: (vt - fits[w][0]) / math.hypot(st, fits[w][1]) for w in WITNESSES}
    zd = [z[w] for w in discriminating]
    ok = bool(discriminating) and vt > 0 and min(zd) >= Z_REQ
    return ok, z


# =============================================================================== circuits
def build_candidate(hw, a, line=None, verbose=True):
    """Line (package rule unless given), transpiled circuits with X-X echoes, exact patterns, discrimination."""
    import run_ibm_arith_dd as RD
    t, L = exponent_bits(a), table_bits(a)
    opts = options(a)
    if line is None:
        _, line = hw.place(LG.build(N, a, t, GENOME, opts["true"], measure=True))
    line = [int(q) for q in line]
    info = {}
    for name, opt in opts.items():
        qc = LG.build(N, a, t, GENOME, opt, measure=True)
        qd = transpile(qc, basis_gates=BASIS, optimization_level=0)
        tq0 = transpile(qd, hw.be, optimization_level=2, seed_transpiler=1, initial_layout=list(line))
        tq = RD.add_dd(tq0, hw.tg, hw.be.coupling_map, MIN_WINDOW)
        cz, czc = GC.count_2q(tq)
        lg_dd = hw.esp(RD.strip_delays(tq), dd=True)
        lg_0 = hw.esp(tq0)
        info[name] = dict(tq=tq, tq0=tq0, opt=opt, cz=cz, cz_cond=czc, n_x_added=RD.added_gates(tq, tq0).get("x", 0),
                          duration=DS.makespan(tq, hw) * 1e6, esp=math.exp(lg_dd["total"]),
                          gates1q_factor=math.exp(lg_dd["gates1q"] - lg_0["gates1q"]), P_model=model_P(a, opt))
    # exact noiseless patterns of the circuits themselves (deferred measurement)
    for name, d in info.items():
        p = DS.register_probabilities(d["tq"])
        p0 = DS.register_probabilities(d["tq0"])
        d["P"] = p
        d["dev_model"] = float(np.abs(p - d["P_model"]).max())
        d["dev_without_dd"] = float(np.abs(p - p0).max())
    T = np.clip(info["true"]["P"], 0, None)
    T = T / T.sum()
    cand = dict(a=a, t=t, L=L, multiplier=multiplier(a), wrong_constant=wrong_constant(a), line=line, info=info,
                template=T, tvd_uniform=tvd_uniform(T), testable=tvd_uniform(T) >= TESTABLE_TVD,
                order_of_a_reported_only=next(k for k in range(1, N) if pow(a, k, N) == 1))
    cand["v_ideal"] = {w: template_proj(info[w]["P"], T) for w in WITNESSES}
    cand["discriminating"] = [w for w in WITNESSES if cand["testable"] and cand["v_ideal"][w] <= DISCRIM_MAX]
    if verbose:
        print(f"a={a:2d}  L={L} t={t} c={cand['multiplier']:2d} c'={cand['wrong_constant'][0]:2d}  "
              f"line {line}  TVD(P_ideal, U)={cand['tvd_uniform']:.4f}  "
              f"{'testable' if cand['testable'] else 'NOT TESTABLE'}  P_ideal={np.round(T, 4)}")
        for name, d in info.items():
            vi = "" if name == "true" else f" v_ideal={cand['v_ideal'][name]:+.3f}" + (
                " (discriminating)" if name in cand["discriminating"] else " (non-discriminating)")
            print(f"     {name:19s} CZ={d['cz']:4d}(+{d['cz_cond']}) +X={d['n_x_added']:3d} "
                  f"dur={d['duration']:5.1f}us ESP={d['esp']:.4f} |P-model|={d['dev_model']:.1e} "
                  f"|P-noDD|={d['dev_without_dd']:.1e}{vi}", flush=True)
    return cand


def shot_plan(cand, budget=BUDGET):
    """Largest plan within the budget (run_ibm.qpu_time model): the true circuit gets sqrt(k) times the shots of
    each witness, k = number of discriminating witnesses (at least 1); rounded down to 100 shots."""
    info = cand["info"]
    k = max(len(cand["discriminating"]), 1)
    w = {n: (math.sqrt(k) if n == "true" else 1.0) for n in info}
    per = sum(w[n] * (RI.REP_DELAY + info[n]["duration"] * 1e-6 + 5e-6) for n in info)
    unit = (0.98 * budget - RI.OVERHEAD * len(info)) / per
    plan = {n: int(unit * w[n]) // 100 * 100 for n in info}
    return plan, RI.qpu_time(plan, info)


def dump_circuits(path, cand):
    with open(path, "wb") as fh:
        qpy.dump([cand["info"][k]["tq"] for k in cand["info"]], fh)


def load_circuits(path):
    with open(path, "rb") as fh:
        return dict(zip(["true", *WITNESSES], qpy.load(fh)))


# =============================================================================== noisy predictions
def _noisy_job(args):
    calib, a, line, name, scale, ntraj, seed = args
    import noise_model as NM
    t = exponent_bits(a)
    st = NM.Setup.__new__(NM.Setup)
    st.N, st.t, st.g, st.a = N, t, GENOME, a
    st.hw = HW(calib)
    st.path = tuple(line)
    st.L = table_bits(a)
    r = st.run(replace(options(a)[name], semi=0), scale, ntraj, dd=True, seed=seed, threads=2)
    P = np.clip(r["Py"], 0, None)
    return a, name, (P / P.sum()).tolist()


def noisy_predictions(calib, cands, scale=1.0, ntraj_true=300, ntraj=150, workers=10, seed=11):
    jobs = [(calib, a, c["line"], name, scale, ntraj_true if name == "true" else ntraj, seed)
            for a, c in cands.items() if c["testable"] for name in c["info"]]
    out = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for a, name, P in ex.map(_noisy_job, jobs):
            out.setdefault(a, {})[name] = np.array(P)
            print(f"  noisy model a={a:2d} {name:19s} P={np.round(P, 4)}  [{time.time() - t0:.0f} s]", flush=True)
    return out


def predict(cand, Pn, plan, kappa=1.0, n_mc=4000, seed=5):
    """Expected V of every circuit (standard model; the true circuit times the X-gate factor and kappa), expected
    significance against each witness, and Monte-Carlo probability of the SIGNAL verdict with the plan."""
    T = cand["template"]
    U = np.full(len(T), 1 / len(T))
    P = {}
    for n, p in Pn.items():
        f = cand["info"][n]["gates1q_factor"] * (kappa if n == "true" else 1.0)
        P[n] = U + f * (p - U)
    ev = {n: expected_fit(P[n], T, plan[n]) for n in P}
    z = {w: (ev["true"][0] - ev[w][0]) / math.hypot(ev["true"][1], ev[w][1]) for w in WITNESSES}
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_mc):
        fits = {n: fit_visibility(rng.multinomial(plan[n], P[n] / P[n].sum()), T) for n in P}
        hits += verdict(fits, cand["discriminating"])[0]
    zd = [z[w] for w in cand["discriminating"]]
    return dict(V={n: round(v[0], 5) for n, v in ev.items()}, se={n: round(v[1], 5) for n, v in ev.items()},
                z={w: round(v, 2) for w, v in z.items()}, z_min_discriminating=round(min(zd), 2) if zd else None,
                p_signal=round(hits / n_mc, 4), P={n: [round(float(x), 5) for x in p] for n, p in P.items()})


# =============================================================================== bitcoin
def http_get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "false-witnesses-shor-blind-draw/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode().strip()


def tip_heights():
    out = {}
    for name, base in EXPLORERS.items():
        try:
            out[name] = int(http_get(f"{base}/blocks/tip/height"))
        except Exception as e:
            out[name] = f"error: {type(e).__name__}: {e}"
    return out


def check_header(header_hex, block_hash):
    """Double SHA-256 of the 80-byte header (little-endian) equals the hash; hash <= target of its nBits."""
    h = bytes.fromhex(header_hex)
    if len(h) != 80:
        return False, "header is not 80 bytes"
    dh = hashlib.sha256(hashlib.sha256(h).digest()).digest()[::-1].hex()
    bits = int.from_bytes(h[72:76], "little")
    target = (bits & 0xFFFFFF) << (8 * ((bits >> 24) - 3))
    ok = dh == block_hash and int(block_hash, 16) <= target
    return ok, dict(double_sha256=dh, nbits=f"{bits:08x}", pow_ok=int(block_hash, 16) <= target,
                    time_utc=datetime.datetime.fromtimestamp(int.from_bytes(h[68:72], "little"),
                                                             datetime.timezone.utc).isoformat())


def block_at(height):
    """Hash of block `height` from every explorer (must agree), header check, confirmations."""
    rec = {"height": height, "explorers": {}}
    hashes, tips = set(), []
    for name, base in EXPLORERS.items():
        e = {}
        try:
            e["tip_height"] = int(http_get(f"{base}/blocks/tip/height"))
            tips.append(e["tip_height"])
            if e["tip_height"] >= height:
                e["hash"] = http_get(f"{base}/block-height/{height}")
                e["header"] = http_get(f"{base}/block/{e['hash']}/header")
                ok, det = check_header(e["header"], e["hash"])
                e["header_check"] = det if isinstance(det, dict) else {"error": det}
                e["header_ok"] = ok
                hashes.add(e["hash"])
        except Exception as ex:
            e["error"] = f"{type(ex).__name__}: {ex}"
        rec["explorers"][name] = e
    rec["fetched_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    good = [e for e in rec["explorers"].values() if e.get("header_ok")]
    rec["agree"] = len(good) == len(EXPLORERS) and len(hashes) == 1
    rec["confirmations"] = (min(tips) - height + 1) if tips else 0
    rec["hash"] = hashes.pop() if rec["agree"] else None
    return rec


# =============================================================================== files
def write_json(path, doc):
    Path(path).write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")


def stamp(files, outdir, label, ots_cmd):
    import timestamp_tools as TT
    res = TT.stamp_all([str(f) for f in files], certs=Path(outdir) / "tsa", ots_cmd=ots_cmd)
    log = Path(outdir) / f"timestamps_{label}.json"
    write_json(log, res)
    print(f"Time-stamps written: {log}")
    return res


def candidate_summary(c, plan, tq_est):
    info = c["info"]
    return dict(
        a=c["a"], t=c["t"], L=c["L"], multiplier_c=c["multiplier"],
        wrong_constant=dict(value=c["wrong_constant"][0], rule=c["wrong_constant"][1]),
        order_of_a=c["order_of_a_reported_only"], physical_line=c["line"],
        tvd_ideal_from_uniform=round(c["tvd_uniform"], 6), testable=bool(c["testable"]),
        template_P_ideal=[round(float(x), 6) for x in c["template"]],
        witness_template_visibility={w: (None if math.isnan(v) else round(v, 4)) for w, v in c["v_ideal"].items()},
        discriminating_witnesses=c["discriminating"],
        non_discriminating_witnesses=[w for w in WITNESSES if w not in c["discriminating"]] if c["testable"] else [],
        circuits={k: dict(cz=d["cz"], cz_conditional=d["cz_cond"], x_added=d["n_x_added"],
                          duration_us=round(d["duration"], 2), esp=round(d["esp"], 5),
                          ideal_P=[round(float(x), 6) for x in d["P"]],
                          max_abs_dev_exact_model=d["dev_model"], max_abs_dev_without_dd=d["dev_without_dd"])
                  for k, d in info.items()},
        shots=plan, qpu_estimate_s=round(tq_est, 1))


WHAT_IT_TESTS = {
    "order 6, t = 3 (a = 2)": "table of the 4 unreduced powers 1, 2, 4, 8 driven by e0, e1; e2 controls one "
                              "coset-encoded multiplication by 16 (an element of order 3): coherence of e2 with "
                              "e0, e1 through a non-trivial modular multiplication (same logical true circuit as "
                              "the 27 Oct runs; new witnesses, new statistic, line from today's calibration)",
    "order 3, t = 2 (a = 4, 16)": "table {1, a} driven by e0; e1 controls one multiplication by a^2 (16 resp. 4, "
                                  "order 3): coherence between e = 0 and e = 3 requires a^3 = 1 to be realised by "
                                  "the multiplication (non-trivial modular arithmetic)",
    "order 2, t = 2 (a = 8, 13)": "a^2 = 1: the multiplication is by c = 1 (identity on residues, full gate "
                                  "structure kept); the pattern comes from the coherence of e1 surviving the "
                                  "multiplication circuit; mult_by_1 is then the same circuit and is "
                                  "non-discriminating: tests coherence through the multiplication, not modular "
                                  "arithmetic beyond the identity",
    "order 6, t = 2 (a = 5, 10, 11, 17, 19)": "the four values a^0..a^3 are distinct, the exponent register is "
                                              "fully entangled with the work register and the ideal pattern is "
                                              "uniform: nothing can be tested; NOT TESTABLE -> redraw with the "
                                              "next block (rule fixed in advance)",
}


# =============================================================================== phases
def phase_prepare(args):
    outdir = Path(args.outdir)
    hw = HW(args.calibration)
    cands = {}
    for a in CANDIDATES:
        cands[a] = build_candidate(hw, a)
    bad = {a: n for a, c in cands.items() for n, d in c["info"].items()
           if d["dev_model"] > EXACT_TOL or d["dev_without_dd"] > 1e-12}
    if bad:
        sys.exit(f"exactness check FAILED: {bad}")
    plans = {a: shot_plan(c) for a, c in cands.items()}
    print("\nShot plans (testable candidates):")
    for a, c in cands.items():
        if c["testable"]:
            print(f"  a={a:2d}: {plans[a][0]}  -> {sum(plans[a][0].values())} shots, ~{plans[a][1]:.1f} s QPU")
    isa = {}
    if args.live_check:
        from archive_job import service
        be = service(args.account).backend(args.backend)
        for a, c in cands.items():
            if c["testable"]:
                isa[a] = PT.isa_errors({k: d["tq"] for k, d in c["info"].items()}, be.target)
        print("ISA check (live target):", {a: (v or "valid") for a, v in isa.items()})
        if any(isa.values()):
            sys.exit("ISA check failed")
    return hw, cands, plans, isa


def phase_prereg(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / PREREG).exists() and not args.force:
        sys.exit(f"{outdir / PREREG} exists (use --force to overwrite BEFORE any time-stamp)")
    hw, cands, plans, isa = phase_prepare(args)
    if args.plan:
        return
    cache = Path(args.noisy_cache) if args.noisy_cache else None
    if cache and cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        if raw["calibration_sha256"] != PT.sha256(args.calibration) or raw["trajectories"] != [args.traj_true,
                                                                                              args.traj]:
            sys.exit(f"{cache} was computed with another calibration or trajectory count")
        noisy = {int(a): {k: np.array(v) for k, v in d.items()} for a, d in raw["P"].items()}
        print(f"\nNoisy predictions read from {cache}")
    else:
        print(f"\nNoisy predictions (standard model with DD, scale 1; {args.traj_true}/{args.traj} trajectories):")
        noisy = noisy_predictions(args.calibration, cands, ntraj_true=args.traj_true, ntraj=args.traj,
                                  workers=args.workers)
        if cache:
            write_json(cache, dict(calibration_sha256=PT.sha256(args.calibration),
                                   trajectories=[args.traj_true, args.traj],
                                   P={str(a): {k: v.tolist() for k, v in d.items()} for a, d in noisy.items()}))
            print(f"Noisy predictions cached: {cache}")
    if args.noisy_only:
        return
    preds = {}
    for a, c in cands.items():
        if not c["testable"]:
            continue
        preds[a] = {"standard_model": predict(c, noisy[a], plans[a][0]),
                    "standard_model_x_kappa_27oct": predict(c, noisy[a], plans[a][0], kappa=KAPPA_27OCT)}
        for k, p in preds[a].items():
            print(f"  a={a:2d} {k:30s} V_true={p['V']['true']:+.4f}+-{p['se']['true']:.4f}  z={p['z']}  "
                  f"P(SIGNAL)={p['p_signal']:.3f}")
    # candidate circuit files
    files = {}
    for a, c in cands.items():
        if c["testable"]:
            f = outdir / f"prereg_blind_circuits_a{a}.qpy"
            dump_circuits(f, c)
            files[a] = f
    # the block height is fixed at the very end, just before writing and time-stamping
    tips = tip_heights()
    nums = [v for v in tips.values() if isinstance(v, int)]
    if len(nums) < 1:
        sys.exit(f"cannot read the block tip: {tips}")
    tip = max(nums)
    H = tip + HEIGHT_OFFSET
    tip_hash = http_get(f"{EXPLORERS['mempool.space']}/block-height/{tip}")
    doc = {
        "experiment": "Blind modular-arithmetic test, N = 21: base drawn from a future Bitcoin block hash",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "status": "pre-registered before the drawing block exists and before any submission",
        "script": "src/run_ibm_blind.py (package false-witnesses-shor)",
        "objection_addressed": "the design of the 27 Oct arithmetic tests used the order of a = 2; here no design "
                               "choice depends on the order of a or on the factors of N",
        "draw": dict(
            candidates_increasing=CANDIDATES,
            rule="k = int(blockhash(H), 16) mod 10; a = candidates[k] (blockhash = big-endian hex of the block "
                 "hash as displayed by block explorers)",
            block_height_H=H, tip_height_when_written=tip, tip_hash_when_written=tip_hash,
            tip_heights_by_explorer=tips,
            explorers=EXPLORERS,
            agreement="the hash of block H must be identical on both explorers; the 80-byte header is fetched and "
                      "its double SHA-256 must equal the hash and satisfy the proof-of-work target of its nBits",
            confirmations_required=CONFIRMATIONS,
            redraw_rule=f"if the drawn base is NOT TESTABLE (list below, fixed now from the exact simulation of "
                        f"the generated circuits: TVD(P_ideal, uniform) < {TESTABLE_TVD}), the draw is repeated with "
                        f"blocks H+1, H+2, ... (same rule, same confirmations), at most {MAX_REDRAWS} blocks; every "
                        f"draw is reported",
            not_testable=[a for a, c in cands.items() if not c["testable"]],
            testable=[a for a, c in cands.items() if c["testable"]]),
        "rules_from_N_and_a_only": dict(
            genome="replace(BASE_GENOME, m0=1)",
            table_bits="L(a) = min(generator_core.lookup_bits(N, a, 2), 2) (unreduced powers a^(2^L - 1) < N)",
            exponent_bits="t(a) = L(a) + 1 (exactly one controlled multiplication, by c = a^(2^L) mod N); t = 3 "
                          "is NOT one multiplication for every candidate (L(a) = 1 for every a != 2), hence the rule",
            physical_line="hardware.HW.place on the true circuit, calibration snapshot below",
            dynamical_decoupling="X-X echoes, run_ibm_arith_dd.add_dd (IBM dynamic-circuit scheduler, ALAP, "
                                 "min window 1 us), on every circuit",
            wrong_constant="smallest unit of Z_21* outside <a> (fallback: smallest unit not in {1, c})",
            shot_plan="largest plan with estimated QPU time <= 0.98 x 70 s (run_ibm.qpu_time: 2 s per circuit + "
                      "shots x (250 us + duration + 5 us)); true circuit sqrt(k) x the shots of each witness, k = "
                      "number of discriminating witnesses; rounded down to 100"),
        "witnesses": {
            "classical_exponent": "every exponent bit measured in Z right after its preparation",
            "decohered_control": "control bit e_L of the multiplication measured in Z right after its preparation",
            "mult_by_1": "multiplier c replaced by 1",
            "wrong_constant": "multiplier c' (rule above)"},
        "statistic": "V = maximum-likelihood visibility of the template P_ideal (exact noiseless distribution of "
                     "the exponent register of the TRUE circuit, dynamic_sim.register_probabilities on the "
                     "submitted circuit) in the model P_y = V P_ideal_y + (1 - V) / 2^t, fitted on the counts of "
                     "EACH circuit with the same template; V unconstrained except P_y > 0; Newton iterations on the "
                     "concave log-likelihood",
        "standard_error": "se(V) = [sum_y n_y d_y^2 / P_y(V_hat)^2]^(-1/2), d = P_ideal - U (observed Fisher "
                          "information); difference of two independent circuits: sqrt(se1^2 + se2^2)",
        "discrimination": f"witness w is discriminating iff v_w = <P_w - U, d> / <d, d> <= {DISCRIM_MAX}, P_w = exact "
                          "noiseless pattern of the witness circuit (computed from the circuits, not from the "
                          "order); non-discriminating witnesses are run and reported as such and do not enter the "
                          "verdict",
        "verdict_rule": "SIGNAL iff V_true > 0 AND (V_true - V_w) / sqrt(se_true^2 + se_w^2) >= 3 for EVERY "
                        "discriminating witness w (at least one required); otherwise NO SIGNAL",
        "reported": ["every draw (block heights, hashes, k, a), including NOT TESTABLE bases and redraws",
                     "V +- se of every circuit, z against every witness (discriminating or not)",
                     "V_true / se_true (significance against 0)", "raw counts of every register"],
        "what_each_case_can_test": WHAT_IT_TESTS,
        "calibration_snapshot": "calibration_prereg.json", "calibration_sha256": PT.sha256(args.calibration),
        "backend": args.backend,
        "budget_s": BUDGET,
        "candidates": {str(a): candidate_summary(c, *plans[a]) for a, c in cands.items()},
        "candidate_circuit_files": {str(a): dict(file=f.name, sha256=PT.sha256(f),
                                                 order=["true", *WITNESSES]) for a, f in files.items()},
        "live_isa_check": {str(a): (v or "valid") for a, v in isa.items()} if isa else "not run",
        "noisy_predictions": {str(a): p for a, p in preds.items()},
        "prediction_notes": "standard noise model (noise_model.py, calibrated depolarising gate errors, readout, "
                            "T1/T2 idle relaxation with the DD model, feed-forward latency; static circuits on the "
                            "same line; 1q-gate error of the added X gates as a factor). Second variant: V of the "
                            f"true circuit times kappa = {KAPPA_27OCT:.3f} = measured / predicted S of the two 27 Oct "
                            "DD runs (a = 2). P(SIGNAL): Monte Carlo of the verdict with the planned shots.",
        "trajectories": {"true": args.traj_true, "witness": args.traj},
    }
    write_json(outdir / PREREG, doc)
    print(f"\nPre-registration written: {outdir / PREREG}  (block H = {H}, tip {tip})")
    sums, lines = PT.write_sums(outdir, patterns=("prereg_*.json", "prereg_*.qpy", "calibration_prereg.json",
                                                  "context_prereg.json"))
    print("\n".join(lines))
    if args.stamp:
        files_to_stamp = [outdir / PREREG, sums, *files.values()]
        for extra in ("prereg_ramsey.json", "prereg_ramsey_circuits.qpy"):
            if (outdir / extra).exists():
                files_to_stamp.append(outdir / extra)
        stamp(files_to_stamp, outdir, "prereg", args.ots)
        tips2 = tip_heights()
        print(f"Tip after time-stamping: {tips2} (H = {H} must not exist yet)")
        nums2 = [v for v in tips2.values() if isinstance(v, int)]
        if nums2 and max(nums2) >= H:
            print("WARNING: block H already exists at time-stamp time: the pre-registration must be redone")


def phase_draw(args):
    outdir = Path(args.outdir)
    pre_path = outdir / PREREG
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    sums = PT.read_sums(outdir) or {}
    h_pre = PT.sha256(pre_path)
    if sums.get(PREREG) != h_pre:
        sys.exit(f"{PREREG} does not match {PT.SUMS}")
    H0 = pre["draw"]["block_height_H"]
    draws = []
    a = None
    for j in range(MAX_REDRAWS + 1):
        rec = block_at(H0 + j)
        if not rec["agree"]:
            sys.exit(f"block {H0 + j}: explorers disagree or block not available yet: "
                     f"{json.dumps(rec['explorers'], indent=1)[:2000]}")
        if rec["confirmations"] < CONFIRMATIONS:
            sys.exit(f"block {H0 + j} has {rec['confirmations']} confirmation(s); {CONFIRMATIONS} required: wait for "
                     f"block {H0 + j + CONFIRMATIONS - 1}")
        k = int(rec["hash"], 16) % len(CANDIDATES)
        cand_a = CANDIDATES[k]
        testable = pre["candidates"][str(cand_a)]["testable"]
        rec.update(k=k, a=cand_a, testable=testable)
        draws.append(rec)
        print(f"block {rec['height']} hash {rec['hash']} -> k = {k}, a = {cand_a} "
              f"({'testable' if testable else 'NOT TESTABLE, redraw with the next block'})")
        if testable:
            a = cand_a
            break
    if a is None:
        sys.exit("no testable base within the redraw limit")
    pc = pre["candidates"][str(a)]
    cf = pre["candidate_circuit_files"][str(a)]
    qfile = outdir / cf["file"]
    if PT.sha256(qfile) != cf["sha256"] or sums.get(cf["file"]) != cf["sha256"]:
        sys.exit(f"{qfile} differs from the pre-registration")
    stored = load_circuits(qfile)
    # independent rebuild from (N, a), the pre-registered line and the calibration snapshot
    hw = HW(str(outdir / pre["calibration_snapshot"]))
    if PT.sha256(outdir / pre["calibration_snapshot"]) != pre["calibration_sha256"]:
        sys.exit("calibration snapshot differs from the pre-registration")
    cand = build_candidate(hw, a, line=pc["physical_line"])
    same = {k: bool(cand["info"][k]["tq"] == stored[k]) for k in stored}
    print("Rebuilt circuits identical to the pre-registered ones:", same)
    if not all(same.values()):
        sys.exit("rebuilt circuits differ from the pre-registered circuits")
    exact = {}
    for k, tq in stored.items():
        p = DS.register_probabilities(tq)
        exact[k] = dict(P=[round(float(x), 8) for x in p],
                        max_abs_dev_prereg=float(np.abs(p - np.array(pc["circuits"][k]["ideal_P"])).max()),
                        max_abs_dev_model=float(np.abs(p - model_P(a, options(a)[k])).max()))
        print(f"  {k:19s} exact P = {np.round(p, 5)}  |P - model| = {exact[k]['max_abs_dev_model']:.1e}")
        if exact[k]["max_abs_dev_model"] > EXACT_TOL:
            sys.exit(f"{k}: exactness check failed")
    shutil.copyfile(qfile, outdir / DRAW_QPY)
    tsa_time = None
    try:
        import timestamp_tools as TT
        tsa_time = TT.tsa_time(str(pre_path) + ".tsr")
    except Exception as e:
        tsa_time = f"unavailable: {type(e).__name__}: {e}"
    doc = dict(
        experiment="Blind modular-arithmetic test N = 21: draw record",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        prereg_file=PREREG, prereg_sha256=h_pre, prereg_tsa_time=tsa_time,
        draws=draws, a=a, t=pc["t"], L=pc["L"], multiplier_c=pc["multiplier_c"],
        wrong_constant=pc["wrong_constant"], physical_line=pc["physical_line"],
        template_P_ideal=exact["true"]["P"], discriminating_witnesses=pc["discriminating_witnesses"],
        non_discriminating_witnesses=pc["non_discriminating_witnesses"],
        shots=pc["shots"], qpu_estimate_s=pc["qpu_estimate_s"],
        circuits_file=DRAW_QPY, circuits_sha256=PT.sha256(outdir / DRAW_QPY), circuits_order=cf["order"],
        circuits_identical_to_prereg_file=cf["file"], rebuilt_identical=same, exactness=exact,
        predictions=pre["noisy_predictions"].get(str(a)))
    write_json(outdir / DRAW, doc)
    print(f"Draw record written: {outdir / DRAW}  (a = {a}, circuits {DRAW_QPY})")
    if args.stamp:
        stamp([outdir / DRAW, outdir / DRAW_QPY], outdir, "draw", args.ots)


def analyse(counts_by, draw):
    T = np.array(draw["template_P_ideal"])
    T = T / T.sum()
    E = len(T)
    disc = draw["discriminating_witnesses"]
    print(f"\n=== Analysis: blind test N = {N}, a = {draw['a']}, t = {draw['t']}; template P_ideal = {np.round(T, 4)} ===")
    fits = {}
    for k, cnt in counts_by.items():
        c = np.zeros(E)
        for y, n in cnt.items():
            c[int(y)] += n
        fits[k] = fit_visibility(c, T)
        print(f"  {k:19s} V = {fits[k][0]:+.4f} +- {fits[k][1]:.4f}  ({fits[k][2]} shots; "
              f"P = {np.round(c / c.sum(), 4)})")
    ok, z = verdict(fits, disc)
    for w in WITNESSES:
        print(f"    V_true - V_{w:19s} = {fits['true'][0] - fits[w][0]:+.4f} ({z[w]:+.1f} sigma)"
              f"{'' if w in disc else '  [non-discriminating, not in the verdict]'}")
    z0 = fits["true"][0] / fits["true"][1]
    v = (f"SIGNAL: V_true > 0 and exceeds every discriminating witness by >= {Z_REQ:.0f} sigma" if ok else
         f"NO SIGNAL (V_true = {fits['true'][0]:+.4f}; smallest discriminating difference "
         f"{min(z[w] for w in disc):+.1f} sigma)")
    print(f"  V_true / se = {z0:+.1f}")
    print("  VERDICT:", v)
    return dict(a=draw["a"], t=draw["t"], fits={k: dict(V=f[0], se=f[1], n=f[2]) for k, f in fits.items()},
                z=z, z_true_vs_0=z0, discriminating=disc, verdict=v, signal=ok)


def counts_y(pub):
    return {int(k, 2): v for k, v in pub.data.y.get_counts().items()}


def phase_ibm(args):
    outdir = Path(args.outdir)
    from archive_job import archive, service
    svc = service(args.account)
    if args.job:
        draw = json.loads((outdir / DRAW).read_text(encoding="utf-8"))
        meta = json.loads((outdir / f"meta_{args.job}.json").read_text(encoding="utf-8"))
        res = svc.job(args.job).result()
        out = analyse({k: counts_y(r) for k, r in zip(meta["order"], res)}, draw)
        write_json(outdir / f"analysis_{args.job}.json", out)
        archive(args.job, outdir, svc)
        return
    pre_path, draw_path, qfile = outdir / PREREG, outdir / DRAW, outdir / DRAW_QPY
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    draw = json.loads(draw_path.read_text(encoding="utf-8"))
    h_pre, h_draw, h_q = PT.sha256(pre_path), PT.sha256(draw_path), PT.sha256(qfile)
    sums = PT.read_sums(outdir) or {}
    a = draw["a"]
    checks = {"prereg in SHA256SUMS": sums.get(PREREG) == h_pre,
              "draw record refers to this prereg": draw["prereg_sha256"] == h_pre,
              "circuits match draw record": draw["circuits_sha256"] == h_q,
              "circuits match pre-registered candidate file": pre["candidate_circuit_files"][str(a)]["sha256"] == h_q,
              "drawn base pre-registered as testable": pre["candidates"][str(a)]["testable"],
              "time-stamp proofs present": all(Path(str(f) + s).exists() for f in (pre_path, draw_path, qfile)
                                               for s in (".ots", ".tsr"))}
    print("Fingerprint checks:", checks)
    if not all(checks.values()):
        sys.exit("fingerprint check failed")
    stored = load_circuits(qfile)
    order = draw["circuits_order"]
    plan = {k: int(v) for k, v in draw["shots"].items()}
    be = svc.backend(args.backend)
    hw = HW(be)
    line = draw["physical_line"]
    info = {k: dict(duration=DS.makespan(stored[k], hw) * 1e6, cz=GC.count_2q(stored[k])[0]) for k in order}
    tq_est = RI.qpu_time(plan, info)
    for k in order:
        if info[k]["cz"] != pre["candidates"][str(a)]["circuits"][k]["cz"]:
            sys.exit(f"{k}: CZ count differs from the pre-registration")
        dev = float(np.abs(DS.register_probabilities(stored[k]) - np.array(draw["exactness"][k]["P"])).max())
        if dev > EXACT_TOL:
            sys.exit(f"{k}: exactness check failed ({dev:.1e})")
    bad = PT.isa_errors(stored, be.target)
    print("ISA check (live target):", bad or "all circuits valid")
    if bad:
        sys.exit("ISA check failed")
    hw_cal = HW(str(outdir / pre["calibration_snapshot"]))
    print("Gate durations changed since the pre-registration (line):",
          PT.duration_changes(hw_cal.tg, be.target, line) or "none")
    print(f"a = {a}, t = {draw['t']}, line {line}\nPlan {plan} -> {sum(plan.values())} shots, estimated QPU "
          f"{tq_est:.1f} s (budget {BUDGET:.0f} s)")
    if tq_est > BUDGET:
        sys.exit("estimated QPU time exceeds the budget")
    if args.plan:
        return
    if not args.yes and input(f"Submit {len(order)} circuits to {be.name}? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return
    from snapshot_calibration import snapshot
    ctx = snapshot(be, svc, outdir, "before_blind")
    rem = ctx["usage"].get("usage_remaining_seconds")
    print(f"  QPU time available on the account before submission: {rem} s")
    if rem is not None and float(rem) < 1.2 * tq_est:
        sys.exit("not enough QPU time left in the usage window")
    tags = [PT.TAG_PREFIX + h_pre, "draw_sha256:" + h_draw]
    job = PT.submit(be, [(stored[k], None, plan[k]) for k in order], BUDGET, tq_est, tags=tags)
    jid = job.job_id()
    meta = dict(experiment="blind_arith_N21", N=N, a=a, t=draw["t"], order=order, plan=plan, backend=be.name,
                physical_line=line, cz={k: info[k]["cz"] for k in order}, prereg=PREREG, prereg_sha256=h_pre,
                draw=DRAW, draw_sha256=h_draw, job_tags=tags,
                date=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                qpu_estimate_s=tq_est)
    write_json(outdir / f"meta_{jid}.json", meta)
    print(f"Job submitted: {jid} (read back: --ibm --job {jid} --outdir {outdir})")
    res = job.result()
    out = analyse({k: counts_y(r) for k, r in zip(order, res)}, draw)
    write_json(outdir / f"analysis_{jid}.json", out)
    archive(jid, outdir, svc)
    snapshot(be, svc, outdir, "after_blind")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prereg-write", action="store_true", help="phase 1: pre-registration (before block H)")
    ap.add_argument("--draw", action="store_true", help="phase 2: draw the base from block H")
    ap.add_argument("--ibm", action="store_true", help="phase 3: REAL device (submits the drawn circuits)")
    ap.add_argument("--plan", action="store_true", help="print circuits and plans (offline or live), then exit")
    ap.add_argument("--calibration", default=None, help="calibration snapshot (.json) for phase 1 / --plan")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--account", default=None, help="name of a saved Qiskit Runtime account")
    ap.add_argument("--live-check", action="store_true", help="phase 1: ISA check against the live target")
    ap.add_argument("--traj", type=int, default=150, help="phase 1: noisy trajectories per witness")
    ap.add_argument("--traj-true", type=int, default=300, help="phase 1: noisy trajectories per true circuit")
    ap.add_argument("--workers", type=int, default=10, help="phase 1: parallel noisy simulations")
    ap.add_argument("--noisy-cache", default=None, help="phase 1: JSON cache of the noisy model distributions")
    ap.add_argument("--noisy-only", action="store_true", help="phase 1: compute (and cache) the noisy model only")
    ap.add_argument("--no-stamp", dest="stamp", action="store_false", help="do not time-stamp (RFC 3161 + OTS)")
    ap.add_argument("--ots", default="ots", help="OpenTimestamps client command (default: ots)")
    ap.add_argument("--force", action="store_true", help="phase 1: overwrite an existing pre-registration")
    ap.add_argument("--job", default=None, help="with --ibm: read back and analyse a submitted job")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()
    if args.ibm:
        phase_ibm(args)
    elif args.draw:
        phase_draw(args)
    elif args.prereg_write or args.plan:
        if not args.calibration:
            sys.exit("--calibration is required")
        if args.plan and not args.prereg_write:
            phase_prepare(args)
        else:
            phase_prereg(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
