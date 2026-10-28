"""Fourth series of hardware runs (ibm_kingston, 28 October 2026): blind arithmetic test and Ramsey / Hahn-echo
characterisation, recomputed from the raw archived data and checked against the analysis printed at run time.

Experiments
    B   blind arithmetic test N = 21: the rules were fixed and time-stamped (RFC 3161) before the Bitcoin block that
        decides the base existed; base a = candidates[int(blockhash, 16) mod 10] among the units 2, 4, 5, 8, 10, 11,
        13, 16, 17, 19 of Z_21, redraw with the next block if the drawn base is NOT TESTABLE (ideal pattern uniform,
        fixed in the pre-registration from the exact simulation of the generated circuits). Statistic: maximum-
        likelihood visibility V of the exact ideal pattern T of the true circuit in V T + (1 - V) U, fitted on every
        circuit; SIGNAL iff V_true > 0 and V_true - V_w >= 3 sigma for every discriminating witness.
        POST HOC (not pre-registered): readout-robust contrast T (component of the ideal deviation orthogonal to the
        constant, the per-bit directions and the witnesses' ideal deviations; here the parity of (y0, y1)), on the
        blind run and, with the same rule, on the two DD runs of the x16 experiment (data of the 27 October series).
    R   Ramsey (artificial detuning 100 kHz, two quadratures) and Hahn echo on qubits 44, 45, 54: idle detuning of
        the day, T2*, T2echo, static ZZ shift of 44 when 45 is in |1>.

This script re-implements both analyses independently of src/run_ibm_blind.py and src/run_ibm_ramsey.py (bounded
scalar maximisation for V instead of Newton iterations; Ramsey and echo fits by a different optimiser with standard
errors from the numerical Hessian of the log-likelihood) and asserts agreement with analysis_<id>.json. It also
checks SHA256SUMS.txt, the SHA-256 recorded in the time-stamp records, the hash chain pre-registration -> draw
record -> submitted circuits -> IBM job tags, the Bitcoin headers of the draw (double SHA-256, proof of work, header
time) and the draw rule itself; if `openssl` is on the PATH, it also verifies the RFC 3161 signatures.

Inputs : data/2026-10-28_ibm_kingston_blind/
Outputs: figures/fig_blind.pdf|png, figures/numbers_blind.json
Run:  python analysis/make_figures_blind.py   (also called at the end of analysis/make_figures.py)
"""
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "2026-10-28_ibm_kingston_blind"
FIG = ROOT / "figures"
JOB_B, JOB_R = "dbgs2e8te5oc738irid0", "dbgs3a9t9uhs73c8s9j0"
WITNESSES = ["classical_exponent", "decohered_control", "mult_by_1", "wrong_constant"]
Z_REQ = 3.0

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def load(name):
    with open(DATA / name, encoding="utf-8") as fh:
        return json.load(fh)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_times(j):
    ts = j["metrics"]["timestamps"]
    return dict(created=ts["created"], running=ts["running"], finished=ts["finished"],
                qpu_s=j["metrics"]["usage"]["qpu_charge_time_seconds"])


# ============================================================================= integrity and time-stamps
def integrity():
    out = {}
    sums = {}
    for ln in (DATA / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            h, f = ln.split(maxsplit=1)
            sums[f.strip().lstrip("*")] = h
    out["sha256sums"] = {f: sha(DATA / f) == h for f, h in sums.items()}
    stamped = {}
    for rec in ("timestamps_prereg.json", "timestamps_draw.json"):
        r = load(rec)
        for f, v in r["files"].items():
            stamped[f] = dict(sha_ok=sha(DATA / f) == v["sha256"], tsa_time=v["rfc3161"]["time"],
                              ots_proof=(DATA / (f + ".ots")).exists())
        for c, h in r["tsa_certificates"].items():
            assert sha(DATA / "tsa" / c) == h, c
    out["stamped"] = stamped
    pre, draw = load("prereg_blind.json"), load("blind_draw.json")
    mb, mr = load(f"meta_{JOB_B}.json"), load(f"meta_{JOB_R}.json")
    chain = dict(
        draw_refers_to_prereg=draw["prereg_sha256"] == sha(DATA / "prereg_blind.json"),
        draw_circuits_hash=draw["circuits_sha256"] == sha(DATA / "blind_draw_circuits.qpy"),
        draw_circuits_equal_prereg_candidate=sha(DATA / "blind_draw_circuits.qpy") == sha(
            DATA / f"prereg_blind_circuits_a{draw['a']}.qpy"),
        prereg_candidate_hashes=all(v["sha256"] == sha(DATA / v["file"]) for v in pre["candidate_circuit_files"].values()),
        submitted_equal_draw=sha(DATA / f"job_{JOB_B}_circuits.qpy") == sha(DATA / "blind_draw_circuits.qpy"),
        tags_blind=sorted(mb["job_tags"]) == sorted([f"prereg_sha256:{sha(DATA / 'prereg_blind.json')}",
                                                     f"draw_sha256:{sha(DATA / 'blind_draw.json')}"]),
        ramsey_submitted_equal_prereg=sha(DATA / f"job_{JOB_R}_circuits.qpy") == sha(DATA / "prereg_ramsey_circuits.qpy"),
        tags_ramsey=mr["job_tags"] == [f"prereg_sha256:{sha(DATA / 'prereg_ramsey.json')}"],
    )
    out["chain"] = chain
    # RFC 3161 signatures (optional: needs openssl)
    ossl = shutil.which("openssl")
    tsa = {}
    if ossl:
        for f in stamped:
            r = subprocess.run([ossl, "ts", "-verify", "-in", str(DATA / (f + ".tsr")), "-data", str(DATA / f),
                                "-CAfile", str(DATA / "tsa" / "cacert.pem"), "-untrusted", str(DATA / "tsa" / "tsa.crt")],
                               capture_output=True, text=True)
            tsa[f] = "Verification: OK" in (r.stdout + r.stderr)
    out["rfc3161_verified"] = tsa if ossl else "openssl not found: not checked"
    assert all(out["sha256sums"].values()), out["sha256sums"]
    assert all(v["sha_ok"] for v in stamped.values()), stamped
    assert all(chain.values()), chain
    if ossl:
        assert all(tsa.values()), tsa
    return out


def header_check(header_hex, block_hash):
    h = bytes.fromhex(header_hex)
    assert len(h) == 80
    dh = hashlib.sha256(hashlib.sha256(h).digest()).digest()[::-1].hex()
    bits = int.from_bytes(h[72:76], "little")
    target = (bits & 0xFFFFFF) * (1 << (8 * ((bits >> 24) - 3)))
    t = datetime.fromtimestamp(int.from_bytes(h[68:72], "little"), tz=timezone.utc)
    return dict(hash_ok=dh == block_hash, pow_ok=int(dh, 16) <= target, time_utc=t.isoformat(),
                prev=h[4:36][::-1].hex())


def check_draw():
    pre, draw = load("prereg_blind.json"), load("blind_draw.json")
    cand = pre["draw"]["candidates_increasing"]
    testable = set(pre["draw"]["testable"])
    H = pre["draw"]["block_height_H"]
    rows, prev_hash, chosen = [], None, None
    for i, d in enumerate(draw["draws"]):
        assert d["height"] == H + i
        hc = [header_check(e["header"], e["hash"]) for e in d["explorers"].values()]
        assert all(x["hash_ok"] and x["pow_ok"] for x in hc), hc
        assert {e["hash"] for e in d["explorers"].values()} == {d["hash"]}, "explorers disagree"
        if prev_hash:
            assert hc[0]["prev"] == prev_hash, "blocks are not consecutive"
        prev_hash = d["hash"]
        k = int(d["hash"], 16) % 10
        a = cand[k]
        assert (k, a, a in testable) == (d["k"], d["a"], d["testable"]), d
        rows.append(dict(height=d["height"], hash=d["hash"], k=k, a=a, testable=a in testable,
                         header_time=hc[0]["time_utc"], confirmations=d["confirmations"]))
        if a in testable:
            chosen = a
            break
    assert chosen == draw["a"]
    # every candidate's status was fixed in the pre-registration from the exact pattern (TVD from uniform >= 0.02)
    tv = {int(a): c["tvd_ideal_from_uniform"] for a, c in pre["candidates"].items()}
    assert {a for a, v in tv.items() if v >= 0.02} == testable
    return dict(H=H, tip_when_written=pre["draw"]["tip_height_when_written"], draws=rows, a=chosen,
                prereg_tsa_time=draw["prereg_tsa_time"], tvd=tv)


# ============================================================================= blind test: visibility fit
def fit_V(counts, T):
    """Independent re-implementation: bounded scalar maximisation of the log-likelihood of V T + (1 - V) U;
    se from the observed Fisher information at the maximum."""
    c = np.asarray(counts, float)
    U = np.full(len(T), 1 / len(T))
    d = np.asarray(T, float) - U
    lo = max([-U[i] / d[i] for i in range(len(d)) if d[i] > 1e-15] + [-1e9])
    hi = min([-U[i] / d[i] for i in range(len(d)) if d[i] < -1e-15] + [1e9])
    eps = 1e-9 * (hi - lo)
    r = minimize_scalar(lambda V: -float((c * np.log(U + V * d)).sum()), bounds=(lo + eps, hi - eps),
                        method="bounded", options=dict(xatol=1e-12))
    V = float(r.x)
    p = U + V * d
    info = float((c * d * d / p ** 2).sum())
    return V, 1 / math.sqrt(info), int(c.sum())


def blind():
    j, meta = load(f"job_{JOB_B}.json"), load(f"meta_{JOB_B}.json")
    draw, pre, rt = load("blind_draw.json"), load("prereg_blind.json"), load(f"analysis_{JOB_B}.json")
    T = np.asarray(draw["template_P_ideal"], float)
    U = np.full(len(T), 1 / len(T))
    d = T - U
    order = meta["order"]
    assert order == draw["circuits_order"]
    counts, P = {}, {}
    for k, circ in zip(order, j["circuits"]):
        c = np.zeros(len(T))
        for y, n in circ["registers"]["y"].items():
            c[int(y, 2)] += n
        counts[k] = c
        P[k] = c / c.sum()
    # discrimination from the exact patterns of the witnesses (draw record), v_w = <P_w - U, d> / <d, d> <= 0.5
    proj = {w: float((np.asarray(draw["exactness"][w]["P"]) - U) @ d / (d @ d)) for w in WITNESSES}
    disc = [w for w in WITNESSES if proj[w] <= 0.5]
    assert disc == draw["discriminating_witnesses"]
    fits = {k: fit_V(counts[k], T) for k in order}
    z = {w: (fits["true"][0] - fits[w][0]) / math.hypot(fits["true"][1], fits[w][1]) for w in WITNESSES}
    signal = fits["true"][0] > 0 and min(z[w] for w in disc) >= Z_REQ
    # agreement with the analysis printed at run time
    for k in order:
        assert abs(fits[k][0] - rt["fits"][k]["V"]) < 1e-6, (k, fits[k], rt["fits"][k])
        assert abs(fits[k][1] / rt["fits"][k]["se"] - 1) < 1e-6 and fits[k][2] == rt["fits"][k]["n"]
    for w in WITNESSES:
        assert abs(z[w] - rt["z"][w]) < 1e-4
    assert signal == rt["signal"] and disc == rt["discriminating"]
    # shots per circuit and plan
    assert {k: fits[k][2] for k in order} == meta["plan"] == draw["shots"]
    # x1 witness: its ideal pattern [0.40625, 0.09375, 0.40625, 0.09375] projects to 0 on the template; the measured
    # asymmetry P(y=0) - P(y=2) is what gives it a positive V
    c1 = counts["mult_by_1"]
    n1 = c1.sum()
    p0, p2 = c1[0] / n1, c1[2] / n1
    asym = p0 - p2
    asym_se = math.sqrt((p0 + p2 - asym ** 2) / n1)
    pred = pre["noisy_predictions"][str(draw["a"])]
    p_signal = {int(a): {m: v["p_signal"] for m, v in pr.items()} for a, pr in pre["noisy_predictions"].items()}
    ratio = fits["true"][0] / pred["standard_model"]["V"]["true"]
    kappa = pred["standard_model_x_kappa_27oct"]["V"]["true"] / pred["standard_model"]["V"]["true"]
    return dict(a=draw["a"], t=draw["t"], c=draw["multiplier_c"], wrong=draw["wrong_constant"]["value"],
                template=T.tolist(), line=draw["physical_line"], cz=meta["cz"], plan=meta["plan"],
                shots=int(sum(meta["plan"].values())), times=job_times(j), P={k: v.tolist() for k, v in P.items()},
                exact_P={k: draw["exactness"][k]["P"] for k in order}, projection=proj, discriminating=disc,
                fits={k: dict(V=f[0], se=f[1], n=f[2]) for k, f in fits.items()}, z=z,
                z_true_vs_0=fits["true"][0] / fits["true"][1], signal=signal,
                verdict="SIGNAL" if signal else "NO SIGNAL", predictions=pred, p_signal=p_signal,
                ratio_to_standard_model=ratio, kappa_used=kappa,
                mult_by_1_asym=dict(P0=p0, P2=p2, diff=asym, se=asym_se),
                counts={k: v.tolist() for k, v in counts.items()})


# ============================================================================= post hoc: readout-robust contrast
def contrast_vector(P_true, P_witnesses):
    """POST HOC (not pre-registered). General rule computed from the exact patterns of the generated circuits only
    (no use of the order): u = P_ideal(true) - U; W = span{constant, per-bit directions (-1)^{y_b}, P_ideal(w) - U for
    every witness}; the contrast is the component of u orthogonal to W, scaled to max |w_y| = 1. Any error acting on
    one measured bit alone (readout bias, relaxation) moves P only inside span{constant, (-1)^{y_b}} and so cannot
    change T = sum_y w_y P(y); every witness has ideal T = 0 by construction. Returns None if u lies in W."""
    E = len(P_true)
    t = int(round(math.log2(E)))
    U = np.full(E, 1 / E)
    cols = [np.ones(E)] + [np.array([(-1.0) ** ((y >> b) & 1) for y in range(E)]) for b in range(t)]
    cols += [np.asarray(p, float) - U for p in P_witnesses]
    Q, s, _ = np.linalg.svd(np.array(cols).T, full_matrices=False)
    Q = Q[:, s > 1e-9 * s.max()]
    u = np.asarray(P_true, float) - U
    w = u - Q @ (Q.T @ u)
    if np.abs(w).max() < 1e-9:
        return None
    return w / np.abs(w).max()


def contrast_T(counts, w):
    """T = sum_y w_y P(y) with its multinomial standard error sqrt((sum w^2 P - T^2) / n)."""
    c = np.asarray(counts, float)
    n = c.sum()
    p = c / n
    T = float(p @ w)
    return dict(T=T, se=math.sqrt((float(p @ (w * w)) - T * T) / n), n=int(n))


def robust_contrast(b):
    """Post hoc analysis of the blind run with the readout-robust contrast, and the same rule applied to the two DD
    runs of the x16 experiment (27 October). For a = 16, t = 2 the rule gives the two-bit parity
    T = P(y0 = y1) - P(y0 != y1) (ideal 0.125 for the true circuit, 0 for every witness); for a = 2, t = 3 it gives the
    parity of (y0, y1) (ideal 0.125; S, the pre-registered statistic, has ideal 0.25)."""
    order = ["true"] + WITNESSES
    w = contrast_vector(b["exact_P"]["true"], [b["exact_P"][k] for k in WITNESSES])
    assert w is not None and np.allclose(w, [1, -1, -1, 1], atol=1e-9), w
    ideal = {k: float(np.asarray(b["exact_P"][k]) @ w) for k in order}
    assert abs(ideal["true"] - 0.125) < 1e-9 and all(abs(ideal[k]) < 1e-9 for k in WITNESSES), ideal
    res = {k: contrast_T(b["counts"][k], w) for k in order}
    for k in order:                                   # T = P(y0 = y1) - P(y0 != y1), written out
        p = np.asarray(b["P"][k])
        assert abs(res[k]["T"] - (p[0] + p[3] - p[1] - p[2])) < 1e-12
    tr = res["true"]
    z = {k: (tr["T"] - res[k]["T"]) / math.hypot(tr["se"], res[k]["se"]) for k in WITNESSES}
    out = dict(rule="post hoc, not pre-registered", w=w.tolist(), ideal=ideal, T=res, z_true_vs_witness=z,
               z_true_vs_0=tr["T"] / tr["se"], z_witness_vs_0={k: res[k]["T"] / res[k]["se"] for k in WITNESSES},
               visibility_true=tr["T"] / ideal["true"], visibility_true_se=tr["se"] / ideal["true"])
    # the pre-registered V is half carried by the y1 marginal (template deviation = (P_y1 + parity) / 32): a bias of
    # y1 alone moves V but not T
    d = np.asarray(b["template"]) - 0.25
    assert np.allclose(d, (np.array([1, 1, -1, -1]) + np.array([1, -1, -1, 1])) / 32, atol=1e-12)
    out["y1_marginal"] = {}
    for k in order:
        p, n = np.asarray(b["P"][k]), b["fits"][k]["n"]
        m = float(p[0] + p[1] - p[2] - p[3])
        out["y1_marginal"][k] = dict(value=m, se=math.sqrt((1 - m * m) / n))
    # same rule on the DD runs of the x16 experiment (jobs 2 and 3), recomputed from their raw counts
    dd = {}
    base_a = ROOT / "data" / "2026-10-27_ibm_kingston_arith"
    base_r = ROOT / "data" / "2026-10-27_ibm_kingston_replication"
    for tag, jid, base, pre_name in (("job2", "dbgfu89j4t8s73ac9l00", base_a, "prereg_arith_dd.json"),
                                     ("job3", "dbgi0bkdgr2s73cri510", base_r, "prereg_replication.json")):
        with open(base / f"job_{jid}.json", encoding="utf-8") as fh:
            j = json.load(fh)
        with open(base / f"meta_{jid}.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        with open(base / pre_name, encoding="utf-8") as fh:
            pre = json.load(fh)
        names = [k for k in meta["order"] if k.endswith("_DD")]
        wits = [k for k in names if k != "true_DD"]
        wd = contrast_vector(pre["circuits"]["true_DD"]["ideal_P"], [pre["circuits"][k]["ideal_P"] for k in wits])
        assert wd is not None and np.allclose(wd, [1, -1, -1, 1] * 2, atol=1e-9), wd
        r = {}
        for k, circ in zip(meta["order"], j["circuits"]):
            if k not in names:
                continue
            c = np.zeros(8)
            for y, n in circ["registers"]["y"].items():
                c[int(y, 2)] += n
            r[k] = contrast_T(c, wd)
            pS = c / c.sum()
            r[k]["S"] = float(pS[0] + pS[4] - pS[2] - pS[6])
        zt = {k: (r["true_DD"]["T"] - r[k]["T"]) / math.hypot(r["true_DD"]["se"], r[k]["se"]) for k in wits}
        dd[tag] = dict(job=jid, T=r, z_true_vs_witness=zt, z_true_vs_0=r["true_DD"]["T"] / r["true_DD"]["se"])
    out["dd_runs_x16"] = dd
    # numbers quoted in the text
    assert abs(tr["T"] - 0.0168) < 5e-4 and abs(tr["se"] - 0.0038) < 1e-4
    assert all(abs(res[k]["T"]) < 0.0055 for k in WITNESSES)
    assert max(abs(v) for v in out["z_witness_vs_0"].values()) < 1.0
    return out

# ============================================================================= Ramsey / echo
def ramsey_family(fam_spec):
    return fam_spec["kind"], fam_spec["probe"], fam_spec["excite"], fam_spec["measure"]


def p1(counts, measure, q, post=None):
    j = measure.index(q)
    jp = measure.index(post) if post is not None else None
    n = k = 0
    for key, v in counts.items():
        b = key[::-1]                                    # bit j of the register = qubit measure[j]
        if jp is not None and b[jp] != "1":
            continue
        n += v
        k += v * (b[j] == "1")
    return k, n


def nll_binom(m, k, n):
    m = np.clip(m, 1e-9, 1 - 1e-9)
    return -float((k * np.log(m) + (n - k) * np.log(1 - m)).sum())


def hess_se(f, x, rel=1e-4):
    """Standard errors from the numerical Hessian of f (negative log-likelihood) at x."""
    x = np.asarray(x, float)
    k = len(x)
    h = rel * np.maximum(np.abs(x), 1e-12)
    H = np.zeros((k, k))
    for i in range(k):
        for jj in range(k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[jj] = h[i], h[jj]
            H[i, jj] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[jj])
    try:
        return np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


def fit_ramsey(tau, th, k, n, f_art):
    p = k / n

    def model(x):
        B, A, T2, dl, ph = x
        return B + A * np.exp(-tau / T2) * np.cos(2 * math.pi * (f_art + dl) * tau + th + ph)
    # start: scan delta on a 0.25 kHz grid with a weighted linear fit of (B, A cos ph, -A sin ph) at T2 = 100 us
    best = None
    for dl in np.arange(-60e3, 60.001e3, 250.0):
        ph = 2 * math.pi * (f_art + dl) * tau + th
        dec = np.exp(-tau / 100e-6)
        X = np.stack([np.ones_like(tau), dec * np.cos(ph), dec * np.sin(ph)], 1)
        w = np.sqrt(n / np.clip(p * (1 - p), 0.02, None))
        cc, *_ = np.linalg.lstsq(X * w[:, None], p * w, rcond=None)
        r = float((((X @ cc) - p) * w) ** 2 @ np.ones_like(p))
        if best is None or r < best[0]:
            best = (r, dl, cc)
    _, dl, cc = best
    x0 = np.array([cc[0], math.hypot(cc[1], cc[2]), 100e-6, dl, math.atan2(-cc[2], cc[1])])
    sc = np.array([1, 1, 1e-4, 1e3, 1])
    f = lambda z: nll_binom(model(z * sc), k, n)
    r = minimize(f, x0 / sc, method="Powell", options=dict(xtol=1e-10, ftol=1e-12, maxiter=200000))
    r = minimize(f, r.x, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=200000))
    x = r.x * sc
    se = hess_se(f, r.x) * sc
    if x[1] < 0:
        x[1], x[4] = -x[1], x[4] + math.pi
    return dict(B=x[0], A=x[1], T2star=x[2], delta=x[3], delta_se=se[3], T2star_se=se[2], nll=r.fun)


def fit_echo(tau, th, k, n):
    def model(x):
        B, C, T2 = x
        return B - np.cos(th) * (C / 2) * np.exp(-tau / T2)
    sc = np.array([1, 1, 1e-4])
    f = lambda z: nll_binom(model(z * sc), k, n)
    best = min((minimize(f, np.array([0.5, 0.95, T]) / sc, method="Nelder-Mead",
                         options=dict(xatol=1e-10, fatol=1e-12, maxiter=200000)) for T in (60e-6, 150e-6, 300e-6)),
               key=lambda r: r.fun)
    x = best.x * sc
    se = hess_se(f, best.x) * sc
    # profile likelihood of T2echo (B, C re-optimised): 68 % interval where 2 [nll_prof - nll_min] <= 1
    def prof(T2):
        g = lambda z: nll_binom(B_C_model(z, T2), k, n)
        return min(minimize(g, z0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12)).fun
                   for z0 in ([x[0], x[1]], [0.5, 0.9]))

    def B_C_model(z, T2):
        return z[0] - np.cos(th) * (z[1] / 2) * np.exp(-tau / T2)
    grid = x[2] * np.linspace(0.6, 1.6, 101)
    pl = np.array([prof(T) for T in grid])
    inside = grid[2 * (pl - best.fun) <= 1.0]
    return dict(B=x[0], C=x[1], T2echo=x[2], T2echo_se=se[2], nll=best.fun,
                T2echo_profile_68=[float(inside.min()), float(inside.max())])


def ramsey():
    j, meta, pre, rt = load(f"job_{JOB_R}.json"), load(f"meta_{JOB_R}.json"), load("prereg_ramsey.json"), \
        load(f"analysis_{JOB_R}.json")
    assert meta["order"] == pre["circuits_order"]
    dt, f_art = pre["dt_s"], pre["artificial_detuning_hz"]
    fams = pre["families"]
    nb = pre["qubits"]["neighbour_on_line"]
    data = {}
    for key, circ in zip(meta["order"], j["circuits"]):
        fam, d_dt = key.split("|")
        spec = fams[fam]
        post = nb if nb in spec["excite"] else None
        for th, cnt in zip(spec["theta"], circ["registers"]["meas"]):
            for q in spec["probe"]:
                k, n = p1(cnt, spec["measure"], q, post)
                data.setdefault((fam, q), []).append((int(d_dt) * dt, th, k, n))
    res = {}
    for (fam, q), rows in data.items():
        tau, th, k, n = (np.array(c, float) for c in zip(*rows))
        key = f"{fam}|q{q}"
        if fams[fam]["kind"] == "ramsey":
            r = fit_ramsey(tau, th, k, n, f_art)
            ref = rt[key]
            assert abs(r["delta"] - ref["delta"]) < max(20.0, 0.05 * ref["delta_se"]), (key, r["delta"], ref["delta"])
            assert abs(r["delta_se"] / ref["delta_se"] - 1) < 0.10, (key, r["delta_se"], ref["delta_se"])
        else:
            r = fit_echo(tau, th, k, n)
            ref = rt[key]
            assert abs(r["T2echo"] / ref["T2echo"] - 1) < 0.01, (key, r["T2echo"], ref["T2echo"])
        r["n_mean"] = float(n.mean())
        res[key] = r
    a, b = res["ramsey_44_45one|q44"], res["ramsey_44_54|q44"]
    zz = a["delta"] - b["delta"]
    zz_se = math.hypot(a["delta_se"], b["delta_se"])
    assert abs(zz - rt["zz_shift_44_by_45"]["value"]) < 40.0
    res["zz_shift_44_by_45"] = dict(value=zz, se=zz_se, calibrated=pre["calibrated_zz_4445_hz"])
    res["times"] = job_times(j)
    res["shots"] = int(sum(sum(sum(c.values()) for c in circ["registers"]["meas"]) for circ in j["circuits"]))
    res["calibrated_T2_us"] = {q: v["T2_us"] for q, v in pre["calibrated"].items()}
    return res


# ============================================================================= figure
def figure(b):
    labels = {"true": "true", "classical_exponent": "classical\nexp.", "decohered_control": "decoh.\ncontrol",
              "mult_by_1": r"$\times1$", "wrong_constant": "wrong\n" + fr"$\times{b['wrong']}$"}
    keys = ["true"] + WITNESSES
    x = np.arange(len(keys))
    V = np.array([b["fits"][k]["V"] for k in keys])
    se = np.array([b["fits"][k]["se"] for k in keys])
    ps = b["predictions"]["standard_model"]["V"]
    pk = b["predictions"]["standard_model_x_kappa_27oct"]["V"]
    fig, ax = plt.subplots(figsize=(3.4, 1.95))
    ax.axhline(0, color=GREY, lw=0.8, zorder=0)
    ax.errorbar(x, V, yerr=se, fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=3)
    ax.scatter(x[:1], V[:1], s=26, color=ORANGE, zorder=4, label="true circuit ($\\pm1\\sigma$)")
    ax.scatter(x[1:], V[1:], s=26, color=BLUE, zorder=4, label="witnesses ($\\pm1\\sigma$)")
    ax.scatter(x + 0.22, [ps[k] for k in keys], marker="D", s=16, facecolor="white", edgecolor=MUTED, zorder=4,
               label="standard model")
    ax.scatter(x[:1] + 0.22, [pk["true"]], marker="v", s=18, facecolor="white", edgecolor=ORANGE, zorder=4,
               label=r"model $\times0.59$")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keys], fontsize=7.5)
    ax.set_ylabel(r"visibility $V$ of the ideal pattern")
    ax.set_xlim(-0.5, len(keys) - 0.3)
    ax.set_ylim(-0.06, 0.235)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="upper right", fontsize=7, handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_blind.{ext}", dpi=300)
    plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return o


def main():
    FIG.mkdir(exist_ok=True)
    integ = integrity()
    dr = check_draw()
    b = blind()
    b["robust_contrast_post_hoc"] = rc = robust_contrast(b)
    r = ramsey()
    figure(b)
    (FIG / "numbers_blind.json").write_text(json.dumps(clean(dict(integrity=integ, draw=dr, blind=b, ramsey=r)),
                                                       indent=1, default=str), encoding="utf-8")
    print("Blind test N=21 (28 Oct): SHA256SUMS", all(integ["sha256sums"].values()), "| stamped files",
          {f: v["tsa_time"] for f, v in integ["stamped"].items()}, "| chain", all(integ["chain"].values()),
          "| RFC 3161", integ["rfc3161_verified"] if isinstance(integ["rfc3161_verified"], str) else all(integ["rfc3161_verified"].values()))
    print(f"  draw: H = {dr['H']} (tip when written {dr['tip_when_written']}, pre-registration signed {dr['prereg_tsa_time']})")
    for row in dr["draws"]:
        print(f"    block {row['height']} {row['hash'][:20]}... header time {row['header_time']} k = {row['k']} "
              f"a = {row['a']} {'testable' if row['testable'] else 'NOT TESTABLE -> redraw'}")
    print(f"  a = {b['a']}, t = {b['t']}, multiplier {b['c']}, wrong constant {b['wrong']}, template {b['template']}, "
          f"CZ {b['cz']}, line {b['line']}")
    print("  projections of the witnesses' exact patterns on the template:", {k: round(v, 6) for k, v in b["projection"].items()})
    for k, f in b["fits"].items():
        print(f"  {k:19s} V = {f['V']:+.4f} +- {f['se']:.4f} ({f['n']} shots)  P = {np.round(b['P'][k], 4)}  "
              f"pred {b['predictions']['standard_model']['V'][k]:+.4f}")
    print("  z:", {k: round(v, 2) for k, v in b["z"].items()}, f"| V_true/se {b['z_true_vs_0']:+.2f} | {b['verdict']}")
    print(f"  measured / standard model {b['ratio_to_standard_model']:.3f}; factor used for the second prediction "
          f"{b['kappa_used']:.3f}; P(SIGNAL) per testable base:", b["p_signal"])
    m1 = b["mult_by_1_asym"]
    print(f"  x1 witness: P(y=0) - P(y=2) = {m1['diff']:+.4f} +- {m1['se']:.4f}")
    print("  POST HOC (not pre-registered) readout-robust contrast w =", np.round(rc["w"], 6),
          "-> T = P(y0 = y1) - P(y0 != y1), ideal", {k: round(v, 6) for k, v in rc["ideal"].items()})
    for k, v in rc["T"].items():
        print(f"    {k:19s} T = {v['T']:+.4f} +- {v['se']:.4f}  ({v['T'] / v['se']:+.2f} sigma from 0)  "
              f"y1 marginal {rc['y1_marginal'][k]['value']:+.4f} +- {rc['y1_marginal'][k]['se']:.4f}")
    print("    T_true - T_w (sigma):", {k: round(v, 2) for k, v in rc["z_true_vs_witness"].items()},
          f"| visibility T_true / 0.125 = {rc['visibility_true']:.3f} +- {rc['visibility_true_se']:.3f}")
    for tag, v in rc["dd_runs_x16"].items():
        print(f"    same rule on x16 DD {tag} ({v['job']}): parity of (y0, y1); T_true = {v['T']['true_DD']['T']:+.4f} +- "
              f"{v['T']['true_DD']['se']:.4f} ({v['z_true_vs_0']:+.2f} sigma from 0); witnesses",
              {k: round(x['T'], 4) for k, x in v['T'].items() if k != 'true_DD'}, "; margins (sigma)",
              {k: round(x, 2) for k, x in v['z_true_vs_witness'].items()})
    t = b["times"]
    print(f"  job {JOB_B}: created {t['created']} running {t['running']} finished {t['finished']} QPU {t['qpu_s']} s, "
          f"shots {b['shots']}")
    print("Ramsey / echo (28 Oct):")
    for k, v in r.items():
        if k.startswith("ramsey"):
            print(f"  {k:22s} delta = {v['delta'] / 1e3:+6.2f} +- {v['delta_se'] / 1e3:.2f} kHz  "
                  f"T2* = {v['T2star'] * 1e6:7.1f} +- {v['T2star_se'] * 1e6:.1f} us")
        elif k.startswith("echo"):
            lo, hi = v["T2echo_profile_68"]
            print(f"  {k:22s} T2echo = {v['T2echo'] * 1e6:6.1f} +- {v['T2echo_se'] * 1e6:.1f} us "
                  f"(profile 68%: {lo * 1e6:.0f}-{hi * 1e6:.0f} us)  B = {v['B']:.3f} C = {v['C']:.3f}")
    z = r["zz_shift_44_by_45"]
    t = r["times"]
    print(f"  ZZ shift of 44 by 45: {z['value'] / 1e3:+.2f} +- {z['se'] / 1e3:.2f} kHz (calibrated {z['calibrated'] / 1e3:.2f} kHz)")
    print(f"  job {JOB_R}: created {t['created']} running {t['running']} finished {t['finished']} QPU {t['qpu_s']} s, "
          f"shots {r['shots']}")


if __name__ == "__main__":
    main()
