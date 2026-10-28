"""Figures, tables and every hardware number of the paper, recomputed from the raw archived data.

Run:  python analysis/make_figures.py   (paths are relative to this file)
Inputs: data/2026-10-27_ibm_kingston/ (raw IBM counts, metadata, calibration snapshots, QPY circuits, run logs)
        data/scaling_results.csv (routed CZ counts of three generator versions, for the scaling figure)
Outputs: figures/*.pdf|png, figures/tab_*.tex, figures/numbers.json
Success rates are recomputed from the IBM counts (criterion C2, fixed before the runs), the AN-code demonstration
from its raw counts, calibration statistics from the backend property snapshots, gate counts from the submitted
QPY circuits. The ideal success, CZ count and calibrated ESP of each circuit, printed before submission, are read
from the run logs (they are recomputed independently by src/verify_package.py).
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = Path(__file__).resolve().parents[1]
D, F = A / "data" / "2026-10-27_ibm_kingston", A / "figures"
F.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
NUM = {}


# ----------------------------------------------------------------------------- criterion C2 ("exact order")
def _primes(n):
    out, p = set(), 2
    while p * p <= n:
        while n % p == 0:
            out.add(p); n //= p
        p += 1
    if n > 1:
        out.add(n)
    return out


def _miller(N, a, M):
    s = (M & -M).bit_length() - 1
    b = pow(a, M >> s, N)
    for _ in range(s + 1):
        c = b * b % N
        if c == 1 and b not in (1, N - 1):
            return True
        b = c
    return False


def _convergent_dens(num, den, qmax):
    out, h0, h1, k0, k1 = [], 0, 1, 1, 0
    while den:
        q = num // den
        h0, h1 = h1, q * h1 + h0
        k0, k1 = k1, q * k1 + k0
        num, den = den, num - q * den
        if k1 > qmax:
            break
        out.append(k1)
    return out


def c2_table(N, a, t):
    """y is a success iff some continued-fraction denominator d <= N of y/2^t IS the order of a
    (a^d = 1, a^(d/p) != 1 for every prime p | d) and Miller's method then splits N.
    Same definition as the analysis printed at run time; implemented a second time here."""
    Q = 1 << t
    ok = np.zeros(Q, bool)
    for y in range(Q):
        for d in _convergent_dens(y, Q, N):
            if d >= 1 and pow(a, d, N) == 1 and all(pow(a, d // p, N) != 1 for p in _primes(d)) and _miller(N, a, d):
                ok[y] = True
                break
    return ok


def load_job(jid):
    job = json.load(open(D / f"job_{jid}.json", encoding="utf-8"))
    meta = json.load(open(D / f"meta_{jid}.json", encoding="utf-8"))
    return job, meta


def analyse(jid):
    job, meta = load_job(jid)
    N, a, t = meta["N"], meta["a"], meta["t"]
    ok = c2_table(N, a, t)
    res = {}
    for i, name in enumerate(meta["order"]):
        c = job["circuits"][i]["registers"]["y"]
        n = sum(c.values())
        k = sum(v for y, v in c.items() if ok[int(y, 2)])
        p = k / n
        res[name] = dict(k=k, n=n, p=p, se=math.sqrt(p * (1 - p) / n),
                         dist={int(y, 2): v / n for y, v in c.items()})
    rand = float(ok.mean())
    v = res["true"]
    for name, r in res.items():
        if name != "true":
            r["z"] = (v["p"] - r["p"]) / math.hypot(v["se"], r["se"])
    v["z_rand"] = (v["p"] - rand) / v["se"]
    ts = job["metrics"]["timestamps"]
    return dict(N=N, a=a, t=t, rand=rand, res=res, ok=ok, meta=meta,
                qpu=job["metrics"]["usage"]["qpu_charge_time_seconds"], ts=ts)


RUN15 = analyse("dbg9qfgiudkc73e424jg")
RUN21 = analyse("dbgb58djm0ks73eg9aig")

# check against the analysis printed at run time (runtime_analysis_*.json)
for jid, S in (("dbg9qfgiudkc73e424jg", RUN15), ("dbgb58djm0ks73eg9aig", RUN21)):
    ref = json.load(open(D / f"runtime_analysis_{jid}.json", encoding="utf-8"))
    for name, r in S["res"].items():
        assert ref[name]["k"] == r["k"] and ref[name]["n"] == r["n"], (jid, name)

# ideal (noiseless) success of each circuit, from the run logs (printed before submission)
def ideal_from_log(fname):
    """Ideal success, calibrated ESP and CZ count per circuit, as printed before submission (run_log_*.txt)."""
    out, esp, cz = {}, {}, {}
    for line in open(D / fname, encoding="utf-8"):
        m = re.match(r"\s+(true|classical_exponent|decohered_e\d|scrambled_constants)\s+CZ=\s*(\d+)"
                     r".*ESP\(calib\.\)=([\d.]+)\s+ideal success=([\d.]+)", line)
        if m:
            cz[m.group(1)] = int(m.group(2)); esp[m.group(1)] = float(m.group(3)); out[m.group(1)] = float(m.group(4))
    return out, esp, cz


ID15, ESP15, CZ15 = ideal_from_log("run_log_shor15.txt")
ID21, ESP21, CZ21 = ideal_from_log("run_log_shor21.txt")
assert len(ID15) == 4 and len(ID21) == 6, (ID15, ID21)

# ----------------------------------------------------------------------------- gate counts of submitted circuits
def qpy_counts(jid):
    try:
        from qiskit import qpy
    except Exception:
        return None
    p = D / f"job_{jid}_circuits.qpy"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        circs = qpy.load(fh)
    out = []
    for c in circs:
        ops = c.count_ops()
        used = {c.find_bit(q).index for inst in c.data for q in inst.qubits}
        out.append(dict(name=c.name, cz=int(ops.get("cz", 0)), qubits=len(used),
                        mid_meas=int(ops.get("measure", 0)), ops={k: int(v) for k, v in ops.items()}))
    return out


QPY = {jid: qpy_counts(jid) for jid in ("dbg9qfgiudkc73e424jg", "dbgb58djm0ks73eg9aig", "dbg9q25jm0ks73eg6hc0")}

# ----------------------------------------------------------------------------- calibration statistics
def calib(fname, path):
    d = json.load(open(D / fname, encoding="utf-8"))
    q = [{p["name"]: p["value"] for p in qq} for qq in d["qubits"]]
    cz = {}
    for g in d["gates"]:
        if g["gate"] == "cz":
            e = [p["value"] for p in g["parameters"] if p["name"] == "gate_error"]
            if e and e[0] < 1:
                cz[tuple(sorted(g["qubits"]))] = e[0]
    edges = [tuple(sorted(e)) for e in zip(path[:-1], path[1:])]
    pe = [cz[e] for e in edges if e in cz]
    med = lambda x: float(np.median(x))
    T1 = [x["T1"] for x in q if "T1" in x]; T2 = [x["T2"] for x in q if "T2" in x]
    ro = [x["readout_error"] for x in q if "readout_error" in x]
    return dict(file=fname, date=d["last_update_date"], n_cz=len(cz), cz_med=med(list(cz.values())),
                cz_path_med=med(pe), cz_path_mean=float(np.mean(pe)), cz_path_max=float(np.max(pe)),
                T1_med=med(T1), T2_med=med(T2), T1_path_med=med([q[i]["T1"] for i in path]),
                T2_path_med=med([q[i]["T2"] for i in path]), ro_med=med(ro),
                ro_path_med=med([q[i]["readout_error"] for i in path]), n_path=len(path), n_edges=len(pe))


CAL15 = calib("calibration_before_N15.json", RUN15["meta"]["path"])
CAL21 = calib("calibration_before_N21.json", RUN21["meta"]["path"])

# ----------------------------------------------------------------------------- AN-code demo (from raw counts)
def an_analysis(jid, w=5, A_code=3):
    """Post-selection of the addition echo on an AN-encoded register: x = low w bits, carries above.
    Filters: none ('raw'), residue 0 mod 3 ('AN'), all carries back to 0 ('carries'), both."""
    job = json.load(open(D / f"job_{jid}.json", encoding="utf-8"))
    meta = json.load(open(D / f"meta_{jid}.json", encoding="utf-8"))
    tot = {k: [0, 0] for k in ("raw", "AN", "carries", "AN+carries")}
    ntot = 0
    for m_, c in zip(meta["circuits"], job["circuits"]):
        for key, n in c["registers"]["meas"].items():
            k = int(key, 2)
            x, anc = k & ((1 << w) - 1), k >> w
            ok, an, ca = x == m_["x0"], x % A_code == 0, anc == 0
            ntot += n
            for name, acc in (("raw", True), ("AN", an), ("carries", ca), ("AN+carries", an and ca)):
                if acc:
                    tot[name][0] += n
                    tot[name][1] += n * ok
    rows = []
    for name, (kept, good) in tot.items():
        p = good / kept
        rows.append((name, p, math.sqrt(p * (1 - p) / kept), kept / ntot))
    return rows


an_rows = an_analysis("dbg9q25jm0ks73eg6hc0")
# consistency with the totals printed at run time
n_checked = 0
for line in open(D / "run_log_an_code.txt", encoding="utf-8"):
    parts = line.split()
    if parts and parts[0] in ("raw", "AN", "carries", "AN+carries") and "success" in line:
        r = [x for x in an_rows if x[0] == parts[0]][0]
        assert abs(r[1] - float(parts[2])) < 6e-4 and abs(r[3] - float(parts[7])) < 6e-3, (r, line)
        n_checked += 1
assert n_checked == 4, n_checked
LAB_AN = {"raw": "no check", "AN": "AN code", "carries": "carry check", "AN+carries": "AN + carry"}

# ============================================================================= FIGURE 1 : N = 15
r15 = RUN15["res"]
Q15 = 1 << RUN15["t"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
ys = np.arange(Q15)
ideal_dist = {y: (0.25 if y % 2 == 0 else 0.0) for y in ys}   # order 4 read with t=3: peaks at 0,2,4,6
meas = [r15["true"]["dist"].get(y, 0) for y in ys]
ax.bar(ys, meas, width=0.62, color=BLUE, label="measured, ibm_kingston", zorder=2)
ax.scatter(ys, [ideal_dist[y] for y in ys], marker="_", s=230, linewidths=2.2, color=INK,
           label="ideal (noiseless)", zorder=3)
ax.axhline(1 / Q15, color=MUTED, lw=1, ls="--", zorder=1, label="uniform 1/8")
for y in (2, 6):
    ax.bar(y, meas[y], width=0.62, color=BLUE, edgecolor=INK, lw=1.0, hatch="////", zorder=2)
ax.set_xticks(ys, [f"{y:03b}" for y in ys])
ax.set_xlabel("measured exponent register $y$")
ax.set_ylabel("probability")
ax.set_ylim(0, 0.27)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.set_title("(a) output distribution, true circuit", loc="left", fontsize=8.5, color=INK)
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, handlelength=1.6,
          columnspacing=1.0)
ax.text(1.0, -0.47, "hatched: $y$ values counted as success (order 4 found)", transform=ax.transAxes,
        ha="right", fontsize=6.5, color=MUTED)

ax = axes[1]
names = ["true", "classical_exponent", "decohered_e2", "scrambled_constants"]
xt = ["true", "classical\nexponent", "decohered\n$e_2$", "scrambled\nconstants"]
xs = np.arange(len(names))
ax.bar(xs, [r15[n]["p"] for n in names], yerr=[2 * r15[n]["se"] for n in names], width=0.6,
       color=[BLUE] + [GREY] * 3, ecolor=INK, capsize=2, zorder=2)
ax.scatter(xs, [ID15[n] for n in names], marker="o", s=26, facecolor="white", edgecolor=INK, lw=1.2, zorder=3,
           label="ideal (noiseless)")
ax.axhline(RUN15["rand"], color=MUTED, lw=1, ls="--", zorder=1, label="uniform random")
for x, n in zip(xs, names):
    ax.text(x, 0.015, f"{r15[n]['p']:.3f}", ha="center", va="bottom", fontsize=7,
            color="white" if n == "true" else INK, zorder=4)
ax.set_xticks(xs, xt, fontsize=7)
ax.set_ylabel("success probability (C2)")
ax.set_ylim(0, 0.55)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.set_title("(b) true circuit vs false witnesses", loc="left", fontsize=8.5, color=INK)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, handlelength=1.6)
fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.33, wspace=0.28)
for ext in ("pdf", "png"):
    fig.savefig(F / f"fig_shor15.{ext}", dpi=220)
plt.close(fig)

# ============================================================================= FIGURE 2 : N = 21
r21 = RUN21["res"]
# the same six circuits with X-X dynamical decoupling, later the same day (pre-registered, time-stamped)
import make_figures_arith  # noqa: E402
T5DD = make_figures_arith.experiment_t5dd()
r21dd = T5DD["res"]
assert T5DD["rand"] == RUN21["rand"] and all(abs(r21dd[n]["ideal"] - ID21[n]) < 6e-5 for n in r21)
names21 = ["true", "classical_exponent", "decohered_e2", "decohered_e3", "decohered_e4", "scrambled_constants"]
xt21 = ["true", "classical\nexponent", "decoh.\n$e_2$", "decoh.\n$e_3$", "decoh.\n$e_4$", "scrambled\nconstants"]
fig, ax = plt.subplots(figsize=(3.45, 2.8))
xs = np.arange(len(names21))
wd = 0.36
ax.bar(xs - wd / 2, [r21[n]["p"] for n in names21], yerr=[2 * r21[n]["se"] for n in names21], width=wd,
       color=[BLUE] + [GREY] * 5, ecolor=INK, capsize=1.5, error_kw=dict(lw=0.8), zorder=2,
       label="no DD ($\\pm2\\sigma$)")
ax.bar(xs + wd / 2, [r21dd[n]["p"] for n in names21], yerr=[2 * r21dd[n]["se"] for n in names21], width=wd,
       color="white", edgecolor=[BLUE] + [MUTED] * 5, hatch="/////", lw=0.8, ecolor=INK, capsize=1.5,
       error_kw=dict(lw=0.8), zorder=2, label="with DD ($\\pm2\\sigma$)")
ax.scatter(xs, [ID21[n] for n in names21], marker="o", s=24, facecolor="white", edgecolor=INK, lw=1.2, zorder=3,
           label="ideal (noiseless)")
ax.axhline(RUN21["rand"], color=MUTED, lw=1, ls="--", zorder=1, label="uniform random 1/16")
ax.set_xticks(xs, xt21, fontsize=6.3)
ax.set_ylabel("success probability (C2)")
ax.set_ylim(0, 0.22)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.5,
          columnspacing=0.8)
fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.30)
for ext in ("pdf", "png"):
    fig.savefig(F / f"fig_shor21.{ext}", dpi=220)
plt.close(fig)

# ============================================================================= FIGURE 3 : AN-code demo
fig, ax = plt.subplots(figsize=(3.45, 2.5))
xs = np.arange(len(an_rows))
ax.bar(xs, [r[1] for r in an_rows], yerr=[2 * r[2] for r in an_rows], width=0.6, color=AQUA, ecolor=INK,
       capsize=2, zorder=2)
for x, r in zip(xs, an_rows):
    ax.text(x, r[1] + 0.03, f"{r[1]:.3f}\nkept {r[3]:.0%}", ha="center", va="bottom", fontsize=6.8, color=INK)
ax.set_xticks(list(xs), [LAB_AN[r[0]] for r in an_rows], fontsize=7)
ax.set_ylabel("success after post-selection")
ax.set_ylim(0, 1.2)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.12)
for ext in ("pdf", "png"):
    fig.savefig(F / f"fig_an.{ext}", dpi=220)
plt.close(fig)

# ============================================================================= FIGURE 4 : the noise wall
# signal retention R = (P - P_rand) / (P_ideal - P_rand)
def retention(S, ideal):
    v = S["res"]["true"]
    den = ideal - S["rand"]
    return (v["p"] - S["rand"]) / den, v["se"] / den


R15, sR15 = retention(RUN15, ID15["true"])
R21, sR21 = retention(RUN21, ID21["true"])
PRED15 = 0.318      # noisy simulation with the day's calibration, logged before submission
Rpred15 = (PRED15 - RUN15["rand"]) / (ID15["true"] - RUN15["rand"])
# 3-sigma detection floor against the exact random baseline with the shots actually used on the true circuit
floor15 = 3 * math.sqrt(RUN15["rand"] * (1 - RUN15["rand"]) / r15["true"]["n"]) / (ID15["true"] - RUN15["rand"])
floor21 = 3 * math.sqrt(RUN21["rand"] * (1 - RUN21["rand"]) / r21["true"]["n"]) / (ID21["true"] - RUN21["rand"])
# effective per-CZ attenuation of the calibrated ESP (gates + readout + idling, from the ESP printed before submission)
k15 = -math.log(ESP15["true"]) / CZ15["true"]
k21 = -math.log(ESP21["true"]) / CZ21["true"]
kap = 0.5 * (k15 + k21)
h15 = math.log(R15) / math.log(ESP15["true"])

fig, ax = plt.subplots(figsize=(7.0, 3.1))
n = np.linspace(100, 4100, 400)
esp = np.exp(-kap * n)
ax.fill_between(n, esp ** 1.0, esp ** 0.55, color=BLUE, alpha=0.13, lw=0, zorder=1,
                label="model: $R=\\mathrm{ESP}^h$, $h\\in[0.55,1]$ (simulations)")
ax.plot(n, esp, color=BLUE, lw=1.6, zorder=2, label="calibrated ESP, $e^{-\\kappa n_{CZ}}$, ibm_kingston")
ax.hlines(floor21, 100, 4100, color=ORANGE, lw=1.4, ls="--", zorder=2,
          label="$3\\sigma$ detection floor (60,000 shots)")
ax.errorbar([CZ15["true"]], [R15], yerr=[[2 * sR15], [2 * sR15]], fmt="o", ms=6, color=INK, capsize=2, zorder=5,
            label="measured, $N=15$")
ax.scatter([CZ15["true"]], [Rpred15], marker="D", s=26, facecolor="white", edgecolor=INK, zorder=5,
           label="pre-run noisy simulation, $N=15$")
up21 = max(R21, 0) + 2 * sR21
ax.scatter([CZ21["true"]], [up21], marker="_", s=140, linewidths=2.2, color=INK, zorder=5,
           label="measured, $N=21$: $2\\sigma$ upper limit")
ax.annotate("", xy=(CZ21["true"], up21 / 4), xytext=(CZ21["true"], up21),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2), zorder=5)
ax.scatter([CZ21["true"]], [ESP21["true"]], marker="s", s=26, facecolor="white", edgecolor=INK, zorder=5,
           label="pre-run ESP, $N=21$")
hist = [(3919, "v1"), (2930, "v2"), (1807, "v3"), (1538, "v4")]
for x, lab in hist:
    ax.axvline(x, color=GREY, lw=0.8, ls=":", zorder=0)
    ax.text(x, 1.6, lab, ha="center", va="bottom", fontsize=6.8, color=MUTED)
ax.text(4100, 1.6, "N=21 generators:", ha="right", va="bottom", fontsize=6.8, color=MUTED, alpha=0)
ax.set_yscale("log")
ax.set_ylim(3e-7, 1.5)
ax.set_xlim(100, 4100)
ax.set_xlabel("two-qubit (CZ) gates on the hardware line, $n_{CZ}$")
ax.set_ylabel("signal retention $R$  /  ESP")
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
ax.legend(fontsize=6.6, loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=1.8)
fig.subplots_adjust(left=0.09, right=0.62, top=0.92, bottom=0.15)
for ext in ("pdf", "png"):
    fig.savefig(F / f"fig_wall.{ext}", dpi=220)
plt.close(fig)

# ============================================================================= FIGURE 5 : scaling
rows = []
for line in open(D.parent / "scaling_results.csv", encoding="utf-8"):
    if line[0].isdigit():
        N_, n_, t_, a_, v1, v2, v3, q_ = (int(x) for x in line.split(","))
        rows.append((N_, n_, v1, v2, v3))
rows = np.array(sorted(rows, key=lambda r: (r[1], r[0])))
fig, ax = plt.subplots(figsize=(3.45, 2.7))
fits = {}
for col, lab, colr, mk in ((2, "v1 (routed by Qiskit)", GREY, "s"), (3, "v2 (routable bricks)", ORANGE, "^"),
                           (4, "v3 line generator", BLUE, "o")):
    sel = rows[:, 1] >= 6
    b, la = np.polyfit(np.log(rows[sel, 1]), np.log(rows[sel, col]), 1)
    fits[lab] = (math.exp(la), b)
    ax.scatter(rows[:, 1], rows[:, col], s=14, marker=mk, color=colr, zorder=3, label=f"{lab}: ${math.exp(la):.0f}\\,n^{{{b:.2f}}}$")
    nn = np.linspace(5, 12.3, 50)
    ax.plot(nn, math.exp(la) * nn ** b, color=colr, lw=1, zorder=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([5, 6, 7, 8, 9, 10, 11, 12], ["5", "6", "7", "8", "9", "10", "11", "12"])
ax.minorticks_off()
ax.set_xlabel("bits $n$ of $N$  ($t=2n$, coherence $C\\geq0.90$)")
ax.set_ylabel("CZ after routing (FakeFez)")
ax.grid(color=GRID, lw=0.6, zorder=0)
ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=1, handlelength=1.2)
fig.subplots_adjust(left=0.19, right=0.97, top=0.97, bottom=0.36)
for ext in ("pdf", "png"):
    fig.savefig(F / f"fig_scaling.{ext}", dpi=220)
plt.close(fig)

# ============================================================================= tables
LAB = {"true": "true circuit", "classical_exponent": "classical exponent", "decohered_e2": "decohered $e_2$",
       "decohered_e3": "decohered $e_3$", "decohered_e4": "decohered $e_4$",
       "scrambled_constants": "scrambled constants"}


def table(S, ID, CZ, names):
    rows = []
    for n in names:
        r = S["res"][n]
        z = "--" if n == "true" else f"{r['z']:+.1f}"
        rows.append(f"{LAB[n]} & {CZ[n]} & {r['n']} & {ID[n]:.3f} & {r['p']:.4f} $\\pm$ {r['se']:.4f} & {z} \\\\")
    zr = S["res"]["true"]["z_rand"]
    rows.append(f"random & -- & -- & {S['rand']:.4f} & {S['rand']:.4f} & {zr:+.1f} \\\\")
    return ("\\begin{tabular}{lrrrrr}\n\\toprule\ncircuit & CZ & shots & ideal & measured & $\\Delta/\\sigma$ \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


(F / "tab_shor15.tex").write_text(table(RUN15, ID15, CZ15, ["true", "classical_exponent", "decohered_e2", "scrambled_constants"]), encoding="utf-8")
def table21():
    """N = 21, t = 5: run without DD and run with DD side by side (measured C2 success and margin of the true
    circuit over each row, in units of the combined standard error)."""
    def cell(r):
        return f"{r['p']:.4f}({round(r['se'] * 1e4):d})"
    short = {"true": "true", "classical_exponent": "classical exp.", "decohered_e2": "decoh.\\ $e_2$", "decohered_e3": "decoh.\\ $e_3$",
             "decohered_e4": "decoh.\\ $e_4$", "scrambled_constants": "scrambled"}
    rows = []
    for n in names21:
        a, b = r21[n], r21dd[n]
        za = "--" if n == "true" else f"{a['z']:+.1f}"
        zb = "--" if n == "true" else f"{b['z']:+.1f}"
        rows.append(f"{short[n]} & {CZ21[n]} & {ID21[n]:.3f} & {cell(a)} & {za} & {cell(b)} & {zb} \\\\")
    rows.append(f"random & -- & {RUN21['rand']:.4f} & {RUN21['rand']:.4f} & {r21['true']['z_rand']:+.1f} & "
                f"{RUN21['rand']:.4f} & {r21dd['true']['z_rand']:+.1f} \\\\")
    return ("\\begin{tabular}{lrrrrrr}\n\\toprule\n & & & \\multicolumn{2}{c}{no DD} & \\multicolumn{2}{c}{with DD} \\\\\n"
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "circuit & CZ & ideal & measured & $\\Delta/\\sigma$ & measured & $\\Delta/\\sigma$ \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


(F / "tab_shor21.tex").write_text(table21(), encoding="utf-8")
(F / "tab_an.tex").write_text(
    "\\begin{tabular}{lrr}\n\\toprule\ncheck & success & kept shots \\\\\n\\midrule\n"
    + "\n".join(f"{LAB_AN[r[0]]} & {r[1]:.3f} $\\pm$ {r[2]:.3f} & {r[3]:.2f} \\\\" for r in an_rows)
    + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def caltab():
    c15, c21 = CAL15, CAL21
    L = [r"\begin{tabular}{lrr}", r"\toprule", r"quantity & $N=15$ run & $N=21$ run \\", r"\midrule",
         f"snapshot (last update, CET) & {c15['date'][11:16]} & {c21['date'][11:16]} \\\\",
         f"CZ error, median, all pairs [$10^{{-3}}$] & {c15['cz_med']*1e3:.2f} & {c21['cz_med']*1e3:.2f} \\\\",
         f"CZ error, median on the line [$10^{{-3}}$] & {c15['cz_path_med']*1e3:.2f} & {c21['cz_path_med']*1e3:.2f} \\\\",
         f"CZ error, worst edge of line [$10^{{-3}}$] & {c15['cz_path_max']*1e3:.2f} & {c21['cz_path_max']*1e3:.2f} \\\\",
         f"$T_1$ median, all / line [$\\mu$s] & {c15['T1_med']:.0f} / {c15['T1_path_med']:.0f} & {c21['T1_med']:.0f} / {c21['T1_path_med']:.0f} \\\\",
         f"$T_2$ median, all / line [$\\mu$s] & {c15['T2_med']:.0f} / {c15['T2_path_med']:.0f} & {c21['T2_med']:.0f} / {c21['T2_path_med']:.0f} \\\\",
         f"readout, median all / line [\\%] & {c15['ro_med']*100:.2f} / {c15['ro_path_med']*100:.2f} & {c21['ro_med']*100:.2f} / {c21['ro_path_med']*100:.2f} \\\\",
         f"qubits on the line & {c15['n_path']} & {c21['n_path']} \\\\", r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


(F / "tab_calib.tex").write_text(caltab(), encoding="utf-8")

# per-bit marginals of the exponent register, N=15: P(y_b = 1) and its standard error, b = 0 (LSB) .. t-1
BITS15 = {}
for name, r in RUN15["res"].items():
    nb = r["n"]
    pb = [sum(v for y, v in r["dist"].items() if (y >> b) & 1) for b in range(RUN15["t"])]
    BITS15[name] = [dict(p1=x, se=math.sqrt(x * (1 - x) / nb)) for x in pb]

# ============================================================================= numbers.json
def strip(S):
    return {k: {kk: vv for kk, vv in v.items() if kk != "dist"} for k, v in S["res"].items()}


NUM = dict(
    N15=dict(rand=RUN15["rand"], res=strip(RUN15), ideal=ID15, esp=ESP15, cz=CZ15, qpu_s=RUN15["qpu"], ts=RUN15["ts"],
             retention=R15, retention_se=sR15, pred=PRED15, pred_retention=Rpred15, floor=floor15, h=h15,
             frac_factor_shots=RUN15["res"]["true"]["p"], path=RUN15["meta"]["path"],
             bit_marginals=BITS15),
    N21=dict(rand=RUN21["rand"], res=strip(RUN21), ideal=ID21, esp=ESP21, cz=CZ21, qpu_s=RUN21["qpu"], ts=RUN21["ts"],
             retention=R21, retention_se=sR21, floor=floor21, path=RUN21["meta"]["path"]),
    N21_DD={k: v for k, v in T5DD.items() if k not in ("times",)} | dict(times={k: str(v) for k, v in T5DD["times"].items()}),
    kappa=dict(k15=k15, k21=k21, mean=kap), calib15=CAL15, calib21=CAL21, an=an_rows, scaling_fits=fits, qpy=QPY)
(F / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float), encoding="utf-8")

print(f"N=15: true {r15['true']['p']:.4f}±{r15['true']['se']:.4f}",
      {n: (round(r15[n]['p'], 4), round(r15[n]['z'], 1)) for n in names[1:]}, f"rand z {r15['true']['z_rand']:+.1f}")
print(f"N=21: true {r21['true']['p']:.4f}±{r21['true']['se']:.4f}",
      {n: (round(r21[n]['p'], 4), round(r21[n]['z'], 1)) for n in names21[1:]}, f"rand z {r21['true']['z_rand']:+.1f}")
print(f"N=21 with DD: true {r21dd['true']['p']:.4f}±{r21dd['true']['se']:.4f}",
      {n: (round(r21dd[n]['p'], 4), round(r21dd[n]['z'], 1)) for n in names21[1:]},
      f"rand z {r21dd['true']['z_rand']:+.1f} {T5DD['verdict']}")
print(f"retention N15 {R15:.3f}±{sR15:.3f} (pred {Rpred15:.3f}, h={h15:.2f}); N21 {R21:.4f}±{sR21:.4f}; "
      f"floors {floor15:.3f} {floor21:.3f}; kappa {k15:.5f} {k21:.5f}")
print("calib15", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in CAL15.items()})
print("calib21", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in CAL21.items()})
print("fits", fits)
for jid, c in QPY.items():
    print(jid, None if c is None else [(x["name"], x["cz"], x["qubits"]) for x in c])

# ============================================================================= second and third series of runs
# one controlled modular multiplication (N = 21, t = 3: diagnostic run without DD, run with DD and its replication),
# the N = 21, t = 5 run with DD and the sine-window Hadamard test: analysis/make_figures_arith.py
make_figures_arith.main()

# ============================================================================= fourth series (28 October 2026)
# blind arithmetic test (base drawn from a Bitcoin block hash) and Ramsey / Hahn-echo characterisation:
# analysis/make_figures_blind.py
import make_figures_blind  # noqa: E402
make_figures_blind.main()
