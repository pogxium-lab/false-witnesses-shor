"""Second and third series of hardware runs (ibm_kingston, 27 October 2026): every number, figure and table of the
paper that concerns them, recomputed from the raw archived data.

Experiments
    M   one non-trivial controlled modular multiplication, N = 21, a = 2, t = 3 (controlled x16 mod 21, order 3),
        true circuit and witnesses on the same 17-qubit line; job 1 without dynamical decoupling (DD, diagnostic
        run), job 2 with X-X echoes inserted in the idle windows (plus the unchanged true circuit), job 3 the
        replication of job 2 with the identical DD circuits (pre-registered and externally time-stamped before
        submission). Pre-registered statistic S = P(y in {0, 4}) - P(y in {2, 6}); ideal 0.25 for the true circuit,
        0 for every witness.
    T5  the N = 21, t = 5 Shor circuit and its five controls with X-X echoes, same line as the run without DD,
        criterion C2, pre-registered and externally time-stamped (the run without DD is analysed in make_figures.py,
        which also calls experiment_t5dd() for fig_shor21 and tab_shor21).
    W   sine versus uniform coset window: Hadamard test of the translation by N on a coset of N = 21
        (M = 8 window indices), <X_c> = sum_j a_j a_{j+1}; pre-registered statistic D = <X>(sine) - <X>(uniform).

Inputs : data/2026-10-27_ibm_kingston_arith/ and data/2026-10-27_ibm_kingston_replication/ (raw counts
         job_<id>.json, submitted circuits job_<id>_circuits.qpy, pre-registration files prereg_*.json written before
         submission, analysis printed at run time analysis_<id>.json, calibration snapshots, SHA256SUMS.txt).
Outputs: figures/fig_arith.pdf|png, figures/fig_sine.pdf|png, figures/tab_arith.tex, figures/numbers_arith.json.

Run:  python analysis/make_figures_arith.py   (also called at the end of analysis/make_figures.py)
"""
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "2026-10-27_ibm_kingston_arith"
DATA_R = ROOT / "data" / "2026-10-27_ibm_kingston_replication"
FIG = ROOT / "figures"

JOB_W, JOB_M1, JOB_M2 = "dbgeesljm0ks73egfmc0", "dbgehntrd2mc73cuiqsg", "dbgfu89j4t8s73ac9l00"
JOB_M3, JOB_T5DD = "dbgi0bkdgr2s73cri510", "dbgi6s8te5oc738i7rr0"
CELLS_A, CELLS_B = (0, 4), (2, 6)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def load(name, base=None):
    with open((base or DATA) / name, encoding="utf-8") as fh:
        return json.load(fh)


def job(jid, base=None):
    return load(f"job_{jid}.json", base), load(f"meta_{jid}.json", base)


def job_times(j):
    ts = j["metrics"]["timestamps"]
    f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dict(running=ts["running"], finished=ts["finished"], qpu_s=j["metrics"]["usage"]["qpu_charge_time_seconds"],
                t_running=f(ts["running"]), t_finished=f(ts["finished"]))


def load_qpy(path):
    from qiskit import qpy
    with open(path, "rb") as fh:
        return qpy.load(fh)


def qpy_counts(jid, base=None):
    """CZ count, number of qubits used and operation counts of each submitted circuit (top level; none of these
    circuits has a CZ inside a conditional block)."""
    circs = load_qpy((base or DATA) / f"job_{jid}_circuits.qpy")
    out = []
    for c in circs:
        ops = c.count_ops()
        used = {c.find_bit(q).index for inst in c.data for q in inst.qubits}
        out.append(dict(cz=int(ops.get("cz", 0)), qubits=len(used), x=int(ops.get("x", 0)),
                        delay=int(ops.get("delay", 0)), if_else=int(ops.get("if_else", 0))))
    return out


# ============================================================================= experiment M (modular multiplication)
def y_counts(circ):
    v = np.zeros(8)
    for k, n in circ["registers"]["y"].items():
        v[int(k, 2)] += n
    return v


def s_stat(c):
    n = c.sum()
    p = c / n
    pa, pb = p[list(CELLS_A)].sum(), p[list(CELLS_B)].sum()
    return float(pa - pb), float(math.sqrt((pa + pb - (pa - pb) ** 2) / n)), int(n)


def phase_fit(c):
    """Exponent register of the true circuit: rho = (1/8)[I + v e^{i phi}|0><6| + v e^{i phi}|1><7| + h.c.]
    (states e and e' carry the same work-register state iff 2^e = 2^e' mod 21, i.e. e' = e + 6), read by the
    Fourier measurement: P(y) = 1/8 + (v/16) cos(3 pi y / 2 - phi). Witnesses have no such coherence.
    Maximum likelihood in (x, z) = (v cos phi, v sin phi); P is linear in (x, z), so the observed information
    is sum_y c_y g g^T / P_y^2 exactly. Returns v, phi (deg), x and their standard errors."""
    y = np.arange(8)
    G = np.vstack([np.cos(1.5 * np.pi * y), np.sin(1.5 * np.pi * y)]).T / 16
    th = np.zeros(2)
    for _ in range(100):                           # Newton iterations on the concave log-likelihood
        P = 1 / 8 + G @ th
        grad = G.T @ (c / P)
        H = (G * (c / P ** 2)[:, None]).T @ G
        step = np.linalg.solve(H, grad)
        th = th + step
        if np.abs(step).max() < 1e-13:
            break
    P = 1 / 8 + G @ th
    cov = np.linalg.inv((G * (c / P ** 2)[:, None]).T @ G)
    x, z = th
    v = math.hypot(x, z)
    gv, gp = np.array([x, z]) / v, np.array([-z, x]) / v ** 2
    two_lnLR = 2 * float((c * np.log(P * 8)).sum())
    return dict(v=v, v_se=float(math.sqrt(gv @ cov @ gv)), phase_deg=math.degrees(math.atan2(z, x)),
                phase_se_deg=math.degrees(math.sqrt(gp @ cov @ gp)), x=float(x), x_se=float(math.sqrt(cov[0, 0])),
                two_lnLR=two_lnLR)


def ideal_S_by_multiplier(N=21, a=2):
    """Ideal S when the controlled multiplication (control e2) uses the constant c instead of a^4 mod N: the work
    register holds a^(e0 + 2 e1) c^(e2) mod N (the encoding depends only on the residue), so exponent states
    e, e' are coherent iff they lead to the same residue; P(y) is their Fourier transform."""
    F = np.exp(2j * np.pi * np.outer(np.arange(8), np.arange(8)) / 8) / math.sqrt(8)
    out = {}
    for c in (x for x in range(1, N) if math.gcd(x, N) == 1):
        w = [pow(a, e & 3, N) * pow(c, e >> 2, N) % N for e in range(8)]
        rho = np.array([[1 / 8 if w[e] == w[f] else 0.0 for f in range(8)] for e in range(8)])
        P = np.real(np.diag(F @ rho @ F.conj().T))
        out[c] = float(P[0] + P[4] - P[2] - P[6])
    return out


def experiment_m():
    out = {}
    out["ideal_S_by_multiplier"] = ideal_S_by_multiplier()
    assert abs(out["ideal_S_by_multiplier"][16] - 0.25) < 1e-12
    pre1, pre2 = load("prereg_arith.json"), load("prereg_arith_dd.json")
    pred1 = pre1["noisy_predictions"]["1.0"]["S"]
    pred2 = pre2["noisy_predictions_standard_model"]["1.0"]["S"]
    runs = {}
    for tag, jid, pred in (("run1", JOB_M1, pred1), ("run2", JOB_M2, pred2)):
        j, meta = job(jid)
        q = qpy_counts(jid)
        res = {}
        for i, name in enumerate(meta["order"]):
            c = y_counts(j["circuits"][i])
            s, se, n = s_stat(c)
            res[name] = dict(S=s, se=se, n=n, P=(c / n).tolist(), P_y0_0=float(c[::2].sum() / n),
                             P_bit1=[float(c[(np.arange(8) >> b) & 1 == 1].sum() / n) for b in range(3)],
                             cz=q[i]["cz"], qubits=q[i]["qubits"], x_gates=q[i]["x"], pred=pred[name],
                             esp=(pre1 if tag == "run1" else pre2)["circuits"][name]["esp"],
                             duration_us=(pre1 if tag == "run1" else pre2)["circuits"][name]["duration_us"],
                             ideal_S=(pre1 if tag == "run1" else pre2)["circuits"][name]["ideal_S"])
            res[name]["counts"] = c
        rt = load(f"analysis_{jid}.json")["S"]
        for name, r in res.items():                # check against the analysis printed at run time
            assert abs(r["S"] - rt[name]["S"]) < 1e-12 and abs(r["se"] - rt[name]["se"]) < 1e-12, name
            assert abs(r["P_y0_0"] - rt[name]["P_y0_0"]) < 1e-12, name
        runs[tag] = dict(job=jid, res=res, meta=meta, times=job_times(j),
                         shots=int(sum(r["n"] for r in res.values())))
    r1, r2 = runs["run1"]["res"], runs["run2"]["res"]

    def z(a, b):
        return (a["S"] - b["S"]) / math.hypot(a["se"], b["se"])
    out["z_run1"] = {w: z(r1["true"], r1[w]) for w in r1 if w != "true"}
    out["z_run2"] = {w: z(r2["true_DD"], r2[w]) for w in ("mult_by_1_DD", "wrong_constant_DD", "classical_exponent_DD")}
    out["z_DD_vs_noDD"] = z(r2["true_DD"], r2["true_noDD"])
    out["z_noDD_vs_run1"] = z(r2["true_noDD"], r1["true"])
    rt2 = load(f"analysis_{JOB_M2}.json")
    for w, v in out["z_run2"].items():
        assert abs(v - rt2["z_primary"][w]) < 1e-9
    assert abs(out["z_DD_vs_noDD"] - rt2["z_secondary"]) < 1e-9
    out["verdict_run1"] = "SIGNAL" if min(out["z_run1"].values()) >= 3 else "NO SIGNAL"
    out["verdict_run2"] = "SIGNAL" if (r2["true_DD"]["S"] > 0 and min(out["z_run2"].values()) >= 3) else "NO SIGNAL"
    # coherence fits of the true circuits; check against the run-time fits (numerical optimiser there)
    out["fit"] = {"run1_true": phase_fit(r1["true"]["counts"]), "run2_true_DD": phase_fit(r2["true_DD"]["counts"]),
                  "run2_true_noDD": phase_fit(r2["true_noDD"]["counts"])}
    for k, name in (("run2_true_DD", "true_DD"), ("run2_true_noDD", "true_noDD")):
        ref = rt2["phase_fit"][name]
        assert abs(out["fit"][k]["v"] - ref["v"]) < 2e-4 and abs(out["fit"][k]["phase_deg"] - ref["phase_deg"]) < 0.2, k
    ref1 = pre2["previous_run"]["phase_fit_true"]
    assert abs(out["fit"]["run1_true"]["phase_deg"] - ref1["phase_deg"]) < 0.2
    # mult_by_1 senses the e2 coherence only through P(y0 = 0) = (1 + v cos phi_2) / 2
    out["mult_by_1_x"] = {"run1": 2 * r1["mult_by_1"]["P_y0_0"] - 1, "run2": 2 * r2["mult_by_1_DD"]["P_y0_0"] - 1}
    out["mult_by_1_pred_P_y0_0"] = {
        "run1": float(sum(pre1["noisy_predictions"]["1.0"]["P"]["mult_by_1"][0::2])),
        "run2": float(sum(pre2["noisy_predictions_standard_model"]["1.0"]["P"]["mult_by_1_DD"][0::2]))}
    out["ratio_measured_to_model"] = {"run1_true": r1["true"]["S"] / r1["true"]["pred"],
                                      "run2_true_DD": r2["true_DD"]["S"] / r2["true_DD"]["pred"]}
    # coherent-error model stored with the second pre-registration (exact simulation of the submitted circuits)
    coh = pre2["coherent_error_model"]["S"]
    comb = pre2["combined_predictions"]
    out["coherent_model"] = {
        "noiseless_true_noDD": {k: [v["true_noDD|asap"]["S"], v["true_noDD|ibm_schedule"]["S"]] for k, v in coh.items()},
        "noiseless_true_DD": {k: v["true_DD"]["S"] for k, v in coh.items()},
        "noiseless_witness_DD_max_abs": max(abs(v[w]["S"]) for v in coh.values()
                                            for w in ("mult_by_1_DD", "wrong_constant_DD", "classical_exponent_DD")),
        "combined_true_noDD": {k: v["S"]["true_noDD"] for k, v in comb.items()},
        "combined_true_DD": {k: v["S"]["true_DD"] for k, v in comb.items()}}
    # calibration: static ZZ on the line, age of T1/T2 of the qubit that holds e1 (44)
    cal = load("calibration_before_arith.json")
    gen = {g["name"]: g["value"] for g in cal["general"]}
    line = runs["run1"]["meta"]["physical_line"]
    zz = {}
    for a, b in zip(line[:-1], line[1:]):
        v = gen.get(f"zz_{a}{b}", gen.get(f"zz_{b}{a}"))
        if v:
            zz[f"{a}-{b}"] = v * 1e6            # GHz -> kHz
    out["zz_khz_line"] = dict(min=min(zz.values()), max=max(zz.values()), n=len(zz), n_edges=len(line) - 1)
    q44 = {p["name"]: p for p in cal["qubits"][44]}
    t_cal = datetime.fromisoformat(q44["T2"]["date"])
    out["q44"] = dict(T1_us=q44["T1"]["value"], T2_us=q44["T2"]["value"], T2_date=q44["T2"]["date"],
                      age_h_at_run1=(runs["run1"]["times"]["t_running"] - t_cal).total_seconds() / 3600)
    out["hours_between_runs"] = (runs["run2"]["times"]["t_running"] - runs["run1"]["times"]["t_running"]).total_seconds() / 3600
    out["idle_hypothesis"] = pre2["hypothesis"]
    # exactness of the submitted circuits (noiseless exact simulation, stored with the pre-registrations)
    out["exactness"] = dict(
        dynamic_max_abs_dev=max(v["max_abs_dev_model"] for v in pre2["exactness_noiseless_dynamic_simulation"].values()),
        dd_vs_no_dd_max_abs_dev=max(v["max_abs_dev_without_dd"] for v in pre2["exactness_noiseless_dynamic_simulation"].values()),
        static_tvd_max=max(pre1["exactness_tvd_statevector_vs_model"].values()),
        true_noDD_identical_to_run1_true=pre2["true_noDD_identical_to_first_run_true"])
    out["empirical_prediction_run2"] = pre2["combined_predictions"]["empirical | first-run visibility, phase refocused"]
    return runs, out


# ============================================================================= replication of the DD run (job 3)
def sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sums(base):
    """Every file listed in SHA256SUMS.txt (pre-registration files and circuits, calibration and context snapshot,
    dry-run logs) is byte-identical to the file that was time-stamped before submission."""
    out = {}
    for line in (base / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, name = line.split()
            out[name] = sha256(base / name) == h
    assert all(out.values()), out
    return out


def experiment_r(runs_m):
    """Job 3: the identical DD circuits of job 2 on the same line, same statistic and primary rule (S > 0 and
    >= 3 sigma above each DD witness)."""
    j, meta = job(JOB_M3, DATA_R)
    pre = load("prereg_replication.json", DATA_R)
    rt = load(f"analysis_{JOB_M3}.json", DATA_R)
    sums = check_sums(DATA_R)
    # the SHA-256 of the pre-registration file is in the IBM job tags; the circuit file matches the pre-registration
    assert sha256(DATA_R / "prereg_replication.json") == meta["prereg_sha256"]
    assert meta["job_tags"] == [f"prereg_sha256:{meta['prereg_sha256']}"]
    assert sha256(DATA_R / "prereg_replication_circuits.qpy") == pre["circuits_sha256"]
    # submitted circuits = pre-registered circuits = the four DD circuits of job 2
    sub, prq = load_qpy(DATA_R / f"job_{JOB_M3}_circuits.qpy"), load_qpy(DATA_R / "prereg_replication_circuits.qpy")
    first = load_qpy(DATA / f"job_{JOB_M2}_circuits.qpy")
    order2 = runs_m["run2"]["meta"]["order"]
    same_as_job2 = all(sub[i] == prq[i] == first[order2.index(name)] for i, name in enumerate(meta["order"]))
    assert same_as_job2
    q = qpy_counts(JOB_M3, DATA_R)
    pred_std = pre["predictions"]["standard_noise_model"]["S"]
    res = {}
    for i, name in enumerate(meta["order"]):
        c = y_counts(j["circuits"][i])
        s_, se, n = s_stat(c)
        res[name] = dict(S=s_, se=se, n=n, P=(c / n).tolist(), P_y0_0=float(c[::2].sum() / n),
                         P_bit1=[float(c[(np.arange(8) >> b) & 1 == 1].sum() / n) for b in range(3)],
                         cz=q[i]["cz"], qubits=q[i]["qubits"], x_gates=q[i]["x"], pred=pred_std[name],
                         esp=pre["circuits"][name]["esp"], duration_us=pre["circuits"][name]["duration_us"],
                         ideal_S=pre["circuits"][name]["ideal_S"], counts=c)
        assert abs(s_ - rt["S"][name]["S"]) < 1e-12 and abs(se - rt["S"][name]["se"]) < 1e-12, name
        assert abs(res[name]["P_y0_0"] - rt["S"][name]["P_y0_0"]) < 1e-12, name
        assert n == meta["plan"][name] == pre["shots"][name], name

    def z(a, b):
        return (a["S"] - b["S"]) / math.hypot(a["se"], b["se"])
    t = res["true_DD"]
    out = dict(sha256sums_ok=sums, identical_to_job2=same_as_job2, prereg_sha256=meta["prereg_sha256"],
               job_tags=meta["job_tags"], prereg_timestamp=pre["timestamp"], qpu_estimate_s=pre["qpu_estimate_s"])
    out["z"] = {w: z(t, res[w]) for w in ("mult_by_1_DD", "wrong_constant_DD", "classical_exponent_DD")}
    for w, v in out["z"].items():
        assert abs(v - rt["z_primary"][w]) < 1e-9, w
    out["verdict"] = "SIGNAL REPLICATED" if (t["S"] > 0 and min(out["z"].values()) >= 3) else "NOT REPLICATED"
    assert rt["verdict_primary"].startswith(out["verdict"])
    # reported, not used for the verdict: comparison with job 2 and inverse-variance mean of the two DD runs
    t2 = runs_m["run2"]["res"]["true_DD"]
    out["z_vs_job2"] = z(t, t2)
    assert abs(out["z_vs_job2"] - rt["z_vs_first_dd_run"]) < 1e-9
    w1, w2 = 1 / t["se"] ** 2, 1 / t2["se"] ** 2
    out["S_combined"] = (w1 * t["S"] + w2 * t2["S"]) / (w1 + w2)
    out["S_combined_se"] = 1 / math.sqrt(w1 + w2)
    out["z_combined_vs_0"] = out["S_combined"] / out["S_combined_se"]
    assert abs(out["S_combined"] - rt["S_combined"]) < 1e-12 and abs(out["S_combined_se"] - rt["S_combined_se"]) < 1e-12
    out["fit_true_DD"] = phase_fit(t["counts"])
    ref = rt["phase_fit"]["true_DD"]
    assert abs(out["fit_true_DD"]["v"] - ref["v"]) < 2e-4 and abs(out["fit_true_DD"]["phase_deg"] - ref["phase_deg"]) < 0.2
    out["mult_by_1_x"] = 2 * res["mult_by_1_DD"]["P_y0_0"] - 1
    out["predictions"] = dict(
        data_based_S=pre["predictions"]["data_based"]["S"]["true_DD"],
        data_based_z_min=pre["predictions"]["data_based"]["z_min"],
        data_based_probability_rule_met=pre["predictions"]["data_based"]["probability_rule_met"],
        standard_model_S=pred_std, standard_model_z_min=pre["predictions"]["standard_noise_model"]["z_min"],
        standard_model_probability_rule_met=pre["predictions"]["standard_noise_model"]["probability_rule_met"])
    out["ratio_measured_to_model"] = t["S"] / pred_std["true_DD"]
    out["exactness_max_abs_dev"] = max(v["max_abs_dev_model"] for v in pre["exactness_noiseless_dynamic_simulation"].values())
    run = dict(job=JOB_M3, res=res, meta=meta, times=job_times(j), shots=int(sum(r["n"] for r in res.values())))
    return run, out


# ============================================================================= N = 21, t = 5 with DD
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


def _miller_splits(N, a, d):
    s = (d & -d).bit_length() - 1
    b = pow(a, d >> s, N)
    for _ in range(s + 1):
        c = b * b % N
        if c == 1 and b not in (1, N - 1):
            return True
        b = c
    return False


def c2_success_set(N, a, t):
    """Criterion C2 ("exact order"): some continued-fraction convergent denominator d <= N of y/2^t is the order
    of a and Miller's method then splits N. Independent implementation, checked against the run-time analysis."""
    Q, ok = 1 << t, []
    for y in range(Q):
        num, den, h0, h1, k0, k1, good = y, Q, 0, 1, 1, 0, False
        while den:
            qq = num // den
            h0, h1 = h1, qq * h1 + h0
            k0, k1 = k1, qq * k1 + k0
            num, den = den, num - qq * den
            if k1 > N:
                break
            d = k1
            if pow(a, d, N) == 1 and all(pow(a, d // p, N) != 1 for p in _primes(d)) and _miller_splits(N, a, d):
                good = True
                break
        ok.append(good)
    return np.array(ok)


T5_LABELS = {"true_DD": "true", "classical_exponent_DD": "classical_exponent", "decohered_e2_DD": "decohered_e2",
             "decohered_e3_DD": "decohered_e3", "decohered_e4_DD": "decohered_e4",
             "scrambled_constants_DD": "scrambled_constants"}


def experiment_t5dd():
    """N = 21, a = 2, t = 5: the six circuits of the run without DD, with X-X echoes, on the same line; criterion C2.
    Results are keyed by the labels of the run without DD (true, classical_exponent, decohered_eK, scrambled_constants)."""
    j, meta = job(JOB_T5DD, DATA_R)
    pre = load("prereg_shor21_t5_dd.json", DATA_R)
    rt = load(f"analysis_{JOB_T5DD}.json", DATA_R)
    check_sums(DATA_R)
    assert sha256(DATA_R / "prereg_shor21_t5_dd.json") == meta["prereg_sha256"]
    assert meta["job_tags"] == [f"prereg_sha256:{meta['prereg_sha256']}"]
    assert sha256(DATA_R / "prereg_shor21_t5_dd_circuits.qpy") == pre["circuits_sha256"]
    sub, prq = load_qpy(DATA_R / f"job_{JOB_T5DD}_circuits.qpy"), load_qpy(DATA_R / "prereg_shor21_t5_dd_circuits.qpy")
    assert len(sub) == len(prq) and all(a_ == b_ for a_, b_ in zip(sub, prq))
    ok = c2_success_set(21, 2, 5)
    rand = float(ok.mean())
    q = qpy_counts(JOB_T5DD, DATA_R)
    res = {}
    for i, name in enumerate(meta["order"]):
        c = j["circuits"][i]["registers"]["y"]
        n = sum(c.values())
        k = sum(v for y, v in c.items() if ok[int(y, 2)])
        p = k / n
        assert k == rt["C2"][name]["k"] and n == rt["C2"][name]["n"] == meta["plan"][name], name
        res[T5_LABELS[name]] = dict(name=name, k=k, n=n, p=p, se=math.sqrt(p * (1 - p) / n), cz=q[i]["cz"],
                                    x_gates=q[i]["x"], ideal=pre["circuits"][name]["ideal_C2_success"],
                                    esp=pre["circuits"][name]["esp"],
                                    esp_without_dd=pre["circuits"][name]["esp_without_dd"],
                                    duration_us=pre["circuits"][name]["duration_us"],
                                    pred=pre["noisy_predictions_standard_model"]["1.0"]["C2_success"][name])
    v = res["true"]
    for lab, r in res.items():
        if lab != "true":
            r["z"] = (v["p"] - r["p"]) / math.hypot(v["se"], r["se"])
            assert abs(r["z"] - rt["z"][r["name"]]) < 1e-9, lab
    v["z_rand"] = (v["p"] - rand) / v["se"]
    assert abs(v["z_rand"] - rt["z"]["random"]) < 1e-9 and rand == rt["random"]
    zmin = min(r["z"] for lab, r in res.items() if lab != "true")
    verdict = "SIGNAL" if (zmin >= 3 and v["z_rand"] >= 3) else "NO SIGNAL"
    assert rt["verdict"].startswith("NO SIGNIFICANT SIGNAL") == (verdict == "NO SIGNAL")
    den = v["ideal"] - rand
    return dict(job=JOB_T5DD, rand=rand, res=res, z_min=zmin, verdict=verdict, retention=(v["p"] - rand) / den,
                retention_se=v["se"] / den, times=job_times(j), shots=int(sum(r["n"] for r in res.values())),
                line=meta["physical_line"], prereg_sha256=meta["prereg_sha256"], prereg_timestamp=pre["timestamp"],
                qpu_estimate_s=pre["qpu_estimate_s"],
                pred_probability_rule_met=pre["noisy_predictions_standard_model"]["1.0"]["probability_rule_met"],
                exactness_max_abs_dev=max(x["max_abs_dev_model"]
                                          for x in pre["exactness_noiseless_dynamic_simulation"].values()),
                identical_without_dd_to_reference=all(pre["checks"]["circuits_without_dd_identical_to_reference"].values()))


# ============================================================================= experiment W (sine window)
def windows(M=8):
    j = np.arange(M)
    W = {"uniform": np.ones(M), "sine": np.sin(np.pi * (j + 1) / (M + 1))}
    p = np.zeros(M); p[M // 2 - 1] = p[M // 2] = 1
    q = np.zeros(M); q[0] = q[M - 1] = 1
    W["middle_pair"], W["edge_pair"] = p, q
    return {k: v / np.linalg.norm(v) for k, v in W.items()}


def lag1(a):
    return float(np.dot(a[:-1], a[1:]))


def experiment_w():
    j, meta = job(JOB_W)
    pre = load("prereg_sine_window.json")
    ideal = {k: lag1(a) for k, a in windows().items()}
    ideal["sine_no_translation"] = 1.0
    assert abs(ideal["sine"] - math.cos(math.pi / 9)) < 1e-12 and abs(ideal["uniform"] - 7 / 8) < 1e-12
    # the sine window maximises sum_j a_j a_{j+1} = a^T A a / 2: top eigenvector of the path-graph adjacency A
    A = np.diag(np.ones(7), 1) + np.diag(np.ones(7), -1)
    w_, V = np.linalg.eigh(A)
    assert abs(w_[-1] / 2 - ideal["sine"]) < 1e-12 and abs(abs(V[:, -1] @ windows()["sine"]) - 1) < 1e-12
    X = {}
    for pub, labels in zip(j["circuits"], meta["pubs"]):
        for counts, lab in zip(pub["registers"]["meas"], labels):
            n = sum(counts.values())
            n0 = sum(v for k, v in counts.items() if k[-1] == "0")      # bit 0 of the register = control c
            x = (2 * n0 - n) / n
            X[lab] = dict(x=x, se=math.sqrt((1 - x * x) / n), n=n, ideal=ideal[lab],
                          pred=pre["noisy_predictions"]["1.0"]["X"][lab])
    rt = load(f"analysis_{JOB_W}.json")
    for k, v in X.items():
        assert abs(v["x"] - rt["X"][k]["x"]) < 1e-12 and abs(v["se"] - rt["X"][k]["se"]) < 1e-12, k
    xu, xs, xe = X["uniform"], X["sine"], X["edge_pair"]
    D, sD = xs["x"] - xu["x"], math.hypot(xs["se"], xu["se"])
    num, den = xs["x"] - xe["x"], xu["x"] - xe["x"]
    r = num / den
    sr = abs(r) * math.sqrt((xs["se"] ** 2 + xe["se"] ** 2) / num ** 2 + (xu["se"] ** 2 + xe["se"] ** 2) / den ** 2)
    labs = ["uniform", "sine", "middle_pair", "edge_pair"]
    Av = np.array([ideal[k] for k in labs]); Xv = np.array([X[k]["x"] for k in labs])
    Wt = np.array([1 / X[k]["se"] ** 2 for k in labs])
    Dm = np.vstack([Av, np.ones_like(Av)]).T
    cov = np.linalg.inv(Dm.T @ (Wt[:, None] * Dm))
    eta, delta = cov @ (Dm.T @ (Wt * Xv))
    chi2 = float((Wt * (Xv - Dm @ np.array([eta, delta])) ** 2).sum())
    out = dict(X=X, D=D, se_D=sD, z=D / sD, D_ideal=ideal["sine"] - ideal["uniform"], D_pred=pre["noisy_predictions"]["1.0"]["D"],
               z_pred=pre["noisy_predictions"]["1.0"]["z_expected"], ratio=r, se_ratio=sr,
               ratio_theory=ideal["sine"] / ideal["uniform"], z_ratio_vs_1=(r - 1) / sr,
               eta=float(eta), eta_se=float(math.sqrt(cov[0, 0])), delta=float(delta),
               delta_se=float(math.sqrt(cov[1, 1])), chi2=chi2)
    for k in ("D", "se_D", "ratio", "se_ratio", "eta", "delta", "chi2"):
        assert abs(out[k] - rt[k]) < 1e-9, k
    # infidelity 1 - |overlap|^2 per overflow, for m window qubits (M = 2^m)
    out["infidelity"] = {m: dict(uniform=1 - (1 - 2 ** -m) ** 2, sine=math.sin(math.pi / (2 ** m + 1)) ** 2)
                         for m in range(1, 7)}
    q = qpy_counts(JOB_W)
    out["cz"], out["qubits"] = [c["cz"] for c in q], [c["qubits"] for c in q]
    out["esp"], out["duration_us"] = pre["esp"], pre["duration_us"]
    out["times"] = job_times(j)
    out["shots"] = int(sum(v["n"] for v in X.values()))
    out["roles"] = meta["roles"]
    return out


# ============================================================================= figures
LAB_M = {"true": "true", "mult_by_1": "×1", "wrong_constant": "×10", "decohered_e2": "decoh. $e_2$",
         "classical_exponent": "classical", "true_DD": "true", "mult_by_1_DD": "×1",
         "wrong_constant_DD": "×10", "classical_exponent_DD": "classical", "true_noDD": "true (no DD)"}


NAVY = "#173f73"


def fig_arith(runs, out, rep=None):
    r1, r2 = runs["run1"]["res"], runs["run2"]["res"]
    r3 = rep[0]["res"] if rep else None
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1, 1.75]})
    ax = axes[0]
    # S and the ideal pattern have period 4 in y; folding y and y + 4 removes any readout error on y2
    y4 = np.arange(4)
    series = [(r1["true"], "job 1, no DD", ORANGE), (r2["true_noDD"], "job 2, no DD", GREY),
              (r2["true_DD"], "job 2, DD", BLUE)]
    if r3:
        series.append((r3["true_DD"], "job 3, DD", NAVY))
    wdt = 0.8 / len(series)
    for k, (r, lab, col) in enumerate(series):
        P = np.array(r["P"])
        P4 = P[:4] + P[4:]
        e = np.sqrt(P4 * (1 - P4) / r["n"])
        ax.bar(y4 + (k - (len(series) - 1) / 2) * wdt, (P4 - 1 / 4) * 100, width=wdt, color=col, label=lab,
               zorder=2, yerr=e * 100, error_kw=dict(lw=0.7, capsize=0, ecolor=INK))
    ideal4 = np.cos(1.5 * np.pi * y4) / 8 * 0.1
    ax.scatter(y4, ideal4 * 100, marker="_", s=420, linewidths=2.0, color=INK, zorder=3, label="ideal × 0.1")
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_xticks(y4, [f"{v:02b}" for v in y4])
    ax.set_xlabel("$y_1y_0$ ($y$ and $y+4$ pooled)")
    ax.set_ylabel("$P(y_1y_0) - 1/4$  [%]")
    ax.set_ylim(-3.75, 2.1)
    ax.set_yticks([-1, 0, 1, 2])
    ax.legend(loc="lower center", fontsize=6.4, ncol=2, handlelength=1.0, columnspacing=0.6, markerscale=0.45,
              title="true circuit", title_fontsize=6.4, borderaxespad=0.2, labelspacing=0.3)
    ax.set_title("(a)", loc="left", fontsize=9)

    ax = axes[1]
    order1 = ["true", "mult_by_1", "wrong_constant", "decohered_e2", "classical_exponent"]
    order2 = ["true_DD", "mult_by_1_DD", "wrong_constant_DD", "classical_exponent_DD", "true_noDD"]
    order3 = ["true_DD", "mult_by_1_DD", "wrong_constant_DD", "classical_exponent_DD"]
    xs1, xs2, xs3 = np.arange(5), np.arange(5) + 5.8, np.arange(4) + 11.6
    groups = [(xs1, order1, r1, ORANGE), (xs2, order2, r2, BLUE)]
    if r3:
        groups.append((xs3, order3, r3, NAVY))
    for xs, order, res, col in groups:
        for xx, name in zip(xs, order):
            r = res[name]
            c = col if name in ("true", "true_DD") else (GREY if name == "true_noDD" else MUTED)
            ax.errorbar(xx, r["S"] * 100, yerr=r["se"] * 100, fmt="o", ms=4.5, color=c, lw=1.1, capsize=2, zorder=3)
            ax.scatter(xx, r["pred"] * 100, marker="D", s=22, facecolors="none", edgecolors=INK, lw=0.9, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(4.9, color=GRID, lw=1.0)
    ticks, labels = list(np.r_[xs1, xs2]), [LAB_M[n] for n in order1 + order2]
    if r3:
        ax.axvline(10.7, color=GRID, lw=1.0)
        ticks += list(xs3)
        labels += [LAB_M[n] for n in order3]
        # inverse-variance mean of the two DD runs (reported, not pre-registered)
        sc, sse = out["rep"]["S_combined"] * 100, out["rep"]["S_combined_se"] * 100
        ax.fill_between([5.5, 15.1], sc - sse, sc + sse, color=BLUE, alpha=0.12, lw=0, zorder=0)
        ax.text(15.1, sc - sse - 0.12, f"mean of DD runs\n{sc:+.2f} ± {sse:.2f} %", ha="right", va="top",
                fontsize=6.5, color=NAVY)
    ax.set_xticks(ticks, labels, fontsize=6.8, rotation=50, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, 15.3)
    ax.set_ylabel("$S$  [%]")
    ax.set_ylim(-2.6, 5.6)
    ax.text(2.0, 4.75, "job 1: diagnostic\nno DD", ha="center", fontsize=7, color=ORANGE, linespacing=1.1)
    ax.text(7.8, 4.75, "job 2: DD\n(last: no DD)", ha="center", fontsize=7, color=BLUE, linespacing=1.1)
    if r3:
        ax.text(13.1, 4.75, "job 3: replication\nDD", ha="center", fontsize=7, color=NAVY, linespacing=1.1)
    ax.scatter([], [], marker="D", s=22, facecolors="none", edgecolors=INK, label="pre-run model prediction")
    ax.errorbar([], [], yerr=[], fmt="o", ms=4.5, color=MUTED, label="measured ± 1σ")
    ax.legend(loc="lower left", fontsize=7, ncol=2, handlelength=1.2, columnspacing=0.8)
    ax.set_title("(b)  ideal: 25 % (true), 0 (witnesses)", loc="left", fontsize=9)
    fig.tight_layout(w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_arith.{ext}", dpi=200)
    plt.close(fig)


def fig_sine(w):
    X = w["X"]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.55), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    aa = np.linspace(0, 1, 2)
    ax.plot(aa, aa, color=GREY, lw=0.9, ls=":", zorder=1)
    ax.plot(aa, w["eta"] * aa + w["delta"], color=BLUE, lw=1.0, zorder=1)
    for k, mk in (("edge_pair", "o"), ("middle_pair", "o"), ("uniform", "o"), ("sine", "o"), ("sine_no_translation", "s")):
        v = X[k]
        ax.errorbar(v["ideal"], v["x"], yerr=v["se"], fmt=mk, ms=4, color=ORANGE if k == "sine" else INK,
                    capsize=1.5, lw=0.9, zorder=3)
        ax.scatter(v["ideal"], v["pred"], marker="D", s=18, facecolors="none", edgecolors=MUTED, lw=0.8, zorder=2)
    ax.annotate("edge", (0, X["edge_pair"]["x"]), xytext=(6, 4), textcoords="offset points", fontsize=7)
    ax.annotate("middle", (0.5, X["middle_pair"]["x"]), xytext=(-5, 7), textcoords="offset points", fontsize=7,
                ha="right")
    ax.annotate("uniform, sine", (0.875, X["uniform"]["x"]), xytext=(4, -28), textcoords="offset points",
                fontsize=7, ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.annotate("no $T_N$", (1.0, X["sine_no_translation"]["x"]), xytext=(-12, 8), textcoords="offset points",
                fontsize=7, ha="center")
    ax.text(0.02, 0.97, "dotted: ideal\nline: fit $\\eta A+\\delta$\n$\\diamond$: pre-run model",
            transform=ax.transAxes, fontsize=6.5, va="top", color=MUTED)
    ax.set_xlabel("ideal overlap $A$")
    ax.set_ylabel("$\\langle X_c\\rangle$")
    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(-0.08, 1.05)
    ax.set_title("(a)", loc="left", fontsize=9)

    ax = axes[1]
    for i, (k, col) in enumerate((("uniform", INK), ("sine", ORANGE))):
        v = X[k]
        ax.errorbar(i, v["x"], yerr=v["se"], fmt="o", ms=4.5, color=col, capsize=2, lw=1.0, zorder=3)
        ax.scatter(i, v["pred"], marker="D", s=20, facecolors="none", edgecolors=MUTED, lw=0.8, zorder=2)
    ax.set_xticks([0, 1], ["uniform", "sine"], fontsize=7.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.52, 0.615)
    ax.text(0.5, 0.583, f"D = {w['D']:+.3f}\n± {w['se_D']:.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("(b)", loc="left", fontsize=9)
    fig.tight_layout(w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_sine.{ext}", dpi=220)
    plt.close(fig)


def tab_arith(runs, out, rep=None):
    r1, r2 = runs["run1"]["res"], runs["run2"]["res"]
    pf = lambda x: f"{0.0 if abs(x) < 5e-4 else x:+.3f}"      # no negative zero
    lab = {"true": "true", "mult_by_1": r"$\times1$", "wrong_constant": r"$\times10$",
           "decohered_e2": r"decoh.\ $e_2$", "classical_exponent": "classical"}

    def row(name, r, zz):
        return (f"{name} & {r['n']} & ${r['S']:+.4f}\\pm{r['se']:.4f}$ & ${pf(r['pred'])}$ & {zz} & "
                f"{r['P_y0_0']:.3f} \\\\")
    rows = [r"\multicolumn{6}{l}{job 1, no DD} \\"]
    for name in ("true", "mult_by_1", "wrong_constant", "decohered_e2", "classical_exponent"):
        zz = "--" if name == "true" else f"${out['z_run1'][name]:+.1f}$"
        rows.append(row(lab[name], r1[name], zz))
    rows += [r"\midrule", r"\multicolumn{6}{l}{job 2, with DD (last row: without)} \\"]
    for name in ("true_DD", "mult_by_1_DD", "wrong_constant_DD", "classical_exponent_DD", "true_noDD"):
        base = name.replace("_noDD", "").replace("_DD", "")
        zz = "--" if name == "true_DD" else (f"${out['z_DD_vs_noDD']:+.1f}$" if name == "true_noDD"
                                              else f"${out['z_run2'][name]:+.1f}$")
        rows.append(row(lab[base] + (" (no DD)" if name == "true_noDD" else ""), r2[name], zz))
    if rep:
        r3, o3 = rep[0]["res"], rep[1]
        rows += [r"\midrule", r"\multicolumn{6}{l}{job 3, replication of job 2 (DD)} \\"]
        for name in ("true_DD", "mult_by_1_DD", "wrong_constant_DD", "classical_exponent_DD"):
            zz = "--" if name == "true_DD" else f"${o3['z'][name]:+.1f}$"
            rows.append(row(lab[name.replace("_DD", "")], r3[name], zz))
    head = [r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"circuit & shots & $S$ & pred. & $\Delta/\sigma$ & $P(y_0{=}0)$ \\", r"\midrule"]
    txt = "\n".join(head + rows + [r"\bottomrule", r"\end{tabular}", ""])
    (FIG / "tab_arith.tex").write_text(txt, encoding="utf-8")


def main():
    FIG.mkdir(exist_ok=True)
    runs, m = experiment_m()
    rep = experiment_r(runs)
    m["rep"] = rep[1]
    t5 = experiment_t5dd()
    w = experiment_w()
    fig_arith(runs, m, rep)
    fig_sine(w)
    tab_arith(runs, m, rep)
    runs = dict(runs, run3=rep[0])

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if k not in ("counts", "t_running", "t_finished")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    num = dict(M=dict(runs={k: dict(job=v["job"], shots=v["shots"], times=v["times"], res=v["res"],
                                    line=v["meta"]["physical_line"]) for k, v in runs.items()}, **m), W=w, T5DD=t5)
    (FIG / "numbers_arith.json").write_text(json.dumps(clean(num), indent=1, default=str), encoding="utf-8")

    r1, r2, r3 = runs["run1"]["res"], runs["run2"]["res"], runs["run3"]["res"]
    print("Modular multiplication N=21, a=2, t=3")
    for tag, res in (("job 1", r1), ("job 2", r2), ("job 3", r3)):
        for k, r in res.items():
            print(f"  {tag} {k:22s} CZ={r['cz']:4d} S={r['S']:+.4f}±{r['se']:.4f} pred={r['pred']:+.4f} "
                  f"P(y0=0)={r['P_y0_0']:.3f} shots={r['n']}")
    print("  exactness:", m["exactness"], "| empirical prediction job 2:", m["empirical_prediction_run2"]["S"],
          m["empirical_prediction_run2"]["z_primary_min"])
    print("  ideal S by multiplier:", {c: round(v, 4) for c, v in m["ideal_S_by_multiplier"].items()})
    print("  z job 1:", {k: round(v, 1) for k, v in m["z_run1"].items()}, m["verdict_run1"])
    print("  z job 2:", {k: round(v, 1) for k, v in m["z_run2"].items()}, m["verdict_run2"],
          f"| DD - noDD {m['z_DD_vs_noDD']:+.1f} | noDD - job1 {m['z_noDD_vs_run1']:+.1f}")
    for k, f in m["fit"].items():
        print(f"  fit {k:15s} v={f['v']:.4f}±{f['v_se']:.4f} phase={f['phase_deg']:+.1f}±{f['phase_se_deg']:.1f} deg "
              f"2lnLR={f['two_lnLR']:.1f}")
    print("  mult_by_1 P(y0=0) measured", {k: round(r["P_y0_0"], 3) for k, r in (("job1", r1["mult_by_1"]), ("job2", r2["mult_by_1_DD"]))},
          "predicted", {k: round(v, 3) for k, v in m["mult_by_1_pred_P_y0_0"].items()})
    print("  P(y_b = 1), b = 0, 1, 2:", {f"{t}:{k}": [round(x, 3) for x in r["P_bit1"]] for t, res in (("job1", r1), ("job2", r2)) for k, r in res.items()})
    print("  measured / model:", {k: round(v, 3) for k, v in m["ratio_measured_to_model"].items()})
    print("  ZZ on line [kHz]:", {k: round(v, 2) if isinstance(v, float) else v for k, v in m["zz_khz_line"].items()},
          "| q44:", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in m["q44"].items()},
          f"| hours between runs {m['hours_between_runs']:.2f}")
    print("  coherent model, noiseless S(true_noDD):", m["coherent_model"]["noiseless_true_noDD"])
    print("  coherent model, combined S(true_noDD):", m["coherent_model"]["combined_true_noDD"])
    print("  times:", {k: (v["times"]["running"], v["times"]["finished"], v["times"]["qpu_s"], v["shots"]) for k, v in runs.items()})
    o3, f3 = m["rep"], m["rep"]["fit_true_DD"]
    print("Replication (job 3): SHA256SUMS", all(o3["sha256sums_ok"].values()), "| circuits identical to job 2:",
          o3["identical_to_job2"], "| tags", o3["job_tags"])
    print("  z job 3:", {k: round(v, 1) for k, v in o3["z"].items()}, o3["verdict"],
          f"| job 3 - job 2 {o3['z_vs_job2']:+.1f} | mean of DD runs {o3['S_combined']:+.4f}±{o3['S_combined_se']:.4f}"
          f" ({o3['z_combined_vs_0']:.1f} sigma from 0)")
    print(f"  fit job3_true_DD   v={f3['v']:.4f}±{f3['v_se']:.4f} phase={f3['phase_deg']:+.1f}±{f3['phase_se_deg']:.1f} deg"
          f" 2lnLR={f3['two_lnLR']:.1f} | mult_by_1 P(y0=0)={r3['mult_by_1_DD']['P_y0_0']:.3f}"
          f" | measured/model {o3['ratio_measured_to_model']:.3f}")
    print("  predictions:", {k: v for k, v in o3["predictions"].items() if k != "standard_model_S"},
          "standard model S(true_DD)", o3["predictions"]["standard_model_S"]["true_DD"])
    print("  P(y_b = 1), job 3:", {k: [round(x, 3) for x in r["P_bit1"]] for k, r in r3.items()})
    print("N=21, t=5 with DD (C2):")
    for k, r in t5["res"].items():
        print(f"  {k:19s} CZ={r['cz']} p={r['p']:.4f}±{r['se']:.4f} ideal={r['ideal']:.4f} pred={r['pred']:.4f} "
              f"n={r['n']}" + (f" z={r['z']:+.1f}" if "z" in r else f" z_rand={r['z_rand']:+.1f}"))
    print(f"  {t5['verdict']} (z_min {t5['z_min']:+.1f}); R={t5['retention']:.4f}±{t5['retention_se']:.4f}; "
          f"times {t5['times']['running']} {t5['times']['finished']} QPU {t5['times']['qpu_s']} s, shots {t5['shots']}")
    print("Sine window")
    for k, v in w["X"].items():
        print(f"  {k:20s} <X>={v['x']:+.4f}±{v['se']:.4f} ideal={v['ideal']:.4f} pred={v['pred']:+.4f} n={v['n']}")
    print(f"  D={w['D']:+.4f}±{w['se_D']:.4f} ({w['z']:+.1f} sigma; ideal {w['D_ideal']:+.4f}, pred {w['D_pred']:+.4f}, "
          f"{w['z_pred']} sigma) ratio={w['ratio']:.4f}±{w['se_ratio']:.4f} (theory {w['ratio_theory']:.4f}, "
          f"{w['z_ratio_vs_1']:+.1f} sigma from 1) eta={w['eta']:.4f}±{w['eta_se']:.4f} delta={w['delta']:+.4f}"
          f"±{w['delta_se']:.4f} chi2={w['chi2']:.1f}")
    print("  CZ", w["cz"], "qubits", w["qubits"], "ESP", w["esp"], "duration", w["duration_us"], "times",
          w["times"]["running"], w["times"]["finished"], w["times"]["qpu_s"], "shots", w["shots"])
    print("  infidelity per overflow:", {m_: {k: round(v, 4) for k, v in d.items()} for m_, d in w["infidelity"].items()})


if __name__ == "__main__":
    main()
