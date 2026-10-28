"""Error-detection demonstration on an IBM device: AN code + uncomputation (carry) check.

Circuit: addition echo with carry propagation (+3K then -3K) on a register holding an AN-encoded value 3v
(w = 5 bits, w - 1 = 4 carry ancillas). The carries are computed with temporary logical-AND gates (relative-phase
Toffoli, Gidney 2018) and then UNCOMPUTED: they must return to 0. Two post-selection filters are compared:
  * AN code: the output residue must be 0 mod 3;
  * carry check: every carry ancilla must be measured in 0 ("algorithmic check by uncomputation").
Inputs are classical, so only bit-flip (X-type) errors are visible; neither filter detects phase errors.

Default: DRY RUN on a local noise model (FakeKingston), no account needed.
  python src/an_code_demo.py                      dry run
  python src/an_code_demo.py --ibm                real device (asks for confirmation)
  python src/an_code_demo.py --ibm --job <ID>     read back and analyse a job
  python src/an_code_demo.py --archived data/2026-10-27_ibm_kingston/job_dbg9q25jm0ks73eg6hc0.json \
      data/2026-10-27_ibm_kingston/meta_dbg9q25jm0ks73eg6hc0.json        re-analyse archived raw counts
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

for _f in (sys.stdout, sys.stderr):
    try:
        _f.reconfigure(encoding="utf-8")
    except Exception:
        pass
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from qiskit import QuantumCircuit, QuantumRegister, transpile  # noqa: E402

AN_W, AN_A = 5, 3
AN_CASES = [(2, 7), (1, 6), (4, 5), (3, 4)]          # (v, K): 3v + 3K < 2^5
REP_DELAY = 250e-6
OVERHEAD = 2.0


# =============================================================================== backend helpers
def esp(t, backend):
    """Estimated success probability prod(1 - error) over gates and measurements (calibration of the day)."""
    tg = backend.target
    p = 1.0
    for ins in t.data:
        name = ins.operation.name
        if name in ("barrier", "delay", "if_else", "reset"):
            continue
        qs = tuple(t.find_bit(q).index for q in ins.qubits)
        try:
            e = tg[name][qs].error or 0.0
        except Exception:
            e = 0.0
        p *= 1 - e
    return p


def circuit_duration(t, backend):
    """Duration (s) of the physical circuit, crude ASAP schedule (for the QPU budget)."""
    tg = backend.target
    free = {}
    for ins in t.data:
        name = ins.operation.name
        if name == "barrier":
            continue
        qs = [t.find_bit(q).index for q in ins.qubits]
        try:
            d = tg[name][tuple(qs)].duration or 0.0
        except Exception:
            d = 2e-6 if name in ("if_else", "reset") else 0.0
        s = max([free.get(q, 0.0) for q in qs] + [0.0])
        for q in qs:
            free[q] = s + d
    return max(free.values()) if free else 0.0


def choose_layout(qc_ref, backend, seeds):
    """Best layout (max ESP over several transpiler seeds); returns (seed, layout)."""
    best = None
    for s in seeds:
        t = transpile(qc_ref, backend, optimization_level=3, seed_transpiler=s)
        sc = esp(t, backend)
        if best is None or sc > best[0]:
            best = (sc, s, t)
    return best[1], best[2].layout.initial_index_layout(filter_ancillas=True)


def n2q(t):
    return sum(c for g, c in t.count_ops().items() if g in ("cz", "cx", "ecr"))


def counts_int(pub_result):
    """{integer outcome: count} for the single classical register of a SamplerV2 result."""
    d = pub_result.data
    name = list(d.keys())[0]
    v = np.array([int(s, 2) for s in getattr(d, name).get_bitstrings()], dtype=np.int64)
    u, c = np.unique(v, return_counts=True)
    return {int(k): int(n) for k, n in zip(u, c)}


# =============================================================================== circuit
def _rccx_dg():
    g = QuantumCircuit(3, name="rccx_dg")
    g.rccx(0, 1, 2)
    return g.inverse().to_gate()


def add_const_ripple(qc, x, cr, k):
    """x (w bits) += k mod 2^w; carries computed into w-1 ancillas (temporary relative-phase AND gates) and then
    UNCOMPUTED (they must return to 0: this is the carry check)."""
    w = len(x)
    kb = [(k >> i) & 1 for i in range(w)]
    dg = _rccx_dg()

    def calc(i):                                # c_{i+1} = MAJ(x_i, k_i, c_i)
        if i == 0:
            if kb[0]:
                qc.cx(x[0], cr[0])
            return
        a, b, t = x[i], cr[i - 1], cr[i]
        if kb[i]:                               # OR = NOT(AND of the negations)
            qc.x(a); qc.x(b); qc.rccx(a, b, t); qc.x(a); qc.x(b); qc.x(t)
        else:
            qc.rccx(a, b, t)

    def uncalc(i):
        if i == 0:
            if kb[0]:
                qc.cx(x[0], cr[0])
            return
        a, b, t = x[i], cr[i - 1], cr[i]
        if kb[i]:
            qc.x(t); qc.x(a); qc.x(b); qc.append(dg, [a, b, t]); qc.x(a); qc.x(b)
        else:
            qc.append(dg, [a, b, t])

    for i in range(w - 1):
        calc(i)
    qc.cx(cr[w - 2], x[w - 1])
    if kb[w - 1]:
        qc.x(x[w - 1])
    for i in range(w - 2, -1, -1):
        uncalc(i)
        if i > 0:
            qc.cx(cr[i - 1], x[i])
        if kb[i]:
            qc.x(x[i])


def an_circuit(A, v, K, w=AN_W):
    x = QuantumRegister(w, "x")
    cr = QuantumRegister(w - 1, "ret")
    qc = QuantumCircuit(x, cr)
    x0 = A * v
    for b in range(w):
        if (x0 >> b) & 1:
            qc.x(x[b])
    qc.barrier()
    add_const_ripple(qc, x, cr, (A * K) % (1 << w))
    qc.barrier()
    add_const_ripple(qc, x, cr, (-A * K) % (1 << w))
    qc.measure_all()
    return qc, x0


def prepare(backend, shots=4000, seeds=10):
    items = [(f"v={v},K={K}", *an_circuit(AN_A, v, K)) for v, K in AN_CASES]
    seed, layout = choose_layout(items[0][1], backend, range(seeds))
    pubs, meta = [], []
    for lab, qc, x0 in items:
        t = transpile(qc, backend, optimization_level=3, seed_transpiler=seed, initial_layout=layout)
        pubs.append((t, None, shots))
        meta.append(dict(label=lab, x0=x0, shots=shots, cz=n2q(t)))
    info = dict(cz=[m["cz"] for m in meta], esp=esp(pubs[0][0], backend), layout=list(layout),
                qubits=items[0][1].num_qubits, duration_s=max(circuit_duration(p[0], backend) for p in pubs))
    print(f"  addition echo +3K/-3K with carry propagation, w={AN_W} + {AN_W - 1} carries; CZ {info['cz']}, "
          f"ESP {info['esp']:.3f}, physical qubits {info['layout']}")
    return pubs, dict(demo="an", circuits=meta, info=info)


# =============================================================================== analysis
def analyse(meta, counts_list):
    """counts_list[i]: {integer outcome: count} of circuit i (x in the low AN_W bits, carries above)."""
    print("\n=== AN code (x3) + carry uncomputation check ===")
    tot = {k: [0, 0] for k in ("raw", "AN", "carries", "AN+carries")}
    ntot = 0
    res = {}
    for m_, cnt in zip(meta["circuits"], counts_list):
        n = sum(cnt.values())
        ntot += n
        S = {k: [0, 0] for k in tot}
        for k, c in cnt.items():
            x = k & ((1 << AN_W) - 1)
            anc = k >> AN_W
            ok = x == m_["x0"]
            an = x % AN_A == 0
            ca = anc == 0
            for name, acc in (("raw", True), ("AN", an), ("carries", ca), ("AN+carries", an and ca)):
                if acc:
                    S[name][0] += c
                    S[name][1] += c * ok
        for k in tot:
            tot[k][0] += S[k][0]
            tot[k][1] += S[k][1]
        res[m_["label"]] = {k: (s[1] / max(s[0], 1), s[0] / n) for k, s in S.items()}
        print(f"  {m_['label']:<9} " + " | ".join(f"{k} {s[1] / max(s[0], 1):.3f} (kept {s[0] / n:.2f})"
                                                for k, s in S.items()))
    print("  Total:")
    for k, s in tot.items():
        p = s[1] / max(s[0], 1)
        sp = math.sqrt(p * (1 - p) / max(s[0], 1))
        print(f"    {k:<11} success {p:.3f} +- {sp:.3f}   kept shots {s[0] / ntot:.2f}   residual error {1 - p:.3f}")
        res[f"total_{k}"] = (p, sp, s[0] / ntot)
    e0 = 1 - tot["raw"][1] / tot["raw"][0]
    for k in ("AN", "carries", "AN+carries"):
        e1 = 1 - tot[k][1] / max(tot[k][0], 1)
        print(f"    errors removed by '{k}': {100 * (1 - e1 / e0):.0f} %")
    print("  Classical inputs: only bit-flip (X) errors are visible; both filters are blind to phase errors.")
    return res


def archived_counts(job_json):
    job = json.loads(Path(job_json).read_text(encoding="utf-8"))
    return [{int(k, 2): v for k, v in c["registers"]["meas"].items()} for c in job["circuits"]]


# =============================================================================== IBM account
def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = os.environ.get("QISKIT_IBM_TOKEN")
    if tok:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=tok,
                                    instance=os.environ.get("QISKIT_IBM_INSTANCE"))
    return QiskitRuntimeService()


def qpu_time(pubs, backend):
    t = OVERHEAD
    for p in pubs:
        t += p[2] * (REP_DELAY + circuit_duration(p[0], backend) + 5e-6)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ibm", action="store_true", help="REAL IBM device (otherwise local dry run)")
    ap.add_argument("--backend", default=None, help="device name (default: ibm_kingston) or FakeXxx for the dry run")
    ap.add_argument("--shots", type=int, default=4000, help="shots per circuit")
    ap.add_argument("--seeds", type=int, default=10, help="transpiler seeds tried for the layout")
    ap.add_argument("--budget", type=float, default=30.0, help="QPU time cap (s)")
    ap.add_argument("--job", default=None, help="ID of a submitted job to read back and analyse")
    ap.add_argument("--archived", nargs=2, metavar=("JOB_JSON", "META_JSON"), help="re-analyse archived raw counts")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    if args.archived:
        meta = json.loads(Path(args.archived[1]).read_text(encoding="utf-8"))
        analyse(meta, archived_counts(args.archived[0]))
        return
    if args.job:
        meta = json.loads((outdir / f"meta_{args.job}.json").read_text(encoding="utf-8"))
        result = service().job(args.job).result()
        analyse(meta, [counts_int(r) for r in result])
        return
    if args.ibm:
        backend = service().backend(args.backend or "ibm_kingston")
    else:
        import qiskit_ibm_runtime.fake_provider as fp
        backend = getattr(fp, args.backend or "FakeKingston")()
    print(f"Backend: {backend.name} ({'REAL DEVICE' if args.ibm else 'dry run, local noisy simulation'})")
    pubs, meta = prepare(backend, args.shots, args.seeds)
    meta.update(backend=backend.name, date=datetime.datetime.now().isoformat(timespec="seconds"))
    tq = qpu_time(pubs, backend)
    meta["qpu_estimate_s"] = tq
    print(f"  {sum(p[2] for p in pubs)} shots, estimated QPU time {tq:.0f} s (budget {args.budget:.0f} s)")
    if tq > args.budget:
        print("Budget exceeded: reduce --shots.")
        return
    if args.ibm and not args.yes:
        if input(f"Submit to {backend.name}? [y/N] ").strip().lower() != "y":
            print("Cancelled.")
            return
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    sampler = Sampler(mode=backend)
    if args.ibm:
        o = sampler.options
        o.dynamical_decoupling.enable = True
        o.dynamical_decoupling.sequence_type = "XpXm"
        o.twirling.enable_gates = True
        o.twirling.enable_measure = True
        o.max_execution_time = int(min(max(60, 2 * tq), args.budget))
    t0 = time.time()
    job = sampler.run(pubs)
    if args.ibm:
        jid = job.job_id()
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"meta_{jid}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"Job submitted: {jid} (read back: --ibm --job {jid})")
    result = job.result()
    if not args.ibm:
        print(f"  local simulation: {time.time() - t0:.0f} s")
    analyse(meta, [counts_int(r) for r in result])


if __name__ == "__main__":
    main()
