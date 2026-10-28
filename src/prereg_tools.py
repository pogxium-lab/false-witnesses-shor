"""Helpers shared by the pre-registered hardware runs: fingerprints of the pre-registration files, live checks of
the stored circuits against today's device, submission with the pre-registration hash in the job tags.

  SHA-256 list of a directory (written next to the files, for external time-stamping):
      python src/prereg_tools.py sums DIR [--out SHA256SUMS.txt]
  Check a directory against its list:
      python src/prereg_tools.py check DIR

External time-stamping is done with the OpenTimestamps client (`ots stamp FILE`, only the hash leaves the
machine) on each pre-registration file, each circuit file and the SHA-256 list; `ots upgrade FILE.ots` later
completes the proofs once the calendars have committed them to the Bitcoin blockchain, `ots verify FILE.ots`
checks them.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

SUMS = "SHA256SUMS.txt"
TAG_PREFIX = "prereg_sha256:"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qpy_path(prereg):
    p = Path(prereg)
    return p.with_name(p.stem + "_circuits.qpy")


def resolve(path, base):
    """A path stored inside a pre-registration file: as written (relative to the working directory or absolute),
    else relative to the directory of that file, else its file name in that directory."""
    p = Path(path)
    for cand in (p, Path(base) / p, Path(base) / p.name):
        if cand.exists():
            return cand
    raise FileNotFoundError(path)


def write_sums(directory, out=SUMS, patterns=("*.json", "*.qpy", "*_log.txt")):
    """SHA-256 of every file of `directory` matching `patterns` (sorted), in the `sha256sum` format."""
    d = Path(directory)
    files = sorted({f for pat in patterns for f in d.glob(pat) if f.is_file() and f.name != out})
    lines = [f"{sha256(f)}  {f.name}" for f in files]
    (d / out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d / out, lines


def read_sums(directory, name=SUMS):
    f = Path(directory) / name
    if not f.exists():
        return None
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, n = line.split(None, 1)
            out[n.strip().lstrip("*")] = h
    return out


def check_fingerprints(prereg, pre):
    """Circuit file against the hash stored in the pre-registration; pre-registration and circuit file against
    the SHA-256 list of their directory (if present); presence of the time-stamp proofs. Exits on a mismatch."""
    prereg = Path(prereg)
    qfile = qpy_path(prereg)
    h_qpy, h_pre = sha256(qfile), sha256(prereg)
    if h_qpy != pre["circuits_sha256"]:
        sys.exit(f"{qfile} differs from the pre-registered circuits (sha256)")
    sums = read_sums(prereg.parent)
    if sums is not None:
        for f, h in ((prereg, h_pre), (qfile, h_qpy)):
            if f.name in sums and sums[f.name] != h:
                sys.exit(f"{f} differs from its entry in {SUMS}")
            if f.name not in sums:
                sys.exit(f"{f} is not listed in {SUMS}")
    ots = {f.name: Path(str(f) + ".ots").exists() for f in (prereg, qfile)}
    print(f"Pre-registration sha256 {h_pre}; circuits sha256 {h_qpy} (matches the pre-registration"
          f"{'' if sums is None else ' and ' + SUMS}); time-stamp proofs present: {ots}")
    return h_pre


def job_tags(prereg):
    return [TAG_PREFIX + sha256(prereg)]


def isa_errors(circuits, target):
    """Instruction-set check of qiskit_ibm_runtime (the one applied at submission); empty if valid."""
    from qiskit_ibm_runtime.utils.utils import is_isa_circuit
    return {k: m for k, m in ((k, is_isa_circuit(c, target)) for k, c in circuits.items()) if m}


def duration_changes(target_a, target_b, qubits, ops=("x", "sx", "measure", "reset")):
    """Gate durations (s) that differ between two targets on `qubits` (single-qubit operations and CZ between
    consecutive qubits of the list). Explicit delays of a scheduled circuit assume the durations of the target
    used for scheduling."""
    out = {}

    def dur(t, name, qs):
        try:
            return t[name][qs].duration
        except Exception:
            return None
    for q in qubits:
        for name in ops:
            a, b = dur(target_a, name, (q,)), dur(target_b, name, (q,))
            if a != b and not (a and b and math.isclose(a, b, rel_tol=1e-9)):
                out[f"{name}{q}"] = (a, b)
    for q1, q2 in zip(qubits, qubits[1:]):
        a = dur(target_a, "cz", (q1, q2)) or dur(target_a, "cz", (q2, q1))
        b = dur(target_b, "cz", (q1, q2)) or dur(target_b, "cz", (q2, q1))
        if a != b and not (a and b and math.isclose(a, b, rel_tol=1e-9)):
            out[f"cz{q1}_{q2}"] = (a, b)
    return out


def submit(be, pubs, budget, tq_est, tags=None):
    """SamplerV2 job, sampler-level dynamical decoupling disabled (echoes, if any, are explicit gates of the
    circuits), execution time capped, optional job tags."""
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    sampler = Sampler(mode=be)
    sampler.options.max_execution_time = int(min(budget, max(60, 1.3 * tq_est)))
    sampler.options.dynamical_decoupling.enable = False
    if tags:
        sampler.options.environment.job_tags = list(tags)
    return sampler.run(pubs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["sums", "check"])
    ap.add_argument("directory")
    ap.add_argument("--out", default=SUMS)
    args = ap.parse_args()
    if args.action == "sums":
        f, lines = write_sums(args.directory, args.out)
        print("\n".join(lines))
        print(f"-> {f}")
        return
    sums = read_sums(args.directory, args.out) or {}
    bad = [n for n, h in sums.items() if sha256(Path(args.directory) / n) != h]
    print(f"{len(sums)} files listed; mismatches: {bad or 'none'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
