"""Archive an IBM job for publication: raw counts of every circuit, metrics, timestamps, backend, and the
submitted circuits (QPY) when available.

Usage: python src/archive_job.py <job_id> <output_directory> [saved_account_name]
The IBM account is the named saved account, else the default saved account (see run_ibm.py --configure), or
the QISKIT_IBM_TOKEN / QISKIT_IBM_INSTANCE environment variables. The output keys (registers, metrics, status, created, estimated_usage) are those of the
archived data of the paper.
"""
import itertools
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def service(account=None):
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = os.environ.get("QISKIT_IBM_TOKEN")
    if tok:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=tok,
                                    instance=os.environ.get("QISKIT_IBM_INSTANCE"))
    return QiskitRuntimeService(name=account) if account else QiskitRuntimeService()


def archive(jid, out, svc=None):
    """Write job_<jid>.json (raw counts, metrics, timestamps) and job_<jid>_circuits.qpy into `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    job = (svc or service()).job(jid)
    res = job.result()
    data = {"job_id": jid, "backend": job.backend().name, "status": str(job.status()),
            "created": str(job.creation_date), "metrics": job.metrics(), "estimated_usage": job.usage_estimation,
            "circuits": []}
    for i, pub in enumerate(res):
        entry = {"index": i, "registers": {}}
        for reg in pub.data:
            try:
                arr = pub.data[reg]
                shape = tuple(getattr(arr, "shape", ()))
                if shape:      # parameter sweep: one count dictionary per parameter set
                    entry["registers"][reg] = [arr.get_counts(loc=idx) for idx in itertools.product(
                        *[range(s) for s in shape])]
                else:
                    entry["registers"][reg] = arr.get_counts()
            except Exception:
                pass
        entry["metadata"] = pub.metadata
        data["circuits"].append(entry)
    with open(out / f"job_{jid}.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, default=str, indent=1)
    try:
        from qiskit import qpy
        pubs = job.inputs.get("pubs", [])
        circs = [p[0] if isinstance(p, (list, tuple)) else p for p in pubs]
        with open(out / f"job_{jid}_circuits.qpy", "wb") as fh:
            qpy.dump(circs, fh)
        print("circuits archived:", len(circs))
    except Exception as e:
        print("circuits not archived:", type(e).__name__, e)
    m = data["metrics"]
    print("archived:", out / f"job_{jid}.json", "| execution:", m.get("timestamps"), "| QPU:", m.get("usage"))
    return data


def main():
    archive(sys.argv[1], sys.argv[2], service(sys.argv[3] if len(sys.argv) > 3 else None))


if __name__ == "__main__":
    main()
