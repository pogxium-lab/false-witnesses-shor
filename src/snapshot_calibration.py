"""Read-only snapshot of an IBM device: full calibration (backend properties) and account context.

Nothing is submitted. Writes
  <outdir>/calibration_<tag>.json : backend.properties().to_dict() (T1, T2, readout and gate errors, durations);
  <outdir>/context_<tag>.json     : timestamp, date of the calibration, pending jobs.
The account usage (QPU time available in the current window) is printed but never written to disk.
The calibration file can be used offline by every script of the package (option --calibration), see
hardware.calibrated_backend.

Usage: python src/snapshot_calibration.py <backend> <outdir> <tag> [--account NAME]
"""
from __future__ import annotations

import argparse
import datetime
import json
import warnings
from pathlib import Path

from archive_job import service

warnings.filterwarnings("ignore")

REDACT = ("instance", "instance_id", "plan_id", "crn", "account", "token", "user", "email")


def redact(d):
    if isinstance(d, dict):
        return {k: ("[REDACTED]" if any(r in k.lower() for r in REDACT) else redact(v)) for k, v in d.items()}
    if isinstance(d, list):
        return [redact(v) for v in d]
    return d


def usage_summary(svc):
    """QPU time available in the current usage window of the account (read-only)."""
    try:
        return redact(svc.usage())
    except Exception as e:  # pragma: no cover - depends on the account type
        return {"error": f"{type(e).__name__}: {e}"}


def snapshot(be, svc, outdir, tag):
    """Write calibration_<tag>.json and context_<tag>.json for the backend object `be`; returns the context."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    props = be.properties(refresh=True).to_dict()
    (out / f"calibration_{tag}.json").write_text(json.dumps(props, default=str), encoding="utf-8")
    try:
        pending = be.status().pending_jobs
    except Exception:
        pending = None
    ctx = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "backend": be.name,
           "last_update_date": str(props.get("last_update_date")), "pending_jobs": pending,
           "usage": usage_summary(svc)}
    saved = {k: v for k, v in ctx.items() if k != "usage"}
    (out / f"context_{tag}.json").write_text(json.dumps(saved, indent=1, default=str), encoding="utf-8")
    return ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend")
    ap.add_argument("outdir")
    ap.add_argument("tag")
    ap.add_argument("--account", default=None, help="name of a saved Qiskit Runtime account")
    args = ap.parse_args()
    svc = service(args.account)
    be = svc.backend(args.backend)
    ctx = snapshot(be, svc, args.outdir, args.tag)
    u = ctx["usage"]
    cal = Path(args.outdir) / f"calibration_{args.tag}.json"
    print(f"{be.name}: calibration of {ctx['last_update_date']} -> {cal}")
    print(f"pending jobs: {ctx['pending_jobs']}; account QPU time available: {u.get('usage_remaining_seconds')} s of "
          f"{u.get('usage_limit_seconds')} s (window {u.get('usage_period')})")


if __name__ == "__main__":
    main()
