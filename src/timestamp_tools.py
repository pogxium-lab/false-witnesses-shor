"""External time-stamps of pre-registration files: signed RFC 3161 time-stamps from a public time-stamping
authority (default FreeTSA) and OpenTimestamps proofs (Bitcoin). Only SHA-256 hashes leave the machine.

RFC 3161 (OpenSSL and an HTTPS POST):
    openssl ts -query -data FILE -sha256 -cert -out FILE.tsq
    POST FILE.tsq to https://freetsa.org/tsr (Content-Type: application/timestamp-query) -> FILE.tsr
    openssl ts -verify -in FILE.tsr -queryfile FILE.tsq -CAfile cacert.pem -untrusted tsa.crt
The TSA certificates (cacert.pem, tsa.crt) are downloaded once into the certificate directory and kept with the
proofs. The signed generation time is read with `openssl ts -reply -in FILE.tsr -text`.

OpenTimestamps: `ots stamp FILE` (FILE.ots, pending until the calendars commit to a Bitcoin block),
`ots upgrade FILE.ots` later completes the proof, `ots verify FILE.ots` checks it.

    python src/timestamp_tools.py stamp FILE [FILE ...] [--certs DIR] [--ots "ots"] [--log LOG.json]
    python src/timestamp_tools.py verify FILE [FILE ...] [--certs DIR]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path

TSA_URL = "https://freetsa.org/tsr"
TSA_CERTS = {"cacert.pem": "https://freetsa.org/files/cacert.pem", "tsa.crt": "https://freetsa.org/files/tsa.crt"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def ensure_certs(certs):
    certs = Path(certs)
    certs.mkdir(parents=True, exist_ok=True)
    for name, url in TSA_CERTS.items():
        f = certs / name
        if not f.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "timestamp-tools/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                f.write_bytes(r.read())
    return {n: sha256(certs / n) for n in TSA_CERTS}


def tsa_time(tsr):
    """Signed generation time of a time-stamp response (UTC, as printed by OpenSSL)."""
    code, out = _run(["openssl", "ts", "-reply", "-in", str(tsr), "-text"])
    for line in out.splitlines():
        if line.strip().startswith("Time stamp:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(out)


def tsa_stamp(file, certs, url=TSA_URL):
    file = Path(file)
    tsq, tsr = Path(str(file) + ".tsq"), Path(str(file) + ".tsr")
    code, out = _run(["openssl", "ts", "-query", "-data", str(file), "-sha256", "-cert", "-out", str(tsq)])
    if code:
        raise RuntimeError(out)
    req = urllib.request.Request(url, data=tsq.read_bytes(), method="POST",
                                 headers={"Content-Type": "application/timestamp-query",
                                          "User-Agent": "timestamp-tools/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        tsr.write_bytes(r.read())
    return tsa_verify(file, certs)


def tsa_verify(file, certs):
    file, certs = Path(file), Path(certs)
    tsq, tsr = Path(str(file) + ".tsq"), Path(str(file) + ".tsr")
    code, out = _run(["openssl", "ts", "-verify", "-in", str(tsr), "-queryfile", str(tsq),
                      "-CAfile", str(certs / "cacert.pem"), "-untrusted", str(certs / "tsa.crt")])
    code2, out2 = _run(["openssl", "ts", "-verify", "-in", str(tsr), "-data", str(file),
                        "-CAfile", str(certs / "cacert.pem"), "-untrusted", str(certs / "tsa.crt")])
    def status(text):
        return next((ln.strip() for ln in text.splitlines() if ln.startswith("Verification")), text[-200:])
    return dict(tsr=tsr.name, time=tsa_time(tsr), verify_query=status(out), verify_data=status(out2),
                ok=(code == 0 and code2 == 0))


def ots_stamp(files, ots_cmd="ots"):
    code, out = _run(shlex.split(ots_cmd, posix=False) + ["stamp", *[str(f) for f in files]])
    return dict(command=f"{ots_cmd} stamp", returncode=code, output=out[-2000:],
                proofs={Path(f).name: Path(str(f) + ".ots").exists() for f in files})


def stamp_all(files, certs, ots_cmd="ots", url=TSA_URL):
    """RFC 3161 time-stamp of each file, then one OpenTimestamps call for all files."""
    out = dict(local_time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               tsa_url=url, tsa_certificates=ensure_certs(certs), files={})
    for f in files:
        try:
            r = tsa_stamp(f, certs, url)
        except Exception as e:
            r = dict(ok=False, error=f"{type(e).__name__}: {e}")
        out["files"][Path(f).name] = dict(sha256=sha256(f), rfc3161=r)
        print(f"  RFC 3161 {Path(f).name}: {r.get('time', r.get('error'))} ({'verified' if r.get('ok') else 'FAILED'})",
              flush=True)
    out["opentimestamps"] = ots_stamp(files, ots_cmd)
    print(f"  OpenTimestamps: return code {out['opentimestamps']['returncode']}, proofs "
          f"{out['opentimestamps']['proofs']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["stamp", "verify"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--certs", default=None, help="TSA certificate directory (default: <dir of first file>/tsa)")
    ap.add_argument("--ots", default="ots", help="OpenTimestamps client command")
    ap.add_argument("--log", default=None, help="stamp: JSON log of the results")
    args = ap.parse_args()
    certs = Path(args.certs) if args.certs else Path(args.files[0]).parent / "tsa"
    if args.action == "stamp":
        res = stamp_all(args.files, certs, args.ots)
        if args.log:
            Path(args.log).write_text(json.dumps(res, indent=1), encoding="utf-8")
        sys.exit(0 if all(v["rfc3161"].get("ok") for v in res["files"].values()) else 1)
    bad = 0
    for f in args.files:
        r = tsa_verify(f, certs)
        bad += not r["ok"]
        print(f"{f}: {r['time']} | {r['verify_query']} | {r['verify_data']}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
