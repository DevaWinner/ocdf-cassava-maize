#!/usr/bin/env python3
"""Verify the integrity of the released artefacts and the pre-registration audit trail.

Checks, in order:
  1. every file listed in CHECKSUMS.sha256 is present and unchanged;
  2. each protocol record matches the SHA-256 written when it was frozen;
  3. the hash chain links the records in the registered order:
       frozen_choices_v2  ->  protocol_v3_registration  ->  frozen_choices_v3  ->  sealed-test opening;
  4. the timestamps are in that order (the sealed test was opened after every choice was frozen);
  5. each deployed model file is the one named, by hash, in its frozen record.

Usage:  python scripts/verify_artifacts.py         (exit code 0 = all checks passed)
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "protocol"
results: list[tuple[bool, str]] = []


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(ok: bool, msg: str) -> None:
    results.append((ok, msg))
    print(("  PASS  " if ok else "  FAIL  ") + msg)


def recorded(path: Path) -> str:
    return path.read_text().split()[0].strip()


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def main() -> int:
    print("1. Release checksums")
    manifest = ROOT / "CHECKSUMS.sha256"
    bad = missing = n = 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        f = ROOT / rel.lstrip("*")
        n += 1
        if not f.exists():
            missing += 1
        elif sha(f) != digest:
            bad += 1
            print(f"        changed: {rel}")
    check(bad == 0 and missing == 0, f"{n} files listed; {missing} missing; {bad} changed")

    print("2. Protocol records match their frozen hashes")
    pairs = [("frozen_choices_v2.json", "frozen_choices_v2.sha256"),
             ("protocol_v3_registration.json", "protocol_v3_registration.json.sha256"),
             ("protocol_v3_registration_text.md", "protocol_v3_registration_text.md.sha256"),
             ("frozen_choices_v3.json", "frozen_choices_v3.sha256")]
    h = {}
    for rec, digest_file in pairs:
        h[rec] = sha(P / rec)
        check(h[rec] == recorded(P / digest_file), f"{rec}  {h[rec][:12]}…")

    print("3. Hash chain")
    reg = json.loads((P / "protocol_v3_registration.json").read_text())
    v3 = json.loads((P / "frozen_choices_v3.json").read_text())
    v2 = json.loads((P / "frozen_choices_v2.json").read_text())
    unsealed = json.loads((P / "d5_unsealed.json").read_text())
    final = json.loads((ROOT / "records/nb11/final_results.json").read_text())
    check(reg["builds_on"]["frozen_choices_v2_sha256"] == h["frozen_choices_v2.json"],
          "registration cites the frozen Phase 1 choices")
    check(v3["registration_sha256"] == h["protocol_v3_registration.json"],
          "Phase 2 freeze cites the registration")
    check(unsealed["registration_sha256"] == h["protocol_v3_registration.json"]
          and unsealed["frozen_choices_v3_sha256"] == h["frozen_choices_v3.json"],
          "sealed-test opening cites the registration and the Phase 2 freeze")
    check(final["frozen_v3_sha256"] == h["frozen_choices_v3.json"],
          "final results cite the Phase 2 freeze")

    print("4. Timeline")
    t = [("Phase 1 choices frozen", v2["frozen_utc"]), ("Phase 2 registered", reg["registered_utc"]),
         ("Phase 2 choices frozen", v3["frozen_utc"]), ("sealed test opened", unsealed["unsealed_utc"])]
    for name, value in t:
        print(f"        {value[:19].replace('T', ' ')} UTC  {name}")
    check(all(ts(a[1]) < ts(b[1]) for a, b in zip(t, t[1:])), "events occurred in the registered order")

    print("5. Deployed models")
    for crop, rec in v2["deployed_instance"].items():
        f = ROOT / "models/phase1" / Path(rec["file"]).name
        check(f.exists() and sha(f) == rec["sha256"], f"Phase 1 {crop}: {f.name}")
    for crop, rec in v3["deployed_instance_v3"].items():
        f = ROOT / "models/phase2_v3" / Path(rec["file"]).name
        check(f.exists() and sha(f) == rec["sha256"], f"Phase 2 {crop}: {f.name}")

    failed = sum(not ok for ok, _ in results)
    print(f"\n{len(results) - failed} of {len(results)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
