#!/usr/bin/env python3
from __future__ import annotations

import base64
import hmac
import hashlib
import json
import os
import sys
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _b64url_decode(s: str) -> bytes:
    # Restore padding if stripped
    pad = '=' * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def main() -> int:
    key = os.getenv("PROOF_SIGN_KEY") or os.getenv("BRAIN_PROOF_SIGN_KEY")
    if not key:
        print("No PROOF_SIGN_KEY provided; skipping verification.")
        return 0

    proof_path = Path(os.getenv("PROOF_JSON_PATH", "proof.json")).resolve()
    sig_path = Path(os.getenv("PROOF_SIG_PATH", "proof.json.sig")).resolve()
    if not proof_path.exists() or not sig_path.exists():
        print(f"ERROR: proof or signature missing: {proof_path}, {sig_path}", file=sys.stderr)
        return 2

    msg = _read(proof_path).encode("utf-8")
    sig_b64 = _read(sig_path).strip()
    try:
        expected = hmac.new(key.encode("utf-8"), msg, hashlib.sha256).digest()
        got = _b64url_decode(sig_b64)
    except Exception as e:
        print(f"ERROR: invalid signature encoding: {e}", file=sys.stderr)
        return 2

    if not hmac.compare_digest(expected, got):
        print("ERROR: proof signature verification FAILED", file=sys.stderr)
        return 1
    print("Proof signature verification: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
