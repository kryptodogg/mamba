"""test_gfx1031_runtime.py — Qualify upstream HIPified Mamba primitives on gfx1030.

Each primitive runs in an isolated subprocess.  Dispatch to chakra fallbacks
is **disabled** (MAMBA_GFX1031=0) so we measure actual upstream behaviour.

Output: a JSONL log of (primitive, status, detail) entries suitable for
appending to GFX1031_MANIFEST.evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ENV = {
    **os.environ,
    "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
    "HSA_ENABLE_SDMA": "0",
    "MAMBA_GFX1031": "0",  # DISABLE chakra dispatch — test upstream
}
_TIMEOUT_S = 90
_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "qualification"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _run(name: str, code: str) -> dict:
    """Run *code* in a subprocess and return {status, detail, rc, wall_s}."""
    started = time.monotonic()
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=_TIMEOUT_S,
            env=_ENV,
        )
        wall = time.monotonic() - started
        if r.returncode == 0:
            return {"status": "pass", "detail": r.stdout.strip()[-200:], "rc": 0, "wall_s": wall}
        else:
            stderr = r.stderr.strip()[-300:] or r.stdout.strip()[-300:]
            return {"status": "fail", "detail": stderr, "rc": r.returncode, "wall_s": wall}
    except subprocess.TimeoutExpired:
        wall = time.monotonic() - started
        return {"status": "timeout", "detail": f"timeout after {_TIMEOUT_S}s", "rc": -1, "wall_s": wall}
    except Exception as e:
        wall = time.monotonic() - started
        return {"status": "crash", "detail": str(e)[-300:], "rc": -1, "wall_s": wall}


# ── Quality-of-life: a helper that routes stdout → /dev/null so we
# only capture stderr on pass.  For verbose pass output use _run_raw. ─

PRIMITIVES = {
    # ── selective_scan_cuda.fwd ──────────────────────────────────
    "selective_scan_cuda": """
import torch
from mamba_ssm.ops.selective_scan_interface import SelectiveScanFn
B, D, L, N = 1, 32, 64, 16
device = "cuda"; dtype = torch.float16
u = torch.randn(B, D, L, device=device, dtype=dtype).contiguous()
delta = torch.randn(B, D, L, device=device, dtype=dtype).contiguous()
A = torch.randn(D, N, device=device, dtype=dtype).contiguous()
Bv = torch.randn(B, 1, N, L, device=device, dtype=dtype).contiguous()
Cv = torch.randn(B, 1, N, L, device=device, dtype=dtype).contiguous()
y = SelectiveScanFn.apply(u, delta, A, Bv, Cv, None, None, None, False)
print(f"PASS shape={list(y.shape)} dtype={y.dtype}")
""",

    # ── causal_conv1d_fn ─────────────────────────────────────────
    "causal_conv1d_fn": """
import torch
from causal_conv1d import causal_conv1d_fn
B, D, L = 1, 64, 128
device = "cuda"; dtype = torch.float16
x = torch.randn(B, D, L, device=device, dtype=dtype).contiguous()
w = torch.randn(D, 4, device=device, dtype=dtype).contiguous()
y = causal_conv1d_fn(x, w, None, True)
print(f"PASS shape={list(y.shape)} dtype={y.dtype}")
""",

    # ── Mamba-1 block (separate path) ────────────────────────────
    "mamba1_block": """
import torch
from mamba_ssm import Mamba
device = "cuda"; dtype = torch.float16
model = Mamba(d_model=64, d_state=16, use_fast_path=False).to(device).to(dtype)
x = torch.randn(1, 32, 64, device=device, dtype=dtype)
y = model(x)
print(f"PASS shape={list(y.shape)} dtype={y.dtype}")
""",

    # ── Mamba-2 block (separate path) ────────────────────────────
    "mamba2_separate": """
import torch
from mamba_ssm import Mamba2
device = "cuda"; dtype = torch.float16
model = Mamba2(d_model=64, d_state=16, use_mem_eff_path=False).to(device).to(dtype)
x = torch.randn(1, 32, 64, device=device, dtype=dtype)
y = model(x)
print(f"PASS shape={list(y.shape)} dtype={y.dtype}")
""",

    # ── Mamba-2 block (fused path) ───────────────────────────────
    "mamba2_fused": """
import torch
from mamba_ssm import Mamba2
device = "cuda"; dtype = torch.float16
model = Mamba2(d_model=64, d_state=16, use_mem_eff_path=True).to(device).to(dtype)
x = torch.randn(1, 32, 64, device=device, dtype=dtype)
y = model(x)
print(f"PASS shape={list(y.shape)} dtype={y.dtype}")
""",

    # ── selective_state_update ───────────────────────────────────
    "selective_state_update": """
import torch
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
B, H, D, N = 1, 4, 16, 8
device = "cuda"; dtype = torch.float16
state = torch.zeros(B, H, D, N, device=device, dtype=dtype)
x = torch.randn(B, H, D, device=device, dtype=dtype)
dt = torch.randn(B, H, D, device=device, dtype=dtype)
A = torch.randn(H, D, N, device=device, dtype=dtype)
Bv = torch.randn(B, 1, N, device=device, dtype=dtype)
Cv = torch.randn(B, 1, N, device=device, dtype=dtype)
y = selective_state_update(state, x, dt, A, Bv, Cv, None, None, None, False)
print(f"PASS shape={list(y.shape)} dtype={y.dtype}")
""",

    # ── Mamba-3 SISO ────────────────────────────────────────────
    "mamba3_siso": """
import torch
from mamba_ssm import Mamba3
device = "cuda"; dtype = torch.float16
model = Mamba3(d_model=64, d_state=16, headdim=32, is_mimo=False).to(device).to(dtype)
x = torch.randn(1, 32, 64, device=device, dtype=dtype)
y = model(x)
print(f"PASS shape={list(y.shape)} dtype={y.dtype}")
""",

    # ── Mamba-3 MIMO (construction only — TileLang unavailable) ──
    "mamba3_mimo": """
import torch
from mamba_ssm import Mamba3
device = "cuda"; dtype = torch.float16
try:
    model = Mamba3(d_model=64, d_state=16, headdim=32, is_mimo=True, mimo_rank=4).to(device).to(dtype)
    print("CONSTRUCT_OK")
except Exception as e:
    print(f"CONSTRUCT_FAIL: {e}")
""",
}


def main() -> int:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    results = {}
    evidence = []

    print(f"=== gfx1030 upstream qualification  {timestamp} ===")
    print(f"    dispatch: OFF (MAMBA_GFX1031=0)")
    print(f"    timeout:  {_TIMEOUT_S}s per primitive")
    print()

    for name, code in PRIMITIVES.items():
        sys.stdout.write(f"  {name:32s} ... ")
        sys.stdout.flush()
        result = _run(name, code)
        results[name] = result
        status = result["status"]
        print(f"{status:7s}  ({result['wall_s']:.1f}s)")

        evidence.append({
            "timestamp": timestamp,
            "primitive": name,
            "status": status,
            "detail": result["detail"],
            "wall_s": result["wall_s"],
        })

    # ── Summary ─────────────────────────────────────────────────
    passed = sum(1 for v in results.values() if v["status"] == "pass")
    failed = sum(1 for v in results.values() if v["status"] in ("fail", "crash", "timeout"))
    print(f"\n  {passed} passed, {failed} failed, {len(results)} total")

    # ── Write evidence file ──────────────────────────────────────
    evidence_path = _OUTPUT_DIR / "upstream_qual.jsonl"
    with open(evidence_path, "a") as f:
        for entry in evidence:
            f.write(json.dumps(entry) + "\n")
    print(f"\nEvidence appended to {evidence_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
