"""Runtime qualification script for gfx1031.

Each primitive is tested in a separate subprocess so a segfault in one
does not hide the status of the others.

Run with:
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python test_gfx1031_runtime.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")


_TESTS = {
    "selective_scan_cuda.fwd": """
        import torch
        import selective_scan_cuda
        B, D, L, N = 1, 16, 32, 8
        u = torch.randn(B, D, L, device='cuda', dtype=torch.float16)
        delta = torch.randn(B, D, L, device='cuda', dtype=torch.float16)
        A = torch.randn(D, N, device='cuda', dtype=torch.float32)
        Bt = torch.randn(B, 1, N, L, device='cuda', dtype=torch.float32)
        C = torch.randn(B, 1, N, L, device='cuda', dtype=torch.float32)
        Dp = torch.randn(D, device='cuda', dtype=torch.float16)
        out, x, z = selective_scan_cuda.fwd(u, delta, A, Bt, C, Dp, None, None, False)
        print(f"out.shape={out.shape}")
    """,
    "causal_conv1d_fn": """
        import torch
        from causal_conv1d import causal_conv1d_fn
        B, D, L, W = 1, 16, 32, 4
        x = torch.randn(B, D, L, device='cuda', dtype=torch.float16)
        weight = torch.randn(D, W, device='cuda', dtype=torch.float16)
        bias = torch.randn(D, device='cuda', dtype=torch.float16)
        y = causal_conv1d_fn(x, weight, bias, activation='silu')
        print(f"y.shape={y.shape}")
    """,
    "Mamba-1 block": """
        import torch
        from mamba_ssm import Mamba
        B, L, D = 1, 64, 64
        block = Mamba(d_model=D, d_state=16, d_conv=4, expand=2).to('cuda').to(torch.float16)
        x = torch.randn(B, L, D, device='cuda', dtype=torch.float16, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        print(f"y.shape={y.shape} grads_ok={x.grad is not None}")
    """,
    "Mamba-2 block": """
        import torch
        from mamba_ssm import Mamba2
        B, L, D = 1, 64, 64
        block = Mamba2(d_model=D, d_state=16, d_conv=4, expand=2, headdim=64, chunk_size=64, ngroups=1)
        block = block.to('cuda').to(torch.float16)
        x = torch.randn(B, L, D, device='cuda', dtype=torch.float16, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        print(f"y.shape={y.shape} grads_ok={x.grad is not None}")
    """,
    "Mamba-3 SISO": """
        import torch
        from mamba_ssm import Mamba3
        B, L, D = 1, 64, 64
        block = Mamba3(d_model=D, d_state=16, headdim=32, is_mimo=False).to('cuda').to(torch.float16)
        x = torch.randn(B, L, D, device='cuda', dtype=torch.float16)
        y = block(x)
        print(f"y.shape={y.shape}")
    """,
    "Mamba-3 MIMO": """
        import torch
        from mamba_ssm import Mamba3
        B, L, D = 1, 64, 64
        block = Mamba3(d_model=D, d_state=16, headdim=32, is_mimo=True, mimo_rank=4).to('cuda').to(torch.float16)
        x = torch.randn(B, L, D, device='cuda', dtype=torch.float16)
        y = block(x)
        print(f"y.shape={y.shape}")
    """,
}


def _run_one(name: str, code: str) -> tuple[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode == 0:
        return "PASS", (proc.stdout.strip() or "ok")
    if proc.returncode == -11:
        return "SIGSEGV", (proc.stderr.strip() or "SIGSEGV")
    return f"FAIL({proc.returncode})", (proc.stderr.strip() or proc.stdout.strip() or "unknown")


if __name__ == "__main__":
    print(f"HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
    print()

    results = {}
    for name, code in _TESTS.items():
        status, detail = _run_one(name, code)
        print(f"[{status}] {name}: {detail}")
        results[name] = (status, detail)

    print()
    print("Summary:")
    for name, (status, _) in results.items():
        print(f"  {status:10} {name}")
