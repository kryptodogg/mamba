"""mamba_ssm/ops/gfx1031 — gfx1031 (RDNA2) compatibility layer.

This package contains:
  * ``IS_GFX1031``: runtime flag True when on RDNA2 gfx1030/gfx1031.
  * ``GFX1031_MANIFEST``: an explicit registry of which upstream native
    extensions work on gfx1031 and which require a fallback.
  * Lazy adapter getters that import from ``chakra.ssm.compat.gfx1031_patches``
    only when chakra is available (i.e. within the Synesthesia repo).

Design principle (from prompt.md):
  1. Use upstream kernels that Hipify successfully and pass qualification.
  2. Patch only the failing boundaries.
  3. Keep PyTorch references as numerical oracles.

Qualification results (2026-07-11, RX 6700 XT, ROCm 7.1):
  * selective_scan_cuda.fwd: FAIL (RuntimeError type mismatch at runtime)
  * causal_conv1d_fn: FAIL (causal_conv1d package not in venv / SIGSEGV)
  * Mamba-1 block: CRASH (SIGSEGV via hip::ihipLaunchKernel)
  * Mamba-2 block: FAIL (causal_conv1d_fwd_function is None)
  * Mamba-3 SISO: PASS (with PTX inline asm → Triton-native math fix)
  * Mamba-3 MIMO: FAIL (TileLang backend unsupported on gfx1031)
"""

from __future__ import annotations

import os
from typing import Callable

__all__ = [
    "IS_GFX1031",
    "GFX1031_MANIFEST",
    "get_selective_scan_adapter",
    "get_causal_conv1d_adapter",
    "get_causal_conv1d_fwd_function_adapter",
    "get_causal_conv1d_update_adapter",
    "get_causal_conv1d_varlen_states_adapter",
    "get_mamba_chunk_scan_combined_adapter",
    "get_selective_state_update_adapter",
]


GFX1031_MANIFEST = {
    "selective_scan_cuda": {
        "build": True,
        "hipifies": True,
        "runtime_status": "crashes_on_gfx1031",
        "qual_result": "FAIL: RuntimeError type mismatch",
        "fallback": "chakra.ssm.compat.gfx1031_patches._patched_selective_scan_fn",
    },
    "causal_conv1d_cuda": {
        "build": True,
        "hipifies": True,
        "runtime_status": "not_installed_or_segfault",
        "qual_result": "FAIL: ModuleNotFoundError for causal_conv1d",
        "fallback": "chakra.ssm.compat.gfx1031_patches._patched_causal_conv1d_fn",
    },
    "mamba1_block": {
        "runtime_status": "crashes_on_gfx1031",
        "qual_result": "CRASH: SIGSEGV via hip::ihipLaunchKernel",
        "fallback": "use_fast_path=False + chakra selective_scan",
    },
    "mamba2_fused_path": {
        "runtime_status": "fails_on_gfx1031",
        "qual_result": "FAIL: causal_conv1d_fwd_function is None",
        "fallback": "use_mem_eff_path=False + chakra mamba_chunk_scan_combined",
    },
    "mamba3_siso": {
        "runtime_status": "passes_on_gfx1031",
        "qual_result": "PASS (with PTX→Triton-native math fix)",
        "fallback": None,
    },
    "mamba3_mimo_tilelang": {
        "runtime_status": "unsupported_backend",
        "qual_result": "FAIL: AttributeError in TileLang",
        "fallback": None,
    },
}


def _detect_gfx1031() -> bool:
    """Return True if the current process targets gfx1030/gfx1031."""
    env = os.environ.get("MAMBA_GFX1031", "")
    if env in {"1", "true", "True", "yes"}:
        return True
    if env in {"0", "false", "False", "no"}:
        return False

    try:
        import torch
    except ImportError:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        name = (getattr(props, "name", "") or "").lower()
        gcn = (getattr(props, "gcnArchName", "") or "").lower()
    except Exception:
        return False

    return (
        "gfx1031" in gcn
        or "gfx1030" in gcn
        or "rx 6700" in name
        or "rx 6800" in name
        or "rx 6900" in name
    )


IS_GFX1031 = _detect_gfx1031()


# ---------------------------------------------------------------------------
# Lazy adapter getters.  Each returns a callable whose signature matches
# the corresponding upstream function, but delegates to the chakra Triton
# kernel.  The import is deferred to avoid hard chakra dependencies when
# the fork is used outside Synesthesia (IS_GFX1031 will be False).
# ---------------------------------------------------------------------------

def get_selective_scan_adapter() -> Callable:
    from chakra.ssm.compat.gfx1031_patches import _patched_selective_scan_fn
    return _patched_selective_scan_fn


def get_causal_conv1d_adapter() -> Callable:
    from chakra.ssm.compat.gfx1031_patches import _patched_causal_conv1d_fn
    return _patched_causal_conv1d_fn


def get_causal_conv1d_fwd_function_adapter() -> Callable:
    from chakra.ssm.compat.gfx1031_patches import _patched_causal_conv1d_fwd_function
    return _patched_causal_conv1d_fwd_function


def get_causal_conv1d_update_adapter() -> Callable:
    from chakra.ssm.compat.gfx1031_patches import _patched_causal_conv1d_update
    return _patched_causal_conv1d_update


def get_causal_conv1d_varlen_states_adapter() -> Callable:
    from chakra.ssm.compat.gfx1031_patches import _patched_causal_conv1d_varlen_states
    return _patched_causal_conv1d_varlen_states


def get_mamba_chunk_scan_combined_adapter() -> Callable:
    from chakra.ssm.compat.gfx1031_patches import _patched_mamba_chunk_scan_combined
    return _patched_mamba_chunk_scan_combined


def get_selective_state_update_adapter() -> Callable:
    from chakra.ssm.compat.gfx1031_patches import _patched_selective_state_update
    return _patched_selective_state_update
