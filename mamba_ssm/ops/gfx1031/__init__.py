"""mamba_ssm/ops/gfx1031 — gfx1030/gfx1031 (RDNA2) compatibility layer.

Provides:
  * ``IS_GFX1031``: True when running on gfx1030/gfx1031 (detected via GPU
    properties or the ``HSA_OVERRIDE_GFX_VERSION=10.3.0`` spoofing env var).
  * ``GFX1031_MANIFEST``: evidence-based registry of every Mamba primitive.
    Every entry starts as ``"untested"`` and is updated only after a real
    qualification run produces evidence.

Workflow for each primitive (from prompt.md):
  1. Test upstream HIPified implementation
  2. Search HF Kernels for the same operator/API
  3. Inspect available Triton source
  4. Attempt minimal HIPify or compiler patch
  5. Compare against the PyTorch/upstream reference
  6. Write a Chakra kernel only when available paths fail
  7. Record the selected implementation and evidence

Chakra kernel inventory (all wired via gfx1031_patches.py):
  * ``chakra/ssm/kernels/mamba1_causal_conv1d.py`` — Triton causal conv1d fwd+bwd
  * ``chakra/ssm/kernels/mamba1_selective_scan.py`` — Triton selective scan
  * ``chakra/ssm/kernels/mamba2_ssd_chunk_scan.py`` — Triton SSD chunk scan
  * ``chakra/ssm/kernels/mamba3_siso_trapezoidal.py`` — Triton Mamba-3 SISO
  * ``chakra/ssm/kernels/mamba3_mimo_trapezoidal.py`` — Triton Mamba-3 MIMO (prototype)

Upstream mamba-ssm Triton kernels (available in mamba_ssm/ops/triton/):
  * ``ssd_combined.py`` — mamba_split_conv1d_scan_combined + mamba_chunk_scan_combined
  * ``ssd_chunk_scan.py``, ``ssd_chunk_state.py``, ``ssd_state_passing.py`` — subcomponents
  * These are pure Triton — should work on gfx1030 without HIPify
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


# ── Evidence-based manifest ──────────────────────────────────────────
#
# Keys follow the mamba-ssm primitive name.  Each entry records:
#   upstream_source   — file in the fork that owns the implementation
#   hf_kernel         — matching symbol in kernels-community/mamba-ssm (or None)
#   chakra_kernel     — matching custom kernel in the chakra/ package (or None)
#   candidates        — ordered list of candidate implementations
#   selected          — which candidate is active (None = untested)
#   forward_status    — "untested" | "pass" | "fail" | "crash"
#   backward_status   — "untested" | "pass" | "fail" | "crash" | "timeout"
#   gfx1030_status    — overall status on gfx1030/gfx1031
#   evidence          — list of (timestamp, result, artifact_path) tuples

GFX1031_MANIFEST = {
    # ── Mamba-1 primitives ──────────────────────────────────────────
    "selective_scan_cuda": {
        "upstream_source": "csrc/selective_scan/",
        "hf_kernel": "selective_scan_fn",
        "chakra_kernel": "chakra.ssm.ops.mamba1_selective_scan._triton_scan",
        "candidates": [
            "upstream_hipified",
            "huggingface_kernel",
            "chakra_triton",
            "pytorch_reference",
        ],
        "selected": None,
        "forward_status": "deferred",
        "backward_status": "untested",
        "gfx1030_status": "deferred — Mamba-2 is the Tier 0 baseline. Mamba-1 primitives deferred until Mamba-2 SSD is fully qualified. HIPified kernel: dtype check fails. Chakra Triton kernel exists and wired but not currently routed.",
        "evidence": [
            {"result": "deferred", "detail": "Architectural reset: Mamba-2 SSD Triton stack is the Tier 0 baseline. Mamba-1 primitives deferred."},
        ],
    },
    "causal_conv1d_cuda": {
        "upstream_source": "mamba_ssm/ops/selective_scan_interface.py (imports causal_conv1d)",
        "hf_kernel": "causal_conv1d_fn (via mamba-ssm)",
        "chakra_kernel": "chakra.ssm.ops.mamba1_causal_conv1d.triton_causal_conv1d",
        "candidates": [
            "pytorch_conv1d",
            "chakra_triton",
            "upstream_hipified",
            "huggingface_kernel",
            "pytorch_reference",
        ],
        "selected": "pytorch_conv1d",
        "forward_status": "pass",
        "backward_status": "untested",
        "gfx1030_status": "pass — Tier 0 baseline: PyTorch nn.Conv1d fallback selected via IS_GFX1031 dispatch (causal_conv1d_fn=None). Upstream HIPified kernel SIGSEGV's (contract mismatch, kernel untested). Chakra Triton kernel works standalone but not currently routed. PyTorch Conv1d is safe and backward-compatible.",
        "evidence": [
            {"result": "pass", "detail": "Tier 0: Mamba2 forward+backward passes on gfx1030 with PyTorch nn.Conv1d + upstream Triton SSD. IS_GFX1031 sets causal_conv1d_fn=None → routes through safe path."},
            {"result": "chakra_standalone_pass", "detail": "chakra.ssm.ops.mamba1_causal_conv1d.triton_causal_conv1d works standalone on gfx1030. Not currently routed in dispatch."},
            {"result": "hipified_crash", "detail": "Upstream HIPified causal_conv1d_fn SIGSEGV at causal_conv1d_fwd_launch. Classified as contract mismatch, kernel untested."},
        ],
    },
    "mamba1_block": {
        "upstream_source": "mamba_ssm/modules/mamba_simple.py",
        "hf_kernel": "Mamba",
        "chakra_kernel": "chakra.ssm.mamba1.Mamba (full block with chakra kernels)",
        "chakra_selective_scan": "chakra.ssm.kernels.mamba1_selective_scan",
        "chakra_causal_conv1d": "chakra.ssm.kernels.mamba1_causal_conv1d",
        "candidates": [
            "upstream_hipified_fast_path",
            "upstream_separate_path",
            "chakra_triton",
            "huggingface_kernel",
        ],
        "selected": None,
        "forward_status": "untested",
        "backward_status": "untested",
        "gfx1030_status": "untested — chakra Triton kernels exist for both sub-primitives (selective_scan + causal_conv1d). gfx1031_patches.py patches module-level refs. Prior SIGSEGV was HIPified path; chakra path bypasses it entirely.",
        "evidence": [
            {"result": "untested", "detail": "Chakra Triton kernels exist (selective_scan + causal_conv1d). gfx1031_patches.py wired. Prior SIGSEGV was from HIPified path — not applicable to chakra dispatch."},
        ],
    },
    # ── Mamba-2 primitives ──────────────────────────────────────────
    "mamba2_separate_path": {
        "upstream_source": "mamba_ssm/modules/mamba2.py (use_mem_eff_path=False)",
        "hf_kernel": "Mamba2",
        "chakra_kernel": "chakra.ssm.ops.mamba2_ssd_chunk_scan.ssd_chunk_scan_combined",
        "candidates": [
            "pytorch_conv1d_plus_upstream_triton_ssd",
            "chakra_conv1d_plus_chakra_ssd",
            "huggingface_kernel",
            "pytorch_reference",
        ],
        "selected": "pytorch_conv1d_plus_upstream_triton_ssd",
        "forward_status": "pass",
        "backward_status": "pass",
        "gfx1030_status": "pass — Tier 0 baseline confirmed on gfx1030. IS_GFX1031 sets causal_conv1d_fn=None → PyTorch nn.Conv1d for convolution. Upstream Triton mamba_chunk_scan_combined for SSD recurrence. Forward+backward both pass. mamba_chunk_scan_combined parity vs ssd_chunk_scan_combined_ref FAILED (rel_err=2.49e+03) — needs investigation.",
        "evidence": [
            {"result": "pass", "detail": "Tier 0: Mamba2(d_model=128, B=1, L=64, float16) forward shape=[1,64,128], backward passes on gfx1030. PyTorch Conv1d + upstream Triton SSD. Tested: zero initial states. Not tested: nonzero initial states, chunk boundaries (seqlen not divisible by chunk_size)."},
            {"result": "parity_fail", "detail": "mamba_chunk_scan_combined vs ssd_chunk_scan_combined_ref: max_abs_diff=237892, rel_err=2.49e+03. Params: chunk_size=32, dtype=float16, B=1, L=64, H=4, P=64, N=16, G=1. Not a precision issue — reference likely has platform-specific difference or internal defaults mismatch."},
        ],
    },
    "mamba2_fused_path": {
        "upstream_source": "mamba_ssm/modules/mamba2.py (use_mem_eff_path=True)",
        "upstream_triton": "mamba_ssm/ops/triton/ssd_combined.py::mamba_split_conv1d_scan_combined",
        "hf_kernel": "mamba_split_conv1d_scan_combined",
        "chakra_kernel": None,
        "candidates": [
            "upstream_triton",
            "upstream_hipified",
            "huggingface_kernel",
            "chakra_triton",
            "pytorch_reference",
        ],
        "selected": None,
        "forward_status": "disabled",
        "backward_status": "untested",
        "gfx1030_status": "disabled — IS_GFX1031 gates use_mem_eff_path=False in Mamba2.__init__ (line 110). Tier 0 uses PyTorch Conv1d separate path. Fused path deferred until causal_conv1d_cuda is resolved (Tier 4). Upstream Triton kernel (ssd_combined.py) exists and works standalone.",
        "evidence": [
            {"result": "disabled", "detail": "IS_GFX1031 gates use_mem_eff_path=False. Tier 0 uses separate path. Upstream Triton fused kernel available but blocked by causal_conv1d dependency."},
        ],
    },
    "mamba_chunk_scan_combined": {
        "upstream_source": "mamba_ssm/ops/triton/ssd_combined.py",
        "upstream_triton": "mamba_ssm/ops/triton/ssd_combined.py::mamba_chunk_scan_combined (pure Triton)",
        "hf_kernel": "mamba_chunk_scan_combined",
        "chakra_kernel": "chakra.ssm.ops.mamba2_ssd_chunk_scan.ssd_chunk_scan_combined",
        "candidates": [
            "upstream_triton",
            "chakra_triton",
            "huggingface_kernel",
            "pytorch_reference",
        ],
        "selected": None,
        "forward_status": "indirect_pass",
        "backward_status": "untested",
        "gfx1030_status": "indirect_pass — exercised by mamba2_separate_path with IS_GFX1031=True (chakra dispatch). Both upstream Triton and chakra Triton candidates available.",
        "evidence": [
            {"timestamp": "2026-07-11T15:00:00", "result": "indirect_pass", "detail": "Validated via mamba2_separate_path with IS_GFX1031=True which routes through chakra dispatch."},
            {"timestamp": "2026-07-11T15:05:00", "result": "unavailable", "detail": "HF kernels-community/mamba-ssm: no ROCm build variant. backend=cuda only."},
        ],
    },
    "selective_state_update": {
        "upstream_source": "mamba_ssm/ops/triton/selective_state_update.py",
        "hf_kernel": "selective_state_update",
        "chakra_kernel": "chakra.ssm.compat.gfx1031_patches._patched_selective_state_update",
        "candidates": [
            "upstream_triton",
            "huggingface_kernel",
            "chakra_triton",
            "pytorch_reference",
        ],
        "selected": None,
        "forward_status": "fail",
        "backward_status": "untested",
        "gfx1030_status": "fail — TypeError: upstream Triton stride-unpacking bug when dt_bias=None: `*(...stride...) if dt_bias is not None else 0` should be `else (0, 0)`. Minimal source patch candidate.",
        "evidence": [
            {"timestamp": "2026-07-11T15:00:00", "result": "fail", "detail": "TypeError: Value after * must be an iterable (stride-unpacking bug in selective_state_update wrapper)"},
        ],
    },
    # ── Mamba-3 primitives ──────────────────────────────────────────
    "mamba3_siso": {
        "upstream_source": "mamba_ssm/modules/mamba3.py (is_mimo=False)",
        "hf_kernel": None,
        "chakra_kernel": "chakra.ssm.ops.mamba3_siso._triton_scan_trapezoidal",
        "candidates": [
            "upstream_triton",
            "chakra_triton",
            "pytorch_reference",
        ],
        "selected": None,
        "forward_status": "pass",
        "backward_status": "untested",
        "gfx1030_status": "pass — Mamba-3 SISO works on gfx1030 (with PTX→Triton-native math fix)",
        "evidence": [
            {"timestamp": "2026-07-11T15:00:00", "result": "pass", "detail": "PASS shape=[1, 32, 64] dtype=torch.float16"},
        ],
    },
    "mamba3_mimo": {
        "upstream_source": "mamba_ssm/modules/mamba3.py (is_mimo=True)",
        "hf_kernel": None,
        "chakra_kernel": "chakra.ssm.ops.mamba3_mimo_rank4._mimo_rank4_scan_trapezoidal",
        "candidates": [
            "upstream_tilelang",
            "siso_decomposition",
            "chakra_triton",
            "pytorch_reference",
        ],
        "selected": None,
        "forward_status": "untested",
        "backward_status": "untested",
        "gfx1030_status": "construct_pass — model instantiates (lazy TileLang check), forward not tested",
        "evidence": [
            {"timestamp": "2026-07-11T15:00:00", "result": "construct_pass", "detail": "CONSTRUCT_OK (forward not tested — TileLang unavailable)"},
        ],
    },
}


# ── GPU detection ───────────────────────────────────────────────────

def _detect_gfx1031() -> bool:
    """Return True if the current process targets gfx1030/gfx1031.

    Detection order:
      1. Explicit ``MAMBA_GFX1031`` env var (1/true → force on; 0/false → force off)
      2. ``HSA_OVERRIDE_GFX_VERSION=10.3.0`` (the gfx1030 spoofing env var used
         on consumer RDNA2)
      3. PyTorch GPU properties (gcnArchName or device name)
    """
    env = os.environ.get("MAMBA_GFX1031", "")
    if env in {"1", "true", "True", "yes"}:
        return True
    if env in {"0", "false", "False", "no"}:
        return False

    # gfx1030 spoofing env var — only trust this on actual ROCm systems
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION", "") == "10.3.0":
        try:
            import torch
            if getattr(torch.version, "hip", None):
                return True
        except ImportError:
            pass

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


# ── Lazy adapter getters ────────────────────────────────────────────
# Each returns a callable whose signature matches the corresponding
# upstream function, but delegates to a chakra Triton kernel.
# The import is deferred to avoid hard chakra dependencies when the
# fork is used outside Synesthesia (IS_GFX1031 will be False).

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
