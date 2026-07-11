"""mamba_ssm/ops/triton/gfx1031 — gfx1031-safe Triton fallbacks.

When the upstream ``selective_scan_cuda`` C extension and
``causal_conv1d`` C extension are unavailable (e.g. on AMD RDNA2
consumer cards such as the RX 6700 XT / gfx1030/gfx1031), the
modules in this package provide Triton-based replacements that
implement the same tensor contracts.
"""

from mamba_ssm.ops.triton.gfx1031.causal_conv1d import (
    triton_causal_conv1d,
    triton_causal_conv1d_fwd_function,
    triton_causal_conv1d_bwd_function,
    triton_causal_conv1d_update,
)
from mamba_ssm.ops.triton.gfx1031.selective_scan import (
    triton_selective_scan_fn,
    triton_selective_scan_cuda_fwd,
    triton_selective_scan_cuda_bwd,
)
from mamba_ssm.ops.triton.gfx1031.ssd_chunk_scan import (
    triton_mamba_chunk_scan_combined,
)

__all__ = [
    "triton_causal_conv1d",
    "triton_causal_conv1d_fwd_function",
    "triton_causal_conv1d_bwd_function",
    "triton_causal_conv1d_update",
    "triton_selective_scan_fn",
    "triton_selective_scan_cuda_fwd",
    "triton_selective_scan_cuda_bwd",
    "triton_mamba_chunk_scan_combined",
]
