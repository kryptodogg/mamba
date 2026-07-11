"""mamba_ssm/ops/triton/gfx1031/selective_scan.py — gfx1031-safe selective scan.

Replaces ``selective_scan_cuda`` with a Triton kernel + PyTorch autograd.
Matches the upstream ``selective_scan_fn`` tensor contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ===========================================================================
# Triton kernel
# ===========================================================================

@triton.jit
def _selective_scan_fwd_kernel(
    delta_ptr,
    A_ptr,
    B_ptr,
    X_ptr,
    H_out_ptr,
    h_prev_ptr,
    B_sz: tl.constexpr,
    L_sz: tl.constexpr,
    D_sz: tl.constexpr,
    N_sz: tl.constexpr,
    s_dt_b, s_dt_l, s_dt_d,
    s_A_d, s_A_n,
    s_Bt_b, s_Bt_l, s_Bt_n,
    s_X_b, s_X_l, s_X_d,
    s_H_b, s_H_l, s_H_d, s_H_n,
    s_hprev_b, s_hprev_d, s_hprev_n,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    d_block = tl.program_id(1)
    n_block = tl.program_id(2)

    d_off = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    n_off = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_d = d_off < D_sz
    mask_n = n_off < N_sz
    mask_dn = mask_d[:, None] & mask_n[None, :]

    h_prev_base = batch_idx * s_hprev_b
    h_prev_ptrs = (h_prev_base + d_off[:, None] * s_hprev_d
                   + n_off[None, :] * s_hprev_n + h_prev_ptr)
    h = tl.load(h_prev_ptrs, mask=mask_dn, other=0.0).to(tl.float32)

    a_ptrs = A_ptr + d_off[:, None] * s_A_d + n_off[None, :] * s_A_n
    A_tile = tl.load(a_ptrs, mask=mask_dn, other=0.0).to(tl.float32)

    dt_base = batch_idx * s_dt_b
    bt_base = batch_idx * s_Bt_b
    x_base = batch_idx * s_X_b
    h_base = batch_idx * s_H_b

    for t in range(0, L_sz):
        dt_ptrs = dt_base + t * s_dt_l + d_off * s_dt_d + delta_ptr
        dt_val = tl.load(dt_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        A_bar = tl.exp(dt_val[:, None] * A_tile)

        b_ptrs = bt_base + t * s_Bt_l + n_off * s_Bt_n + B_ptr
        B_val = tl.load(b_ptrs, mask=mask_n, other=0.0).to(tl.float32)
        B_bar = dt_val[:, None] * B_val[None, :]

        x_ptrs = x_base + t * s_X_l + d_off * s_X_d + X_ptr
        x_val = tl.load(x_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        h = A_bar * h + B_bar * x_val[:, None]

        h_ptrs = (h_base + t * s_H_l + d_off[:, None] * s_H_d
                  + n_off[None, :] * s_H_n + H_out_ptr)
        tl.store(h_ptrs, h.to(tl.float16), mask=mask_dn)


# ===========================================================================
# Autograd wrapper
# ===========================================================================

class _SelectiveScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, z, delta_bias, delta_softplus):
        if delta_bias is not None:
            delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
        if delta_softplus:
            delta = torch.nn.functional.softplus(delta)

        B_sz, D_sz, L_sz = u.shape
        N_sz = A.shape[1]

        ssm_state = torch.zeros(B_sz, D_sz, N_sz, device=u.device, dtype=u.dtype)
        H_out = torch.empty(B_sz, L_sz, D_sz, N_sz, device=u.device, dtype=u.dtype)

        BLOCK_D = 128 if u.dtype == torch.float16 else 64
        BLOCK_N = 64
        grid = (B_sz, triton.cdiv(D_sz, BLOCK_D), triton.cdiv(N_sz, BLOCK_N))

        _selective_scan_fwd_kernel[grid](
            delta, A, B, u,
            H_out,
            ssm_state,
            B_sz, L_sz, D_sz, N_sz,
            delta.stride(0), delta.stride(1), delta.stride(2),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1), B.stride(2),
            u.stride(0), u.stride(1), u.stride(2),
            H_out.stride(0), H_out.stride(1), H_out.stride(2), H_out.stride(3),
            ssm_state.stride(0), ssm_state.stride(1), ssm_state.stride(2),
            BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
        )

        Y_out = torch.einsum("bdn,bldn->bld",
                             C.float(), H_out.float()).to(u.dtype)
        if D is not None:
            Y_out = Y_out + D.unsqueeze(0).unsqueeze(-1) * u
        if z is not None:
            Y_out = Y_out * torch.nn.functional.silu(z)

        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, H_out, ssm_state)
        ctx.delta_softplus = delta_softplus
        return Y_out

    @staticmethod
    def backward(ctx, grad_y):
        u, delta, A, B, C, D, z, delta_bias, H_out, ssm_state = ctx.saved_tensors
        B_sz, D_sz, L_sz = u.shape
        N_sz = A.shape[1]

        if delta_bias is not None:
            delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
        if ctx.delta_softplus:
            delta = torch.nn.functional.softplus(delta)

        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        grad_delta = torch.zeros_like(delta)
        grad_A = torch.zeros_like(A)
        grad_B = torch.zeros_like(B)
        grad_C = torch.zeros_like(C)
        grad_X = torch.zeros_like(u)
        grad_D = torch.zeros(D_sz, device=u.device, dtype=u.dtype) if D is not None else None

        if D is not None:
            grad_D[:] = torch.einsum("bdl,bdl->d", grad_y.float(), u.float()).to(grad_D.dtype)

        grad_C[:] = torch.einsum(
            "bdl,bldn->bdn", grad_y.float(), H_out.float()
        ).to(grad_C.dtype)

        G = torch.zeros(B_sz, D_sz, N_sz, device=u.device, dtype=torch.float32)

        for t in range(L_sz - 1, -1, -1):
            G_contrib = (grad_y[:, :, t].unsqueeze(-1)
                         * C[:, :, t].unsqueeze(1).float())
            if t < L_sz - 1:
                G = G_contrib + G * A_bar[:, :, t + 1].float()
            else:
                G = G_contrib

            h_prev = ssm_state if t == 0 else H_out[:, t - 1, :, :]

            dt_t = delta[:, :, t]
            x_t = u[:, :, t]
            A_bar_t = A_bar[:, :, t].float()

            d_delta = (G
                       * (A.unsqueeze(0).float()
                          * A_bar_t
                          * h_prev.float()
                          + B[:, :, t].unsqueeze(1).float()
                          * x_t.unsqueeze(-1).float())
                       ).sum(dim=-1)
            grad_delta[:, :, t] = d_delta.to(delta.dtype)

            dA = (G
                  * dt_t.unsqueeze(-1).float()
                  * A_bar_t
                  * h_prev.float()
                  ).sum(dim=0)
            grad_A += dA.to(A.dtype)

            dB_t = (G
                    * dt_t.unsqueeze(-1).float()
                    * x_t.unsqueeze(-1).float()
                    ).sum(dim=1)
            grad_B[:, :, t] = dB_t.to(B.dtype)

            dX_direct = grad_y[:, :, t] * D.unsqueeze(0) if D is not None else 0.0
            dX_ssm = (G
                      * dt_t.unsqueeze(-1).float()
                      * B[:, :, t].unsqueeze(1).float()
                      ).sum(dim=-1)
            grad_X[:, :, t] = (dX_direct + dX_ssm).to(u.dtype)

        return (grad_X, grad_delta, grad_A, grad_B, grad_C,
                grad_D, None, None, None)


# ===========================================================================
# Public API matching upstream contracts
# ===========================================================================

def triton_selective_scan_fn(
    u, delta, A, B, C, D=None, z=None,
    delta_bias=None, delta_softplus=False,
    return_last_state=False,
):
    """Matches upstream ``selective_scan_fn`` contract (u: B,D,L)."""
    y = _SelectiveScanFunction.apply(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus,
    )
    if return_last_state:
        raise NotImplementedError("return_last_state not implemented for gfx1031 fallback")
    return y


def triton_selective_scan_cuda_fwd(
    u, delta, A, B, C, D, z, delta_bias, delta_softplus,
):
    """Matches ``selective_scan_cuda.fwd`` return shape used by MambaInnerFn."""
    y = triton_selective_scan_fn(
        u, delta, A, B, C, D=D, z=z,
        delta_bias=delta_bias, delta_softplus=delta_softplus,
    )
    return y, u, z


def triton_selective_scan_cuda_bwd(*args, **kwargs):
    raise NotImplementedError("selective_scan_cuda.bwd not implemented for gfx1031 fallback")
