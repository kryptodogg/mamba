"""
Mamba-3 Util Functions.

Copyright (c) 2025, Dao AI Lab, Goombalab
"""

import triton
import triton.language as tl

# We use PTX approximations instead of triton built-in functions
# to trade off a bit of accuracy for much faster speed.

@triton.jit
def cos_approx(x):
    """
    Cosine using Triton-native math (gfx1031-safe).

    The upstream PTX inline-assembly version (``cos.approx.f32``) cannot be
    lowered by the HIP compiler on RDNA2 (gfx1030/gfx1031).  We use the
    standard Triton built-in which is slightly slower but portable.
    """
    return tl.cos(x.to(tl.float32))


@triton.jit
def sin_approx(x):
    """
    Sine using Triton-native math (gfx1031-safe).
    """
    return tl.sin(x.to(tl.float32))


@triton.jit
def tanh_approx(x):
    """
    tanh using Triton-native math (gfx1031-safe).
    """
    return 2.0 * tl.sigmoid(2.0 * x.to(tl.float32)) - 1.0


@triton.jit
def sech2_approx(x):
    """
    sech^2 using Triton-native math (gfx1031-safe).
    """
    tanh_x = tanh_approx(x)
    return 1.0 - tanh_x * tanh_x

@triton.jit
def sigmoid_approx(x):
    """
    (Fast) Sigmoid approximation using PTX inline assembly.

    Formula: sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))
    Leverages fast tanh approximation for speed.

    Args:
        x: Input triton tensor (any shape) in float32
    Returns:
        Approximate sigmoid values in float32
    """
    # tanh_half_x = tl.inline_asm_elementwise(
    #     "tanh.approx.f32 $0, $1;",
    #     constraints="=f,f",
    #     args=[0.5 * x],
    #     dtype=tl.float32,
    #     is_pure=True,
    #     pack=1,
    # )
    # return 0.5 * (1.0 + tanh_half_x)
    # NOTE: We ended up using the built-in sigmoid for better performance, as the PTX approximation was not faster in this case.
    return tl.sigmoid(x)

@triton.jit
def silu(x):
    """
    SiLU (Swish) activation function: x * sigmoid(x).

    Formula: silu(x) = 0.5*x * (1 + tanh(0.5*x)) + 0.5*x.
    Leverages fast tanh_approx for speed.
    
    Args:
        x: Input triton tensor (any shape) in float32
    
    Returns:
        SiLU activation output in float32
    """
    # x_half = 0.5 * x
    # return x_half * tanh_approx(x_half) + x_half
    # NOTE: We ended up using the built-in sigmoid for better performance, as the PTX approximation was not faster in this case.
    return x*tl.sigmoid(x)