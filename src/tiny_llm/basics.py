import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    x_max = mx.max(x, axis=axis, keepdims=True)
    x_stablized = x - x_max
    exp_x = mx.exp(x_stablized)
    return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    if bias is not None:
        return mx.addmm(bias, x, w.T)
    return mx.matmul(x, w.T)


def silu(x: mx.array) -> mx.array:
    return x / (1 + mx.exp(-x))
