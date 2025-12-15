import math
import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.theta = theta
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = mx.rsqrt(self.head_dim)
        self.rope = RoPE(self.head_dim, max_seq_len, theta)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        *B, L, _ = x.shape
        q, k, v = (
            linear(x, self.wq, self.bq),
            linear(x, self.wk, self.bk),
            linear(x, self.wv, self.bv),
        )
        q = q.reshape(*B, L, self.num_heads, self.head_dim)
        k = k.reshape(*B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(*B, L, self.num_kv_heads, self.head_dim)
        q = self.rope(
            q,
            slice(0, L),
        )
        k = self.rope(
            k,
            slice(0, L),
        )
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = scaled_dot_product_attention_grouped(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype)
        out = out.transpose(0, 2, 1, 3).reshape(*B, L, self.hidden_size)
        return linear(out, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.hidden_dim = hidden_dim
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        gate_projection = linear(x, self.w_gate)  # gate 分支
        up_projection = linear(x, self.w_up)  # up 分支

        # SwiGLU: silu(gate) * up
        activated = silu(gate_projection) * up_projection

        return linear(activated, self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
