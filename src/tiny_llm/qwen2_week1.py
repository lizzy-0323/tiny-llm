import math
import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights


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
        self.RMSNorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.PostRMSNorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, rms_norm_eps
        )
        self.MLP = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        self.MultiHeadAttn = Qwen2MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            wq,
            wk,
            wv,
            wo,
            bq,
            bk,
            bv,
            max_seq_len,
            theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # 1. rms norm
        out = self.RMSNorm(x)
        # 2. multi head attn (传递 mask)
        out = self.MultiHeadAttn(out, mask=mask)
        # 3. add residual
        out_stage_one = out + x
        # 4. rms norm
        out = self.PostRMSNorm(out_stage_one)
        # 5. mlp
        out = self.MLP(out)
        # 6. add residual
        out = out + out_stage_one

        return out


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.layers_inner = []
        self.embedding = Embedding(
            mlx_model.args.vocab_size,
            mlx_model.args.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens),
        )
        for i in range(mlx_model.args.num_hidden_layers):
            self.layers_inner.append(
                Qwen2TransformerBlock(
                    num_attention_heads=mlx_model.args.num_attention_heads,
                    num_kv_heads=mlx_model.args.num_key_value_heads,
                    hidden_size=mlx_model.args.hidden_size,
                    intermediate_size=mlx_model.args.intermediate_size,
                    rms_norm_eps=mlx_model.args.rms_norm_eps,
                    wq=dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj),
                    wk=dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj),
                    wv=dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj),
                    wo=dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj),
                    bq=mlx_model.model.layers[i].self_attn.q_proj.bias,
                    bk=mlx_model.model.layers[i].self_attn.k_proj.bias,
                    bv=mlx_model.model.layers[i].self_attn.v_proj.bias,
                    w_gate=dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj),
                    w_up=dequantize_linear(mlx_model.model.layers[i].mlp.up_proj),
                    w_down=dequantize_linear(mlx_model.model.layers[i].mlp.down_proj),
                    w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight,
                    w_post_attention_layernorm=mlx_model.model.layers[
                        i
                    ].post_attention_layernorm.weight,
                    max_seq_len=mlx_model.args.max_position_embeddings,
                    theta=mlx_model.args.rope_theta,
                )
            )
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight,
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.w_lm_head = None

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        # 1. embedding
        output = self.embedding(inputs)
        # 2. transformer blocks
        for layer in self.layers_inner:
            output = layer(output, mask="causal")
        # 3. norm
        output = self.norm(output)
        # 4. linear
        if self.w_lm_head is not None:
            return linear(output, self.w_lm_head)
        else:
            return self.embedding.as_linear(output)
