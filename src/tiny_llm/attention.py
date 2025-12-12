import mlx.core as mx
from mlx.nn import Linear
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # input: N.. x L x H x D
    out = mx.matmul(query, key.transpose(*range(key.ndim - 2), -1, -2))
    if scale is None:
        scale = mx.rsqrt(query.shape[-1])
    out = out * scale
    if mask is not None:
        out += mask
    scores = softmax(out, axis=-1)
    out = mx.matmul(scores, value)
    return out


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    # 1. 映射得到三个矩阵
    # 2. 对hidden_size 进行切分
    # 3. 进行注意力运算，此时得到的注意力矩阵实际上也有多个头
    # 4. 合并结果 并乘以Wo矩阵
    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        d_k = self.hidden_size // self.num_heads

        # input: (batch, L, E)
        # wq: (hidden_size, E)
        # print(self.wq.shape)
        # print(query.shape)
        q = linear(query, self.wq)
        k = linear(key, self.wk)
        v = linear(value, self.wv)
        # print(q.shape)
        # q: (batch, hidden_size, L)

        # Reshape: (batch, hidden_size,) -> (batch, L, num_heads, d_k)
        q = q.reshape(*query.shape[:-1], self.num_heads, d_k)
        k = k.reshape(*key.shape[:-1], self.num_heads, d_k)
        v = v.reshape(*value.shape[:-1], self.num_heads, d_k)
        # print(q.shape)

        # Transpose: (batch, L, num_heads, d_k) -> (batch, num_heads, L, d_k)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = scaled_dot_product_attention_simple(q, k, v, mask=mask)
        out = out.transpose(0, 2, 1, 3)
        out = out.reshape(query.shape)
        out = linear(out, self.wo)
        return out


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.tril(mx.ones((L, S), dtype=dtype), k=(S - L))
    mask_matrix = mx.where(mask > 0, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask_matrix


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    *B, H_q, L_q, D_h = query.shape
    *B, H, L_k, _ = key.shape
    n_repeat = H_q // H

    # 将 query 从 (*B, H_q, L_q, D_h) reshape 成 (*B, H, n_repeat, L_q, D_h)
    # 这样 heads [0,1,2] 共享 key head 0, heads [3,4,5] 共享 key head 1, 等等
    # 这里可以记住，reshape 18到 6,3时， 意思是把18变成6个3
    query = query.reshape(*B, H, n_repeat, L_q, D_h)

    # 将 key 和 value reshape 成 (*B, H, 1, L_k, D_h)
    # 1 维度会在 matmul 时广播到 n_repeat，这里主要是因为广播是从右边开始广播的
    key = key.reshape(*B, H, 1, L_k, D_h)
    value = value.reshape(*B, H, 1, L_k, D_h)

    # 计算 attention scores: (*B, H, n_repeat, L_q, D_h) @ (*B, H, 1, D_h, L_k)
    # 结果: (*B, H, n_repeat, L_q, L_k)
    out = mx.matmul(query, key.transpose(*range(key.ndim - 2), -1, -2))

    if scale is None:
        scale = mx.rsqrt(D_h)
    out = out * scale

    if isinstance(mask, str) and mask == "causal":
        mask = causal_mask(L_q, L_k, mx.float32)
    elif mask is not None:
        # mask 从 (*B, H_q, L_q, L_k) reshape 到 (*B, H, n_repeat, L_q, L_k)
        mask = mask.reshape(*B, H, n_repeat, L_q, L_k)

    if mask is not None:
        out += mask
    scores = softmax(out, axis=-1)

    # 计算输出: (*B, H, n_repeat, L_q, L_k) @ (*B, H, 1, L_k, D_h)
    # 结果: (*B, H, n_repeat, L_q, D_h)
    out = mx.matmul(scores, value)

    # reshape 回原始形状: (*B, H, n_repeat, L_q, D_h) → (*B, H_q, L_q, D_h)
    out = out.reshape(*B, H_q, L_q, D_h)
    return out


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
