import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.cos_freqs = None
        self.sin_freqs = None
        self._precompute_rope()

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        seq_len = x.shape[1]

        if offset is None:
            cos = self.cos_freqs[0:seq_len].reshape(1, seq_len, 1, self.dims // 2)
            sin = self.sin_freqs[0:seq_len].reshape(1, seq_len, 1, self.dims // 2)
        else:
            cos = self.cos_freqs[offset.start : offset.stop].reshape(
                1, seq_len, 1, self.dims // 2
            )
            sin = self.sin_freqs[offset.start : offset.stop].reshape(
                1, seq_len, 1, self.dims // 2
            )
        x_out = self._apply_rope(x, cos, sin)
        return x_out

    def _precompute_rope(self):
        # 1. 计算每一个位置对应的 theta
        # i  1, 2 ,d/2
        # 2i 2, 4, d
        # 2i -2 , 0,2 , d-2

        inv_freqs = 1.0 / (
            self.base ** (mx.arange(0, self.dims, 2, dtype=mx.float32) / self.dims)
        )

        # 2. 计算位置
        t = mx.arange(self.seq_len, dtype=mx.float32)

        # 3. 计算角度
        freqs = mx.outer(t, inv_freqs)
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)

    def _apply_rope(self, x, cos, sin):
        if self.traditional:
            x_split = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
            x1, x2 = x_split[..., 0], x_split[..., 1]
            rotate_x1 = x1 * cos - x2 * sin
            rotate_x2 = x2 * cos + x1 * sin
            x_out = mx.stack([rotate_x1, rotate_x2], axis=-1)
            x_out = x_out.reshape(*x_out.shape[:-2], x.shape[-1])
        else:
            half_dim = x.shape[-1] // 2
            x0, xhalf = (x[..., :half_dim], x[..., half_dim:])
            rotate_x0 = x0 * cos - xhalf * sin
            rotate_xhalf = xhalf * cos + x0 * sin
            x_out = mx.concatenate([rotate_x0, rotate_xhalf], axis=-1)
        return x_out
