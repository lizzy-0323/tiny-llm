import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        output = model(y)
        output_logits = output[:, -1, :]
        next_token = mx.argmax(output_logits, axis=-1).item()
        return next_token

    tokenized_prompt = tokenizer.encode(prompt)
    detokenizer = tokenizer.detokenizer

    while True:
        next_token = _step(model, tokenized_prompt)
        if next_token == tokenizer.eos_token_id:
            break
        tokenized_prompt.append(next_token)
        detokenizer.add_token(next_token)
        print(detokenizer.text, end="", flush=True)
        detokenizer.reset()


def log_sum_exp(x: mx.array):
    a = mx.max(x)
    e_x_minus_a = (x - a).exp()
    sum_exp = e_x_minus_a.sum(dim=1, keepdims=True)
    log_sum_exp = sum_exp.log()
    return a + log_sum_exp


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
