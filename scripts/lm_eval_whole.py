#!/usr/bin/env python3
"""15.7: lm_eval, with a Hugging Face model whose replies are kept whole.

lm_eval's `hf` model decodes each reply with skip_special_tokens=True and cuts
it at `think_end_token`, keeping what follows. So a Gemma 4 reply lost its
<|channel>…<channel|> (special tokens) and was never cut — scored with its
thinking — and a Qwen3.5 reply capped inside its thinking (its template opens
<think>, and no </think> came) was kept whole as if it were the answer.

`hf-whole` is `hf` with neither: every token decoded, special ones included,
and nothing cut — lm_eval still needs a think_end_token to turn the thinking
on, and is given one, but never cuts at it. DeviceMark's scoring then splits
the reply itself (devicemark.split_reply), as DeviceMark does. The runner
uses it for a DeviceMark task with the thinking on, and says so beside the
answers (reply_form.json).

    python scripts/lm_eval_whole.py --model hf-whole --model_args … (as lm_eval)
"""

from __future__ import annotations

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

MODEL = "hf-whole"


@register_model(MODEL)
class WholeHFLM(HFLM):
    """lm_eval's hf model, each reply kept whole"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # given (enable_thinking needs one), never cut at: the scorer splits
        self.think_end_token = None

    def tok_decode(self, tokens, skip_special_tokens: bool = True):
        return self.tokenizer.decode(tokens, skip_special_tokens=False)


def main() -> None:
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()


if __name__ == "__main__":
    main()
