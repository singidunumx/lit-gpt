import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import lightning as L
import lm_eval.api.instance
import lm_eval.api.model
import lm_eval.evaluator
import lm_eval.tasks
import lm_eval.utils
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    gptq_quantization,
    load_checkpoint,
)


class EvalHarnessLM(lm_eval.api.model.LM):
    """https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/docs/model_guide.md"""

    def __init__(self, fabric: L.Fabric, model: GPT, tokenizer: Tokenizer, batch_size: int) -> int:
        super().__init__()
        self.fabric = fabric
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def loglikelihood(self, requests: List[lm_eval.api.instance.Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests: List[lm_eval.api.instance.Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError

    def generate_until(self, requests: List[lm_eval.api.instance.Instance]) -> List[str]:
        """https://github.com/EleutherAI/lm-evaluation-harness/blob/afda6551e9e8d8021c1fdd35d2aad0fbe63f3919/lm_eval/models/huggingface.py#L823"""
        res = defaultdict(list)
        re_ords = {}

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        grouper = lm_eval.utils.Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            # within each set of reqs for given kwargs, we reorder by token length, descending.
            re_ords[key] = lm_eval.utils.Reorderer([req.args for req in reqs], _collate)

        # for each different set of kwargs, we execute all requests, by batch.
        for key, re_ord in re_ords.items():
            chunks = lm_eval.utils.chunks(
                re_ord.get_reordered(),
                n=self.batch_size
            )
            for chunk in chunks:
                contexts, all_gen_kwargs = zip(*chunk)
                # we assume all gen kwargs in the batch are the same
                # this is safe to assume because the `grouper` object ensures it.
                gen_kwargs = all_gen_kwargs[0]
                # unpack our keyword arguments.
                until: Optional[List[str]] = None
                if isinstance(gen_kwargs, dict):
                    kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [until]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                            )
                else:
                    raise ValueError(
                        f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}"
                    )
                if not until:
                    eos_id = self.tokenizer.eos_id
                else:
                    eos_id = self.tokenizer.encode(until[0])
                if "max_gen_toks" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_gen_toks")
                else:
                    max_gen_toks = self.model.config.block_size

                context_enc = torch.tensor(self.tok_encode(contexts[0]), device=self.fabric.device)

                if "max_returned_tokens" not in kwargs:
                    kwargs["max_returned_tokens"] = min(context_enc.shape[0] + max_gen_toks, self.model.config.block_size)

                # set the max length in tokens of inputs ("context_enc")
                with self.fabric.init_tensor():
                    self.model.max_seq_length = kwargs["max_returned_tokens"]
                    self.model.set_kv_cache(batch_size=self.batch_size)

                # unused
                kwargs.pop("do_sample", None)

                # perform batched generation
                cont = generate(self.model, context_enc, eos_id=eos_id.to(context_enc.device), **kwargs)

                s = self.tok_decode(cont)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]

                res[key].append(s)

                self.cache_hook.add_partial(
                    "generate_until", (contexts[0], gen_kwargs), s
                )
            # reorder this group of results back to original unsorted form
            res[key] = re_ord.get_original(res[key])

        return grouper.get_original(res)

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, bos=False, eos=False).tolist()

    def tok_decode(self, tokens: Union[torch.Tensor, List[int]]) -> str:
        return self.tokenizer.decode(tokens)

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        raise NotImplementedError


@torch.inference_mode()
def run_eval_harness(
    checkpoint_dir: Path,
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    eval_tasks: List[str] = ["arc_challenge", "piqa", "hellaswag", "hendrycksTest-*"],
    save_filepath: Optional[Path] = None,
    **kwargs: Any,
):
    if precision is None:
        precision = get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

    check_valid_checkpoint_dir(checkpoint_dir)
    tokenizer = Tokenizer(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")

    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
        model = GPT(config)
    model.eval()
    model = fabric.setup_module(model)
    load_checkpoint(fabric, model, checkpoint_path)

    # https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/docs/interface.md
    lm_obj = EvalHarnessLM(fabric, model, tokenizer, 1)
    lm_eval.tasks.initialize_tasks()
    results = lm_eval.evaluator.simple_evaluate(model=lm_obj, tasks=list(eval_tasks), **kwargs)
    if save_filepath is None:
        print(results)
    else:
        print(f"Saving results to {str(save_filepath)!r}")
        data = json.dumps(results)
        with open(save_filepath, "w") as fw:
            fw.write(data)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(run_eval_harness, as_positional=False)
