import gc
import json
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from lightning.fabric.utilities.load import _NotYetLoadedTensor as NotYetLoadedTensor

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.utils import incremental_save, lazy_load


def copy_weights_llama(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    llama_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
) -> None:
    weight_map = {
        "tok_embeddings.weight": "transformer.wte.weight",
        "layers.{}.attention_norm.weight": "transformer.h.{}.norm_1.weight",
        "layers.{}.attention.wq.weight": None,
        "layers.{}.attention.wk.weight": None,
        "layers.{}.attention.wv.weight": None,
        "layers.{}.attention.wo.weight": "transformer.h.{}.attn.proj.weight",
        "layers.{}.self_attn.rotary_emb.inv_freq": None,
        "layers.{}.ffn_norm.weight": "transformer.h.{}.norm_2.weight",
        "norm.weight": "transformer.ln_f.weight",
        "output.weight": "lm_head.weight",

        "layers.{}.feed_forward.gate.weight": "transformer.h.{}.mlp.gate",
        "layers.{}.feed_forward.experts.0.w1.weight": "transformer.h.{}.mlp.experts.0.fc_1.weight",
        "layers.{}.feed_forward.experts.0.w3.weight": "transformer.h.{}.mlp.experts.0.fc_2.weight",
        "layers.{}.feed_forward.experts.0.w2.weight": "transformer.h.{}.mlp.experts.0.proj.weight",
        "layers.{}.feed_forward.experts.1.w1.weight": "transformer.h.{}.mlp.experts.1.fc_1.weight",
        "layers.{}.feed_forward.experts.1.w3.weight": "transformer.h.{}.mlp.experts.1.fc_2.weight",
        "layers.{}.feed_forward.experts.1.w2.weight": "transformer.h.{}.mlp.experts.1.proj.weight",
        "layers.{}.feed_forward.experts.2.w1.weight": "transformer.h.{}.mlp.experts.2.fc_1.weight",
        "layers.{}.feed_forward.experts.2.w3.weight": "transformer.h.{}.mlp.experts.2.fc_2.weight",
        "layers.{}.feed_forward.experts.2.w2.weight": "transformer.h.{}.mlp.experts.2.proj.weight",
        "layers.{}.feed_forward.experts.3.w1.weight": "transformer.h.{}.mlp.experts.3.fc_1.weight",
        "layers.{}.feed_forward.experts.3.w3.weight": "transformer.h.{}.mlp.experts.3.fc_2.weight",
        "layers.{}.feed_forward.experts.3.w2.weight": "transformer.h.{}.mlp.experts.3.proj.weight",
        "layers.{}.feed_forward.experts.4.w1.weight": "transformer.h.{}.mlp.experts.4.fc_1.weight",
        "layers.{}.feed_forward.experts.4.w3.weight": "transformer.h.{}.mlp.experts.4.fc_2.weight",
        "layers.{}.feed_forward.experts.4.w2.weight": "transformer.h.{}.mlp.experts.4.proj.weight",
        "layers.{}.feed_forward.experts.5.w1.weight": "transformer.h.{}.mlp.experts.5.fc_1.weight",
        "layers.{}.feed_forward.experts.5.w3.weight": "transformer.h.{}.mlp.experts.5.fc_2.weight",
        "layers.{}.feed_forward.experts.5.w2.weight": "transformer.h.{}.mlp.experts.5.proj.weight",
        "layers.{}.feed_forward.experts.6.w1.weight": "transformer.h.{}.mlp.experts.6.fc_1.weight",
        "layers.{}.feed_forward.experts.6.w3.weight": "transformer.h.{}.mlp.experts.6.fc_2.weight",
        "layers.{}.feed_forward.experts.6.w2.weight": "transformer.h.{}.mlp.experts.6.proj.weight",
        "layers.{}.feed_forward.experts.7.w1.weight": "transformer.h.{}.mlp.experts.7.fc_1.weight",
        "layers.{}.feed_forward.experts.7.w3.weight": "transformer.h.{}.mlp.experts.7.fc_2.weight",
        "layers.{}.feed_forward.experts.7.w2.weight": "transformer.h.{}.mlp.experts.7.proj.weight",
    }

    for name, param in llama_weights.items():
        if "layers" in name:
            from_name, number = layer_template(name, 1)
            qkv = qkv_weights.setdefault(number, [None, None, None])
            if "wq" in name:
                qkv[0] = param
            elif "wk" in name:
                qkv[1] = param
            elif "wv" in name:
                qkv[2] = param
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

    for i, (q, k, v) in list(qkv_weights.items()):
        if q is None or k is None or v is None:
            # split across different .bin files
            continue
        q = load_param(q, f"layer {i} q", dtype)
        k = load_param(k, f"layer {i} k", dtype)
        v = load_param(v, f"layer {i} v", dtype)
        q_per_kv = config.n_head // config.n_query_groups
        qs = torch.split(q, config.head_size * q_per_kv)
        ks = torch.split(k, config.head_size)
        vs = torch.split(v, config.head_size)
        cycled = [t for group in zip(qs, ks, vs) for t in group]
        qkv = torch.cat(cycled)
        state_dict[f"transformer.h.{i}.attn.attn.weight"] = qkv
        del qkv_weights[i]


def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number


def load_param(param: Union[torch.Tensor, NotYetLoadedTensor], name: str, dtype: Optional[torch.dtype]) -> torch.Tensor:
    if hasattr(param, "_load_tensor"):
        # support tensors loaded via `lazy_load()`
        print(f"Loading {name!r} into RAM")
        param = param._load_tensor()
    if dtype is not None and type(dtype) is not NotYetLoadedTensor and dtype != param.dtype:
        print(f"Converting {name!r} from {param.dtype} to {dtype}")
        param = param.to(dtype)
    return param


@torch.inference_mode()
def convert_llama_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/mistralai/mixtral-8x7b-32kseqlen"),
    model_name: Optional[str] = None,
    dtype: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name
    if dtype is not None:
        dtype = getattr(torch, dtype)

    config = Config.from_name(model_name)
    config_dict = asdict(config)
    print(f"Model config {config_dict}")
    with open(checkpoint_dir / "lit_config.json", "w") as json_config:
        json.dump(config_dict, json_config)

    # holder to reconstitute the split q, k, v
    qkv_weights = {}
    copy_fn = partial(copy_weights_llama, config, qkv_weights)

    # initialize a new empty state dict to hold our new weights
    sd = {}

    llama_file = checkpoint_dir / "consolidated.00.pth"

    with incremental_save(checkpoint_dir / "lit_model.pth") as saver:
        llama_weights = lazy_load(llama_file)
        copy_fn(sd, llama_weights, saver=saver, dtype=dtype)
        gc.collect()
        print("Saving converted checkpoint")
        saver.save(sd)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_llama_checkpoint)
