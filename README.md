<div align="center">
<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Badge.png" alt="Lit-GPT" width="128"/>

# ⚡ Lit-GPT

<!--
<p align="center">
  <a href="https://www.lightning.ai/">Lightning.ai</a> •
  <a href="https://lightning.ai/docs/pytorch/stable/">PyTorch Lightning</a> •
  <a href="https://lightning.ai/docs/fabric/stable/">Fabric</a>
</p>
-->

![cpu-tests](https://github.com/lightning-AI/lit-stablelm/actions/workflows/cpu-tests.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-stablelm/blob/master/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)

<img src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM.gif" alt="Lit-GPT and pineapple pizza" width="500px"/>

</div>

# ⚡ Lit-GPT

Fast, simple implementations of state-of-the-art large language models (LLM).

With Lit-GPT you can:

* generate text or code, or make your documents interactive
* fine-tune models on your own data on your own hardware
* train brand new models from scratch on your datasets of choice

Lit-GPT can run on your own device or scale up to large clusters, thanks to [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡.

## Get started

Get started in minutes:

* [Setup the repo](#setup)
* [Get a checkpoint](#get-a-checkpoint)
* Generate
* Fine-tune
* Pretrain

## Setup

**Clone the repo**

```bash
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
```

Lit-GPT currently relies on PyTorch nightly. Until PyTorch 2.1 is released you'll need to install nightly manually:

**On GPU**

```bash
pip install -r requirements/pytorch_nightly_gpu.txt
```

**On CPU (incl Macs)**

```bash
pip install -r requirements/pytorch_nightly_cpu.txt
```

All good, now **install the dependencies**:

```bash
pip install -r requirements.txt
```

You are all set! 🎉

## Get a checkpoint

In order to use Lit-GPT to generate and fine-tune, you need to download a checkpoint to start from, and convert it so that it works with Lit-GPT.

To download and conver, use this script and pass a checkpoint to it:

```bash
python scripts/download_and_convert.py --checkpoint <checkpoint>
```

For instance, to download and convert the OpenLLaMA 7B checkpoint do:

```bash
python scripts/download_and_convert.py --checkpoint openlm-research/open_llama_7b
```

Here's a list of all checkpoints supported by Lit-GPT along with their checkpoint id:

<!--
| Model | Description | Checkpoints |
| -- | -- | -- | -- | -- |
| [Falcon](tutorials/download_falcon.md)    | GPT-like model trained on [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) | `tiiuae/falcon-7b` <br/> `tiiuae/falcon-7b-instruct` <br/> `tiiuae/falcon-40b` <br/> `tiiuae/falcon-40b-instruct` |
| [OpenLLaMA](tutorials/download_openllama.md)        | Open source LLaMA trained on [RedPajama](https://www.together.xyz/blog/redpajama) | `openlm-research/open_llama_3b` <br/> `openlm-research/open_llama_7b` <br/> `openlm-research/open_llama_13b` |
| [Vicuna](tutorials/download_vicuna.md)           | LLaMA fine-tuned on conversation | `lmsys/vicuna-7b-v1.3` <br/> `lmsys/vicuna-13b-v1.3` <br/> `lmsys/vicuna-33b-v1.3` |
| [LongChat](tutorials/download_longchat.md)         | LLaMA fine-tuned on long context lengths (16k) | `lmsys/longchat-7b-16k` <br/> `lmsys/longchat-13b-16k` |
| [RedPajama-INCITE](tutorials/download_redpajama_incite.md) | GPT-like model trained on [RedPajama](https://www.together.xyz/blog/redpajama) | `togethercomputer/RedPajama-INCITE-7B-Base` <br/> `togethercomputer/RedPajama-INCITE-7B-Chat` <br/> `togethercomputer/RedPajama-INCITE-7B-Instruct`
| [StableLM](tutorials/download_stablelm.md)         | GPT-like model trained on [The Pile](https://pile.eleuther.ai) | `stabilityai/stablelm-base-alpha-3b` <br/> `stabilityai/stablelm-tuned-alpha-3b` <br/> `stabilityai/stablelm-base-alpha-7b` <br/> `stabilityai/stablelm-tuned-alpha-7b`
| [Pythia](tutorials/download_pythia.md)           | Collection of models trained on [The Pile](https://pile.eleuther.ai) | `EleutherAI/pythia-70m` <br/> `EleutherAI/pythia-160m` <br/> `EleutherAI/pythia-410m` <br/> `EleutherAI/pythia-1b` <br/> `EleutherAI/pythia-1.4b` <br/> `EleutherAI/pythia-2.8b` <br/> `EleutherAI/pythia-6.9b` <br/> `EleutherAI/pythia-12b`
-->

| Model | Description | Recommended | More options |
| -- | -- | -- | -- | -- |
| [OpenLLaMA](tutorials/download_openllama.md) | Open-source LLaMA trained on [RedPajama](https://www.together.xyz/blog/redpajama) | `openlm-research/open_llama_7b` | <details> <summary> view all checkpoints </summary> `openlm-research/open_llama_3b` <br/> `openlm-research/open_llama_7b` <br/> `openlm-research/open_llama_13b` </details> |
| [Vicuna](tutorials/download_vicuna.md) | LLaMA fine-tuned on conversation | `lmsys/vicuna-7b-v1.3` | <details> <summary> view all checkpoints </summary> `lmsys/vicuna-7b-v1.3` <br/> `lmsys/vicuna-13b-v1.3` <br/> `lmsys/vicuna-33b-v1.3` </details> |
| [LongChat](tutorials/download_longchat.md) | LLaMA fine-tuned on long context lengths (16k) | `lmsys/longchat-7b-16k` | <details> <summary> view all checkpoints </summary> `lmsys/longchat-7b-16k` <br/> `lmsys/longchat-13b-16k` </details> |
| [Falcon](tutorials/download_falcon.md) | GPT-like model trained on [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) | `tiiuae/falcon-7b-instruct` | <details> <summary> view all checkpoints </summary> `tiiuae/falcon-7b` <br/> `tiiuae/falcon-7b-instruct` <br/> `tiiuae/falcon-40b` <br/> `tiiuae/falcon-40b-instruct` </details> |
| [RedPajama-INCITE](tutorials/download_redpajama_incite.md) | GPT-like model trained on [RedPajama](https://www.together.xyz/blog/redpajama) | `togethercomputer/RedPajama-INCITE-7B-Instruct` | <details> <summary> view all checkpoints </summary> `togethercomputer/RedPajama-INCITE-7B-Base` <br/> `togethercomputer/RedPajama-INCITE-7B-Chat` <br/> `togethercomputer/RedPajama-INCITE-7B-Instruct` </details> |
| [StableLM](tutorials/download_stablelm.md) | GPT-like model trained on [The Pile](https://pile.eleuther.ai) | `stabilityai/stablelm-tuned-alpha-3b` | <details> <summary> view all checkpoints </summary> `stabilityai/stablelm-base-alpha-3b` <br/> `stabilityai/stablelm-tuned-alpha-3b` <br/> `stabilityai/stablelm-base-alpha-7b` <br/> `stabilityai/stablelm-tuned-alpha-7b` </details> |
| [Pythia](tutorials/download_pythia.md) | GPT-like model trained on [The Pile](https://pile.eleuther.ai) | `EleutherAI/pythia-6.9b` | <details> <summary> view all checkpoints </summary> `EleutherAI/pythia-70m` <br/> `EleutherAI/pythia-160m` <br/> `EleutherAI/pythia-410m` <br/> `EleutherAI/pythia-1b` <br/> `EleutherAI/pythia-1.4b` <br/> `EleutherAI/pythia-2.8b` <br/> `EleutherAI/pythia-6.9b` <br/> `EleutherAI/pythia-12b` </details> |

If you don't find your favorite checkpoint and the model family is the same (e.g. LLaMA, GPTNeoX, Falcon), then adding it to the supported checkpoints is going to be quick. Please post an issue about it!

## Generate

Got your checkpoint? Great, you can now have a conversation with Lit-GPT:

```bash
python generate/base.py --prompt "Hello, my name is" --checkpoint openlm-research/open_llama_7b
```

This will run the 3B pre-trained model and require ~7 GB of GPU memory using the `bfloat16` datatype.

You can also chat with the model interactively:

```bash
python chat/base.py
```

For more options for generation, including quantization to run large models on consumer devices, follow [this tutorial](tutorials/inference.md).

## Design principles

The Lit-GPT codebase builds on top of [Lit-LLaMA](https://github.com/lightning-AI/lit-llama) and [nanoGPT](https://github.com/karpathy/nanoGPT), and it's **powered by [Lightning Fabric](https://lightning.ai/docs/fabric/stable/) ⚡**.


This repository follows the main principle of **openness through clarity**.

**Lit-GPT** is:

- **Simple:** Single-file implementation without boilerplate.
- **Correct:** Numerically equivalent to the original models.
- **Optimized:** Runs fast on consumer hardware or at scale.
- **Open-source:** Apache 2.0, no strings attached.

Avoiding code duplication is **not** a goal. **Readability** and **hackability** are.

## Get involved!

[Join our Discord](https://discord.gg/VptPCZkGNa) to build high-performance, truly open-source models for the common benefit of the community.

## Use the model

Run inference:

```bash
python generate/base.py --prompt "Hello, my name is"
```

This will run the 3B pre-trained model and require ~7 GB of GPU memory using the `bfloat16` datatype.

[Full guide for generating samples from the model](tutorials/inference.md).

You can also chat with the model interactively:

```bash
python chat/base.py
```

### Run large models on smaller consumer devices

We support LLM.int8 and GPTQ.int4 inference by following [this guide](tutorials/inference.md#run-large-models-on-consumer-devices).

## Finetune the model

We provide a simple training scripts (`finetune/adapter.py`, `finetune/adapter_v2.py`, and `finetune/lora.py`) that instruction-tunes a pretrained model on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset.

1. Download the data and generate an instruction tuning dataset:

```bash
python scripts/prepare_alpaca.py
```

2. Run the finetuning script

For example, you can either use

Adapter ([Zhang et al. 2023](https://arxiv.org/abs/2303.16199)):

```bash
python finetune/adapter.py
```

or Adapter v2 ([Gao et al. 2023](https://arxiv.org/abs/2304.15010)):

```bash
python finetune/adapter_v2.py
```

or LoRA ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)):

```bash
python finetune/lora.py
```

(Please see the [tutorials/finetune_adapter](tutorials/finetune_adapter.md) for details on the differences between the two adapter methods.)

The finetuning requires at least one GPU with ~12 GB memory (RTX 3060).

It is expected that you have downloaded the pretrained weights as described above.
More details about each finetuning method and how you can apply it to your own data can be found in our technical how-to guides.

### Finetuning How-To Guides

These technical tutorials illustrate how to run the finetuning code.

- [Finetune with Adapters](tutorials/finetune_adapter.md)
- [Finetune with LoRA](tutorials/finetune_lora.md)

### Understanding Finetuning -- Conceptual Tutorials

Looking for conceptual tutorials and explanations? We have some additional articles below:

- [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)

- [Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (LoRA)](https://lightning.ai/pages/community/tutorial/lora-llm/)

## Pre-training

Porting from Lit-LLaMA in progress 👷

## Get involved!

We are on a quest towards fully open source AI.

<img align="right" src="https://pl-public-data.s3.amazonaws.com/assets_lightning/LitStableLM_Illustration.png" alt="Lit-GPT" width="128"/>

Join us and start contributing, especially on the following areas:

- [ ] [Pre-training](https://github.com/Lightning-AI/lit-gpt/labels/pre-training)
- [ ] [Fine-tuning](https://github.com/Lightning-AI/lit-gpt/labels/fine-tuning)
- [ ] [Quantization](https://github.com/Lightning-AI/lit-gpt/labels/quantization)
- [ ] [Sparsification](https://github.com/Lightning-AI/lit-gpt/labels/sparsification)

We welcome all individual contributors, regardless of their level of experience or hardware. Your contributions are valuable, and we are excited to see what you can accomplish in this collaborative and supportive environment. 

Unsure about contributing? Check out our [Contributing to Lit-LLaMA: A Hitchhiker’s Guide to the Quest for Fully Open-Source AI](https://lightning.ai/pages/community/tutorial/contributing-to-lit-llama-a-hitchhikers-guide-to-the-quest-for-fully-open-source-ai/) guide. The same guidelines apply to Lit-GPT.

Don't forget to [join our Discord](https://discord.gg/VptPCZkGNa)!

## Acknowledgements

- [@karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT)
- [@EleutherAI](https://github.com/karpathy) for [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
- [@TimDettmers](https://github.com/TimDettmers) for [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [@IST-DASLab](https://github.com/IST-DASLab) for [GPTQ](https://github.com/IST-DASLab/gptq)
- [@Microsoft](https://github.com/microsoft) for [LoRA](https://github.com/microsoft/LoRA)


## License

Lit-GPT is released under the [Apache 2.0](https://github.com/Lightning-AI/lit-gpt/blob/main/LICENSE) license.
