#!/usr/bin/env python3
"""
Loads models and tokenizers for the DAS pipeline.

Every supported model is listed in _MODEL_CONFIGS with the kwargs it needs.
load_model_and_tokenizer() is the entry point; it returns (model, tokenizer).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _get_max_memory(reserve_gb: float = 4.0) -> dict:
    """
    Per-GPU memory budget for device_map placement.

    We leave reserve_gb free on each GPU for activations and the intervention
    tensors, and set the CPU budget to 0 so nothing gets offloaded there
    (offloading would break the interventions).
    """
    n = torch.cuda.device_count()
    max_mem = {}
    for i in range(n):
        total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        max_mem[i] = f"{max(0.0, total_gb - reserve_gb):.0f}GiB"
    max_mem["cpu"] = "0GiB"
    return max_mem


_MODEL_CONFIGS = {
    "meta-llama/Llama-3.1-8B-Instruct": dict(
        model_cls=AutoModelForCausalLM,
        tok_cls=AutoTokenizer,
        model_kwargs=dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        ),
    ),
    "allenai/OLMo-2-0425-1B-Instruct": dict(
        model_cls=AutoModelForCausalLM,
        tok_cls=AutoTokenizer,
        model_kwargs=dict(
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
        ),
    ),
    "allenai/OLMo-2-1124-7B-Instruct": dict(
        model_cls=AutoModelForCausalLM,
        tok_cls=AutoTokenizer,
        model_kwargs=dict(
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
        ),
    ),
    "allenai/OLMo-2-1124-13B-Instruct": dict(
        model_cls=AutoModelForCausalLM,
        tok_cls=AutoTokenizer,
        model_kwargs=dict(
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
        ),
    ),
    "Qwen/Qwen2.5-7B-Instruct": dict(
        model_cls=AutoModelForCausalLM,
        tok_cls=AutoTokenizer,
        model_kwargs=dict(
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="eager",
        ),
    ),
    "google/gemma-2-9b-it": dict(
        model_cls=AutoModelForCausalLM,
        tok_cls=AutoTokenizer,
        model_kwargs=dict(
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",   #fits on one GPU
            attn_implementation="eager",
        ),
    ),
}


def _register_olmo2_with_pyvene():
    """
    Tell pyvene how to hook into OLMo-2.

    pyvene ships mappings for common architectures but not Olmo2ForCausalLM, so
    we add the dimension and module mappings by hand. Returns early if it's
    already registered.
    """
    from pyvene.models.modeling_utils import (
        type_to_dimension_mapping,
        type_to_module_mapping,
        CONST_OUTPUT_HOOK,
        CONST_INPUT_HOOK,
    )
    from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM

    if Olmo2ForCausalLM in type_to_module_mapping:
        return

    type_to_dimension_mapping[Olmo2ForCausalLM] = {
        "block_output":              ("hidden_size",),
        "block_input":               ("hidden_size",),
        "attention_output":          ("hidden_size",),
        "attention_input":           ("hidden_size",),
        "mlp_output":                ("hidden_size",),
        "mlp_input":                 ("hidden_size",),
        "head_attention_value_output": ("hidden_size // num_attention_heads", "num_attention_heads"),
    }
    type_to_module_mapping[Olmo2ForCausalLM] = {
        "block_output":              ("model.layers[%s]",             CONST_OUTPUT_HOOK),
        "block_input":               ("model.layers[%s]",             CONST_INPUT_HOOK),
        "attention_output":          ("model.layers[%s].self_attn",   CONST_OUTPUT_HOOK),
        "attention_input":           ("model.layers[%s].self_attn",   CONST_INPUT_HOOK),
        "mlp_output":                ("model.layers[%s].mlp",         CONST_OUTPUT_HOOK),
        "mlp_input":                 ("model.layers[%s].mlp",         CONST_INPUT_HOOK),
        "head_attention_value_output": ("model.layers[%s].self_attn", CONST_OUTPUT_HOOK),
    }


def _resolve_model_cls(cls_or_name):
    # model_cls is usually a class already. If it's a string, look it up here.
    if isinstance(cls_or_name, str):
        from transformers import Gemma3ForConditionalGeneration
        mapping = {"Gemma3ForConditionalGeneration": Gemma3ForConditionalGeneration}
        if cls_or_name not in mapping:
            raise ValueError(f"Unknown model class string: {cls_or_name!r}")
        return mapping[cls_or_name]
    return cls_or_name


def load_model_and_tokenizer(model_name: str):
    if model_name not in _MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name!r}\n"
            f"Supported: {list(_MODEL_CONFIGS)}"
        )
    cfg       = _MODEL_CONFIGS[model_name]
    model_cls = _resolve_model_cls(cfg["model_cls"])
    tok_cls   = cfg["tok_cls"]
    tok_kwargs = cfg.get("tok_kwargs", {})

    print(f"Loading tokenizer ({tok_cls.__name__}) for {model_name} ...")
    tokenizer = tok_cls.from_pretrained(model_name, **tok_kwargs)

    # For large models, compute the memory budget at runtime. This keeps weights
    # off the CPU and avoids pointing device_map at GPUs this job wasn't given.
    model_kwargs = dict(cfg["model_kwargs"])
    if cfg.get("large_model", False):
        reserve_gb = cfg.get("reserve_gb", 8.0)
        model_kwargs["max_memory"] = _get_max_memory(reserve_gb=reserve_gb)
        model_kwargs["device_map"] = "balanced"
        print(f"  max_memory: {model_kwargs['max_memory']}")
        print(f"  device_map: balanced (forced for large model)")

    print(f"Loading model ({model_cls.__name__}) for {model_name} ...")
    model = model_cls.from_pretrained(model_name, **model_kwargs)
    model.eval()
    print(f"  Actual model class: {type(model).__name__}")

    if "olmo-2" in model_name.lower() or "olmo2" in model_name.lower():
        try:
            _register_olmo2_with_pyvene()
        except ModuleNotFoundError:
            pass

    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def model_slug(model_name: str) -> str:
    """Filesystem-safe name, e.g. 'llama-3.1-8b-instruct'."""
    return model_name.split("/")[-1].lower()