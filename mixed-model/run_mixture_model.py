#!/usr/bin/env python3
"""
run_mixture_model.py

Fits the lightweight mixture model over empirical pronoun distributions. The distributions come from inference-only interchange interventions (VanillaIntervention, no training):
 patch the source residual stream into the base run at each mechanism's layer and read the softmax over he/she/they.

The mixture combines the three mechanisms (G, R, S) with learned weights and we
report Jensen-Shannon similarity against the empirical distributions, plus the single- and double-mechanism ablations.

Use:
python run_mixture_model.py --model meta-llama/Llama-3.1-8B-Instruct --layer_G 8 --layer_R 16 --layer_S 8
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import gc
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import t as t_dist
from tqdm import tqdm
from transformers import (
    AutoConfig, AutoTokenizer,
    LlamaForCausalLM, Qwen2ForCausalLM,
)
from pyvene.models.modeling_utils import type_to_dimension_mapping, type_to_module_mapping
_original_torch_gather = torch.gather

def _safe_torch_gather(input, dim, index, *, sparse_grad=False, out=None):
    if index.device != input.device:
        index = index.to(input.device)
    if out is not None:
        return _original_torch_gather(input, dim, index, sparse_grad=sparse_grad, out=out)
    return _original_torch_gather(input, dim, index, sparse_grad=sparse_grad)

torch.gather = _safe_torch_gather

# if architectures which pyvene doesn't have mappings for; reuse a compatible base class.
_FALLBACK_MAP = {
    "Olmo2ForCausalLM": LlamaForCausalLM,
    "Qwen3ForCausalLM": Qwen2ForCausalLM,
}
for _name, _base_cls in _FALLBACK_MAP.items():
    try:
        import transformers as _tf
        _cls = getattr(_tf, _name, None)
        if _cls is not None:
            if _cls not in type_to_dimension_mapping:
                type_to_dimension_mapping[_cls] = type_to_dimension_mapping[_base_cls]
            if _cls not in type_to_module_mapping:
                type_to_module_mapping[_cls] = type_to_module_mapping[_base_cls]
            print(f"  pyvene registry: mapped {_name} -> {_base_cls.__name__}")
    except Exception as e:
        print(f"  pyvene registry: could not map {_name}: {e}")

from pyvene import IntervenableModel, IntervenableConfig, RepresentationConfig, VanillaIntervention

from huggingface_hub import login
login(os.environ.get("HF_TOKEN"))  # set HF_TOKEN


# Inlined from model_registry so this script can run standalone.
def _get_max_memory(reserve_gb=4.0):
    n = torch.cuda.device_count()
    max_mem = {}
    for i in range(n):
        total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        max_mem[i] = f"{max(0.0, total_gb - reserve_gb):.0f}GiB"
    max_mem["cpu"] = "0GiB"
    return max_mem

def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()

def load_model_and_tokenizer(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name = model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "olmo" in name:
        model_kwargs = dict(
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
        )
    elif "gemma" in name:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    elif "qwen" in name:
        model_kwargs = dict(
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    else:  # llama and fallback
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )

    print(f"Loading model {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    # OLMo-2 isn't built into pyvene; register the block hooks we need.
    if "olmo" in name:
        try:
            from pyvene.models.modeling_utils import (
                type_to_dimension_mapping, type_to_module_mapping,
                CONST_OUTPUT_HOOK, CONST_INPUT_HOOK,
            )
            from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM
            if Olmo2ForCausalLM not in type_to_module_mapping:
                type_to_dimension_mapping[Olmo2ForCausalLM] = {
                    "block_output": ("hidden_size",),
                    "block_input":  ("hidden_size",),
                }
                type_to_module_mapping[Olmo2ForCausalLM] = {
                    "block_output": ("model.layers[%s]", CONST_OUTPUT_HOOK),
                    "block_input":  ("model.layers[%s]", CONST_INPUT_HOOK),
                }
        except Exception as e:
            print(f"  OLMo-2 pyvene registration skipped: {e}")

    return model, tokenizer


# Inlined from das_utils.
def normalize_pronoun(pronoun):
    if not pronoun:
        return None
    pronoun = str(pronoun).lower().strip()
    if pronoun in ("he", "him", "his"):
        return "he"
    if pronoun in ("she", "her", "hers"):
        return "she"
    if pronoun in ("they", "them", "their", "theirs"):
        return "they"
    return pronoun

def get_pronoun_token_ids(tokenizer):
    return {
        "nominative": {
            "he":   tokenizer.encode("he",    add_special_tokens=False)[0],
            "she":  tokenizer.encode("she",   add_special_tokens=False)[0],
            "they": tokenizer.encode("they",  add_special_tokens=False)[0],
        },
        "accusative": {
            "he":   tokenizer.encode("him",   add_special_tokens=False)[0],
            "she":  tokenizer.encode("her",   add_special_tokens=False)[0],
            "they": tokenizer.encode("them",  add_special_tokens=False)[0],
        },
        "possessive": {
            "he":   tokenizer.encode("his",   add_special_tokens=False)[0],
            "she":  tokenizer.encode("her",   add_special_tokens=False)[0],
            "they": tokenizer.encode("their", add_special_tokens=False)[0],
        },
    }

def _resolve_source_for_row(row, mechanism):
    # G reads the base-pronoun source; R the distractor; S the stereotype pronoun.
    if mechanism == "G":
        prompt_col   = "source_prompt_G"
        expected_col = "intervention_expected_G"
        sentence_col = "source_sentence_G"
        if prompt_col not in row.index or pd.isna(row.get(prompt_col)):
            return None
        return {
            "source_prompt":         row[prompt_col],
            "source_pronoun":        normalize_pronoun(row["base_pronoun"]),
            "intervention_expected": normalize_pronoun(row[expected_col]),
            "source_sentence":       row.get(sentence_col, ""),
        }
    confuse_pron    = normalize_pronoun(row["confuse_pronoun"])
    stereotype_pron = normalize_pronoun(row["stereotype_pronoun"])
    if mechanism == "R":
        source_pron = confuse_pron
    elif mechanism == "S":
        source_pron = stereotype_pron
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    prompt_col   = f"source_prompt_{source_pron}"
    expected_col = f"intervention_expected_{source_pron}"
    sentence_col = f"source_sentence_{source_pron}"
    if prompt_col not in row.index or pd.isna(row.get(prompt_col)):
        return None
    return {
        "source_prompt":         row[prompt_col],
        "source_pronoun":        source_pron,
        "intervention_expected": normalize_pronoun(row[expected_col]),
        "source_sentence":       row.get(sentence_col, ""),
    }

def dataset_slug(model_name: str) -> str:
    slug = model_name.split("/")[-1].lower()
    for suffix in ["-instruct", "-it"]:
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
    return re.sub(r"(olmo-\d+)-\d{4}-(\d+b)", r"\1-\2", slug)


# Inlined from das_layersearch (avoids the pronoun_token_registry).
def get_num_layers_from_config(model_name: str) -> int:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name)
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
        return cfg.text_config.num_hidden_layers
    raise AttributeError(f"Cannot read num_hidden_layers from config: {type(cfg)}")

def get_layers_to_test(num_layers: int, stride: int = 2, start_layer: int = 0) -> list:
    return list(range(start_layer, num_layers, stride))

def get_default_stride(model_name: str) -> int:
    name = model_name.lower()
    if any(k in name for k in ["13b"]):
        return 8
    if "gemma" in name and "9b" in name:
        return 5
    return 2


PRONOUN_TO_IDX = {"he": 0, "she": 1, "they": 2}
IDX_TO_PRONOUN = {0: "he", 1: "she", 2: "they"}

DATA_DIR = Path("add/path/data")  # add path later

# Per-model CSV filenames, keyed by dataset_slug. all_fixed = GR dataset. #Todo: Change that name to GR!
_CSV_PATHS = {
    "llama-3.1-8b":  ("diagnostic_pairs_all_fixed_llama-3.1-8b.csv",
                      "diagnostic_pairs_GS_llama-3.1-8b.csv",
                      "diagnostic_pairs_RS_llama-3.1-8b.csv"),
    "olmo-2-1b":     ("diagnostic_pairs_all_fixed_olmo-2-1b.csv",
                      "diagnostic_pairs_GS_olmo-2-1b.csv",
                      "diagnostic_pairs_RS_olmo-2-1b.csv"),
    "olmo-2-7b":     ("diagnostic_pairs_all_fixed_olmo-2-7b.csv",
                      "diagnostic_pairs_GS_olmo-2-7b.csv",
                      "diagnostic_pairs_RS_olmo-2-7b.csv"),
    "olmo-2-13b":    ("diagnostic_pairs_all_fixed_olmo-2-13b.csv",
                      "diagnostic_pairs_GS_olmo-2-13b.csv",
                      "diagnostic_pairs_RS_olmo-2-13b.csv"),
    "qwen2.5-7b":    ("diagnostic_pairs_all_fixed_qwen2.5-7b.csv",
                      "diagnostic_pairs_GS_qwen2.5-7b.csv",
                      "diagnostic_pairs_RS_qwen2.5-7b.csv"),
    "gemma-2-9b":    ("diagnostic_pairs_all_fixed_gemma-2-9b.csv",
                      "diagnostic_pairs_GS_gemma-2-9b.csv",
                      "diagnostic_pairs_RS_gemma-2-9b.csv"),
}

def get_csv_paths(model_name: str, data_dir: Path) -> list:
    dslug = dataset_slug(model_name)
    if dslug not in _CSV_PATHS:
        raise ValueError(
            f"No CSV paths for slug '{dslug}'. Known slugs: {list(_CSV_PATHS)}"
        )
    return [data_dir / f for f in _CSV_PATHS[dslug]]


def build_intervenable(model, layer):
    # VanillaIntervention = straight residual-stream patch, no learned rotation.
    config = IntervenableConfig(
        model_type=type(model),
        representations=[RepresentationConfig(layer, "block_output")],
        intervention_types=VanillaIntervention,
    )
    return IntervenableModel(config, model)


def _find_intervention_position(base_ids, source_ids, pad_token_id):
    # First non-pad token where base and source differ (the swapped pronoun).
    diff = (base_ids != source_ids).nonzero(as_tuple=False).squeeze(-1)
    non_pad = base_ids != pad_token_id
    valid = diff[non_pad[diff]] if len(diff) > 0 else diff
    if len(valid) == 0:
        return (base_ids != pad_token_id).nonzero()[-1].item()
    return valid[0].item()


def _run_single_intervention(intervenable, base_ids, source_ids, intervention_pos,
                              pronoun_token_ids, tokenizer):
    # softmax over pronouns at the response position after patching source-> base.
    base_input   = base_ids.unsqueeze(0)
    source_input = source_ids.unsqueeze(0)

    with torch.no_grad():
        _, out = intervenable(
            {"input_ids": base_input},
            [{"input_ids": source_input}],
            {"sources->base": intervention_pos},
        )

    non_pad  = base_ids != tokenizer.pad_token_id
    resp_pos = non_pad.nonzero()[-1].item()
    logits   = out.logits[0, resp_pos]

    ids = torch.tensor([
        pronoun_token_ids["nominative"]["he"],
        pronoun_token_ids["nominative"]["she"],
        pronoun_token_ids["nominative"]["they"],
    ], device=logits.device)
    pronoun_logits = logits[ids].float()
    return F.softmax(pronoun_logits, dim=0).cpu().numpy()


def find_best_layer(model, tokenizer, model_name, gr_csv_path, candidate_layers,
                    n_subset=50, mechanism="R"):
    """
    Inference-only layer pick: the last layer before the source pronoun first
    starts winning over the base, i.e. just before the intervention takes hold.
    `mechanism` selects which source prompt drives the patch.
    """
    df = pd.read_csv(gr_csv_path)
    df = df.dropna(subset=["base_prompt"]).sample(
        n=min(n_subset, len(df)), random_state=42
    )
    pronoun_token_ids = get_pronoun_token_ids(tokenizer)
    layer_results = {}

    print(f"\nLayer search ({mechanism}) on {len(df)} examples across layers: {candidate_layers}")

    for layer in tqdm(candidate_layers, desc="Layer search"):
        intervenable = build_intervenable(model, layer)
        base_wins = source_wins = total = 0

        with torch.no_grad():
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Layer {layer}", leave=False):
                resolved = _resolve_source_for_row(row, mechanism)
                if resolved is None:
                    continue
                base_pronoun   = normalize_pronoun(row["base_pronoun"])
                source_pronoun = resolved["source_pronoun"]
                if base_pronoun == source_pronoun:
                    continue

                base_tok = tokenizer(
                    row["base_prompt"], max_length=512,
                    padding="max_length", truncation=True, return_tensors="pt",
                )
                source_tok = tokenizer(
                    resolved["source_prompt"], max_length=512,
                    padding="max_length", truncation=True, return_tensors="pt",
                )
                device      = next(model.parameters()).device
                base_ids    = base_tok["input_ids"][0].to(device)
                source_ids  = source_tok["input_ids"][0].to(device)
                intv_pos    = _find_intervention_position(
                    base_ids, source_ids, tokenizer.pad_token_id
                )

                try:
                    probs = _run_single_intervention(
                        intervenable, base_ids, source_ids, intv_pos,
                        pronoun_token_ids, tokenizer,
                    )
                    pred = IDX_TO_PRONOUN[int(np.argmax(probs))]
                    total += 1
                    if pred == base_pronoun:
                        base_wins += 1
                    elif pred == source_pronoun:
                        source_wins += 1
                except Exception as e:
                    continue

        base_frac   = base_wins / total if total else 0.0
        source_frac = source_wins / total if total else 0.0
        layer_results[layer] = {"base_frac": base_frac, "source_frac": source_frac,
                                 "total": total}
        print(f"  Layer {layer:3d}: base={base_frac:.3f}  source={source_frac:.3f}  n={total}")

        del intervenable
        torch.cuda.empty_cache()

    # For bigger models, we find the first layer where source overtakes base, then take the layer just before it.
    flip_idx = None
    for i, layer in enumerate(candidate_layers):
        r = layer_results[layer]
        if r["source_frac"] > r["base_frac"]:
            flip_idx = i
            break

    if flip_idx is None:
        # Source never wins -> no clear transition; use the second-to-last layer.
        best_layer = candidate_layers[-2] if len(candidate_layers) >= 2 else candidate_layers[-1]
        print(f"  WARNING: source never dominates -- no clear flip, using layer {best_layer}")
    elif flip_idx == 0:
        # Source already wins at the first layer.
        best_layer = candidate_layers[0]
        print(f"  WARNING: flip at first layer, using layer {best_layer}")
    else:
        best_layer = candidate_layers[flip_idx - 1]
        print(f"  Flip at layer {candidate_layers[flip_idx]} -- best layer (just before): {best_layer}")

    return best_layer, layer_results


def collect_empirical_distributions(model, tokenizer, layer_per_mechanism, csv_paths,
                                     model_name, max_samples=None):
    """
    Mean softmax over (he, she, they) for each (g_idx, r_idx, s_idx) triple.

    Each row is run once per mechanism at that mechanism's layer, so up to 3
    interventions per row feed the triple's accumulator. layer_per_mechanism has
    keys "G", "R", "S".
    """
    pronoun_token_ids = get_pronoun_token_ids(tokenizer)
    device = next(model.parameters()).device

    print(f"\n  Layer per mechanism: G={layer_per_mechanism['G']}  "
          f"R={layer_per_mechanism['R']}  S={layer_per_mechanism['S']}")

    # One intervenable per unique layer (mechanisms can share a layer).
    unique_layers = {v for v in layer_per_mechanism.values()}
    intervenables = {layer: build_intervenable(model, layer) for layer in unique_layers}

    # triple -> list of (3,) softmax arrays
    accum = {}

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["base_prompt"])
        if max_samples and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=42)

        print(f"\n  Processing {csv_path.name} ({len(df)} rows)...")

        row_count = skipped = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=csv_path.stem):
            base_pronoun       = normalize_pronoun(str(row.get("base_pronoun", "")))
            confuse_pronoun    = normalize_pronoun(str(row.get("confuse_pronoun", "")))
            stereotype_pronoun = normalize_pronoun(str(row.get("stereotype_pronoun", "")))

            if not all([base_pronoun, confuse_pronoun, stereotype_pronoun]):
                skipped += 1
                continue
            if any(p not in PRONOUN_TO_IDX
                   for p in [base_pronoun, confuse_pronoun, stereotype_pronoun]):
                skipped += 1
                continue

            g_idx  = PRONOUN_TO_IDX[base_pronoun]
            r_idx  = PRONOUN_TO_IDX[confuse_pronoun]
            s_idx  = PRONOUN_TO_IDX[stereotype_pronoun]
            triple = (g_idx, r_idx, s_idx)

            base_tok = tokenizer(
                row["base_prompt"], max_length=512,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            base_ids = base_tok["input_ids"][0].to(device)

            # Run each mechanism at its own layer.
            for mech in ("G", "R", "S"):
                resolved = _resolve_source_for_row(row, mech)
                if resolved is None:
                    continue

                source_tok = tokenizer(
                    resolved["source_prompt"], max_length=512,
                    padding="max_length", truncation=True, return_tensors="pt",
                )
                source_ids   = source_tok["input_ids"][0].to(device)
                intv_pos     = _find_intervention_position(
                    base_ids, source_ids, tokenizer.pad_token_id
                )
                intervenable = intervenables[layer_per_mechanism[mech]]

                try:
                    probs = _run_single_intervention(
                        intervenable, base_ids, source_ids, intv_pos,
                        pronoun_token_ids, tokenizer,
                    )
                    accum.setdefault(triple, []).append(probs)
                    row_count += 1
                except Exception:
                    continue

        print(f"    Collected {row_count} mechanism-interventions, skipped {skipped} rows")

    for layer, intv in intervenables.items():
        del intv
    torch.cuda.empty_cache()
    gc.collect()

    # Average within each triple.
    distributions = {triple: np.stack(arrs).mean(axis=0)
                     for triple, arrs in accum.items()}

    print(f"\n  Total unique (g,r,s) triples: {len(distributions)}")
    for triple, dist in sorted(distributions.items()):
        g, r, s = triple
        print(f"    ({IDX_TO_PRONOUN[g]},{IDX_TO_PRONOUN[r]},{IDX_TO_PRONOUN[s]}): "
              f"he={dist[0]:.3f} she={dist[1]:.3f} they={dist[2]:.3f}  "
              f"(n={len(accum[triple])})")

    return distributions


class MixturePronounModel(nn.Module):
    """
    logit(a) = w_G * 1{a==g} + w_R[r] * 1{a==r} + w_S[s] * 1{a==s}

    pki_idx: LongTensor[B, 3], columns (g_idx, r_idx, s_idx).
    w_G is a scalar; w_R and w_S are pronoun-indexed vectors.
    """
    def __init__(self, active_G=True, active_R=True, active_S=True):
        super().__init__()
        self.active_G = active_G
        self.active_R = active_R
        self.active_S = active_S
        if active_G:
            self.w_G = nn.Parameter(torch.tensor(1.0))
        if active_R:
            self.w_R = nn.Parameter(torch.ones(3))
        if active_S:
            self.w_S = nn.Parameter(torch.ones(3))

    def forward(self, pki_idx):
        B = pki_idx.shape[0]
        g_idx = pki_idx[:, 0]  # [B]
        r_idx = pki_idx[:, 1]
        s_idx = pki_idx[:, 2]

        logits = torch.zeros(B, 3, device=pki_idx.device)

        # G: one-hot at g_idx
        if self.active_G:
            g_oh = F.one_hot(g_idx, num_classes=3).float()
            logits = logits + self.w_G * g_oh

        # R: indexed weight at r_idx
        if self.active_R:
            r_oh = F.one_hot(r_idx, num_classes=3).float()
            w_r  = self.w_R[r_idx].unsqueeze(1)  # [B,1]
            logits = logits + w_r * r_oh

        # S: indexed weight at s_idx
        if self.active_S:
            s_oh = F.one_hot(s_idx, num_classes=3).float()
            w_s  = self.w_S[s_idx].unsqueeze(1)
            logits = logits + w_s * s_oh

        return logits


def jsd_loss(pred_logits, target_probs):
    # Jensen-Shannon divergence (log base2), averaged over batch.
    pred_probs = F.softmax(pred_logits, dim=-1).clamp(min=1e-8)
    target     = target_probs.clamp(min=1e-8)
    m = 0.5 * (pred_probs + target)
    kl1 = (target * (torch.log2(target) - torch.log2(m))).sum(-1)
    kl2 = (pred_probs * (torch.log2(pred_probs) - torch.log2(m))).sum(-1)
    return (0.5 * kl1 + 0.5 * kl2).mean()


def _build_tensors(distributions):
    # dict -> (pki triple tensor, target prob tensor)
    triples = sorted(distributions.keys())
    pki     = torch.tensor(triples, dtype=torch.long)
    targets = torch.tensor(
        np.stack([distributions[t] for t in triples], axis=0), dtype=torch.float32
    )
    return pki, targets


def train_mixture_model(distributions, epochs=2000, patience=200, lr=0.05,
                        seed=42, batch_size=512, active_G=True, active_R=True,
                        active_S=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    pki, targets = _build_tensors(distributions)
    N = len(pki)
    idx = torch.randperm(N)
    n_train = int(0.70 * N)
    n_val   = int(0.15 * N)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]
    pki_train, tgt_train = pki[train_idx], targets[train_idx]
    pki_val,   tgt_val   = pki[val_idx],   targets[val_idx]
    pki_test,  tgt_test  = pki[test_idx],  targets[test_idx]

    model = MixturePronounModel(active_G=active_G, active_R=active_R, active_S=active_S)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_loss = float("inf")
    best_state    = {k: v.clone() for k, v in model.state_dict().items()}  # init from epoch 0
    patience_cnt  = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(pki_train))
        for start in range(0, len(pki_train), batch_size):
            bidx  = perm[start:start + batch_size]
            loss  = jsd_loss(model(pki_train[bidx]), tgt_train[bidx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = jsd_loss(model(pki_val), tgt_val).item()
        history.append(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    def _eval_metrics(pki_t, tgt_t):
        # JSS = 1 - JSD; also report both KL directions for reference.
        model.eval()
        with torch.no_grad():
            pred_logits = model(pki_t)
            pred_probs  = F.softmax(pred_logits, dim=-1).clamp(min=1e-8)
            tgt_c       = tgt_t.clamp(min=1e-8)
            jss_vals    = 1.0 - 0.5 * (
                (tgt_c * (torch.log2(tgt_c) - torch.log2(0.5 * (tgt_c + pred_probs)))).sum(-1) +
                (pred_probs * (torch.log2(pred_probs) - torch.log2(0.5 * (tgt_c + pred_probs)))).sum(-1)
            )
            kl_tp = (tgt_c * (torch.log(tgt_c) - torch.log(pred_probs))).sum(-1)
            kl_pt = (pred_probs * (torch.log(pred_probs) - torch.log(tgt_c))).sum(-1)
        return jss_vals.numpy(), kl_tp.numpy(), kl_pt.numpy()

    def _ci(arr, confidence=0.95):
        n    = len(arr)
        mean = float(np.mean(arr))
        se   = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        h    = t_dist.ppf((1 + confidence) / 2, df=max(n - 1, 1)) * se
        return mean, float(np.std(arr, ddof=1)), h

    jss_arr, kl_tp_arr, kl_pt_arr = _eval_metrics(pki_test, tgt_test)
    jss_mean, jss_std, jss_ci     = _ci(jss_arr)
    kl_tp_mean, kl_tp_std, _      = _ci(kl_tp_arr)
    kl_pt_mean, kl_pt_std, _      = _ci(kl_pt_arr)

    weights = {}
    if active_G:
        weights["w_G"] = float(model.w_G.item())
    if active_R:
        weights["w_R"] = model.w_R.detach().tolist()
    if active_S:
        weights["w_S"] = model.w_S.detach().tolist()

    info = {
        "jss":       {"mean": jss_mean,   "std": jss_std,   "ci95": jss_ci},
        "kl_tp":     {"mean": kl_tp_mean, "std": kl_tp_std},
        "kl_pt":     {"mean": kl_pt_mean, "std": kl_pt_std},
        "weights":   weights,
        "n_train":   n_train,
        "n_val":     n_val,
        "n_test":    len(pki_test),
        "epochs_run": len(history),
        "history":   history,
    }
    print(f"  Test JSS={jss_mean:.4f} +/- {jss_std:.4f}  (95% CI +/-{jss_ci:.4f})")
    print(f"  KL(t||p)={kl_tp_mean:.4f}  KL(p||t)={kl_pt_mean:.4f}")
    return model, info

# Full model plus every single- and double-mechanism ablation.
ABLATIONS = [
    ("M\\{G}",    False, True,  True),
    ("M\\{R}",    True,  False, True),
    ("M\\{S}",    True,  True,  False),
    ("M\\{R,S}",  True,  False, False),
    ("M\\{G,S}",  False, True,  False),
    ("M\\{G,R}",  False, False, True),
]


def run_ablations(distributions, epochs=2000, patience=200, lr=0.05, seed=42,
                  batch_size=512):
    rows = []

    # Full model first.
    _, info = train_mixture_model(distributions, epochs=epochs, patience=patience,
                                  lr=lr, seed=seed, batch_size=batch_size)
    rows.append({"model": "M (full)", "JSS": info["jss"]["mean"],
                 "KL_tp": info["kl_tp"]["mean"], "KL_pt": info["kl_pt"]["mean"]})

    for name, aG, aR, aS in ABLATIONS:
        _, info = train_mixture_model(distributions, epochs=epochs,
                                      patience=patience, lr=lr, seed=seed,
                                      batch_size=batch_size,
                                      active_G=aG, active_R=aR, active_S=aS)
        rows.append({"model": name, "JSS": info["jss"]["mean"],
                     "KL_tp": info["kl_tp"]["mean"],
                     "KL_pt": info["kl_pt"]["mean"]})

        print(f"  Ablation {name}: JSS={rows[-1]['JSS']:.4f}")

    return pd.DataFrame(rows)


# def plot_training_curve(history, out_path):
#     plt.figure(figsize=(8, 4))
#     plt.plot(history)
#     plt.xlabel("Epoch")
#     plt.ylabel("Val JSD")
#     plt.title("Training curve (JSD loss)")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()


# def plot_learned_weights(weights, out_path):
#     labels, vals = [], []
#     if "w_G" in weights:
#         labels.append("w_G"); vals.append(weights["w_G"])
#     if "w_R" in weights:
#         for i, v in enumerate(weights["w_R"]):
#             labels.append(f"w_R[{IDX_TO_PRONOUN[i]}]"); vals.append(v)
#     if "w_S" in weights:
#         for i, v in enumerate(weights["w_S"]):
#             labels.append(f"w_S[{IDX_TO_PRONOUN[i]}]"); vals.append(v)
#     plt.figure(figsize=(8, 4))
#     colors = (["tab:blue"] * (1 if "w_G" in weights else 0) +
#               ["tab:orange"] * (3 if "w_R" in weights else 0) +
#               ["tab:green"] * (3 if "w_S" in weights else 0))
#     plt.bar(labels, vals, color=colors)
#     plt.xticks(rotation=30, ha="right")
#     plt.ylabel("Weight value")
#     plt.title("Learned mixture weights")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()


# def plot_ablations(abl_df, out_path):
#     plt.figure(figsize=(10, 4))
#     colors = ["tab:blue" if r["model"] == "M (full)" else "tab:orange"
#               for _, r in abl_df.iterrows()]
#     plt.bar(abl_df["model"], abl_df["JSS"], color=colors)
#     plt.xticks(rotation=30, ha="right")
#     plt.ylabel("JSS")
#     plt.ylim(0, 1)
#     plt.title("Ablation JSS scores")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=150)
#     plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--layer_G",     type=int, default=None,
                        help="Layer for G (group entity binding)")
    parser.add_argument("--layer_R",     type=int, default=None,
                        help="Layer for R (recency / distractor)")
    parser.add_argument("--layer_S",     type=int, default=None,
                        help="Layer for S (lexical stereotype)")
    parser.add_argument("--max_samples",    type=int, default=None)
    parser.add_argument("--layer_search_n", type=int, default=50,
                        help="Examples per layer in the layer search (default 50)")
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--output_dir",  default=None)
    parser.add_argument("--epochs",      type=int,  default=2000)
    parser.add_argument("--patience",    type=int,  default=200)
    args = parser.parse_args()

    model_name = args.model
    slug       = model_slug(model_name)
    dslug      = dataset_slug(model_name)
    data_dir   = Path(args.data_dir)

    out_dir = Path(args.output_dir) if args.output_dir else \
              Path(__file__).resolve().parent / "results" / "mixture_model" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"MIXTURE MODEL  |  {model_name}")
    print(f"Output: {out_dir}")
    print("=" * 80)

    csv_paths = get_csv_paths(model_name, data_dir)
    print(f"CSV paths:")
    for p in csv_paths:
        print(f"  {p}  ({'EXISTS' if p.exists() else 'MISSING'})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers_provided = all(x is not None for x in [args.layer_G, args.layer_R, args.layer_S])

    if layers_provided:
        layer_per_mechanism = {"G": args.layer_G, "R": args.layer_R, "S": args.layer_S}
        layer_search_info   = {"layer_per_mechanism": layer_per_mechanism, "source": "cli"}
        print(f"Using provided layers: G={args.layer_G}  R={args.layer_R}  S={args.layer_S}")
    else:
        # No (or partial) layers given -> search all three.
        if any(x is not None for x in [args.layer_G, args.layer_R, args.layer_S]):
            print("WARNING: partial layers provided -- running layer search for all three.")

        num_layers = get_num_layers_from_config(model_name)
        stride     = get_default_stride(model_name)
        candidates = get_layers_to_test(num_layers, stride)

        print("Loading model for layer search...")
        model, _ = load_model_and_tokenizer(model_name)

        # G and R search on the GR (all_fixed) set; S searches on RS.
        _paths            = get_csv_paths(model_name, data_dir)
        all_fixed_csv, _, rs_csv = _paths
        layer_per_mechanism = {}
        all_ls_results      = {}

        for mech, csv_path in [("G", all_fixed_csv), ("R", all_fixed_csv),
                                ("S", rs_csv)]:
            print(f"\n--- Layer search for mechanism {mech} ---")
            best, ls_results = find_best_layer(
                model, tokenizer, model_name, csv_path, candidates,
                n_subset=args.layer_search_n, mechanism=mech,
            )
            layer_per_mechanism[mech] = best
            all_ls_results[mech]      = ls_results

        layer_search_info = {"layer_per_mechanism": layer_per_mechanism,
                             "results": all_ls_results}
        (out_dir / "layer_search.json").write_text(
            json.dumps(layer_search_info, default=str, indent=2)
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nFinal layers: G={layer_per_mechanism['G']}  "
          f"R={layer_per_mechanism['R']}  S={layer_per_mechanism['S']}")

    # Collect the empirical distributions.
    print(f"\nLoading model for distribution collection...")
    model, _ = load_model_and_tokenizer(model_name)

    distributions = collect_empirical_distributions(
        model, tokenizer, layer_per_mechanism, csv_paths, model_name,
        max_samples=args.max_samples,
    )
    torch.save(distributions, out_dir / "empirical_distributions.pt")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if len(distributions) < 4:
        print("WARNING: fewer than 4 unique triples collected -- mixture model may be unreliable.")

    # Full mixture model.
    print("\nTraining full mixture model...")
    mixture_model, info = train_mixture_model(
        distributions, epochs=args.epochs, patience=args.patience,
    )
    torch.save(mixture_model.state_dict(), out_dir / "mixture_model_weights.pt")

    # Ablations:
    print("\nRunning ablations...")
    abl_df = run_ablations(distributions, epochs=args.epochs, patience=args.patience)

    results = {
        "model":              model_name,
        "layer_per_mechanism": layer_per_mechanism,
        "jss":                info["jss"],
        "kl_tp":              info["kl_tp"],
        "kl_pt":              info["kl_pt"],
        "weights":            info["weights"],
        "n_train":            info["n_train"],
        "n_val":              info["n_val"],
        "n_test":             info["n_test"],
        "n_triples":          len(distributions),
        "ablations":          abl_df.to_dict(orient="records"),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # plot_training_curve(info["history"], out_dir / "training_curve.png")
    # plot_learned_weights(info["weights"], out_dir / "learned_weights.png")
    # plot_ablations(abl_df, out_dir / "ablations.png")

    print(f"\nDone. Results saved to {out_dir}")
    print(f"  JSS={info['jss']['mean']:.4f} +/- {info['jss']['std']:.4f}")
    print(f"  Layers: G={layer_per_mechanism['G']}  R={layer_per_mechanism['R']}  "
          f"S={layer_per_mechanism['S']}")
    print("\nAblation summary:")
    print(abl_df.to_string(index=False))


if __name__ == "__main__":
    main()