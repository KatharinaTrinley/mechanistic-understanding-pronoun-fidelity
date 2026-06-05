#!/usr/bin/env python3
"""
Layer search for the DAS mechanisms G, S and R.
For each mechanism we sweep across layers (every Nth, see the stride),
train a short DAS run at each, and keep the layer with the highest IIA. The
dataset used is the GR diagnostic set (the file is named ..all_fixed.csv).

Use:
python das_layersearch.py
python das_layersearch.py --model google/...
python das_layersearch.py --model Qwen/Qwen2.5-7B-Instruct --start_layer 10

Each layer writes a checkpoint_layer{n}.json
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM, OlmoForCausalLM
from pyvene.models.modeling_utils import type_to_dimension_mapping, type_to_module_mapping

import pyvene.models.modeling_utils as _pyvene_utils

# torch.gather errors out if index and input sit on different devices, which happens under multi-GPU device_map. Wrap it to move index onto input's device.
_original_torch_gather = torch.gather

def _safe_torch_gather(input, dim, index, *, sparse_grad=False, out=None):
    if index.device != input.device:
        index = index.to(input.device)
    if out is not None:
        return _original_torch_gather(input, dim, index, sparse_grad=sparse_grad, out=out)
    return _original_torch_gather(input, dim, index, sparse_grad=sparse_grad)

torch.gather = _safe_torch_gather

# In case of newer architectures pyvene doesn't ship mappings for: we reuse a compatible base class's mappings so block_output interventions work.
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

from model_registry import load_model_and_tokenizer, model_slug
from pronoun_token_registry import print_pronoun_tokens
from das_utils import (
    CSVDatasetLoader,
    check_label_distribution,
    compute_iia_forced_choice,
    calculate_loss,
)

from huggingface_hub import login
login("xxxxxx") # add HF token

from pyvene import IntervenableModel, IntervenableConfig, RepresentationConfig, \
                   BoundlessRotatedSpaceIntervention, set_seed


SEEDS      = [42, 43, 44]
MECHANISMS = ["G", "S", "R"]
DATA_DIR   = Path("add/path/data") # add path later

# Wider layer strides for bigger models.
_STRIDE_TIERS = [
    (5, ["13b"]),
]
_STRIDE_EXCEPTIONS = [
    (5, "gemma", ["9b"]),
]


def get_num_hidden_layers(model) -> int:
    # Handles both flat configs and the nested text_config used by some models.
    cfg = model.config
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
        return cfg.text_config.num_hidden_layers
    raise AttributeError(
        f"Cannot determine num_hidden_layers from {type(cfg).__name__}."
    )
def get_default_stride(model_name: str) -> int:
    # Exceptions win over the general tiers; everything else uses stride 2.
    name = model_name.lower()
    for stride, family, keywords in _STRIDE_EXCEPTIONS:
        if family in name and any(k in name for k in keywords):
            return stride
    for stride, keywords in _STRIDE_TIERS:
        if any(k in name for k in keywords):
            return stride
    return 2


def get_layers_to_test(num_layers: int, stride: int = 2):
    return list(range(0, num_layers, stride))

def _get_layer_device(model, layer: int) -> torch.device:
    """Device that transformer layer `layer` actually lives on (multi-GPU safe)."""
    try:
        module = model.model.layers[layer]
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def simple_boundless_das_config(model_type, layer):
    return IntervenableConfig(
        model_type=model_type,
        representations=[RepresentationConfig(layer, "block_output")],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )


def prepare_all_variant_dataset():
    # Loads the GR diagnostic set (stored as diagnostic_pairs_all_fixed.csv).
    path = DATA_DIR / "diagnostic_pairs_all_fixed.csv"
    if not path.exists():
        raise FileNotFoundError(f"GR dataset not found: {path}\nRun fix_g_source.py first.")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows from diagnostic_pairs_all_fixed.csv")
    return path, df

def checkpoint_path_for_layer(output_dir: Path, layer: int) -> Path:
    return output_dir / f"checkpoint_layer{layer}.json"

def save_layer_checkpoint(output_dir: Path, layer: int, final_iia: float,
                           best_iia: float, history: dict):
    path = checkpoint_path_for_layer(output_dir, layer)
    data = {
        "layer":     layer,
        "final_iia": final_iia,
        "best_iia":  best_iia,
        "history":   history,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Checkpoint saved: {path.name}")


def load_completed_layers(output_dir: Path, layers_to_test: list) -> dict:
    """Read back any checkpoint_layer*.json so a rerun skips finished layers."""
    completed = {}
    for layer in layers_to_test:
        ckpt = checkpoint_path_for_layer(output_dir, layer)
        if not ckpt.exists():
            continue
        try:
            with open(ckpt) as f:
                data = json.load(f)
            completed[layer] = {
                "seeds": {
                    SEEDS[0]: {
                        "final_iia": data["final_iia"],
                        "best_iia":  data["best_iia"],
                    }
                },
                "mean_iia":           data["final_iia"],
                "std_iia":            0.0,
                "best_iia":           data["best_iia"],
                "training_histories": [data["history"]],
            }
            print(f"  Resumed layer {layer} from checkpoint (best IIA={data['best_iia']:.4f})")
        except Exception as e:
            print(f"  WARNING: could not load checkpoint for layer {layer}: {e}")
    return completed


def train_single_layer(model, tokenizer, train_loader, eval_loader, eval_metadata,
                       layer, epochs=2, verbose=False):
    # One DAS training run at a single layer, returning (final IIA, best IIA, adn history).
    layer_device = _get_layer_device(model, layer)

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    config       = simple_boundless_das_config(type(model), layer)
    intervenable = IntervenableModel(config, model)

    # Put the intervention params on the same device as the layer we hook.
    for v in intervenable.interventions.values():
        intervention = v[0] if isinstance(v, list) else v
        intervention.to(layer_device)

    intervenable.disable_model_gradients()

    # Train only the rotation and boundary scalars.
    optimizer_params = []
    for v in intervenable.interventions.values():
        intervention = v[0] if isinstance(v, list) else v
        optimizer_params += [{"params": intervention.rotate_layer.parameters(), "lr": 1e-3}]
        optimizer_params += [{"params": intervention.intervention_boundaries,   "lr": 1e-2}]

    optimizer     = torch.optim.Adam(optimizer_params)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * len(train_loader) * epochs),
        num_training_steps=len(train_loader) * epochs,
    )
    # Anneal boundary temperature 50 -> 0.1 so the soft mask sharpen.
    temp_schedule = torch.linspace(50.0, 0.1, len(train_loader) * epochs).to(layer_device)

    history  = {"epoch_losses": [], "eval_iias": []}
    step     = 0
    best_iia = 0.0

    for epoch in range(epochs):
        intervenable.model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Layer {layer}, Epoch {epoch}", disable=not verbose)
        for inputs in pbar:
            for k, v in inputs.items():
                if k != "intervention_ids" and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(layer_device)

            intervenable.set_temperature(temp_schedule[step])
            intervention_positions = inputs["intervention_ids"]
            if isinstance(intervention_positions, torch.Tensor):
                intervention_positions = intervention_positions.tolist()

            # Each example has its own intervention position, so step through one pos at a time and accumulat gradients before the optimizer step.
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=layer_device, requires_grad=False)
            for i in range(len(inputs["input_ids"])):
                pos = intervention_positions[i]
                if isinstance(pos, torch.Tensor):
                    pos = pos.item()
                _, output = intervenable(
                    {"input_ids": inputs["input_ids"][i:i+1]},
                    [{"input_ids": inputs["source_input_ids"][i:i+1]}],
                    {"sources->base": pos},
                )
                loss_i = calculate_loss(output.logits, inputs["labels"][i:i+1], intervenable)
                loss_i.backward()
                batch_loss = batch_loss + loss_i.detach()
                del output, loss_i
                torch.cuda.empty_cache()

            avg_loss_val = (batch_loss / len(inputs["input_ids"])).item()
            epoch_losses.append(avg_loss_val)
            if verbose:
                pbar.set_postfix({"loss": f"{avg_loss_val:.3f}"})

            optimizer.step()
            scheduler.step()
            step += 1

        avg_loss = float(np.mean(epoch_losses))
        history["epoch_losses"].append(avg_loss)

        eval_metrics = compute_iia_forced_choice(
            intervenable, eval_loader, eval_metadata, tokenizer, device=layer_device
        )
        iia = eval_metrics["iia_overall"]
        history["eval_iias"].append(iia)
        best_iia = max(best_iia, iia)

        print(f"  Epoch {epoch}: loss={avg_loss:.3f}  IIA={iia:.4f}  "
              f"attr={eval_metrics['attribution_proportions']}")

    final_iia = history["eval_iias"][-1]
    del intervenable
    torch.cuda.empty_cache()
    return final_iia, best_iia, history


def layer_search(model, tokenizer, train_loader, eval_loader, eval_metadata,
                 num_layers, epochs_per_layer=2, output_dir=".", stride=2, start_layer=0):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_layers = get_layers_to_test(num_layers, stride)

    # Drop layers below start_layer before checking checkpoints; included cause it could take a long time to complete layer search.
    if start_layer > 0:
        skipped = [l for l in all_layers if l < start_layer]
        active_layers = [l for l in all_layers if l >= start_layer]
        print(f"  --start_layer={start_layer}: skipping layers {skipped}")
    else:
        active_layers = list(all_layers)

    # Of the active layers, skip ones that already have a checkpoint.
    completed      = load_completed_layers(output_dir, active_layers)
    layers_to_test = [l for l in active_layers if l not in completed]

    print("\n" + "=" * 80)
    print(f"All layers      : {all_layers}  (stride={stride})")
    print(f"Active (>=start): {active_layers}")
    print(f"Already done    : {sorted(completed.keys())}")
    print(f"Remaining       : {layers_to_test}")
    print(f"Seeds           : {SEEDS}")
    print(f"Epochs / layer  : {epochs_per_layer}")
    print("=" * 80)

    results = {
        "layers_tested": all_layers,
        "stride":        stride,
        "seeds":         SEEDS,
        "layer_results": dict(completed),  # start with resumed layers
        "best_layer":    None,
        "best_iia":      0.0,
    }

    # Seed the running best from the resumed layers.
    for layer, lr in completed.items():
        if lr["best_iia"] > results["best_iia"]:
            results["best_iia"]  = lr["best_iia"]
            results["best_layer"] = layer

    for layer in layers_to_test:
        print(f"\n{'=' * 80}\nTESTING LAYER {layer}/{num_layers - 1}\n{'=' * 80}")
        layer_results = {"seeds": {}, "mean_iia": 0.0, "std_iia": 0.0,
                         "best_iia": 0.0, "training_histories": []}
        seed_iias = []

        for seed_idx, seed in enumerate(SEEDS):
            print(f"\nSeed {seed_idx + 1}/{len(SEEDS)}  (seed={seed})")
            set_seed(seed)
            try:
                final_iia, best_iia, history = train_single_layer(
                    model, tokenizer, train_loader, eval_loader, eval_metadata,
                    layer, epochs=epochs_per_layer, verbose=(seed_idx == 0)
                )
            except torch.cuda.OutOfMemoryError:
                # Skip this seed on OOM rather than crashing the whole search.
                print(f"  !! OOM at layer {layer}, seed {seed} -- skipping seed.")
                torch.cuda.empty_cache()
                final_iia = best_iia = float("nan")
                history = {"epoch_losses": [], "eval_iias": []}

            layer_results["seeds"][seed]            = {"final_iia": final_iia, "best_iia": best_iia}
            layer_results["training_histories"].append(history)
            seed_iias.append(final_iia)
            if not np.isnan(final_iia):
                layer_results["best_iia"] = max(layer_results["best_iia"], final_iia)

        valid_iias = [x for x in seed_iias if not np.isnan(x)]
        layer_results["mean_iia"] = float(np.nanmean(seed_iias)) if valid_iias else float("nan")
        layer_results["std_iia"]  = float(np.nanstd(seed_iias))  if valid_iias else float("nan")

        if valid_iias:
            print(f"\nLayer {layer}: mean={layer_results['mean_iia']:.4f} "
                  f"+/- {layer_results['std_iia']:.4f}  best={layer_results['best_iia']:.4f}")
        else:
            print(f"\nLayer {layer}: all seeds OOM -- skipped.")

        results["layer_results"][layer] = layer_results

        if layer_results["best_iia"] > results["best_iia"]:
            results["best_iia"]  = layer_results["best_iia"]
            results["best_layer"] = layer

        # Checkpoint this layer and flush the full results JSON.
        save_layer_checkpoint(
            output_dir, layer,
            final_iia=layer_results["mean_iia"],
            best_iia=layer_results["best_iia"],
            history=layer_results["training_histories"][0],
        )
        with open(output_dir / "layer_search_results.json", "w") as f:
            json.dump(results, f, indent=2)

        torch.cuda.empty_cache()

    # Plot whatever layers have results.
    visualize_layer_search(results, output_dir)
    return results


def write_run_log(results, model_name, num_layers, mechanism, output_dir):
    layers = results["layers_tested"]
    lines  = [
        "=" * 60, "LAYER SEARCH RUN LOG", "=" * 60,
        f"Model:         {model_name}",
        f"Mechanism:     {mechanism}",
        f"Num layers:    {num_layers}",
        f"Stride:        {results.get('stride', '?')}",
        f"Layers tested: {layers}",
        f"Seeds:         {SEEDS}", "",
        "Per-layer results:",
    ]
    for layer in layers:
        r = results["layer_results"].get(layer)
        if r is None:
            lines.append(f"  Layer {layer:3d}: skipped (below --start_layer)")
            continue
        lines.append(f"  Layer {layer:3d}: mean IIA = {r['mean_iia']:.4f} "
                     f"+/- {r['std_iia']:.4f}  (best = {r['best_iia']:.4f})")
    lines += ["", f"Best layer:    {results['best_layer']}",
              f"Best IIA:      {results['best_iia']:.4f}", "=" * 60]
    text = "\n".join(lines) + "\n"
    (Path(output_dir) / "run.log").write_text(text)
    print(text)


def visualize_layer_search(results, output_dir):
    layers = [l for l in results["layers_tested"] if l in results["layer_results"]]
    if not layers:
        print("  No completed layers to plot yet.")
        return

    mean_iias = [results["layer_results"][l]["mean_iia"] for l in layers]
    std_iias  = [results["layer_results"][l]["std_iia"]  for l in layers]
    best_iias = [results["layer_results"][l]["best_iia"] for l in layers]

    plt.figure(figsize=(20, 8))

    ax1 = plt.subplot(1, 3, 1)
    ax1.errorbar(layers, mean_iias, yerr=std_iias, marker="o", capsize=5, linewidth=2)
    if results["best_layer"] in layers:
        ax1.axvline(results["best_layer"], color="red", linestyle="--", alpha=0.5,
                    label=f'Best ({results["best_layer"]})')
    ax1.set_xlabel("Layer"); ax1.set_ylabel("Mean IIA")
    ax1.set_title("Mean IIA Across Layers", fontweight="bold")
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(layers, best_iias, marker="s", linewidth=2, color="green")
    if results["best_layer"] in layers:
        ax2.axvline(results["best_layer"], color="red", linestyle="--", alpha=0.5,
                    label=f'Best ({results["best_layer"]})')
    ax2.set_xlabel("Layer"); ax2.set_ylabel("Best IIA")
    ax2.set_title("Best IIA Across Layers", fontweight="bold")
    ax2.grid(True, alpha=0.3); ax2.legend()

    ax3 = plt.subplot(1, 3, 3)
    best_layer = results["best_layer"]
    if best_layer is not None and best_layer in results["layer_results"]:
        for i, hist in enumerate(results["layer_results"][best_layer]["training_histories"]):
            ax3.plot(hist["eval_iias"], marker="o", label=f"Seed {SEEDS[i]}", alpha=0.7)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Eval IIA")
    ax3.set_title(f"Best Layer ({best_layer}) Training", fontweight="bold")
    ax3.grid(True, alpha=0.3); ax3.legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "layer_search.png", dpi=300, bbox_inches="tight")
    plt.savefig(Path(output_dir) / "layer_search.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DAS layer search on the GR diagnostic dataset")
    parser.add_argument("--model",        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--train_split",  type=float, default=0.8)
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--max_samples",  type=int,   default=None,
                        help="Cap total examples before train/eval split (default: use all)")
    parser.add_argument("--mechanisms",   nargs="+",  default=["G", "S", "R"],
                        choices=["G", "S", "R"],
                        help="Mechanisms to search (default: G S R)")
    parser.add_argument("--layer_stride", type=int,   default=None,
                        help="Layer search stride (default: per-model, see get_default_stride)")
    parser.add_argument("--start_layer",  type=int,   default=0,
                        help="Skip all layers below this index (checkpoints above it as usual)")
    args = parser.parse_args()

    model_name = args.model
    slug       = model_slug(model_name)
    stride     = args.layer_stride if args.layer_stride is not None else get_default_stride(model_name)


    print(f"DAS LAYER SEARCH   model={model_name}  mechanisms={args.mechanisms}")
    print(f"  train_split  : {args.train_split}")
    print(f"  epochs/layer : {args.epochs}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  seeds        : {SEEDS}")
    print(f"  layer_stride : {stride}")
    print(f"  start_layer  : {args.start_layer}")

    csv_path, all_df = prepare_all_variant_dataset()

    model, tokenizer = load_model_and_tokenizer(model_name)
    print(f"Model loaded on: {next(model.parameters()).device}")
    print_pronoun_tokens(tokenizer, model_name)

    num_layers     = get_num_hidden_layers(model)
    layers_to_test = get_layers_to_test(num_layers, stride)
    print(f"\nModel has {num_layers} hidden layers.")
    print(f"Layer search scheme (stride={stride}): {layers_to_test}")

    script_dir  = Path(__file__).resolve().parent
    output_root = script_dir / "results" / "layer_search" / slug

    for mechanism in args.mechanisms:
        print(f"MECHANISM: DAS-{mechanism}")
        check_label_distribution(all_df, "GR")

        loader = CSVDatasetLoader(csv_path, tokenizer, max_length=512)
        if args.max_samples is not None:
            loader.df = loader.df.sample(
                n=min(args.max_samples, len(loader.df)), random_state=42
            )
            print(f"  Subsampled to {len(loader.df):,} rows (--max_samples={args.max_samples})")

        train_loader, eval_loader = loader.create_dataloaders(
            mechanism=mechanism,
            batch_size=args.batch_size,
            train_split=args.train_split,
        )

        output_dir = output_root / mechanism / "all"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = layer_search(
            model, tokenizer, train_loader, eval_loader, loader.eval_metadata,
            num_layers, epochs_per_layer=args.epochs,
            output_dir=str(output_dir),
            stride=stride,
            start_layer=args.start_layer,
        )

        write_run_log(results, model_name, num_layers, mechanism, output_dir)
        print(f"\nDAS-{mechanism} best layer: {results['best_layer']}  "
              f"IIA: {results['best_iia']:.4f}")
        torch.cuda.empty_cache()

    print("DONE")


if __name__ == "__main__":
    main()