#!/usr/bin/env python3
"""
train.py

Trains DAS-G, DAS-R and DAS-S at their best layers (from the layer search) on
the three diagnostic datasets:
- all : diagnostic_pairs_all_fixed_{model_slug}.csv  (the GR dataset)
- GS  : diagnostic_pairs_GS_{model_slug}.csv
- RS  : diagnostic_pairs_RS_{model_slug}.csv

Use:
python run_best_mechanism.py --best_layer_G 10 --best_layer_R 18 --best_layer_S 6
"""

import os
import re
import json
import argparse
import torch
import pandas as pd
from pathlib import Path
from pyvene import set_seed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Force the plain attention kernel: flash SDP don't go nicely with the intervention hooks.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from model_registry import load_model_and_tokenizer, model_slug
from pronoun_token_registry import print_pronoun_tokens
from das_utils import (
    CSVDatasetLoader,
    check_label_distribution,
    train_best_layer,
)


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

CONFIG = {
    "epochs":                      3,
    "batch_size":                  4,
    "gradient_accumulation_steps": 16,
    "max_length":                  512,
    "train_split":                 0.8,
    "eval_every_steps":            200,
    "eval_max_examples":           200,
}

# Fallback layers if --best_layer_* aren't passed; override per model.
DEFAULT_BEST_LAYERS = {"G": 8, "R": 16, "S": 8}

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = Path("path/data")  # add path to data

DATASET_FILENAME = {
    "all": "diagnostic_pairs_all_fixed_{slug}.csv",
    "GS":  "diagnostic_pairs_GS_{slug}.csv",
    "RS":  "diagnostic_pairs_RS_{slug}.csv",
}


def dataset_slug(model_name: str) -> str:
    # Build the slug used in dataset filenames.
    slug = model_name.split("/")[-1].lower()
    for suffix in ["-instruct", "-it"]:
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
    slug = re.sub(r"(olmo-\d+)-\d{4}-(\d+b)", r"\1-\2", slug)
    return slug


def load_dataset(model_name: str, dataset_name: str,
                 max_samples: int = None) -> tuple[Path, pd.DataFrame]:
    slug  = dataset_slug(model_name)
    fname = DATASET_FILENAME[dataset_name].format(slug=slug)
    path  = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"  Capped to {len(df):,} rows (--max_samples={max_samples})")
    print(f"  Loaded {len(df):,} rows from {fname}")
    print(f"  By base_pronoun: {dict(df.groupby('base_pronoun').size())}")
    return path, df


def main():
    parser = argparse.ArgumentParser(
        description="DAS best-layer training for G, R and/or S across datasets"
    )
    parser.add_argument("--model",        default=DEFAULT_MODEL)
    parser.add_argument("--mechanisms",   nargs="+", default=["G", "R", "S"],
                        choices=["G", "R", "S"])
    parser.add_argument("--datasets",     nargs="+", default=["all", "GS", "RS"],
                        choices=["all", "GS", "RS"])
    parser.add_argument("--best_layer_G", type=int, default=DEFAULT_BEST_LAYERS["G"])
    parser.add_argument("--best_layer_R", type=int, default=DEFAULT_BEST_LAYERS["R"])
    parser.add_argument("--best_layer_S", type=int, default=DEFAULT_BEST_LAYERS["S"])
    args = parser.parse_args()

    model_name  = args.model
    slug        = model_slug(model_name)
    best_layers = {
        "G": args.best_layer_G,
        "R": args.best_layer_R,
        "S": args.best_layer_S,
    }

    print("=" * 80)
    print(f"DAS BEST-LAYER TRAINING  |  model={model_name}")
    print(f"Mechanisms  : {args.mechanisms}")
    print(f"Datasets    : {args.datasets}")
    print(f"Max samples : {args.max_samples}")
    print(f"Best layers : " + "  ".join(f"{m}={best_layers[m]}" for m in args.mechanisms))
    print("=" * 80)
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded, device={device}")
    print_pronoun_tokens(tokenizer, model_name)

    output_root = SCRIPT_DIR / "results" / "mixed_best_layers" / slug
    all_results = {}

    for dataset_name in args.datasets:
        try:
            csv_path, df = load_dataset(model_name, dataset_name,
                                        max_samples=args.max_samples)
        except FileNotFoundError as e:
            print(f"\nWARNING: {e}\nSkipping dataset '{dataset_name}'.")
            continue

        print(f"\n{'=' * 80}")
        print(f"DATASET: {dataset_name}  ({len(df):,} rows)")
        print("=" * 80)

        for mechanism in args.mechanisms:
            layer = best_layers[mechanism]
            print(f"\n{'#' * 80}\nMECHANISM=DAS-{mechanism}  LAYER={layer}  DATASET={dataset_name}\n{'#' * 80}")
            check_label_distribution(df, dataset_name)

            output_dir = output_root / dataset_name / mechanism
            output_dir.mkdir(parents=True, exist_ok=True)

            loader = CSVDatasetLoader(
                csv_path, tokenizer,
                max_length=CONFIG["max_length"],
                model_name=model_name,
            )
            # Reuse the already-sampled df
            loader.df = df

            train_loader, eval_loader = loader.create_dataloaders(
                mechanism=mechanism,
                batch_size=CONFIG["batch_size"],
                train_split=CONFIG["train_split"],
            )

            set_seed(42)
            best_iia, history = train_best_layer(
                model, tokenizer, train_loader, eval_loader,
                loader.eval_metadata, layer, mechanism=mechanism,
                epochs=CONFIG["epochs"], device=device,
                gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
                save_dir=str(output_dir),
                eval_every_steps=CONFIG["eval_every_steps"],
                eval_max_examples=CONFIG["eval_max_examples"],
            )

            result = {
                "model":       model_name,
                "dataset":     dataset_name,
                "mechanism":   mechanism,
                "layer":       layer,
                "max_samples": args.max_samples,
                "best_iia":    best_iia,
                "history":     history,
            }
            all_results[f"{dataset_name}_{mechanism}"] = result

            with open(output_dir / "training_results.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved training_results.json -> {output_dir}")
            torch.cuda.empty_cache()

    summary_path = output_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 80}\nALL DONE -- summary -> {summary_path}\n{'=' * 80}")


if __name__ == "__main__":
    main()