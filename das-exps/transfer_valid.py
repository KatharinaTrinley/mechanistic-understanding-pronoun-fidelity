#!/usr/bin/env python3
"""
transfer_valid.py

Cross-mechanism specificity check for the Boundless DAS rotations.
Builds a 3x3 matrix over (rotation, dataset). 
All three rotations are the ones trained on GR. We freeze each and evaluate
it on every dataset. 

Per cell, over the scored uids:
- hit_source : prediction = isolated mechanism's intervention_expected
- hit_rot    : prediction = rotation mechanism's intervention_expected
- no_flip    : prediction = base(unintervened) prediction
- other      : none of the above

--> random set of --n_sample uids per dataset, fixed seed.

Use:
python transfer_eval.py --model google/gemma-2-9b-it --n_sample 250
"""

import os
import re
import json
import glob
import random
import argparse
from pathlib import Path
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from model_registry import load_model_and_tokenizer, model_slug
from das_utils import (
    CSVDatasetLoader,
    get_pronoun_token_ids,
    compute_log_likelihood_for_candidates,
    simple_boundless_das_config,
    dataset_slug,
)
from pyvene import IntervenableModel


MECHANISMS = ("G", "R", "S")

# Each dataset agrees on two mechanisms, so it isolates the third.
DATASET_ISOLATES = {"all": "S", "GS": "R", "RS": "G"}
DATASETS = ("all", "GS", "RS")

DATASET_FILENAME = {
    "all": "diagnostic_pairs_all_fixed_{slug}.csv",
    "GS":  "diagnostic_pairs_GS_{slug}.csv",
    "RS":  "diagnostic_pairs_RS_{slug}.csv",
}

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

SCRIPT_DIR = Path(__file__).resolve().parent

# Where the trained rotations sit (output of full training (train.py)), and the diagnostic CSVs.
ROTATION_ROOT = Path("results/mixed_best_layers")  # change path later
DATA_DIR = Path("data")  # change path later


def find_rotation(model_dir_slug: str, mechanism: str) -> dict:
    # Locate the `all`-trained rotation (+ boundaries + layer) for one mechanism.
    mech_dir = ROTATION_ROOT / model_dir_slug / "all" / mechanism
    if not mech_dir.is_dir():
        raise FileNotFoundError(f"Missing rotation dir: {mech_dir}")

    rot_files = glob.glob(str(mech_dir / "rotation_matrix_layer*.pt"))
    if not rot_files:
        raise FileNotFoundError(f"No rotation_matrix_layer*.pt in {mech_dir}")
    rot_path = Path(rot_files[0])

    # The layer index is in the filename
    m = re.search(r"layer(\d+)\.pt$", rot_path.name)
    if m is None:
        raise ValueError(f"Cannot parse layer from {rot_path.name}")
    layer = int(m.group(1))

    bnd_path = mech_dir / f"intervention_boundaries_layer{layer}.pt"
    if not bnd_path.exists():
        raise FileNotFoundError(f"Missing boundaries file: {bnd_path}")

    train_iia = None
    tr_json = mech_dir / "training_results.json"
    if tr_json.exists():
        with open(tr_json) as f:
            tr = json.load(f)
        train_iia = tr.get("best_iia")
        if tr.get("layer") is not None and tr["layer"] != layer:
            print(f"  WARNING: layer mismatch for {mechanism}: "
                  f"file={layer} json={tr['layer']}")

    return {
        "mechanism":       mechanism,
        "layer":           layer,
        "rotation_path":   str(rot_path),
        "boundaries_path": str(bnd_path),
        "train_iia":       train_iia,
    }


def load_intervenable_for(model, layer: int,
                          rotation_path: str, boundaries_path: str):
    # Fresh IntervenableModel at `layer` with the saved rotation loaded in.
    config       = simple_boundless_das_config(type(model), layer)
    intervenable = IntervenableModel(config, model)

    # Pin intervention tensors to the device of the layer we hook, not the first model parameter.
    try:
        layer_device = next(
            model.model.layers[layer].parameters()
        ).device
    except (AttributeError, IndexError):
        layer_device = next(model.parameters()).device

    for k in intervenable.interventions:
        intervention = intervenable.interventions[k]
        for p in intervention.parameters():
            p.data = p.data.to(layer_device)
        for b in intervention.buffers():
            b.data = b.data.to(layer_device)

    rot_ckpt = torch.load(rotation_path,   map_location=layer_device)
    bnd_ckpt = torch.load(boundaries_path, map_location=layer_device)
    for v in intervenable.interventions.values():
        intervention = v[0] if isinstance(v, list) else v
        intervention.rotate_layer.load_state_dict(rot_ckpt["rotate_layer"])
        intervention.intervention_boundaries.data = (
            bnd_ckpt["intervention_boundaries"].to(layer_device)
        )
        break

    intervenable.disable_model_gradients()
    intervenable.model.eval()
    return intervenable


def build_indexed(loader, isolated_mechanism):
    """
    Tokenize one dataset for its isolated mechanism, keyed by uid.

    Source, intervention position and the expected label all come from the
    isolated mechanism -- that's the single source every rotation is scored
    against for this dataset. Falls back to positional index if uid is absent.
    """
    src, base, labels, pos, meta = loader.prepare_intervention_data(isolated_mechanism)
    indexed = {}
    for i, m in enumerate(meta):
        key = m.get("uid", i)
        indexed[key] = {"base": base[i], "source": src[i], "pos": pos[i], "meta": m}
    return indexed


@torch.no_grad()
def base_prediction(model, base_ids, pronoun_tokens, pronoun_case,
                    tokenizer, device):
    # Forced-choice prediction on the base prompt with no intervention.
    base_ids = base_ids.unsqueeze(0).to(device)
    out      = model(input_ids=base_ids)
    non_pad  = base_ids[0] != tokenizer.pad_token_id
    resp_pos = non_pad.nonzero()[-1].item()
    logits   = out.logits[0, resp_pos]
    lp       = torch.log_softmax(logits, dim=0)
    scores   = {p: lp[pronoun_tokens[pronoun_case][p]].item()
                for p in ("he", "she", "they")}
    return max(scores, key=scores.get)


@torch.no_grad()
def score_cell(intervenable, model, indexed, rotation_mech,
               base_pred_cache, tokenizer, device, sample_uids=None):
    """
    Score one (rotation, dataset) cell.

    indexed       : the dataset's uid-indexed data (source = isolated mechanism).
    rotation_mech : which mechanism's rotation is loaded, used only for the
                    hit_rot leakage signal.
    base_pred_cache : {uid: base prediction}, caller-scoped per dataset.
    """
    uids = list(indexed.keys())
    if sample_uids is not None:
        uids = [u for u in uids if u in sample_uids]

    counts = {"hit_source": 0, "hit_rot": 0, "no_flip": 0, "other": 0}
    total  = 0
    pron_t = get_pronoun_token_ids(tokenizer)

    for u in tqdm(uids, desc=f"  rot={rotation_mech}", leave=False):
        d    = indexed[u]
        case = d["meta"]["pronoun_case"]

        if u not in base_pred_cache:
            base_pred_cache[u] = base_prediction(
                model, d["base"], pron_t, case, tokenizer, device
            )
        base_pred = base_pred_cache[u]

        scores = compute_log_likelihood_for_candidates(
            intervenable,
            d["base"].unsqueeze(0).to(device),
            d["source"].unsqueeze(0).to(device),
            d["pos"], pron_t, case, tokenizer, device,
        )
        predicted = max(scores, key=scores.get)

        # isolated mechanism's counterfactual label for this row
        src_expected = d["meta"]["intervention_expected"]
        # what the rotation's OWN mechanism would predict, if recorded
        rot_expected = d["meta"].get(f"intervention_expected_{rotation_mech}")

        total += 1
        matched = False
        if predicted == src_expected:
            counts["hit_source"] += 1
            matched = True
        if rot_expected is not None and predicted == rot_expected:
            counts["hit_rot"] += 1
            matched = True
        if predicted == base_pred:
            counts["no_flip"] += 1
            matched = True
        if not matched:
            counts["other"] += 1

    props = ({k: round(v / total, 4) for k, v in counts.items()}
             if total else {k: 0.0 for k in counts})
    return {"n": total, "counts": counts, "proportions": props}


def main():
    ap = argparse.ArgumentParser(
        description="Rotation x dataset DAS specificity matrix")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n_sample", type=int, default=1200,
                    help="Random uids scored per dataset (0 = all)")
    args = ap.parse_args()

    model_name = args.model
    slug       = model_slug(model_name)
    ds_slug    = dataset_slug(model_name)

    print("=" * 80)
    print(f"TRANSFER EVAL (rotation x dataset)  |  model={model_name}")
    print(f"  n_sample : {args.n_sample if args.n_sample else 'ALL'}")
    print("=" * 80)

    # Locate the three `all`-trained rotations.
    rotations = {}
    for mech in MECHANISMS:
        rotations[mech] = find_rotation(slug, mech)
        r = rotations[mech]
        print(f"  rotation {mech}: layer={r['layer']:>2}  "
              f"train_iia={r['train_iia']}  {Path(r['rotation_path']).name}")

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Index each dataset
    print("\nIndexing datasets...")
    indexed_by_ds   = {}
    sample_by_ds    = {}
    available_ds    = []
    for ds in DATASETS:
        iso  = DATASET_ISOLATES[ds]
        path = DATA_DIR / DATASET_FILENAME[ds].format(slug=ds_slug)
        if not path.exists():
            print(f"  WARNING: {path} missing -- skipping dataset '{ds}'")
            continue
        loader  = CSVDatasetLoader(path, tokenizer, model_name=model_name)
        indexed = build_indexed(loader, iso)
        if not indexed:
            print(f"  WARNING: dataset '{ds}' isolates {iso} but no rows "
                  f"resolved (missing source/label column?) -- skipping")
            continue
        rng = random.Random(42)
        if args.n_sample and args.n_sample < len(indexed):
            uids = set(rng.sample(sorted(indexed.keys()), args.n_sample))
        else:
            uids = set(indexed.keys())
        indexed_by_ds[ds] = indexed
        sample_by_ds[ds]  = uids
        available_ds.append(ds)
        print(f"  {ds} (isolates {iso}): {len(indexed):,} rows, "
              f"scoring {len(uids)}")

    if not available_ds:
        raise RuntimeError("No datasets available -- nothing to evaluate.")

    # matrix[rotation,dataset] 
    matrix = {m: {} for m in MECHANISMS}

    for rot in MECHANISMS:
        r = rotations[rot]
        print(f"\n{'#' * 80}\nROTATION = {rot}  (layer {r['layer']})\n{'#' * 80}")
        intervenable = load_intervenable_for(
            model, r["layer"], r["rotation_path"], r["boundaries_path"])

        for ds in available_ds:
            # Base-prediction cache is per dataset (base prompts differ by ds).
            base_pred_cache = matrix.setdefault("_bpc", {}).setdefault(ds, {})
            cell = score_cell(
                intervenable, model,
                indexed_by_ds[ds], rotation_mech=rot,
                base_pred_cache=base_pred_cache,
                tokenizer=tokenizer, device=device,
                sample_uids=sample_by_ds[ds],
            )
            matrix[rot][ds] = cell
            iso = DATASET_ISOLATES[ds]
            tag = "  <-- diagonal" if iso == rot else ""
            print(f"  {ds} (isolates {iso}): n={cell['n']:>5}  "
                  f"IIA={cell['proportions']['hit_source']:.4f}  "
                  f"no_flip={cell['proportions']['no_flip']:.4f}  "
                  f"hit_rot={cell['proportions']['hit_rot']:.4f}{tag}")

        # Diagonal sanity check: rotation M on the dataset isolating M should roughly reproduce its training IIA (which it doesn't).
        diag_ds = next((d for d in available_ds
                        if DATASET_ISOLATES[d] == rot), None)
        if diag_ds is not None and r["train_iia"] is not None:
            diag  = matrix[rot][diag_ds]["proportions"]["hit_source"]
            delta = abs(diag - r["train_iia"])
            flag  = "  WARNING: >0.05 from train_iia" if delta > 0.05 else "  OK"
            print(f"  [check] diagonal IIA={diag:.4f} ({diag_ds}) vs "
                  f"train_iia={r['train_iia']:.4f} (delta={delta:.4f}){flag}")

        del intervenable
        torch.cuda.empty_cache()

    matrix.pop("_bpc", None)

    out_dir = SCRIPT_DIR / "results" / "transfer" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model":            model_name,
        "n_sample":         args.n_sample,
        "dataset_isolates": {d: DATASET_ISOLATES[d] for d in available_ds},
        "rotations":        rotations,
        "matrix":           {m: matrix[m] for m in MECHANISMS},
    }
    json_path = out_dir / "transfer_matrix.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {json_path}")

    tex_path = out_dir / "transfer_matrix.tex"
    write_latex(matrix, rotations, available_ds, model_name, tex_path)
    print(f"Saved {tex_path}")

    # Console summary of the IIA matrix.
    print(f"\n{'=' * 80}")
    print("IIA matrix  -- rows=rotation, cols=dataset (isolated mechanism)")
    print("=" * 80)
    header = "         " + "".join(
        f"{ds + '/' + DATASET_ISOLATES[ds]:>14}" for ds in available_ds)
    print(header)
    for rot in MECHANISMS:
        row = f"  rot {rot:<4}" + "".join(
            f"{matrix[rot][ds]['proportions']['hit_source']:>14.4f}"
            for ds in available_ds)
        print(row)
    print("=" * 80)
    print("Diagonal = rotation evaluated on the dataset isolating its own")
    print("mechanism (expect high). Off-diagonal expect ~no_flip if specific.")


if __name__ == "__main__":
    main()