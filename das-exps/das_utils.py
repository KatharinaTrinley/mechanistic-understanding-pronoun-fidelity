"""
Shared helpers for the training scripts: the dataset
loader, stereotype lookup, pronoun handling, IIA evaluation and the training
loops.
"""

import re
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

VALID_MECHANISMS = ("G", "R", "S")

# Per-model stereotype maps: for each occupation, the pronoun the model predicts most often when probed context-free on RUFF. These define mechanism S (the
# stereotype target) and feed IIA_stereotype. Measured empirically, not assumed:
OCCUPATION_STEREOTYPES_BY_MODEL = {
    "llama-3.1-8b": {
        "accountant": "he",    "administrator": "they", "advisor": "he",
        "appraiser": "he",     "architect": "he",       "auditor": "he",
        "baker": "he",         "bartender": "he",       "broker": "they",
        "carpenter": "he",     "cashier": "she",        "chef": "he",
        "chemist": "he",       "clerk": "she",          "counselor": "she",
        "dietitian": "she",    "dispatcher": "she",     "doctor": "he",
        "educator": "she",     "electrician": "he",     "engineer": "he",
        "examiner": "he",      "firefighter": "he",     "hairdresser": "she",
        "hygienist": "she",    "inspector": "he",       "instructor": "she",
        "investigator": "he",  "janitor": "he",         "lawyer": "he",
        "librarian": "she",    "machinist": "he",       "manager": "they",
        "mechanic": "he",      "nurse": "she",          "nutritionist": "she",
        "officer": "he",       "painter": "he",         "paralegal": "she",
        "paramedic": "they",   "pathologist": "he",     "pharmacist": "he",
        "physician": "he",     "planner": "she",        "plumber": "he",
        "practitioner": "they","programmer": "he",      "psychologist": "she",
        "receptionist": "she", "salesperson": "they",   "scientist": "he",
        "secretary": "she",    "specialist": "he",      "supervisor": "he",
        "surgeon": "he",       "teacher": "she",        "technician": "he",
        "therapist": "she",    "veterinarian": "she",   "worker": "they",
    },
    "olmo-2-1b": {
        occ: "he" for occ in [
            "accountant", "administrator", "advisor", "appraiser", "architect", "auditor",
            "baker", "bartender", "broker", "carpenter", "cashier", "chef", "chemist",
            "clerk", "counselor", "dietitian", "dispatcher", "doctor", "educator",
            "electrician", "engineer", "examiner", "firefighter", "hairdresser",
            "hygienist", "inspector", "instructor", "investigator", "janitor", "lawyer",
            "librarian", "machinist", "manager", "mechanic", "nurse", "nutritionist",
            "officer", "painter", "paralegal", "paramedic", "pathologist", "pharmacist",
            "physician", "planner", "plumber", "practitioner", "programmer", "psychologist",
            "receptionist", "salesperson", "scientist", "secretary", "specialist",
            "supervisor", "surgeon", "teacher", "technician", "therapist", "veterinarian",
            "worker",
        ]
    },
    "olmo-2-7b": {
        "accountant": "she",   "administrator": "she",  "advisor": "she",
        "appraiser": "she",    "architect": "she",      "auditor": "she",
        "baker": "she",        "bartender": "she",      "broker": "she",
        "carpenter": "he",     "cashier": "she",        "chef": "she",
        "chemist": "she",      "clerk": "she",          "counselor": "she",
        "dietitian": "she",    "dispatcher": "she",     "doctor": "she",
        "educator": "she",     "electrician": "he",     "engineer": "she",
        "examiner": "she",     "firefighter": "she",    "hairdresser": "she",
        "hygienist": "she",    "inspector": "she",      "instructor": "she",
        "investigator": "she", "janitor": "she",        "lawyer": "she",
        "librarian": "she",    "machinist": "she",      "manager": "she",
        "mechanic": "he",      "nurse": "she",          "nutritionist": "she",
        "officer": "she",      "painter": "she",        "paralegal": "she",
        "paramedic": "he",     "pathologist": "she",    "pharmacist": "she",
        "physician": "she",    "planner": "she",        "plumber": "he",
        "practitioner": "she", "programmer": "she",     "psychologist": "she",
        "receptionist": "she", "salesperson": "she",    "scientist": "she",
        "secretary": "she",    "specialist": "she",     "supervisor": "she",
        "surgeon": "she",      "teacher": "she",        "technician": "she",
        "therapist": "she",    "veterinarian": "she",   "worker": "she",
    },
    "olmo-2-13b": {
        "accountant": "he",    "administrator": "he",   "advisor": "he",
        "appraiser": "he",     "architect": "he",       "auditor": "he",
        "baker": "he",         "bartender": "he",       "broker": "he",
        "carpenter": "he",     "cashier": "they",       "chef": "he",
        "chemist": "he",       "clerk": "he",           "counselor": "they",
        "dietitian": "she",    "dispatcher": "he",      "doctor": "he",
        "educator": "they",    "electrician": "he",     "engineer": "he",
        "examiner": "they",    "firefighter": "he",     "hairdresser": "they",
        "hygienist": "she",    "inspector": "he",       "instructor": "he",
        "investigator": "he",  "janitor": "he",         "lawyer": "he",
        "librarian": "they",   "machinist": "he",       "manager": "he",
        "mechanic": "he",      "nurse": "she",          "nutritionist": "they",
        "officer": "he",       "painter": "he",         "paralegal": "they",
        "paramedic": "he",     "pathologist": "he",     "pharmacist": "he",
        "physician": "he",     "planner": "they",       "plumber": "he",
        "practitioner": "they","programmer": "he",      "psychologist": "he",
        "receptionist": "she", "salesperson": "they",   "scientist": "he",
        "secretary": "he",     "specialist": "he",      "supervisor": "he",
        "surgeon": "he",       "teacher": "they",       "technician": "he",
        "therapist": "they",   "veterinarian": "he",    "worker": "he",
    },
    "qwen2.5-7b": {
        "accountant": "he",    "administrator": "he",   "advisor": "he",
        "appraiser": "he",     "architect": "he",       "auditor": "he",
        "baker": "he",         "bartender": "he",       "broker": "he",
        "carpenter": "he",     "cashier": "he",         "chef": "he",
        "chemist": "he",       "clerk": "he",           "counselor": "he",
        "dietitian": "he",     "dispatcher": "he",      "doctor": "he",
        "educator": "he",      "electrician": "he",     "engineer": "he",
        "examiner": "he",      "firefighter": "he",     "hairdresser": "he",
        "hygienist": "he",     "inspector": "he",       "instructor": "he",
        "investigator": "he",  "janitor": "he",         "lawyer": "he",
        "librarian": "he",     "machinist": "he",       "manager": "he",
        "mechanic": "he",      "nurse": "she",          "nutritionist": "he",
        "officer": "he",       "painter": "he",         "paralegal": "he",
        "paramedic": "he",     "pathologist": "he",     "pharmacist": "he",
        "physician": "he",     "planner": "he",         "plumber": "he",
        "practitioner": "he",  "programmer": "he",      "psychologist": "he",
        "receptionist": "he",  "salesperson": "he",     "scientist": "he",
        "secretary": "he",     "specialist": "he",      "supervisor": "he",
        "surgeon": "he",       "teacher": "he",         "technician": "he",
        "therapist": "he",     "veterinarian": "he",    "worker": "he",
    },
    "gemma-2-9b": {
        "accountant": "he",    "administrator": "he",   "advisor": "he",
        "appraiser": "he",     "architect": "he",       "auditor": "he",
        "baker": "he",         "bartender": "he",       "broker": "he",
        "carpenter": "he",     "cashier": "he",         "chef": "he",
        "chemist": "he",       "clerk": "he",           "counselor": "he",
        "dietitian": "she",    "dispatcher": "he",      "doctor": "he",
        "educator": "he",      "electrician": "he",     "engineer": "he",
        "examiner": "he",      "firefighter": "he",     "hairdresser": "she",
        "hygienist": "she",    "inspector": "he",       "instructor": "he",
        "investigator": "he",  "janitor": "he",         "lawyer": "he",
        "librarian": "she",    "machinist": "he",       "manager": "he",
        "mechanic": "he",      "nurse": "she",          "nutritionist": "she",
        "officer": "he",       "painter": "he",         "paralegal": "she",
        "paramedic": "he",     "pathologist": "he",     "pharmacist": "he",
        "physician": "he",     "planner": "he",         "plumber": "he",
        "practitioner": "he",  "programmer": "he",      "psychologist": "he",
        "receptionist": "she", "salesperson": "he",     "scientist": "he",
        "secretary": "she",    "specialist": "he",      "supervisor": "he",
        "surgeon": "he",       "teacher": "she",        "technician": "he",
        "therapist": "she",    "veterinarian": "he",    "worker": "he",
    },
}

# model_slug() produces names like 'llama-3.1-8b-instruct'; map those to the keys used above.
_MODEL_SLUG_ALIASES = {
    "llama-3.1-8b-instruct":  "llama-3.1-8b",
    "olmo-2-1b-instruct":     "olmo-2-1b",
    "olmo-2-7b-instruct":     "olmo-2-7b",
    "olmo-2-13b-instruct":    "olmo-2-13b",
    "qwen2.5-7b-instruct":    "qwen2.5-7b",
    "gemma-2-9b-it":          "gemma-2-9b",
}


OCCUPATION_STEREOTYPES = OCCUPATION_STEREOTYPES_BY_MODEL["llama-3.1-8b"]


def get_stereotype_pronoun(occupation: str, model_name: str = "llama-3.1-8b") -> str:
    """
    Stereotypical pronoun for an occupation under the given model. 
    """
    key = model_name.lower().split("/")[-1]
    key = _MODEL_SLUG_ALIASES.get(key, key)
    stereotype_map = OCCUPATION_STEREOTYPES_BY_MODEL.get(key)
    return stereotype_map.get(occupation.lower().strip(), "they")


def dataset_slug(model_name: str) -> str:
    slug = model_name.split("/")[-1].lower()
    for suffix in ["-instruct", "-it"]:
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
    slug = re.sub(r"(olmo-\d+)-\d{4}-(\d+b)", r"\1-\2", slug)
    return slug


def normalize_pronoun(pronoun):
    """Collapse any case form (him/his, her/hers, them/their...) to he/she/they."""
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
    """Token IDs for he/she/they in each of the three grammatical cases."""
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


def detect_pronoun_case_from_prompt(prompt):
    """
    Work out which case the blank expects so IIA scores the right token set.
    Checks template placeholders first, then surface cues, defaulting to nominative.
    """
    if "$NOM_PRONOUN"  in prompt or "NOM_PRONOUN"  in prompt: return "nominative"
    if "$ACC_PRONOUN"  in prompt or "ACC_PRONOUN"  in prompt: return "accusative"
    if "$POSS_PRONOUN" in prompt or "POSS_PRONOUN" in prompt: return "possessive"
    pl = prompt.lower()
    if any(w in pl for w in ["his", "her", "their", "its", "'s"]): return "possessive"
    if any(w in pl for w in ["saw", "told", "gave", "asked", "showed", "sent"]): return "accusative"
    return "nominative"


def _resolve_source_for_row(row, mechanism):
    """
    The source prompt and the expected post-intervention label for one row.
    """
    if mechanism == "G":
        prompt_col   = "source_prompt_G"
        sentence_col = "source_sentence_G"
        expected_col = "intervention_expected_G"
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


def verify_pronoun_token_ids(tokenizer):
    # Print how each pronoun form tokenizes; flags leading spaces and multi-token splits, both would throw off the log-prob scoring.
    print("\n" + "=" * 60)
    print("PRONOUN TOKEN ID VERIFICATION")
    print("=" * 60)
    forms = ["he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"]
    for form in forms:
        ids     = tokenizer.encode(form, add_special_tokens=False)
        decoded = tokenizer.decode([ids[0]])
        flags   = ""
        if decoded.startswith(" "):  flags += "  WARNING: LEADING SPACE"
        if len(ids) > 1:             flags += f"  WARNING: MULTI-TOKEN ({len(ids)})"
        print(f"  {form!r:10s} -> id={ids[0]:6d}  decoded={decoded!r}{flags}")
    print("=" * 60)


def check_label_distribution(df, variant_name):

    counts = df["base_pronoun"].value_counts()
    total  = len(df)
    print(f"\nLabel distribution for '{variant_name}' (n={total:,}):")
    for cls, cnt in counts.items():
        pct  = cnt / total * 100
        flag = "  WARNING: BELOW 20%" if pct < 20.0 else ""
        print(f"  {cls}: {cnt:,} ({pct:.1f}%){flag}")


class CSVDatasetLoader:
    """
    Reads diagnostic_pairs_all_fixed.csv and turns it into tokenized,
    base/source-paired DAS training data for one mechanism.
    """

    def __init__(self, csv_path, tokenizer, max_length=512, model_name="llama-3.1-8b"):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.model_name = model_name
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Diagnostic CSV not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df):,} rows from {csv_path.name}")

    def _find_intervention_position(self, base_ids, source_ids):
        # The intervention happens at the first token where base and source differ (the swapped pronoun).
        diff_positions = (base_ids != source_ids).nonzero(as_tuple=False).squeeze(-1)
        non_pad_mask   = base_ids != self.tokenizer.pad_token_id
        valid_diff     = diff_positions[non_pad_mask[diff_positions]]
        if len(valid_diff) == 0:
            return (base_ids != self.tokenizer.pad_token_id).nonzero()[-1].item()
        return valid_diff[0].item()

    def prepare_intervention_data(self, mechanism, num_samples=None):
        if mechanism not in VALID_MECHANISMS:
            raise ValueError(f"mechanism must be one of {VALID_MECHANISMS}")

        subset = (
            self.df.sample(n=num_samples, random_state=42)
            if num_samples and num_samples < len(self.df)
            else self.df.copy()
        )

        base_input_ids, source_input_ids = [], []
        labels, intervention_positions, metadata = [], [], []
        skipped = 0

        for _, row in tqdm(subset.iterrows(), total=len(subset),
                           desc=f"Tokenizing (DAS-{mechanism})"):
            resolved = _resolve_source_for_row(row, mechanism)
            if resolved is None:
                skipped += 1
                continue

            base_prompt  = row["base_prompt"]
            base_pronoun = normalize_pronoun(row["base_pronoun"])
            occupation   = row.get("occupation", "").lower().strip()
            stereotype_pronoun = get_stereotype_pronoun(occupation, self.model_name)

            base_tokens = self.tokenizer(
                base_prompt, max_length=self.max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            base_ids     = base_tokens["input_ids"][0]
            # Response position=last non-pad token.
            response_pos = (base_ids != self.tokenizer.pad_token_id).nonzero()[-1].item()

            source_tokens = self.tokenizer(
                resolved["source_prompt"], max_length=self.max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            source_ids       = source_tokens["input_ids"][0]
            intervention_pos = self._find_intervention_position(base_ids, source_ids)

            # The label is the counterfactual pronoun, placed only at the response position; everything else is -100 so the loss ignores it.
            intervention_token = self.tokenizer.encode(
                resolved["intervention_expected"], add_special_tokens=False
            )[0]
            label = torch.full_like(base_ids, -100)
            label[response_pos] = intervention_token

            meta = {
                "mechanism":             mechanism,
                "base_pronoun":          base_pronoun,
                "source_pronoun":        resolved["source_pronoun"],
                "confuse_pronoun":       normalize_pronoun(row["confuse_pronoun"]),
                "stereotype_pronoun":    stereotype_pronoun,
                "intervention_expected": resolved["intervention_expected"],
                "occupation":            occupation,
                "base_sentence":         row.get("base_sentence", ""),
                "source_sentence":       resolved["source_sentence"],
                "base_prompt":           base_prompt,
                "source_prompt":         resolved["source_prompt"],
                "pronoun_case":          detect_pronoun_case_from_prompt(base_prompt),
            }

            base_input_ids.append(base_ids.clone())
            source_input_ids.append(source_ids)
            labels.append(label)
            intervention_positions.append(intervention_pos)
            metadata.append(meta)

        print(f"  Prepared {len(metadata):,} DAS-{mechanism} examples (skipped {skipped})")
        return source_input_ids, base_input_ids, labels, intervention_positions, metadata

    def create_dataloaders(self, mechanism, batch_size, train_split=0.8):
        all_data   = self.prepare_intervention_data(mechanism)
        actual     = len(all_data[0])
        train_size = int(actual * train_split)
        print(f"  Split: train={train_size:,}, eval={actual - train_size:,}")

        # all_data is (source, base, labels, positions, metadata). input_ids is the base run and source_input_ids is the source.
        train_data = {
            "input_ids":        all_data[1][:train_size],
            "source_input_ids": all_data[0][:train_size],
            "labels":           all_data[2][:train_size],
            "intervention_ids": all_data[3][:train_size],
        }
        eval_data = {
            "input_ids":        all_data[1][train_size:],
            "source_input_ids": all_data[0][train_size:],
            "labels":           all_data[2][train_size:],
            "intervention_ids": all_data[3][train_size:],
        }
        self.train_metadata = all_data[4][:train_size]
        self.eval_metadata  = all_data[4][train_size:]

        train_loader = DataLoader(
            Dataset.from_dict(train_data).with_format("torch"),
            batch_size=batch_size, shuffle=True,
        )
        eval_loader = DataLoader(
            Dataset.from_dict(eval_data).with_format("torch"),
            batch_size=batch_size,
        )
        return train_loader, eval_loader


def save_rotation_matrix(intervenable, save_dir, layer):
    # Save the learned rotation and the soft boundary params for this layer. 
    save_dir = Path(save_dir)
    for v in intervenable.interventions.values():
        intervention = v[0] if isinstance(v, list) else v
        torch.save(
            {"rotate_layer": intervention.rotate_layer.state_dict()},
            save_dir / f"rotation_matrix_layer{layer}.pt",
        )
        torch.save(
            {"intervention_boundaries": intervention.intervention_boundaries.detach().cpu()},
            save_dir / f"intervention_boundaries_layer{layer}.pt",
        )
        print(f"  Saved rotation_matrix_layer{layer}.pt + intervention_boundaries_layer{layer}.pt")
        break


def load_rotation_matrix(intervenable, load_path):
    checkpoint = torch.load(load_path)
    for v in intervenable.interventions.values():
        intervention = v[0] if isinstance(v, list) else v
        intervention.rotate_layer.load_state_dict(checkpoint["rotate_layer"])
        intervention.intervention_boundaries.data = checkpoint["intervention_boundaries"]
        break
    return intervenable


def compute_log_likelihood_for_candidates(intervenable, base_input, source_input,
                                          intervention_pos, pronoun_tokens, pronoun_case,
                                          tokenizer, device):
    # Intervened forward pass run; and read the log-probs of he/she/they at the response position (the last non-pad token).
    _, counterfactual_outputs = intervenable(
        {"input_ids": base_input},
        [{"input_ids": source_input}],
        {"sources->base": intervention_pos},
    )
    non_pad_mask  = base_input[0] != tokenizer.pad_token_id
    response_pos  = non_pad_mask.nonzero()[-1].item()
    logits_at_pos = counterfactual_outputs.logits[0, response_pos]
    log_probs     = F.log_softmax(logits_at_pos, dim=0)
    return {p: log_probs[pronoun_tokens[pronoun_case][p]].item()
            for p in ["he", "she", "they"]}


def compute_iia_forced_choice(intervenable, eval_loader, metadata, tokenizer,
                               device="cuda", max_examples=None):
    # IIA = fraction of examples where the intervened prediction matches the counterfactual label. The prediction is the argmax over just he/she/they (forced choice), since other tokens like "the" can otherwise win.
    # attr_counts tracks what the model picked instead: the distractor/group pronoun (POS_G_R), the stereotype (S), or neither (none).
    intervenable.model.eval()
    total_count = correct_count = 0
    attr_counts    = {"POS_G_R": 0, "S": 0, "none": 0}
    pronoun_tokens = get_pronoun_token_ids(tokenizer)

    with torch.no_grad():
        for batch_idx, inputs in enumerate(eval_loader):
            if max_examples and total_count >= max_examples:
                break
            for k, v in inputs.items():
                if k != "intervention_ids" and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            intervention_positions = inputs["intervention_ids"]
            if isinstance(intervention_positions, torch.Tensor):
                intervention_positions = intervention_positions.tolist()

            for i in range(len(inputs["input_ids"])):
                if max_examples and total_count >= max_examples:
                    break
                single_pos = intervention_positions[i]
                if isinstance(single_pos, torch.Tensor):
                    single_pos = single_pos.item()
                meta_idx = batch_idx * eval_loader.batch_size + i
                if meta_idx >= len(metadata):
                    break
                meta   = metadata[meta_idx]
                scores = compute_log_likelihood_for_candidates(
                    intervenable,
                    inputs["input_ids"][i:i+1],
                    inputs["source_input_ids"][i:i+1],
                    single_pos, pronoun_tokens, meta["pronoun_case"], tokenizer, device,
                )
                predicted = max(scores, key=scores.get)
                total_count += 1
                if predicted == meta["intervention_expected"]:
                    correct_count += 1
                if predicted == meta["confuse_pronoun"]:
                    attr_counts["POS_G_R"] += 1
                elif predicted == meta["stereotype_pronoun"]:
                    attr_counts["S"] += 1
                else:
                    attr_counts["none"] += 1

    iia       = round(correct_count / total_count, 4) if total_count else 0.0
    attr_props = (
        {k: round(v / total_count, 4) for k, v in attr_counts.items()}
        if total_count else {k: 0.0 for k in attr_counts}
    )
    return {
        "iia_overall":             iia,
        "total_count":             total_count,
        "correct_count":           correct_count,
        "attribution_counts":      attr_counts,
        "attribution_proportions": attr_props,
    }


# any pyvene doesn't know natively, we pass as "llama", which has a compatible residual-stream layout for block_output interventions.
_PYVENE_NATIVE_TYPES = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "GPT2LMHeadModel",
    "GPTNeoXForCausalLM",
    "BloomForCausalLM",
    "OPTForCausalLM",
}

def simple_boundless_das_config(model_type, layer):
    from pyvene import (
        BoundlessRotatedSpaceIntervention, IntervenableConfig, RepresentationConfig,
    )
    type_name     = model_type.__name__ if hasattr(model_type, "__name__") else str(model_type)
    resolved_type = model_type if type_name in _PYVENE_NATIVE_TYPES else "llama"
    return IntervenableConfig(
        model_type=resolved_type,
        representations=[RepresentationConfig(layer, "block_output")],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )


def calculate_loss(logits, labels, intervenable):
    # Cross-entropy against the counterfactual label, plus an L1 penalty on the boundary scalars. The penalty pushes the learned subspace to stay small.
    ce_loss = CrossEntropyLoss(ignore_index=-100)(
        logits.reshape(-1, logits.size(-1)), labels.view(-1)
    )
    boundary_reg = 0.0
    for v in intervenable.interventions.values():
        intervention = v[0] if isinstance(v, list) else v
        if hasattr(intervention, "intervention_boundaries"):
            boundary_reg += intervention.intervention_boundaries.sum()
    return ce_loss + 1.0 * boundary_reg


def train_best_layer(model, tokenizer, train_loader, eval_loader, eval_metadata,
                     layer, mechanism, epochs=3, device="cuda",
                     gradient_accumulation_steps=16, save_dir=None,
                     eval_every_steps=200, eval_max_examples=200):
    # Full DAS training at a single chosen layer (used after the layer search picks the best layer). 
    from pyvene import IntervenableModel, set_seed
    from transformers import get_linear_schedule_with_warmup

    config       = simple_boundless_das_config(type(model), layer)
    intervenable = IntervenableModel(config, model)

    # Move the intervention params/buffers onto the model's first device.
    first_device = next(model.parameters()).device
    for k in intervenable.interventions:
        intervention = intervenable.interventions[k]
        for param in intervention.parameters():
            param.data = param.data.to(first_device)
        for buf in intervention.buffers():
            buf.data = buf.data.to(first_device)

    intervenable.disable_model_gradients()

    # Train the rotation and the boundary scalars.
    optimizer_params = []
    for v in intervenable.interventions.values():
        intervention = v[0] if isinstance(v, list) else v
        optimizer_params += [{"params": intervention.rotate_layer.parameters(), "lr": 1e-3}]
        optimizer_params += [{"params": intervention.intervention_boundaries,   "lr": 1e-2}]

    optimizer        = torch.optim.Adam(optimizer_params)
    total_grad_steps = (len(train_loader) * epochs) // gradient_accumulation_steps
    scheduler        = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_grad_steps,
    )
    # Anneal the boundary temperature from 50 to 0.1 so soft masks sharpen to discrete subspace by the end.
    total_fwd_steps = len(train_loader) * epochs
    temp_schedule   = torch.linspace(50.0, 0.1, total_fwd_steps).to(device)

    history   = {"losses": [], "eval_steps": [], "eval_iias": [], "boundary_values": []}
    fwd_step  = 0
    grad_step = 0
    best_iia  = 0.0

    for epoch in range(epochs):
        intervenable.model.train()
        pbar = tqdm(train_loader, desc=f"Layer {layer}, Epoch {epoch}")

        for batch_idx, inputs in enumerate(pbar):
            for k, v in inputs.items():
                if k != "intervention_ids" and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            intervenable.set_temperature(temp_schedule[fwd_step])
            intervention_positions = inputs["intervention_ids"]
            if isinstance(intervention_positions, torch.Tensor):
                intervention_positions = intervention_positions.tolist()

            #each example is run through its own intervention (pos differ per example), then batch the logits for one loss.
            batch_outputs = []
            for i in range(len(inputs["input_ids"])):
                pos = intervention_positions[i]
                if isinstance(pos, torch.Tensor):
                    pos = pos.item()
                _, output = intervenable(
                    {"input_ids": inputs["input_ids"][i:i+1]},
                    [{"input_ids": inputs["source_input_ids"][i:i+1]}],
                    {"sources->base": pos},
                )
                batch_outputs.append(output.logits)

            loss = calculate_loss(
                torch.cat(batch_outputs, dim=0), inputs["labels"], intervenable
            )
            (loss / gradient_accumulation_steps).backward()
            history["losses"].append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "grad_step": grad_step})
            fwd_step += 1

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                grad_step += 1

                if grad_step % eval_every_steps == 0:
                    eval_metrics = compute_iia_forced_choice(
                        intervenable, eval_loader, eval_metadata, tokenizer,
                        device=device, max_examples=eval_max_examples,
                    )
                    iia          = eval_metrics["iia_overall"]
                    boundary_val = None
                    for v in intervenable.interventions.values():
                        intervention = v[0] if isinstance(v, list) else v
                        if hasattr(intervention, "intervention_boundaries"):
                            boundary_val = intervention.intervention_boundaries.item()
                            break
                    history["eval_steps"].append(grad_step)
                    history["eval_iias"].append(iia)
                    history["boundary_values"].append(boundary_val)
                    print(f"\n  [step {grad_step}] IIA={iia:.4f}  "
                          f"boundary={boundary_val:.4f}  "
                          f"Attr={eval_metrics['attribution_proportions']}")
                    if save_dir is not None and iia > best_iia:
                        save_rotation_matrix(intervenable, save_dir, layer)
                    best_iia = max(best_iia, iia)
                    intervenable.model.train()

        # final optimizer step if the last batch didn't land on a gradient-accumulation boundary.
        if len(train_loader) % gradient_accumulation_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            grad_step += 1

    print(f"\n  Training complete. Best IIA: {best_iia:.4f}")
    del intervenable
    torch.cuda.empty_cache()
    return best_iia, history