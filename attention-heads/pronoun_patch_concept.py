"""
adapted from: https://github.com/sfeucht/dual-route-induction/
pronoun_patch_concept.py

Clean vs. corrupt forced-choice accuracy for the OLMo models, no patching.
Writes a patching_summary_n<N>.json per model.

Use:
python pronoun_patch_concept.py --models allenai/OLMo-2-1124-7B-Instruct
"""

import sys, os
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts')))

import json, argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
login(os.environ.get("HF_TOKEN"))  # set HF_TOKEN

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
_CACHE_ROOT = os.path.join(_REPO_ROOT, 'cache')

STANDARD_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers',
    'they', 'them', 'their', 'theirs',
}
PRONOUN_FORMS = {
    'he':   ['he', 'him', 'his'],
    'she':  ['she', 'her', 'hers'],
    'they': ['they', 'them', 'their', 'theirs'],
}
MODELS = [
    'allenai/OLMo-2-0425-1B-Instruct',
    'allenai/OLMo-2-1124-7B-Instruct',
    'allenai/OLMo-2-1124-13B-Instruct',
]


def _tok_single(tokenizer, word, model_name):
    # Single token id for ' word', dropping a leading BOS/pad if the tokenizer
    # prepends one. Returns None if the word isn't a single token.
    ids = tokenizer(' ' + word)['input_ids']
    if len(ids) > 1 and ids[0] in (tokenizer.bos_token_id, tokenizer.pad_token_id):
        ids = ids[1:]
    return ids[0] if len(ids) == 1 else None

def get_candidate_ids(tokenizer, correct_pronoun, confuse_pronoun, model_name):
    # Token ids for the correct and confuse pronouns, plus the full set of
    # candidate ids (all case forms of both) for the forced-choice argmax.
    correct_id = _tok_single(tokenizer, correct_pronoun, model_name)
    confuse_id = _tok_single(tokenizer, confuse_pronoun, model_name)
    if correct_id is None or confuse_id is None:
        return None, None, None
    all_ids = set()
    for base in [correct_pronoun.lower().strip(), confuse_pronoun.lower().strip()]:
        for surface in PRONOUN_FORMS.get(base, [base]):
            tid = _tok_single(tokenizer, surface, model_name)
            if tid is not None:
                all_ids.add(tid)
    all_ids |= {correct_id, confuse_id}
    return correct_id, confuse_id, list(all_ids)

def tokenize_prefix(prompt, tokenizer, model_name):
    # Everything up to and including the ___ blank. Ensure exactly one BOS.
    prefix = prompt.split('___')[0] + '___'
    ids = tokenizer(prefix, return_tensors='pt')['input_ids']
    if tokenizer.bos_token_id is None:
        return ids
    if ids[0, 0].item() != tokenizer.bos_token_id:
        bos = torch.tensor([[tokenizer.bos_token_id]])
        ids = torch.cat([bos, ids], dim=1)
    return ids


@torch.no_grad()
def forced_choice(model, input_ids, correct_id, confuse_id, all_ids, device):
    # Predict the next pronoun by argmax over the candidate ids only.
    logits = model(input_ids.to(device)).logits[0, -1]
    lp     = F.log_softmax(logits, dim=-1)
    predicted = max(all_ids, key=lambda t: lp[t].item())
    return {
        'correct':      int(predicted == correct_id),
        'correct_prob': lp[correct_id].item(),
        'confuse_prob': lp[confuse_id].item(),
        'logit_diff':   (logits[correct_id] - logits[confuse_id]).item(),
    }


def evaluate_model(model_name, df):
    print(f'\n{"="*60}\n{model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto')
    except RuntimeError:
        print('GPU load failed, falling back to CPU.')
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='cpu')
    model.eval()
    device = next(model.parameters()).device

    clean_stats   = {'correct': 0, 'correct_prob': [], 'logit_diff': []}
    corrupt_stats = {'correct': 0, 'correct_prob': [], 'logit_diff': []}
    n, skipped = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=model_name.split('/')[-1]):
        correct_id, confuse_id, all_ids = get_candidate_ids(
            tokenizer, str(row['correct_pronoun']), str(row['confuse_pronoun']), model_name)
        if correct_id is None:
            skipped += 1
            continue

        clean_ids = tokenize_prefix(str(row['p_clean']),   tokenizer, model_name)
        corr_ids  = tokenize_prefix(str(row['p_corrupt']), tokenizer, model_name)

        cl = forced_choice(model, clean_ids,  correct_id, confuse_id, all_ids, device)
        co = forced_choice(model, corr_ids,   correct_id, confuse_id, all_ids, device)

        clean_stats['correct']      += cl['correct']
        clean_stats['correct_prob'].append(cl['correct_prob'])
        clean_stats['logit_diff'].append(cl['logit_diff'])

        corrupt_stats['correct']      += co['correct']
        corrupt_stats['correct_prob'].append(co['correct_prob'])
        corrupt_stats['logit_diff'].append(co['logit_diff'])

        n += 1

    clean_acc   = clean_stats['correct']   / n if n > 0 else float('nan')
    corrupt_acc = corrupt_stats['correct'] / n if n > 0 else float('nan')

    print(f'  n={n}  skipped={skipped}')
    print(f'  Clean acc:          {clean_acc:.3f}')
    print(f'  Corrupt acc:        {corrupt_acc:.3f}')
    print(f'  Clean mean logP:    {np.mean(clean_stats["correct_prob"]):.4f}')
    print(f'  Corrupt mean logP:  {np.mean(corrupt_stats["correct_prob"]):.4f}')
    print(f'  Clean logit diff:   {np.mean(clean_stats["logit_diff"]):.4f}')
    print(f'  Corrupt logit diff: {np.mean(corrupt_stats["logit_diff"]):.4f}')

    summary = {
        'clean_acc':               round(clean_acc,  4),
        'corrupt_acc':             round(corrupt_acc, 4),
        'clean_mean_logprob':      round(float(np.mean(clean_stats['correct_prob'])),   4),
        'corrupt_mean_logprob':    round(float(np.mean(corrupt_stats['correct_prob'])), 4),
        'clean_mean_logit_diff':   round(float(np.mean(clean_stats['logit_diff'])),     4),
        'corrupt_mean_logit_diff': round(float(np.mean(corrupt_stats['logit_diff'])),   4),
        'n':       n,
        'skipped': skipped,
    }

    out_dir = os.path.join(_CACHE_ROOT, 'pronoun_patching', model_name.split('/')[-1])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'patching_summary_n{n}.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Saved {out_path}')

    del model
    torch.cuda.empty_cache()
    return summary


def main(args):
    torch.manual_seed(8)
    np.random.seed(8)

    df = pd.read_csv(args.data_path, sep='\t')
    mask = (df['correct_pronoun'].isin(STANDARD_PRONOUNS) &
            df['confuse_pronoun'].isin(STANDARD_PRONOUNS))
    df = df[mask].reset_index(drop=True)
    df = df.iloc[args.skip_n:]
    if args.max_n is not None:
        df = df.iloc[:args.max_n]
    print(f'Examples: {len(df)}')

    models = args.models if args.models else MODELS
    all_results = {}
    for model_name in models:
        all_results[model_name] = evaluate_model(model_name, df)

    print('\n\nSummary:')
    print(f'{"Model":<35} {"Clean acc":>10} {"Corrupt acc":>12} {"Delta acc":>10}')
    print('-' * 70)
    for m, r in all_results.items():
        delta = round(r['clean_acc'] - r['corrupt_acc'], 4)
        print(f'{m.split("/")[-1]:<35} {r["clean_acc"]:>10.3f} {r["corrupt_acc"]:>12.3f} {delta:>10.3f}')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        default=os.path.join(_REPO_ROOT, 'data',
                                             'eo_ep_task_induction_pairs.tsv'))
    parser.add_argument('--max_n',  type=int, default=50)
    parser.add_argument('--skip_n', type=int, default=50,
                        help='Skip first N rows (default 50 = training set)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Subset of models to run (default: all OLMo)')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)