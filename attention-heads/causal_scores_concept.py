"""
adapted from: https://github.com/sfeucht/dual-route-induction/
causal_scores_concept.py

Activation patching (clean->corrupt) for per-head causal scores. The primary
metric is the accuracy delta; log-prob and logit-diff are tracked alongside.
Head count is read from the o_proj input width, so GQA models (Gemma-2 etc.)
work without special-casing.

cache/:
  causal_scores/<model>/causal_scores_n<N>.json
  head_orderings/<model>/pronoun_copying_n<N>.json
  causal_scores/<model>/n<N>.pkl
  pronoun_patching/<model>/patching_summary_n<N>.json

Use:
python causal_scores.py --model google/gemma-2-9b-it --max_n 50
"""

import sys, os
sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts')))

import json, pickle, argparse
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


def _tok_single(tokenizer, word):
    # Single token id for ' word', dropping a leading BOS/pad if present.
    ids = tokenizer(' ' + word)['input_ids']
    if len(ids) > 1 and ids[0] in (tokenizer.bos_token_id, tokenizer.pad_token_id):
        ids = ids[1:]
    return ids[0] if len(ids) == 1 else None


def get_candidate_ids(tokenizer, correct_pronoun, confuse_pronoun):
    correct_id = _tok_single(tokenizer, correct_pronoun)
    confuse_id = _tok_single(tokenizer, confuse_pronoun)
    if correct_id is None or confuse_id is None:
        return None, None, None
    all_ids = set()
    for base in [correct_pronoun.lower().strip(), confuse_pronoun.lower().strip()]:
        for surface in PRONOUN_FORMS.get(base, [base]):
            tid = _tok_single(tokenizer, surface)
            if tid is not None:
                all_ids.add(tid)
    all_ids |= {correct_id, confuse_id}
    return correct_id, confuse_id, list(all_ids)


def tokenize_prefix(prompt, tokenizer):
    prefix = prompt.split('___')[0] + '___'
    ids = tokenizer(prefix, return_tensors='pt')['input_ids']
    if tokenizer.bos_token_id is None:
        return ids
    if ids[0, 0].item() != tokenizer.bos_token_id:
        bos = torch.tensor([[tokenizer.bos_token_id]])
        ids = torch.cat([bos, ids], dim=1)
    return ids


def pronoun_stats(logits_1d, correct_id, confuse_id, all_ids):
    # (is_correct, logP_correct, logP_confuse, logit_correct, logit_confuse).
    lp = F.log_softmax(logits_1d, dim=-1)
    predicted = max(all_ids, key=lambda t: lp[t].item())
    return (
        float(predicted == correct_id),
        lp[correct_id].item(),
        lp[confuse_id].item(),
        logits_1d[correct_id].item(),
        logits_1d[confuse_id].item(),
    )


class HeadSaver:
    # Running sums over examples for one set of heads; divide by n at the end.
    def __init__(self, name, n_heads):
        self.name, self.n_heads, self.n = name, n_heads, 0
        self.correct       = torch.zeros(n_heads)
        self.correct_prob  = torch.zeros(n_heads)
        self.confuse_prob  = torch.zeros(n_heads)
        self.correct_logit = torch.zeros(n_heads)
        self.confuse_logit = torch.zeros(n_heads)

    def update(self, correct, correct_prob, confuse_prob, correct_logit, confuse_logit):
        self.n             += len(correct)
        self.correct       += correct.sum(dim=0)
        self.correct_prob  += correct_prob.sum(dim=0)
        self.confuse_prob  += confuse_prob.sum(dim=0)
        self.correct_logit += correct_logit.sum(dim=0)
        self.confuse_logit += confuse_logit.sum(dim=0)

    def acc(self):        return self.correct       / self.n
    def mean_prob(self):  return self.correct_prob  / self.n
    def logit_diff(self): return (self.correct_logit - self.confuse_logit) / self.n


def get_o_proj(model, layer):
    # The attention output projection, where per-head contributions still
    # live separately in the input (pythia calls it 'dense').
    if hasattr(model, 'gpt_neox'):
        return model.gpt_neox.layers[layer].attention.dense
    return model.model.layers[layer].self_attn.o_proj


def get_head_dim(model):
    cfg = model.config
    if hasattr(cfg, 'head_dim'):
        return cfg.head_dim
    return cfg.hidden_size // cfg.num_attention_heads


@torch.no_grad()
def collect_clean_acts(model, input_ids):
    # Pre-o_proj activations per layer, kept flat as (1, seq, hidden_actual).
    # We reshape by head_dim later, so the real tensor width sets the head count: correct for both MHA and GQA.
    n_layers = model.config.num_hidden_layers
    acts     = [None] * n_layers
    hooks    = []

    def make_hook(l):
        def hook(module, inp, out):
            acts[l] = inp[0].detach().cpu()
        return hook

    for l in range(n_layers):
        hooks.append(get_o_proj(model, l).register_forward_hook(make_hook(l)))
    try:
        model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return acts


@torch.no_grad()
def patch_all_heads(model, clean_acts, corr_ids, correct_id, confuse_id, all_ids):
    # For each (layer, head): patch the clean head output at the last token into the corrupt run and record the resulting pronoun stats.
    head_dim = get_head_dim(model)
    n_layers = model.config.num_hidden_layers
    device   = next(model.parameters()).device

    corrects, correct_probs, confuse_probs, correct_logits, confuse_logits = \
        [], [], [], [], []

    total_heads = sum(act.shape[-1] // head_dim for act in clean_acts)
    pbar = tqdm(total=total_heads, desc='patching', leave=False)

    for layer in range(n_layers):
        ca_flat      = clean_acts[layer].to(device)   # (1, seq, hidden_actual)
        n_heads_here = ca_flat.shape[-1] // head_dim
        ca           = ca_flat.view(1, -1, n_heads_here, head_dim)

        for head_idx in range(n_heads_here):
            # Replace just this head's last-token slice with the clean version.
            def pre_hook(module, inp, _h=head_idx, _ca=ca, _n=n_heads_here):
                x = inp[0].clone()
                r = x.view(x.shape[0], x.shape[1], _n, head_dim)
                r[:, -1, _h, :] = _ca[:, -1, _h, :]
                return (r.view(x.shape),)

            handle = get_o_proj(model, layer).register_forward_pre_hook(pre_hook)
            try:
                logits = model(corr_ids).logits[0, -1].cpu()
            finally:
                handle.remove()

            c, cp, fp, cl, fl = pronoun_stats(logits, correct_id, confuse_id, all_ids)
            corrects.append(c); correct_probs.append(cp); confuse_probs.append(fp)
            correct_logits.append(cl); confuse_logits.append(fl)
            pbar.update(1)
    pbar.close()

    return (
        torch.tensor(corrects,       dtype=torch.float32),
        torch.tensor(correct_probs,  dtype=torch.float32),
        torch.tensor(confuse_probs,  dtype=torch.float32),
        torch.tensor(correct_logits, dtype=torch.float32),
        torch.tensor(confuse_logits, dtype=torch.float32),
    )


def main(args):
    torch.manual_seed(8)
    np.random.seed(8)
    model_name = args.model.split('/')[-1]

    print(f'Loading {args.model}...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map='auto',
            attn_implementation='eager', trust_remote_code=True)
    except RuntimeError:
        print('GPU load failed, falling back to CPU.')
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map='cpu',
            attn_implementation='eager', trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device

    head_dim   = get_head_dim(model)
    n_layers   = model.config.num_hidden_layers
    print(f'  head_dim={head_dim}  n_layers={n_layers}')

    df = pd.read_csv(args.data_path, sep='\t')
    mask = (df['correct_pronoun'].isin(STANDARD_PRONOUNS) &
            df['confuse_pronoun'].isin(STANDARD_PRONOUNS))
    df = df[mask].reset_index(drop=True)
    if args.max_n is not None:
        df = df.iloc[:args.max_n]
    print(f'Examples: {len(df)}')

    # We don't know the total head count until we see the first activation, thats why patched_saver is created after the first example.
    clean_saver   = HeadSaver('clean',   1)
    corrupt_saver = HeadSaver('corrupt', 1)
    patched_saver = None
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        correct_id, confuse_id, all_ids = get_candidate_ids(
            tokenizer, str(row['correct_pronoun']), str(row['confuse_pronoun']))
        if correct_id is None:
            skipped += 1
            continue

        clean_ids = tokenize_prefix(str(row['p_clean']),   tokenizer).to(device)
        corr_ids  = tokenize_prefix(str(row['p_corrupt']), tokenizer).to(device)

        with torch.no_grad():
            cl_logits = model(clean_ids).logits[0, -1].cpu()
            co_logits = model(corr_ids).logits[0, -1].cpu()

        clean_saver.update(*[torch.tensor([[v]])
                             for v in pronoun_stats(cl_logits, correct_id, confuse_id, all_ids)])
        corrupt_saver.update(*[torch.tensor([[v]])
                               for v in pronoun_stats(co_logits, correct_id, confuse_id, all_ids)])

        clean_acts  = collect_clean_acts(model, clean_ids)

        if patched_saver is None:
            n_heads_total = sum(a.shape[-1] // head_dim for a in clean_acts)
            print(f'  n_heads_total (from tensors): {n_heads_total}')
            patched_saver = HeadSaver('patched', n_heads_total)

        patch_stats = patch_all_heads(model, clean_acts, corr_ids, correct_id, confuse_id, all_ids)
        patched_saver.update(*[s.unsqueeze(0) for s in patch_stats])

    if patched_saver is None:
        print('No examples processed -- check data and tokenisation.')
        return

    processed = clean_saver.n
    print(f'\nProcessed: {processed}  Skipped: {skipped}')
    print(f'Clean acc:   {clean_saver.acc().item():.3f}')
    print(f'Corrupt acc: {corrupt_saver.acc().item():.3f}')

    suffix          = f'n{processed}'
    acc_delta       = patched_saver.acc()        - corrupt_saver.acc()
    logprob_delta   = patched_saver.mean_prob()  - corrupt_saver.mean_prob()
    logitdiff_delta = patched_saver.logit_diff() - corrupt_saver.logit_diff()

    # Per-layer head counts, recomputed from a fresh activation sample, so the flat index maps back to (layer, head) correctly.
    clean_acts_sample = collect_clean_acts(model, tokenize_prefix(
        str(df.iloc[0]['p_clean']), tokenizer).to(device))
    heads_per_layer = [a.shape[-1] // head_dim for a in clean_acts_sample]

    score_dir = os.path.join(_CACHE_ROOT, 'causal_scores',    model_name)
    rank_dir  = os.path.join(_CACHE_ROOT, 'head_orderings',   model_name)
    patch_dir = os.path.join(_CACHE_ROOT, 'pronoun_patching', model_name)
    for d in (score_dir, rank_dir, patch_dir):
        os.makedirs(d, exist_ok=True)

    records = []
    flat_i  = 0
    for layer, n_h in enumerate(heads_per_layer):
        for head in range(n_h):
            records.append({
                'layer':           layer,
                'head_idx':        head,
                'score':           round(acc_delta[flat_i].item(),        4),
                'logprob_delta':   round(logprob_delta[flat_i].item(),    4),
                'logitdiff_delta': round(logitdiff_delta[flat_i].item(),  4),
            })
            flat_i += 1
    records.sort(key=lambda r: r['score'], reverse=True)

    scores_path = os.path.join(score_dir, f'causal_scores_{suffix}.json')
    with open(scores_path, 'w') as f:
        json.dump(records, f, indent=2)
    print(f'Saved causal scores -> {scores_path}')

    ranked = [[r['layer'], r['head_idx']] for r in records]
    with open(os.path.join(rank_dir, f'pronoun_copying_{suffix}.json'), 'w') as f:
        json.dump(ranked, f)
    print(f'Saved head ranking -> {rank_dir}/pronoun_copying_{suffix}.json')

    pkl_path = os.path.join(score_dir, f'{suffix}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump([clean_saver, corrupt_saver, patched_saver], f)
    print(f'Saved raw savers -> {pkl_path}')

    summary = {
        'clean_acc':            round(clean_saver.acc().item(),          4),
        'corrupt_acc':          round(corrupt_saver.acc().item(),        4),
        'clean_mean_logprob':   round(clean_saver.mean_prob().item(),    4),
        'corrupt_mean_logprob': round(corrupt_saver.mean_prob().item(),  4),
        'clean_logit_diff':     round(clean_saver.logit_diff().item(),   4),
        'corrupt_logit_diff':   round(corrupt_saver.logit_diff().item(), 4),
        'n':       processed,
        'skipped': skipped,
    }
    summary_path = os.path.join(patch_dir, f'patching_summary_{suffix}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved patching summary -> {summary_path}')

    print(f'\nTop-10 heads by accuracy delta:')
    print(f'  {"Rank":>4}  {"Layer":>5}  {"Head":>4}  {"d_acc":>8}  {"d_logprob":>10}')
    for rank, r in enumerate(records[:10]):
        print(f'  {rank+1:>4}  {r["layer"]:>5}  {r["head_idx"]:>4}  '
              f'{r["score"]:>8.4f}  {r["logprob_delta"]:>10.4f}')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='HuggingFace model ID')
    parser.add_argument('--data_path',
                        default=os.path.join(_REPO_ROOT, 'data',
                                             'eo_ep_task_induction_pairs.tsv'))
    parser.add_argument('--max_n', type=int, default=50)
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)