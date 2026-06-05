#!/usr/bin/env python3
"""
Just a short check. 

Resolves the single-token IDs for ' he', ' she', ' they' for any supported tokenizer.

"""

from __future__ import annotations

# The leading space is how BPE tokenizers represent a word mid-sentence, which
# is the position we care about (the blank comes after "that"/"said that").
_PRONOUN_SURFACE = {
    "he":   " he",
    "she":  " she",
    "they": " they",
}


def _unwrap(tokenizer):
    # Gemma's AutoProcessor wraps the real tokenizer under .tokenizer; unwrap once.
    if not hasattr(tokenizer, "encode") and hasattr(tokenizer, "tokenizer"):
        return tokenizer.tokenizer
    return tokenizer


def _encode_no_special(tok, text: str) -> list[int]:
    # add_special_tokens=False keeps EOS out so we get just the word's tokens.
    ids = tok.encode(text, add_special_tokens=False)
    return ids


def get_pronoun_token_ids(tokenizer, model_name: str) -> dict[str, int]:
    """
    Return {label: token_id} for each pronoun, checking each is a single token.
    Raises ValueError listing any pronoun that came out multi-token.
    """
    tok = _unwrap(tokenizer)
    result: dict[str, int] = {}

    multi_token_errors = []
    for label, surface in _PRONOUN_SURFACE.items():
        ids = _encode_no_special(tok, surface)
        # SentencePiece models sometimes give a space token first. If so, drop it and keep the pronoun.
        if len(ids) == 2:
            decoded_first = tok.decode([ids[0]])
            if decoded_first.strip() == "":
                ids = ids[1:]

        if len(ids) != 1:
            decoded = [tok.decode([i]) for i in ids]
            multi_token_errors.append(
                f"  '{surface}' -> {ids} ({decoded}) for model {model_name!r}"
            )
        else:
            result[label] = ids[0]

    if multi_token_errors:
        raise ValueError(
            "IIA computation requires single-token pronouns, but the following "
            "resolved to multiple tokens:\n" + "\n".join(multi_token_errors) + "\n"
            "Consider adapting the surface forms or using a different evaluation "
            "strategy for this model."
        )

    return result


def print_pronoun_tokens(tokenizer, model_name: str) -> None:
    """Print the resolved ids and decoded forms; handy for sanity checks."""
    tok = _unwrap(tokenizer)
    print(f"Pronoun token resolution for {model_name!r}:")
    for label, surface in _PRONOUN_SURFACE.items():
        ids = _encode_no_special(tok, surface)
        decoded = [tok.decode([i]) for i in ids]
        marker = "OK" if len(ids) == 1 else "MULTI-TOKEN"
        print(f"  [{marker}] '{surface}' -> ids={ids}, decoded={decoded}")