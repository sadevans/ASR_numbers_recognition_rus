from __future__ import annotations

from collections import defaultdict

__all__ = [
    'char_levenshtein',
    'cer_over_utterances',
    'cer_by_speaker',
    'exact_match_rate',
    'harmonic_mean',
]


def char_levenshtein(ref, hyp):
    a, b = ref, hyp
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1]
            if ca != cb:
                sub += 1
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[len(b)]


def cer_over_utterances(refs, hyps):
    total_edits = 0
    total_chars = 0
    for r, h in zip(refs, hyps):
        total_edits += char_levenshtein(r, h)
        total_chars += max(len(r), 1)
    if total_chars == 0:
        return 0.0
    return total_edits / total_chars


def cer_by_speaker(
    refs,
    hyps,
    spk_ids,
):
    edits = defaultdict(int)
    chars = defaultdict(int)
    for r, h, spk in zip(refs, hyps, spk_ids):
        spk = str(spk)
        edits[spk] += char_levenshtein(r, h)
        chars[spk] += max(len(r), 1)
    return {s: edits[s] / chars[s] if chars[s] else 0.0 for s in chars}


def exact_match_rate(refs, hyps):
    rs, hs = list(refs), list(hyps)
    if not rs:
        return 0.0
    return sum(1 for r, h in zip(rs, hs) if r == h) / len(rs)


def harmonic_mean(a, b, *, eps = 1e-8):
    if a < eps or b < eps:
        return 0.0
    return 2.0 * a * b / (a + b)
