"""Length-normalized, differentiable decoder hypothesis ranking."""
import torch


def token_mean_log_probability(logits, labels, special_ids, eos_id):
    # Keep EOS as a scored stopping decision. Exclude control prefix and padding.
    valid = labels.ne(-100)
    for token in special_ids:
        if token != eos_id:
            valid &= labels.ne(token)
    lengths = valid.sum(-1)
    if (lengths == 0).any():
        raise ValueError('No scored content/EOS tokens')
    safe_labels = labels.masked_fill(labels == -100, 0)
    # Avoid materializing a second full log_softmax tensor (B,S,51865).
    values = logits.float().gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    values = values - logits.float().logsumexp(-1)
    return values.masked_fill(~valid, 0).sum(-1) / lengths


def correction_margin(reference_scores, hypothesis_scores, margin):
    gaps = margin + hypothesis_scores - reference_scores
    return gaps.clamp_min(0).mean(), gaps.gt(0)
