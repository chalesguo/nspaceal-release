"""BADGE acquisition for active learning.

BADGE selects a batch with k-means++ seeding in gradient-embedding space.
The implementation supports random projection so class-by-feature gradient
embeddings remain practical for datasets such as CIFAR-100.
"""

import torch


def _projected_badge_embeddings(
    features,
    logits,
    proj_dim=256,
    chunk_size=256,
    seed=0,
):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    n_samples, feat_dim = features.shape
    num_classes = probs.shape[1]
    full_dim = feat_dim * num_classes

    if proj_dim and proj_dim < full_dim:
        gen = torch.Generator(device=features.device)
        gen.manual_seed(seed)
        projector = torch.randn(
            full_dim, proj_dim, generator=gen, device=features.device
        ) / (proj_dim ** 0.5)
        output = torch.empty(n_samples, proj_dim, device=features.device)
    else:
        projector = None
        output = torch.empty(n_samples, full_dim, device=features.device)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        coeff = probs[start:end].clone()
        coeff[torch.arange(end - start, device=features.device), preds[start:end]] -= 1.0
        emb = coeff.unsqueeze(2) * features[start:end].unsqueeze(1)
        emb = emb.reshape(end - start, full_dim)
        if projector is not None:
            emb = emb @ projector
        output[start:end] = emb

    return output


def kmeans_plus_plus_select(embeddings, num_select, seed=0):
    """Return k-means++ seed indices for a tensor of embeddings."""
    n_samples = embeddings.size(0)
    if num_select >= n_samples:
        return list(range(n_samples))

    gen = torch.Generator(device=embeddings.device)
    gen.manual_seed(seed)
    selected = []
    first = int(torch.randint(n_samples, (1,), generator=gen, device=embeddings.device).item())
    selected.append(first)

    min_dist_sq = torch.cdist(embeddings, embeddings[first:first + 1]).squeeze(1).pow(2)
    min_dist_sq[first] = 0.0

    while len(selected) < num_select:
        total = min_dist_sq.sum()
        if total <= 0:
            remaining = [i for i in range(n_samples) if i not in set(selected)]
            selected.extend(remaining[: num_select - len(selected)])
            break
        probs = min_dist_sq / total
        nxt = int(torch.multinomial(probs, 1, generator=gen).item())
        if nxt in selected:
            min_dist_sq[nxt] = 0.0
            continue
        selected.append(nxt)
        new_dist_sq = torch.cdist(embeddings, embeddings[nxt:nxt + 1]).squeeze(1).pow(2)
        min_dist_sq = torch.minimum(min_dist_sq, new_dist_sq)
        min_dist_sq[selected] = 0.0

    return selected


def badge_select(
    features,
    logits,
    num_select,
    proj_dim=256,
    chunk_size=256,
    seed=0,
):
    """Select samples using BADGE gradient embeddings and k-means++."""
    embeddings = _projected_badge_embeddings(
        features,
        logits,
        proj_dim=proj_dim,
        chunk_size=chunk_size,
        seed=seed,
    )
    return kmeans_plus_plus_select(embeddings, num_select, seed=seed)
