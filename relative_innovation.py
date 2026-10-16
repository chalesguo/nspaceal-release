"""Relative-innovation utilities for nullspace active learning."""

import time

import torch


def _sync_cuda(device=None):
    if torch.cuda.is_available():
        if device is not None and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()


def adaptive_nullspace_dim(S, method="spectral_gap", threshold=0.05):
    """Estimate the effective learned-subspace dimension from singular values."""
    S = S.float()
    S_norm = S / (S.max() + 1e-8)

    if method == "spectral_gap":
        diffs = S_norm[:-1] - S_norm[1:]
        gap_idx = torch.argmax(diffs).item()
        return gap_idx + 1

    if method == "energy_ratio":
        energy = torch.cumsum(S_norm ** 2, dim=0)
        total = energy[-1]
        eff_dims = (energy < 0.99 * total).sum().item() + 1
        return max(1, eff_dims)

    if method == "parallel_analysis":
        n_random = 100
        d = len(S)
        random_eigenvals = torch.randn(n_random, d, device=S.device).square() * (
            S.sum().item() / d
        )
        cutoff = torch.quantile(random_eigenvals.mean(dim=0), 0.95)
        eff_dims = (S > cutoff).sum().item()
        return max(1, eff_dims)

    if method == "fixed":
        eff_dims = (S_norm > threshold).sum().item()
        return max(1, eff_dims)

    raise ValueError(f"Unknown dimension method: {method}")


def compute_nullspace_projection(
    features, adaptive_threshold=0.05, dim_method="spectral_gap"
):
    """Compute the projection matrix onto the complement of the learned subspace."""
    centered = features - features.mean(dim=0, keepdim=True)
    _, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    eff_dims = adaptive_nullspace_dim(
        S, method=dim_method, threshold=adaptive_threshold
    )
    V_eff = Vt[:eff_dims, :].t()
    feat_dim = features.size(1)

    P_null = torch.eye(feat_dim, device=features.device) - V_eff @ V_eff.t()
    return P_null, eff_dims, S


def _clip_by_percentile(x, percentile):
    if percentile is None or percentile <= 0 or percentile >= 100 or x.numel() == 0:
        return x
    cutoff = torch.quantile(x.detach().float(), percentile / 100.0)
    return torch.clamp(x, max=cutoff.to(x.device))


def compute_relative_innovation(
    L_t,
    U_t,
    beta=0.1,
    dim_method="spectral_gap",
    adaptive_threshold=0.05,
    prev_P_null=None,
    prev_residuals=None,
    eps=1e-8,
    ratio_clip_percentile=95.0,
):
    """Compute relative innovation scores for unlabeled features.

    Preferred usage passes ``prev_residuals`` aligned with the current unlabeled
    pool. Those values are cached from the previous active-learning cycle and
    therefore compare cycle-specific residual magnitudes:

        ||P_null^(t) h_t(x)|| / (cached ||P_null^(t-1) h_(t-1)(x)|| + eps).

    If previous residuals are unavailable, the first-cycle convention is used:
    ``r_t(x)=1`` and acquisition relies on current residual coverage only.
    """
    device = L_t.device
    _sync_cuda(device)
    t_svd = time.perf_counter()
    P_null_t, dims_t, _ = compute_nullspace_projection(
        L_t, adaptive_threshold=adaptive_threshold, dim_method=dim_method
    )
    _sync_cuda(device)
    svd_time = time.perf_counter() - t_svd

    Z_perp_t = U_t @ P_null_t.t()
    inn_now = torch.norm(Z_perp_t, dim=1)

    used_cached_residuals = prev_residuals is not None
    if prev_residuals is not None:
        inn_prev = prev_residuals.to(device).float()
        dd = torch.zeros(1, device=device)
    else:
        inn_prev = None
        dd = torch.zeros(1, device=device)

    if inn_prev is None:
        relative = torch.ones_like(inn_now)
    else:
        relative = inn_now / (inn_prev + eps)
    relative = _clip_by_percentile(relative, ratio_clip_percentile)

    _sync_cuda(device)
    t_density = time.perf_counter()
    density = _compute_density(Z_perp_t, bandwidth=1.0)
    _sync_cuda(device)
    density_time = time.perf_counter() - t_density
    density_probs = density / (density.sum() + 1e-12)
    density_probs = torch.clamp(density_probs, min=1e-12)
    coverage_ent = -torch.log(density_probs)

    isd = relative * coverage_ent * torch.exp(beta * dd)
    info = {
        "svd_time_s": svd_time,
        "density_time_s": density_time,
        "used_cached_residuals": used_cached_residuals,
        "relative_clip_percentile": ratio_clip_percentile,
    }
    return isd, relative, density, inn_now, dims_t, P_null_t, info


def compute_standard_nullspace(
    U_t, L_t, beta=0.1, adaptive_threshold=0.05, dim_method="spectral_gap"
):
    """Compute the baseline absolute nullspace innovation score."""
    _sync_cuda(L_t.device)
    t_svd = time.perf_counter()
    P_null, dims, _ = compute_nullspace_projection(
        L_t, adaptive_threshold=adaptive_threshold, dim_method=dim_method
    )
    _sync_cuda(L_t.device)
    svd_time = time.perf_counter() - t_svd

    Z_perp = U_t @ P_null.t()
    innovation = torch.norm(Z_perp, dim=1)

    _sync_cuda(L_t.device)
    t_density = time.perf_counter()
    density = _compute_density(Z_perp, bandwidth=1.0)
    _sync_cuda(L_t.device)
    density_time = time.perf_counter() - t_density
    density_probs = density / (density.sum() + 1e-12)
    density_probs = torch.clamp(density_probs, min=1e-12)
    coverage_ent = -torch.log(density_probs)

    dd = torch.zeros(1, device=L_t.device)
    isd = innovation * coverage_ent * torch.exp(beta * dd)
    info = {"svd_time_s": svd_time, "density_time_s": density_time}
    return isd, innovation, density, dims, P_null, info


def _compute_density(Z_perp, bandwidth=1.0):
    """Estimate feature density with a Gaussian kernel."""
    n = Z_perp.size(0)
    sigma = bandwidth
    scores = torch.zeros(n, device=Z_perp.device)
    for i in range(0, n, 512):
        batch = Z_perp[i : i + 512]
        dists = torch.cdist(batch, Z_perp, p=2)
        scores[i : i + 512] = torch.sum(
            torch.exp(-(dists ** 2) / (sigma ** 2 + 1e-12)), dim=1
        )
    return scores


def rank_normalize(x):
    """Rank-normalize a vector to the [0, 1] interval."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    dev = x.device
    order = torch.argsort(x)
    r = torch.zeros_like(x, dtype=torch.float32, device=dev)
    if x.numel() == 1:
        r[order] = 1.0
    else:
        r[order] = torch.linspace(0.0, 1.0, steps=x.size(0), device=dev)
    return r


def compute_entropy_from_logits(logits):
    """Compute predictive entropy from logits."""
    probs = torch.softmax(logits, dim=1)
    logp = torch.log(torch.clamp(probs, 1e-12, 1.0))
    return -torch.sum(probs * logp, dim=1)


def maximin_selection(Z_perp, final_scores, num_select, lambda_div=0.3):
    """Select samples with a score-plus-diversity maximin rule."""
    n = Z_perp.size(0)
    if num_select >= n:
        return list(range(n))

    selected = []
    available = torch.ones(n, dtype=torch.bool, device=Z_perp.device)

    first = int(torch.argmax(final_scores).item())
    selected.append(first)
    available[first] = False

    min_dists = torch.full((n,), float("inf"), device=Z_perp.device)
    while len(selected) < num_select:
        last = selected[-1]
        dists_to_last = torch.norm(Z_perp - Z_perp[last].unsqueeze(0), dim=1)
        min_dists = torch.minimum(min_dists, dists_to_last)
        min_dists[~available] = 0.0

        denom = min_dists[available].max().clamp(min=1e-12)
        scores = final_scores + lambda_div * (min_dists / denom)
        scores = scores.masked_fill(~available, -float("inf"))

        nxt = int(torch.argmax(scores).item())
        selected.append(nxt)
        available[nxt] = False

    return selected


def select_with_relative_innovation(
    L_t,
    U_t,
    U_logits,
    pred_loss_vals,
    num_select,
    args,
    cycle=0,
    prev_P_null=None,
    prev_residuals=None,
    mode="relative",
):
    """Select unlabeled samples with relative-innovation-enhanced scoring."""
    beta = getattr(args, "beta", 0.1)
    w_isd = getattr(args, "w_isd", 0.6)
    w_ent = getattr(args, "w_ent", 0.25)
    w_pl = getattr(args, "w_pl", 0.15)
    use_maximin = getattr(args, "use_maximin", True)
    lambda_div = getattr(args, "lambda_div", 0.7)
    dim_method = getattr(args, "dim_method", "spectral_gap")
    adaptive_threshold = getattr(args, "adaptive_dim_threshold", 0.05)
    eps = getattr(args, "relative_eps", 1e-8)
    ratio_clip_percentile = getattr(args, "ratio_clip_percentile", 95.0)

    timing = {}
    if mode == "ablation_compare":
        isd_rel, rel, _, inn, dims, P_null, rel_info = compute_relative_innovation(
            L_t,
            U_t,
            beta=beta,
            dim_method=dim_method,
            adaptive_threshold=adaptive_threshold,
            prev_P_null=prev_P_null,
            prev_residuals=prev_residuals,
            eps=eps,
            ratio_clip_percentile=ratio_clip_percentile,
        )
        isd_std, inn_std, _, dims_std, _, std_info = compute_standard_nullspace(
            U_t,
            L_t,
            beta=beta,
            adaptive_threshold=adaptive_threshold,
            dim_method=dim_method,
        )
        isd = isd_rel
        innovation_score = rel
        inn_now = inn
        timing.update(rel_info)
        info = {
            "isd_relative": isd_rel.cpu(),
            "isd_standard": isd_std.cpu(),
            "innovation_relative": rel.cpu(),
            "innovation_standard": inn_std.cpu(),
            "dims_t": dims,
            "dims_prev": dims_std,
            "standard_svd_time_s": std_info["svd_time_s"],
            "standard_density_time_s": std_info["density_time_s"],
        }
    elif mode == "relative":
        isd, innovation_score, _, inn_now, dims, P_null, rel_info = (
            compute_relative_innovation(
                L_t,
                U_t,
                beta=beta,
                dim_method=dim_method,
                adaptive_threshold=adaptive_threshold,
                prev_P_null=prev_P_null,
                prev_residuals=prev_residuals,
                eps=eps,
                ratio_clip_percentile=ratio_clip_percentile,
            )
        )
        timing.update(rel_info)
        info = {"dims_t": dims}
    else:
        isd, innovation_score, _, dims, P_null, std_info = compute_standard_nullspace(
            U_t,
            L_t,
            beta=beta,
            adaptive_threshold=adaptive_threshold,
            dim_method=dim_method,
        )
        inn_now = innovation_score
        timing.update(std_info)
        info = {"dims_t": dims}

    Z_perp = U_t @ P_null.t()
    entropy_scores = compute_entropy_from_logits(U_logits)
    pred_loss_vals = pred_loss_vals.to(Z_perp.device)

    isd_n = rank_normalize(isd)
    ent_n = rank_normalize(entropy_scores)
    pl_n = rank_normalize(pred_loss_vals)

    w_sum = max(1e-12, w_isd + w_ent + w_pl)
    final_scores = (
        (w_isd / w_sum) * isd_n
        + (w_ent / w_sum) * ent_n
        + (w_pl / w_sum) * pl_n
    )

    _sync_cuda(Z_perp.device)
    t_select = time.perf_counter()
    if use_maximin:
        selected = maximin_selection(Z_perp, final_scores, num_select, lambda_div)
    else:
        _, topk = torch.topk(final_scores, num_select)
        selected = topk.cpu().numpy().tolist()
    _sync_cuda(Z_perp.device)
    timing["select_time_s"] = time.perf_counter() - t_select

    info.update(timing)
    info["current_residuals"] = inn_now.detach().cpu()
    return selected, info, P_null
