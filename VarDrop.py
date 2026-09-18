import torch
import numpy as np


# ============================================================
# Original VarDrop
# ============================================================

def hash_func(k_freqs):
    """
    Input:
        k_freqs: [k, N] or [N, k]

    Output:
        hash_values: [N]
    """
    return np.array([
        '-'.join(map(str, k_freq.tolist()))
        for k_freq in k_freqs.T
    ])


def k_dominant_frequency_hashing(
    batch_x,
    k,
    freq_list=None,
    min_thres=None
):
    """
    k-Dominant Frequency Hashing (k-DFH)

    batch_x: [B, L, N]
    """

    x_amp = torch.fft.rfft(batch_x, dim=1).abs()

    # [B, F, N] -> [F, N]
    x_amp = torch.mean(x_amp, dim=0)

    if freq_list is not None:

        freq_list = list(freq_list)

        freq_idx = torch.tensor(
            freq_list,
            device=x_amp.device,
            dtype=torch.long
        )

        # Keep only valid frequencies
        freq_idx = freq_idx[
            (freq_idx >= 0) &
            (freq_idx < x_amp.shape[0])
        ]

        if len(freq_idx) == 0:
            raise ValueError(
                "freq_list does not contain any valid frequency index."
            )

        k = min(k, len(freq_idx))

        filtered_amp = x_amp.index_select(0, freq_idx)

        k_amps, rel_freqs = torch.topk(
            filtered_amp,
            k=k,
            dim=0
        )

        k_freqs = freq_idx[rel_freqs]

    else:

        k = min(k, x_amp.shape[0])

        k_amps, k_freqs = torch.topk(
            x_amp,
            k=k,
            dim=0
        )

    if min_thres is not None:

        k_freqs = torch.where(
            k_amps >= min_thres,
            k_freqs,
            torch.full_like(k_freqs, 99)
        )

    return hash_func(
        k_freqs.detach().cpu().numpy()
    )


# ============================================================
# Helper: temporal representation
# ============================================================

def _temporal_profile(batch_x):
    """
    Build a normalized temporal representation for every variate.

    Input:
        batch_x: [B, L, N]

    Output:
        profile: [N, L]
    """

    x = batch_x.detach().float()

    # Normalize each variate independently over time
    x = x - x.mean(dim=1, keepdim=True)

    x = x / (
        x.std(
            dim=1,
            keepdim=True,
            unbiased=False
        ) + 1e-6
    )

    # [B, L, N] -> [N, L]
    profile = x.permute(0, 2, 1).mean(dim=0)

    # L2 normalize
    profile = profile / (
        torch.norm(
            profile,
            dim=1,
            keepdim=True
        ) + 1e-8
    )

    return profile.cpu().numpy()


# ============================================================
# Version 0
# Original VarDrop
# ============================================================

def efficient_sampler_v0(
    x,
    k,
    group_size,
    freq_list,
    min_thres=None
):
    """
    Original VarDrop sampler.

    Frequency grouping +
    random sampling inside each group.
    """

    hash_values = k_dominant_frequency_hashing(
        x,
        k,
        freq_list,
        min_thres
    )

    sparse_indices = []

    for value in np.unique(hash_values):

        group_indices = np.where(
            hash_values == value
        )[0].tolist()

        sample_size = min(
            group_size,
            len(group_indices)
        )

        # Original behavior
        sampled = np.random.choice(
            group_indices,
            sample_size,
            replace=True
        )

        sparse_indices.extend(
            sampled.tolist()
        )

    # Keep original behavior
    sparse_indices = np.unique(
        sparse_indices
    )

    return sparse_indices.tolist()


# ============================================================
# Version 1
# Temporal Representative Sampling
# ============================================================

def efficient_sampler_v1(
    x,
    k,
    group_size,
    freq_list,
    min_thres=None
):
    """
    Frequency-based grouping +
    temporal representative selection.

    Instead of randomly selecting variables from a frequency group,
    selects variables closest to the temporal centroid.
    """

    hash_values = k_dominant_frequency_hashing(
        x,
        k,
        freq_list,
        min_thres
    )

    profile = _temporal_profile(x)

    sparse_indices = []

    for value in np.unique(hash_values):

        group_indices = np.where(
            hash_values == value
        )[0]

        sample_size = min(
            group_size,
            len(group_indices)
        )

        if sample_size == len(group_indices):

            selected = group_indices

        else:

            group_profile = profile[group_indices]

            # Temporal centroid
            centroid = group_profile.mean(axis=0)

            centroid = centroid / (
                np.linalg.norm(centroid) + 1e-8
            )

            # Cosine similarity to centroid
            similarity = (
                group_profile @ centroid
            )

            # Most representative first
            order = np.argsort(
                -similarity
            )

            selected = group_indices[
                order[:sample_size]
            ]

        sparse_indices.extend(
            selected.tolist()
        )

    return sorted(
        list(set(sparse_indices))
    )


# ============================================================
# Version 2
# Adaptive Token Budget
# ============================================================

def efficient_sampler_v2(
    x,
    k,
    group_size,
    freq_list,
    target_tokens,
    min_thres=None
):
    """
    Adaptive Token Budget.

    1. Group variables according to dominant frequencies.
    2. Measure temporal heterogeneity of each group.
    3. Allocate a global token budget adaptively.
    4. Select temporal representatives.

    target_tokens controls the final number of selected variables.
    """

    hash_values = k_dominant_frequency_hashing(
        x,
        k,
        freq_list,
        min_thres
    )

    profile = _temporal_profile(x)

    unique_groups = np.unique(
        hash_values
    )

    groups = []

    for value in unique_groups:

        indices = np.where(
            hash_values == value
        )[0]

        group_profile = profile[indices]

        # Temporal centroid
        centroid = group_profile.mean(axis=0)

        centroid = centroid / (
            np.linalg.norm(centroid) + 1e-8
        )

        # Average distance from centroid
        dispersion = np.mean(
            1.0 -
            (group_profile @ centroid)
        )

        groups.append({
            "indices": indices,
            "profile": group_profile,
            "dispersion": float(dispersion)
        })

    N = len(hash_values)
    G = len(groups)

    budget = min(
        int(target_tokens),
        N
    )

    # --------------------------------------------------------
    # Allocation
    # --------------------------------------------------------

    sizes = np.array([
        len(g["indices"])
        for g in groups
    ], dtype=np.int32)

    dispersion = np.array([
        g["dispersion"]
        for g in groups
    ], dtype=np.float32)

    # Larger and more heterogeneous groups get more budget
    weights = (
        np.sqrt(sizes) *
        (0.5 + dispersion)
    )

    allocation = np.zeros(
        G,
        dtype=np.int32
    )

    # Ensure every selected group initially gets one token
    min_groups = min(
        budget,
        G
    )

    if min_groups > 0:

        initial_groups = np.argsort(
            -weights
        )[:min_groups]

        allocation[
            initial_groups
        ] = 1

    current = int(
        allocation.sum()
    )

    # --------------------------------------------------------
    # Distribute remaining budget
    # --------------------------------------------------------

    while current < budget:

        valid = np.where(
            allocation < sizes
        )[0]

        if len(valid) == 0:
            break

        priority = (
            weights[valid] /
            (allocation[valid] + 1.0)
        )

        best_local = np.argmax(
            priority
        )

        best_group = valid[
            best_local
        ]

        allocation[
            best_group
        ] += 1

        current += 1

    # --------------------------------------------------------
    # Representative selection
    # --------------------------------------------------------

    sparse_indices = []

    for group, n_select in zip(
        groups,
        allocation
    ):

        if n_select <= 0:
            continue

        indices = group["indices"]

        if n_select >= len(indices):

            selected = indices

        else:

            group_profile = group["profile"]

            centroid = group_profile.mean(
                axis=0
            )

            centroid = centroid / (
                np.linalg.norm(centroid) + 1e-8
            )

            similarity = (
                group_profile @ centroid
            )

            order = np.argsort(
                -similarity
            )

            selected = indices[
                order[:n_select]
            ]

        sparse_indices.extend(
            selected.tolist()
        )

    return sorted(
        list(set(sparse_indices))
    )


# ============================================================
# Cached Adaptive Sampler
# ============================================================

class CachedAdaptiveSampler:
    """
    Version 3.

    Runs Version 2 only every 'refresh_every' batches.
    Between refreshes, the same variable subset is reused.
    """

    def __init__(
        self,
        k,
        group_size,
        freq_list,
        target_tokens,
        refresh_every=64,
        min_thres=None
    ):

        self.k = k
        self.group_size = group_size
        self.freq_list = freq_list
        self.target_tokens = target_tokens
        self.refresh_every = max(
            1,
            int(refresh_every)
        )
        self.min_thres = min_thres

        self.cached_indices = None
        self.step = 0

    def __call__(self, batch_x):

        should_refresh = (
            self.cached_indices is None
            or
            self.step % self.refresh_every == 0
        )

        if should_refresh:

            self.cached_indices = efficient_sampler_v2(
                batch_x,
                k=self.k,
                group_size=self.group_size,
                freq_list=self.freq_list,
                target_tokens=self.target_tokens,
                min_thres=self.min_thres
            )

        self.step += 1

        return self.cached_indices


# ============================================================
# Version 4
# Random Fixed-Budget Sampling
# ============================================================

def random_sampler(
    x,
    target_tokens
):
    """
    Uniform random sampling with a fixed global token budget.

    Input:
        x: [B, L, N]

    Output:
        sorted selected variate indices
    """

    N = x.shape[-1]

    budget = min(
        int(target_tokens),
        N
    )

    if budget <= 0:
        return []

    indices = np.random.choice(
        N,
        size=budget,
        replace=False
    )

    return np.sort(
        indices
    ).astype(
        np.int64
    ).tolist()


# ============================================================
# Version 5
# Frequency-Grouped Fixed-Budget Random Sampling
# ============================================================

def efficient_sampler_freq_budget_random(
    x,
    k,
    group_size,
    freq_list,
    target_tokens,
    min_thres=None
):
    """
    Frequency-aware fixed-budget random sampling.

    1. Group variates using k-DFH.
    2. Allocate one global token budget proportional
       to group size.
    3. Randomly sample from each frequency group.

    This version intentionally does NOT use:
        - temporal profiles
        - temporal centroids
        - dispersion
        - heterogeneity-aware allocation

    It is designed as a controlled baseline for V2.
    """

    hash_values = k_dominant_frequency_hashing(
        x,
        k,
        freq_list,
        min_thres
    )

    unique_groups = np.unique(
        hash_values
    )

    groups = []

    for value in unique_groups:

        indices = np.where(
            hash_values == value
        )[0]

        groups.append(
            indices
        )

    N = len(hash_values)

    budget = min(
        int(target_tokens),
        N
    )

    if budget <= 0:
        return []

    sizes = np.array(
        [
            len(group)
            for group in groups
        ],
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Allocate global budget proportional to group size.
    # --------------------------------------------------------

    raw_allocation = (
        budget *
        sizes /
        sizes.sum()
    )

    allocation = np.floor(
        raw_allocation
    ).astype(
        np.int32
    )

    allocation = np.minimum(
        allocation,
        sizes
    )

    # --------------------------------------------------------
    # Largest remainder correction
    # --------------------------------------------------------

    current = int(
        allocation.sum()
    )

    remaining = budget - current

    if remaining > 0:

        fractional = (
            raw_allocation -
            np.floor(
                raw_allocation
            )
        )

        valid = np.where(
            allocation < sizes
        )[0]

        order = valid[
            np.argsort(
                -fractional[valid]
            )
        ]

        for index in order:

            if remaining <= 0:
                break

            allocation[index] += 1
            remaining -= 1

    # --------------------------------------------------------
    # Random sampling inside each group
    # --------------------------------------------------------

    sparse_indices = []

    for group, n_select in zip(
        groups,
        allocation
    ):

        if n_select <= 0:
            continue

        if n_select >= len(group):

            selected = group

        else:

            selected = np.random.choice(
                group,
                size=n_select,
                replace=False
            )

        sparse_indices.extend(
            selected.tolist()
        )

    return sorted(
        list(
            set(
                sparse_indices
            )
        )
    )


# ============================================================
# Unified sampler
# ============================================================

def efficient_sampler(
    x,
    k,
    group_size,
    freq_list,
    version=0,
    target_tokens=None,
    sampler=None,
    min_thres=None
):
    """
    Unified VarDrop interface.

    version:
        0 -> Original VarDrop
        1 -> Temporal Representative Sampling
        2 -> Adaptive Token Budget
        3 -> Cached Adaptive Sampling
        4 -> Random Fixed-Budget Sampling
        5 -> Frequency-Grouped Fixed-Budget Random Sampling

    For version 3, pass the CachedAdaptiveSampler object
    through 'sampler'.
    """

    version = int(version)

    if version == 0:

        return efficient_sampler_v0(
            x,
            k=k,
            group_size=group_size,
            freq_list=freq_list,
            min_thres=min_thres
        )

    elif version == 1:

        return efficient_sampler_v1(
            x,
            k=k,
            group_size=group_size,
            freq_list=freq_list,
            min_thres=min_thres
        )

    elif version == 2:

        if target_tokens is None:
            raise ValueError(
                "target_tokens is required for VarDrop version 2."
            )

        return efficient_sampler_v2(
            x,
            k=k,
            group_size=group_size,
            freq_list=freq_list,
            target_tokens=target_tokens,
            min_thres=min_thres
        )

    elif version == 3:

        if sampler is None:
            raise ValueError(
                "CachedAdaptiveSampler object is required "
                "for VarDrop version 3."
            )

        return sampler(x)

    elif version == 4:

        if target_tokens is None:
            raise ValueError(
                "target_tokens is required for "
                "VarDrop version 4."
            )

        return random_sampler(
            x,
            target_tokens=target_tokens
        )

    elif version == 5:

        if target_tokens is None:
            raise ValueError(
                "target_tokens is required for "
                "VarDrop version 5."
            )

        return efficient_sampler_freq_budget_random(
            x,
            k=k,
            group_size=group_size,
            freq_list=freq_list,
            target_tokens=target_tokens,
            min_thres=min_thres
        )

    else:

        raise ValueError(
            f"Unknown VarDrop version: {version}. "
            f"Use 0, 1, 2, 3, 4 or 5."
        )