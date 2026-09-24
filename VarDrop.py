import torch
import numpy as np
import time
from itertools import chain


def hash_func(k_freqs):
    """ Return an ordered list of the dominant frequencies as a string hash value.
    Input:
        - k_freqs. # .shape = [k, n_vars] 
    Output:
        - hash_values.
    """
    return np.array(['-'.join(map(str, k_freq.tolist())) for k_freq in k_freqs.T])

def k_dominant_frequency_hashing(batch_x, k, freq_list=None, min_thres=None):
    """ k-Dominant Frequency Hashing (k-DFH) """
    
    x_amp = torch.fft.rfft(batch_x, dim=1).abs()
    x_amp = torch.mean(x_amp, dim=0) # Averaging for batch: [batch_size, n_freq, n_var] -> [n_freq,n_var]

    if freq_list != None:
        k_amps, k_freqs = torch.topk(x_amp[freq_list], k=k, dim=0)  # LPF(Low pass filter)
        k_freqs += freq_list[0]
    else:
        k_amps, k_freqs = torch.topk(x_amp, k=k, dim=0)  # without LPF

    if min_thres != None:
        k_freqs = k_freqs * (k_amps >= min_thres) # non-dominant top-k
        k_freqs += (k_amps < min_thres) * 99

    hash_values = hash_func(k_freqs)
    return hash_values

def efficient_sampler(x, k, group_size, freq_list, min_thres=None, return_group=False):

    # k-Dominant Frequency Hashing(k-DFH).
    hash_values = k_dominant_frequency_hashing(x, k, freq_list, min_thres)
    
    sparse_indices = []

    if return_group: 
        group_dict = {}

    # Iteration for each group
    for value in np.unique(hash_values):
        group_indices = np.where(hash_values == value)[0].tolist() # get group indices
        group_indices = np.random.choice(group_indices, min(group_size, len(group_indices)), replace=True) # Any sampling method can be utilized.
        sparse_indices.append(group_indices)

        if return_group:
            group_dict[value] = group_indices

    sample_indices = sorted(list(chain.from_iterable(sparse_indices)))

    if return_group:
        return sample_indices, group_dict

    return sample_indices

# ============================================================
# GPT3: Stability-Gated VarDrop
# ============================================================

def _k_dfh_signature(batch_x, k, freq_list=None, min_thres=None, return_hash=False):
    """
    Compute the same ordered top-k dominant-frequency signature used by
    the original k-DFH, but return it as a numeric [n_vars, k] array.

    The original public functions above are intentionally left unchanged.
    This helper is used only by StabilityGatedSampler.
    """
    x_amp = torch.fft.rfft(batch_x, dim=1).abs()
    x_amp = torch.mean(x_amp, dim=0)

    if freq_list is not None:
        # Preserve the original VarDrop indexing semantics exactly.
        freq_list = list(freq_list)
        k_amps, k_freqs = torch.topk(x_amp[freq_list], k=k, dim=0)
        k_freqs += freq_list[0]
    else:
        k_amps, k_freqs = torch.topk(x_amp, k=k, dim=0)

    if min_thres is not None:
        k_freqs = k_freqs * (k_amps >= min_thres)
        k_freqs += (k_amps < min_thres) * 99

    # One synchronized transfer instead of repeatedly calling .tolist()
    # on GPU tensors. Shape: [n_vars, k].
    signatures = (
        k_freqs.detach()
        .cpu()
        .transpose(0, 1)
        .contiguous()
        .numpy()
        .astype(np.int64, copy=False)
    )

    if not return_hash:
        return signatures

    # Build hashes with the same string representation as hash_func so
    # group ordering and the sampling behavior of a full refresh match
    # original VarDrop.
    hash_values = np.array([
        '-'.join(map(str, row.tolist()))
        for row in signatures
    ])

    return signatures, hash_values


def _groups_from_hash_values(hash_values):
    """Build groups in the same np.unique order as original VarDrop."""
    return [
        np.where(hash_values == value)[0].tolist()
        for value in np.unique(hash_values)
    ]


def _sample_from_groups_original(groups, group_size):
    """
    Apply the original VarDrop within-group sampling rule exactly:
    np.random.choice(..., replace=True).
    """
    sparse_indices = []

    for group_indices in groups:
        sampled = np.random.choice(
            group_indices,
            min(group_size, len(group_indices)),
            replace=True
        )
        sparse_indices.append(sampled)

    return sorted(list(chain.from_iterable(sparse_indices)))


class StabilityGatedSampler:
    """
    Stability-Gated VarDrop (GPT3).

    Goal
    ----
    Preserve the original VarDrop random sampling rule while avoiding a
    full-batch k-DFH computation when the cached frequency grouping is still
    supported by a small deterministic probe from the current batch.

    Decision rule
    -------------
    1. First call of every epoch: full k-DFH refresh.
    2. Later calls: compute k-DFH only on `probe_size` examples.
    3. Measure whether each cached redundancy group remains internally
       coherent under the probe (weighted group purity).
    4. If cached-group purity >= `stability_threshold`, reuse cached groups
       and perform ORIGINAL random sampling inside those groups.
    5. Otherwise, run full k-DFH and refresh the cache.
    6. `max_stale_batches` forces a periodic full refresh as a safety bound.

    The probe indices are deterministic and do not consume NumPy RNG state,
    so the only NumPy randomness remains the original VarDrop sampling.
    """

    def __init__(
        self,
        k,
        group_size,
        freq_list,
        probe_size=4,
        stability_threshold=0.98,
        max_stale_batches=64,
        min_thres=None,
        log_every=100
    ):
        self.k = int(k)
        self.group_size = int(group_size)
        self.freq_list = list(freq_list) if freq_list is not None else None
        self.probe_size = int(probe_size)
        self.stability_threshold = float(stability_threshold)
        self.max_stale_batches = int(max_stale_batches)
        self.min_thres = min_thres
        self.log_every = int(log_every)

        if self.k <= 0:
            raise ValueError('k must be positive.')
        if self.group_size <= 0:
            raise ValueError('group_size must be positive.')
        if self.probe_size <= 0:
            raise ValueError('probe_size must be positive.')
        if not 0.0 <= self.stability_threshold <= 1.0:
            raise ValueError('stability_threshold must be in [0, 1].')
        if self.max_stale_batches < 0:
            raise ValueError('max_stale_batches must be >= 0.')

        self._cache_signatures = None
        self._cache_groups = None
        self._cache_n_vars = None
        self._stale_batches = 0

        self.total_calls = 0
        self.full_refreshes = 0
        self.cache_reuses = 0
        self.unstable_refreshes = 0
        self.forced_refreshes = 0
        self.probe_checks = 0
        self.exact_agreement_sum = 0.0
        self.slot_agreement_sum = 0.0
        self.group_purity_sum = 0.0
        self.last_exact_agreement = None
        self.last_slot_agreement = None
        self.last_group_purity = None

        self.total_sampler_time = 0.0
        self.full_refresh_time = 0.0
        self.probe_time = 0.0
        self.sample_time = 0.0

    def reset_epoch(self):
        """Force the first batch of each epoch to establish a fresh cache."""
        self._cache_signatures = None
        self._cache_groups = None
        self._cache_n_vars = None
        self._stale_batches = 0

    @staticmethod
    def _probe_indices(batch_size, probe_size, device):
        probe_size = min(int(probe_size), int(batch_size))
        if probe_size <= 0:
            raise RuntimeError('Cannot probe an empty batch.')

        # Deterministic, approximately evenly spaced examples.
        # This intentionally avoids consuming the global NumPy RNG state.
        idx = (
            torch.arange(probe_size, device=device, dtype=torch.long)
            * int(batch_size)
            // probe_size
        )
        return idx

    def _sample_cached_groups(self):
        start = time.perf_counter()
        result = _sample_from_groups_original(
            self._cache_groups,
            self.group_size
        )
        self.sample_time += time.perf_counter() - start
        return result

    def _full_refresh(self, x, reason):
        start = time.perf_counter()

        signatures, hash_values = _k_dfh_signature(
            x,
            k=self.k,
            freq_list=self.freq_list,
            min_thres=self.min_thres,
            return_hash=True
        )

        groups = _groups_from_hash_values(hash_values)

        self._cache_signatures = signatures.copy()
        self._cache_groups = groups
        self._cache_n_vars = int(x.shape[-1])
        self._stale_batches = 0

        self.full_refreshes += 1
        if reason == 'unstable':
            self.unstable_refreshes += 1
        elif reason == 'forced':
            self.forced_refreshes += 1

        self.full_refresh_time += time.perf_counter() - start
        return self._sample_cached_groups()

    def _cached_group_purity(self, probe_signatures):
        """
        Measure whether cached redundancy groups remain coherent under
        the current probe. Singleton groups are ignored because reusing a
        singleton cannot merge dissimilar variates.

        For each cached non-singleton group G, compute the fraction of G
        sharing its most common probe signature, then aggregate by group
        size. A value of 1.0 means no cached group is split by the probe.
        """
        coherent = 0
        relevant = 0

        for group in self._cache_groups:
            if len(group) <= 1:
                continue

            rows = probe_signatures[np.asarray(group, dtype=np.int64)]
            _, counts = np.unique(rows, axis=0, return_counts=True)

            coherent += int(counts.max())
            relevant += len(group)

        if relevant == 0:
            return 1.0

        return float(coherent) / float(relevant)

    def _probe_stability(self, x):
        start = time.perf_counter()

        probe_idx = self._probe_indices(
            batch_size=x.shape[0],
            probe_size=self.probe_size,
            device=x.device
        )
        probe_x = x.index_select(0, probe_idx)

        probe_signatures = _k_dfh_signature(
            probe_x,
            k=self.k,
            freq_list=self.freq_list,
            min_thres=self.min_thres,
            return_hash=False
        )

        if probe_signatures.shape != self._cache_signatures.shape:
            exact_agreement = 0.0
            slot_agreement = 0.0
            group_purity = 0.0
        else:
            # Diagnostic only: absolute signature agreement can be low even
            # when the GROUPING relation remains valid.
            exact_agreement = float(
                np.mean(
                    np.all(
                        probe_signatures == self._cache_signatures,
                        axis=1
                    )
                )
            )
            slot_agreement = float(
                np.mean(
                    probe_signatures == self._cache_signatures
                )
            )

            # Gate on partition stability, not absolute frequency labels.
            group_purity = self._cached_group_purity(probe_signatures)

        self.probe_checks += 1
        self.exact_agreement_sum += exact_agreement
        self.slot_agreement_sum += slot_agreement
        self.group_purity_sum += group_purity
        self.last_exact_agreement = exact_agreement
        self.last_slot_agreement = slot_agreement
        self.last_group_purity = group_purity
        self.probe_time += time.perf_counter() - start

        return group_purity, exact_agreement, slot_agreement

    def __call__(self, x):
        total_start = time.perf_counter()
        self.total_calls += 1

        n_vars = int(x.shape[-1])

        # First call, epoch reset, or a shape change: full refresh.
        if (
            self._cache_signatures is None
            or self._cache_groups is None
            or self._cache_n_vars != n_vars
        ):
            result = self._full_refresh(x, reason='initial')

        # Safety bound: never keep a cache indefinitely.
        elif (
            self.max_stale_batches > 0
            and self._stale_batches >= self.max_stale_batches
        ):
            result = self._full_refresh(x, reason='forced')

        else:
            group_purity, _, _ = self._probe_stability(x)

            if group_purity >= self.stability_threshold:
                self.cache_reuses += 1
                self._stale_batches += 1
                result = self._sample_cached_groups()
            else:
                result = self._full_refresh(x, reason='unstable')

        self.total_sampler_time += time.perf_counter() - total_start

        if self.log_every > 0 and self.total_calls % self.log_every == 0:
            print(self.format_status(prefix='[SG-VarDrop]'))

        return result

    def get_stats(self):
        mean_exact = (
            self.exact_agreement_sum / self.probe_checks
            if self.probe_checks > 0 else float('nan')
        )
        mean_slot = (
            self.slot_agreement_sum / self.probe_checks
            if self.probe_checks > 0 else float('nan')
        )
        mean_group_purity = (
            self.group_purity_sum / self.probe_checks
            if self.probe_checks > 0 else float('nan')
        )
        reuse_rate = (
            self.cache_reuses / self.total_calls
            if self.total_calls > 0 else 0.0
        )

        return {
            'calls': self.total_calls,
            'full_refreshes': self.full_refreshes,
            'cache_reuses': self.cache_reuses,
            'reuse_rate': reuse_rate,
            'unstable_refreshes': self.unstable_refreshes,
            'forced_refreshes': self.forced_refreshes,
            'probe_checks': self.probe_checks,
            'mean_exact_agreement': mean_exact,
            'mean_slot_agreement': mean_slot,
            'mean_group_purity': mean_group_purity,
            'last_exact_agreement': self.last_exact_agreement,
            'last_slot_agreement': self.last_slot_agreement,
            'last_group_purity': self.last_group_purity,
            'sampler_time_sec': self.total_sampler_time,
            'full_refresh_time_sec': self.full_refresh_time,
            'probe_time_sec': self.probe_time,
            'sample_time_sec': self.sample_time,
        }

    def format_status(self, prefix='[SG-VarDrop]'):
        stats = self.get_stats()
        mean_exact = stats['mean_exact_agreement']
        mean_slot = stats['mean_slot_agreement']
        mean_purity = stats['mean_group_purity']

        return (
            f"{prefix} calls={stats['calls']} "
            f"refresh={stats['full_refreshes']} "
            f"reuse={stats['cache_reuses']} "
            f"reuse_rate={stats['reuse_rate']:.3f} "
            f"unstable={stats['unstable_refreshes']} "
            f"forced={stats['forced_refreshes']} "
            f"mean_purity={mean_purity:.4f} "
            f"mean_exact={mean_exact:.4f} "
            f"mean_slot={mean_slot:.4f} "
            f"sampler_time={stats['sampler_time_sec']:.3f}s"
        )

