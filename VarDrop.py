import torch
import numpy as np
import time
import os
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
# GPT3b: Exact Fast k-DFH VarDrop
# ============================================================

# ============================================================
# GPT3c: Anchor-Set k-DFH
# ============================================================

def _canonicalize_anchor_set(k_freqs, k_amps):
    """
    Keep the strongest dominant frequency as an ordered anchor, but treat
    ranks 2..k as an unordered set.

    This removes a brittle distinction in ranked k-DFH: two variates can have
    the same dominant-frequency support and the same strongest period, yet be
    assigned to different groups only because weaker peaks swap amplitude
    order from one batch to another.

    k_amps follow the same permutation so min_thres semantics stay aligned.
    """
    if k_freqs.shape[0] <= 2:
        return k_freqs, k_amps

    tail_order = torch.argsort(k_freqs[1:], dim=0)
    tail_freqs = torch.gather(k_freqs[1:], 0, tail_order)
    tail_amps = torch.gather(k_amps[1:], 0, tail_order)

    k_freqs = torch.cat([k_freqs[:1], tail_freqs], dim=0)
    k_amps = torch.cat([k_amps[:1], tail_amps], dim=0)
    return k_freqs, k_amps


def _resolve_gpt3c_hash_mode(hash_mode=None):
    """Resolve GPT3b ranked mode vs GPT3c Anchor-Set mode."""
    if hash_mode is None:
        hash_mode = os.environ.get('GPT3C_HASH_MODE', 'anchor_set')

    hash_mode = str(hash_mode).strip().lower()
    aliases = {
        'ranked': 'ranked',
        'gpt3b': 'ranked',
        'exact': 'ranked',
        'anchor_set': 'anchor_set',
        'anchor': 'anchor_set',
        'gpt3c': 'anchor_set',
    }
    if hash_mode not in aliases:
        raise ValueError(
            f"Unknown GPT3C_HASH_MODE={hash_mode!r}. "
            "Use 'ranked' or 'anchor_set'."
        )
    return aliases[hash_mode]


def k_dominant_frequency_hashing_fast_exact(
    batch_x,
    k,
    freq_list=None,
    min_thres=None,
    hash_mode=None
):
    """
    Exact k-DFH equivalent to k_dominant_frequency_hashing(), but avoids
    repeatedly synchronizing GPU tensors through per-variate .tolist() calls.

    IMPORTANT:
        - FFT, mean spectrum, top-k, thresholding, string hash format,
          group ordering, and random sampling semantics are unchanged.
        - The only optimization is one batched device-to-CPU transfer of the
          [k, n_vars] top-k frequency tensor before Python string construction.

    GPT3c extension:
        - hash_mode='ranked' is exact GPT3b behavior.
        - hash_mode='anchor_set' preserves rank-1 and canonicalizes ranks 2..k.
    """
    x_amp = torch.fft.rfft(batch_x, dim=1).abs()
    x_amp = torch.mean(x_amp, dim=0)

    if freq_list is not None:
        freq_list = list(freq_list)
        k_amps, k_freqs = torch.topk(x_amp[freq_list], k=k, dim=0)
        # Preserve original VarDrop indexing semantics exactly.
        k_freqs += freq_list[0]
    else:
        k_amps, k_freqs = torch.topk(x_amp, k=k, dim=0)

    hash_mode = _resolve_gpt3c_hash_mode(hash_mode)
    if hash_mode == 'anchor_set':
        k_freqs, k_amps = _canonicalize_anchor_set(k_freqs, k_amps)

    if min_thres is not None:
        k_freqs = k_freqs * (k_amps >= min_thres)
        k_freqs += (k_amps < min_thres) * 99

    # Original hash_func() calls .tolist() once per variate on k_freqs.T.
    # On CUDA that can induce many small synchronizations/transfers.
    # GPT3b performs ONE synchronized transfer for the whole tensor.
    k_freqs_cpu = (
        k_freqs.detach()
        .cpu()
        .transpose(0, 1)
        .contiguous()
        .numpy()
    )

    # Keep the exact original string representation and np.unique ordering.
    hash_values = np.array([
        '-'.join(map(str, row.tolist()))
        for row in k_freqs_cpu
    ])

    return hash_values


def efficient_sampler_fast_exact(
    x,
    k,
    group_size,
    freq_list,
    min_thres=None,
    return_group=False,
    hash_mode=None,
    return_stats=False
):
    """
    Exact-behavior fast implementation of original VarDrop sampling.

    This function intentionally preserves:
        1. k-DFH definition,
        2. string hash representation,
        3. np.unique group ordering,
        4. np.random.choice(..., replace=True),
        5. sorted flattened output.

    It changes only how the top-k frequency tensor is transferred from the
    device before hash construction.
    """
    hash_values = k_dominant_frequency_hashing_fast_exact(
        x,
        k=k,
        freq_list=freq_list,
        min_thres=min_thres,
        hash_mode=hash_mode
    )

    hash_mode = _resolve_gpt3c_hash_mode(hash_mode)
    sparse_indices = []

    if return_group:
        group_dict = {}

    unique_hash_values = np.unique(hash_values)

    for value in unique_hash_values:
        group_indices = np.where(hash_values == value)[0].tolist()
        group_indices = np.random.choice(
            group_indices,
            min(group_size, len(group_indices)),
            replace=True
        )
        sparse_indices.append(group_indices)

        if return_group:
            group_dict[value] = group_indices

    sample_indices = sorted(list(chain.from_iterable(sparse_indices)))

    stats = {
        'hash_mode': hash_mode,
        'n_groups': int(len(unique_hash_values)),
        'n_unique_tokens': int(len(set(sample_indices))),
    }

    if return_group:
        if return_stats:
            return sample_indices, group_dict, stats
        return sample_indices, group_dict

    if return_stats:
        return sample_indices, stats

    return sample_indices


class ExactFastVarDropSampler:
    """
    Thin measurement wrapper around efficient_sampler_fast_exact().

    There is NO cache, probe, threshold, stale state, or approximation.
    Every batch recomputes full k-DFH exactly as original VarDrop.
    """

    def __init__(
        self,
        k,
        group_size,
        freq_list,
        min_thres=None,
        log_every=100
    ):
        self.k = int(k)
        self.group_size = int(group_size)
        self.freq_list = list(freq_list) if freq_list is not None else None
        self.min_thres = min_thres
        self.log_every = int(log_every)
        self.hash_mode = _resolve_gpt3c_hash_mode()

        if self.k <= 0:
            raise ValueError('k must be positive.')
        if self.group_size <= 0:
            raise ValueError('group_size must be positive.')

        self.total_calls = 0
        self.total_sampler_time = 0.0
        self.total_groups = 0
        self.total_unique_tokens = 0

    def __call__(self, x):
        start = time.perf_counter()

        result, sampler_stats = efficient_sampler_fast_exact(
            x,
            k=self.k,
            group_size=self.group_size,
            freq_list=self.freq_list,
            min_thres=self.min_thres,
            hash_mode=self.hash_mode,
            return_stats=True
        )

        self.total_calls += 1
        self.total_sampler_time += time.perf_counter() - start
        self.total_groups += sampler_stats['n_groups']
        self.total_unique_tokens += sampler_stats['n_unique_tokens']

        if self.log_every > 0 and self.total_calls % self.log_every == 0:
            print(self.format_status())

        return result

    def get_stats(self):
        mean_ms = (
            1000.0 * self.total_sampler_time / self.total_calls
            if self.total_calls > 0 else 0.0
        )
        mean_groups = (
            self.total_groups / self.total_calls
            if self.total_calls > 0 else 0.0
        )
        mean_unique_tokens = (
            self.total_unique_tokens / self.total_calls
            if self.total_calls > 0 else 0.0
        )
        return {
            'calls': self.total_calls,
            'sampler_time_sec': self.total_sampler_time,
            'mean_sampler_ms': mean_ms,
            'hash_mode': self.hash_mode,
            'mean_groups': mean_groups,
            'mean_unique_tokens': mean_unique_tokens,
        }

    def format_status(self, prefix='[FastVarDrop]'):
        stats = self.get_stats()
        return (
            f"{prefix} hash={stats['hash_mode']} "
            f"calls={stats['calls']} "
            f"sampler_time={stats['sampler_time_sec']:.3f}s "
            f"mean_sampler={stats['mean_sampler_ms']:.3f}ms "
            f"mean_groups={stats['mean_groups']:.2f} "
            f"mean_unique_tokens={stats['mean_unique_tokens']:.2f}"
        )
