import torch
import numpy as np
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

def efficient_sampler_with_mass(x, k, group_size, freq_list, min_thres=None):
    """
    Original VarDrop sampling plus multiplicity/mass information.

    The sampling behavior is intentionally identical to efficient_sampler:
    - k-DFH grouping
    - random sampling inside each group
    - sampling with replacement

    Returns
    -------
    sample_indices : np.ndarray[int64]
        Unique, sorted selected variate indices.

    sample_masses : np.ndarray[float32]
        Mass aligned with sample_indices. For a retained variate from group g,

            mass = |G_g| / |S_g_unique|

        where |G_g| is the original group size and |S_g_unique| is the actual
        number of unique retained variates from that group.

    The masses conserve the original number of variates:

        sum(sample_masses) == x.shape[-1]
    """

    hash_values = k_dominant_frequency_hashing(
        x, k, freq_list, min_thres
    )

    selected_indices = []
    selected_masses = []

    for value in np.unique(hash_values):
        group_indices = np.where(hash_values == value)[0].tolist()
        group_size_original = len(group_indices)

        sampled = np.random.choice(
            group_indices,
            min(group_size, group_size_original),
            replace=True
        )

        # The original experiment applies np.unique after efficient_sampler.
        # We make that explicit here because the mass must correspond to the
        # actual unique tokens passed to the model.
        sampled_unique = np.unique(sampled).astype(np.int64)

        if len(sampled_unique) == 0:
            continue

        mass = float(group_size_original) / float(len(sampled_unique))

        selected_indices.extend(sampled_unique.tolist())
        selected_masses.extend([mass] * len(sampled_unique))

    if len(selected_indices) == 0:
        raise RuntimeError("Mass VarDrop selected zero variates.")

    # Each variate belongs to exactly one k-DFH group, so duplicates across
    # groups cannot occur. Sort indices and masses together to preserve the
    # same channel order as np.unique in the original experiment.
    order = np.argsort(np.asarray(selected_indices, dtype=np.int64))
    sample_indices = np.asarray(selected_indices, dtype=np.int64)[order]
    sample_masses = np.asarray(selected_masses, dtype=np.float32)[order]

    expected_mass = float(x.shape[-1])
    actual_mass = float(sample_masses.sum())

    if not np.isclose(actual_mass, expected_mass, rtol=1e-5, atol=1e-5):
        raise RuntimeError(
            f"Mass conservation failed: expected {expected_mass}, "
            f"got {actual_mass}."
        )

    return sample_indices, sample_masses

