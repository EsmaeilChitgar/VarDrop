import copy
import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.OURS import Model


class Cfg:
    seq_len = 96
    pred_len = 96
    output_attention = False
    use_norm = True
    embed = 'timeF'
    d_model = 512
    freq = 'h'
    dropout = 0.1
    class_strategy = 'projection'
    factor = 1
    n_heads = 8
    e_layers = 4
    d_ff = 512
    activation = 'gelu'
    enc_in = 862
    use_lpra = False
    lpra_rank = 32
    lpra_period = 168


def main():
    base_cfg = Cfg()
    lpra_cfg = copy.copy(base_cfg)
    lpra_cfg.use_lpra = True

    torch.manual_seed(2023)
    base = Model(base_cfg)
    rng_after_base = torch.get_rng_state().clone()

    torch.manual_seed(2023)
    enhanced = Model(lpra_cfg)
    rng_after_enhanced = torch.get_rng_state().clone()

    # Enabling LPRA must not perturb GPT3b backbone initialization or global RNG.
    base_state = base.state_dict()
    enhanced_state = enhanced.state_dict()
    for key, value in base_state.items():
        if key == 'lpra_alpha':
            continue
        other = enhanced_state[key]
        if not torch.equal(value, other):
            raise AssertionError('Backbone mismatch at ' + key)
    if not torch.equal(rng_after_base, rng_after_enhanced):
        raise AssertionError('LPRA construction changed global PyTorch RNG state')

    ids = torch.tensor([0, 17, 861], dtype=torch.long)
    phase = torch.tensor([[0, 1, 167], [24, 25, 26]], dtype=torch.long)
    corr = enhanced.lpra_correction(ids, phase)
    if corr.shape != (2, 3, 3):
        raise AssertionError('Unexpected correction shape: {}'.format(tuple(corr.shape)))
    if corr.abs().max().item() != 0.0:
        raise AssertionError('LPRA must be exact-zero at initialization')

    # Zero output must not mean a dead branch: phase factors must get a gradient.
    loss = enhanced.lpra_correction(ids, phase).sum()
    loss.backward()
    phase_grad = enhanced.lpra.phase_factor.weight.grad
    if phase_grad is None or phase_grad.abs().sum().item() == 0.0:
        raise AssertionError('LPRA phase factor received no first-step gradient')

    enhanced.set_lpra_alpha(0.0)
    fallback = enhanced.lpra_alpha * enhanced.lpra_correction(ids, phase)
    if fallback.abs().max().item() != 0.0:
        raise AssertionError('alpha=0 must be exact GPT3b fallback')

    print('[LPRA SANITY] PASS')
    print('  adapter params   : {}'.format(enhanced.lpra.num_parameters))
    print('  zero-init output : exact zero')
    print('  backbone init    : exact GPT3b')
    print('  RNG preservation : PASS')
    print('  first-step grad  : PASS')
    print('  alpha=0 fallback : exact GPT3b')


if __name__ == '__main__':
    main()
