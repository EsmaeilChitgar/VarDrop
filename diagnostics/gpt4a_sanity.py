import argparse
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.OURS import Model


def make_cfg(attn_dim, d_ff):
    return SimpleNamespace(
        seq_len=96, pred_len=96, output_attention=False, use_norm=True,
        embed='timeF', freq='h', dropout=0.0, class_strategy='projection',
        factor=1, d_model=512, n_heads=8, e_layers=4, d_ff=d_ff,
        activation='gelu', gpt4_attn_dim=attn_dim,
    )


def encoder_params(model):
    return sum(p.numel() for p in model.encoder.parameters())


def core_macs(tokens, d_model, attn_dim, d_ff):
    # Q/K/V + O projections, QK^T + AV, and two FFN projections.
    return (
        4 * tokens * d_model * attn_dim
        + 2 * tokens * tokens * attn_dim
        + 2 * tokens * d_model * d_ff
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=int, default=229,
                   help='Representative selected-token count used only for MAC reporting')
    args = p.parse_args()

    torch.manual_seed(2023)
    baseline = Model(make_cfg(None, 512)).eval()
    torch.manual_seed(2023)
    gpt4a = Model(make_cfg(256, 128)).train()

    for i, layer in enumerate(gpt4a.encoder.attn_layers):
        a = layer.attention
        checks = [
            (a.query_projection.in_features, 512, 'q.in'),
            (a.query_projection.out_features, 256, 'q.out'),
            (a.key_projection.out_features, 256, 'k.out'),
            (a.value_projection.out_features, 256, 'v.out'),
            (a.out_projection.in_features, 256, 'o.in'),
            (a.out_projection.out_features, 512, 'o.out'),
            (layer.conv1.in_channels, 512, 'ffn1.in'),
            (layer.conv1.out_channels, 128, 'ffn1.out'),
            (layer.conv2.in_channels, 128, 'ffn2.in'),
            (layer.conv2.out_channels, 512, 'ffn2.out'),
        ]
        for got, expected, name in checks:
            if got != expected:
                raise RuntimeError(f'Layer {i} {name}: got {got}, expected {expected}')

    # Small real forward/backward through the complete forecasting model.
    x = torch.randn(2, 96, 16)
    y = gpt4a(x, None, None, None)
    if tuple(y.shape) != (2, 96, 16):
        raise RuntimeError(f'Unexpected output shape: {tuple(y.shape)}')
    y.square().mean().backward()
    missing = [n for n, q in gpt4a.named_parameters() if q.requires_grad and q.grad is None]
    if missing:
        raise RuntimeError(f'Missing gradients: {missing[:5]}')

    p0 = encoder_params(baseline)
    p1 = encoder_params(gpt4a)
    pred = 100.0 * (p0 - p1) / p0

    base_macs = core_macs(args.tokens, 512, 512, 512)
    gpt4a_macs = core_macs(args.tokens, 512, 256, 128)
    mac_red = 100.0 * (base_macs - gpt4a_macs) / base_macs

    # Guard invalid head partition.
    try:
        Model(make_cfg(250, 128))
        raise RuntimeError('Invalid gpt4_attn_dim=250 was unexpectedly accepted')
    except ValueError:
        pass

    print('[GPT4-A SANITY] PASS')
    print(f'  residual d_model       : 512')
    print(f'  attention total width  : 256 (32 per head x 8)')
    print(f'  FFN hidden width       : 128')
    print(f'  output shape           : {tuple(y.shape)}')
    print(f'  encoder params baseline: {p0:,}')
    print(f'  encoder params GPT4-A  : {p1:,}')
    print(f'  encoder param reduction: {pred:.2f}%')
    print(f'  core MAC estimate @S={args.tokens}: {base_macs:,} -> {gpt4a_macs:,} ({mac_red:.2f}% reduction)')


if __name__ == '__main__':
    main()
