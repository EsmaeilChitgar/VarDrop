import torch
import torch.nn as nn


class LowRankPeriodicResidualAdapter(nn.Module):
    """Low-rank sensor x periodic-phase residual correction.

    For channel n and forecast phase p, the correction is
        <sensor_factor[n], phase_factor[p]>.

    phase_factor is zero-initialized, so the adapter contributes exactly zero
    before calibration while still receiving a non-zero first-step gradient.
    """

    def __init__(self, num_channels, period=168, rank=32, init_std=0.02):
        super().__init__()
        self.num_channels = int(num_channels)
        self.period = int(period)
        self.rank = int(rank)

        if self.num_channels <= 0:
            raise ValueError('num_channels must be positive')
        if self.period <= 0:
            raise ValueError('period must be positive')
        if self.rank <= 0:
            raise ValueError('rank must be positive')

        self.sensor_factor = nn.Embedding(self.num_channels, self.rank)
        self.phase_factor = nn.Embedding(self.period, self.rank)

        nn.init.normal_(self.sensor_factor.weight, mean=0.0, std=float(init_std))
        nn.init.zeros_(self.phase_factor.weight)

    def forward(self, channel_ids, phase_indices):
        """
        channel_ids: [N] original channel ids.
        phase_indices: [B, H] periodic phase for each forecast step.
        returns: [B, H, N]
        """
        if channel_ids.dtype != torch.long:
            channel_ids = channel_ids.long()
        if phase_indices.dtype != torch.long:
            phase_indices = phase_indices.long()

        if channel_ids.ndim != 1:
            raise ValueError('channel_ids must have shape [N]')
        if phase_indices.ndim != 2:
            raise ValueError('phase_indices must have shape [B, H]')


        phase_indices = torch.remainder(phase_indices, self.period)
        sensor = self.sensor_factor(channel_ids)      # [N, R]
        phase = self.phase_factor(phase_indices)      # [B, H, R]
        return torch.einsum('bhr,nr->bhn', phase, sensor)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
