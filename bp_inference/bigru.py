"""New family: bidirectional GRU, PPG-only regression.

A conv stem downsamples the window, then a bidirectional GRU reads the reduced
sequence both directions and the head pools its outputs, a recurrent baseline
(the prior DS4400 project's family). Single PPG channel in, or +VPG/+APG, two
regression heads.
"""
from pathlib import Path

from torch import nn

from bp_inference import train


class BiGRUNet(nn.Module):
    def __init__(self, in_channels: int, T: int, model_cfg: dict, n_targets: int):
        super().__init__()
        c = int(model_cfg.get("channels", 32))
        hidden = int(model_cfg.get("hidden", 64))
        n_layers = int(model_cfg.get("n_layers", 2))
        ds = int(model_cfg.get("downsample", 4))
        dropout = float(model_cfg.get("dropout", 0.2))
        self.stem = nn.Conv1d(in_channels, c, kernel_size=2 * ds + 1, stride=ds, padding=ds)
        self.gru = nn.GRU(c, hidden, n_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, n_targets)

    def forward(self, x):                            # (B, T, C)
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)   # (B, T', c)
        out, _ = self.gru(h)                                # (B, T', 2*hidden)
        return self.head(out.mean(dim=1))


def _factory(in_channels, T, model_cfg, n_targets):
    return BiGRUNet(in_channels, T, model_cfg, n_targets)


def run_from_dir(run_dir: Path, data_root: Path) -> dict:
    return train.run_from_dir_with_factory(_factory, run_dir, data_root, "bigru")
