"""Backbones: FCN (UCR and real-world datasets) and the CA-TCC encoder (UEA datasets)."""

import torch
import torch.nn as nn

__all__ = ["FCN", "CATCCEncoder", "build_backbone"]


class FCN(nn.Module):
    """Fully convolutional network (Wang et al., 2017): three conv-BN-ReLU blocks and
    global average pooling over time."""

    def __init__(self, in_channels, n_classes, dropout=0.05, **_):
        super().__init__()

        def block(ci, co, k):
            return nn.Sequential(nn.Conv1d(ci, co, k, padding=k // 2), nn.BatchNorm1d(co), nn.ReLU())

        self.net = nn.Sequential(block(in_channels, 128, 8), block(128, 256, 5), block(256, 128, 3))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):                                    # (B, T, C) -> (B, K)
        return self.head(self.drop(self.net(x.transpose(1, 2)).mean(dim=2)))


class CATCCEncoder(nn.Module):
    """The convolutional encoder of CA-TCC (Eldele et al., 2023) with a linear head on the
    flattened features (architecture only, no self-supervised pre-training)."""

    def __init__(self, in_channels, n_classes, seq_len, dropout=0.05, hidden=128, **_):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv1d(in_channels, 32, 8, 1, padding=4, bias=False), nn.BatchNorm1d(32),
                                    nn.ReLU(), nn.MaxPool1d(2, 2, padding=1), nn.Dropout(dropout))
        self.block2 = nn.Sequential(nn.Conv1d(32, 64, 8, 1, padding=4, bias=False), nn.BatchNorm1d(64),
                                    nn.ReLU(), nn.MaxPool1d(2, 2, padding=1))
        self.block3 = nn.Sequential(nn.Conv1d(64, hidden, 8, 1, padding=4, bias=False), nn.BatchNorm1d(hidden),
                                    nn.ReLU(), nn.MaxPool1d(2, 2, padding=1))
        with torch.no_grad():
            width = self._trunk(torch.zeros(1, in_channels, seq_len)).flatten(1).shape[1]
        self.head = nn.Linear(width, n_classes)

    def _trunk(self, x):
        return self.block3(self.block2(self.block1(x)))

    def forward(self, x):                                    # (B, T, C) -> (B, K)
        return self.head(self._trunk(x.transpose(1, 2)).flatten(1))


def build_backbone(name, in_channels, n_classes, seq_len):
    if name == "fcn":
        return FCN(in_channels, n_classes)
    if name == "catcc":
        return CATCCEncoder(in_channels, n_classes, seq_len)
    raise ValueError(f"unknown backbone {name!r} (fcn | catcc)")
