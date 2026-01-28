from __future__ import annotations
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TrialityHeadConfig:
    coord_dim: int = 8
    lowrank_r: int = 4
    use_frame: bool = True


@dataclass
class TrialityCoords:
    geo: torch.Tensor   # [*, coord_dim]
    spec: torch.Tensor  # [*, coord_dim]
    mem: torch.Tensor   # [*, coord_dim]


def apply_frame(x: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """
    Applies a right-multiply frame transform: x @ frame^T

    x:     [..., n]
    frame: [n, n]
    """
    if frame.dim() != 2 or frame.shape[0] != frame.shape[1]:
        raise ValueError("frame must be [n, n].")
    if x.shape[-1] != frame.shape[0]:
        raise ValueError("frame dimension must match x last dim.")
    return torch.matmul(x, frame.transpose(0, 1))


class TrialityCoordinateHead(nn.Module):
    def __init__(self, d_model: int, cfg: TrialityHeadConfig):
        super().__init__()
        self.cfg = cfg

        self.W_geo = nn.Linear(d_model, cfg.coord_dim, bias=False)
        self.W_spec = nn.Linear(d_model, cfg.coord_dim, bias=False)
        self.V_mem = nn.Linear(d_model, cfg.lowrank_r, bias=False)
        self.U_mem = nn.Linear(cfg.lowrank_r, cfg.coord_dim, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        frame_geo: Optional[torch.Tensor] = None,
        frame_spec: Optional[torch.Tensor] = None,
    ) -> TrialityCoords:
        geo = self.W_geo(h)
        spec = self.W_spec(h)
        mem = self.U_mem(self.V_mem(h))

        if self.cfg.use_frame:
            if frame_geo is not None:
                geo = apply_frame(geo, frame_geo)
            if frame_spec is not None:
                spec = apply_frame(spec, frame_spec)

        return TrialityCoords(geo=geo, spec=spec, mem=mem)