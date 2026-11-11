from __future__ import annotations
import torch
import torch.nn as nn
import torchdiffeq

class ODE(nn.Module):
    """Minimal ODE solver used by the active scNODE baseline."""

    def __init__(self, input_dim: int, drift_net: nn.Module, ode_method: str, step_size: float | None=None):
        super().__init__()
        self.input_dim = input_dim
        self.net = drift_net
        self.ode_method = ode_method
        self.step_size = step_size
        self.rtol = 1e-05
        self.atol = 1e-05

    def forward(self, first_data: torch.Tensor, tp_to_pred: torch.Tensor) -> torch.Tensor:
        options = {'step_size': self.step_size} if self.step_size else None
        pred = torchdiffeq.odeint(self.net.forwardWithTime, first_data, tp_to_pred, method=self.ode_method, rtol=self.rtol, atol=self.atol, options=options)
        return torch.moveaxis(pred, [0, 1, 2], [1, 0, 2])
