import torch
from torch import nn


class MLPVectorField(nn.Module):

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        *,
        state_dim: int | None = None,
        output_dim: int | None = None,
        time_dim: int = 1,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = y_dim if state_dim is None else state_dim
        self.output_dim = y_dim if output_dim is None else output_dim
        self.time_dim = time_dim

        input_dim = x_dim + self.state_dim + time_dim

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        ]

        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            ])

        layers.append(nn.Linear(hidden_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def _time_feature(
        self,
        state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.time_dim == 0:
            return state.new_empty(*state.shape[:-1], 0)

        t = torch.as_tensor(t, device=state.device, dtype=state.dtype)

        if t.ndim == 0:
            return t.reshape(1).expand(*state.shape[:-1], self.time_dim)

        if t.shape == state.shape[:-1]:
            t = t.unsqueeze(-1)

        return t.expand(*state.shape[:-1], self.time_dim)

    def forward(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([state, x, self._time_feature(state, t)], dim=-1))
