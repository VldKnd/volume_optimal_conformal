import torch
from torch import nn

class MLPVectorField(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
    ):
        super().__init__()

        input_dim = x_dim + y_dim + 1

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        ]

        for _ in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                ]
            )

        layers.append(nn.Linear(hidden_dim, y_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([state, x, t], dim=-1))
