
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@dataclass(frozen=True)
class LSTMConfig:
    input_size: int = 128
    hidden_size: int = 256
    num_layers: int = 1
    actor_hidden_size: int = 128

    def __post_init__(self) -> None:
        if self.input_size < 1:
            raise ValueError("input_size must be >= 1")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.actor_hidden_size < 1:
            raise ValueError("actor_hidden_size must be >= 1")

    def to_dict(self) -> Dict[str, int]:
        return {"model": "lstm", **asdict(self)}


class LSTMPolicy(nn.Module):
    """Actor-only LSTM recurrent policy for the POPGym REINFORCE loop."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: LSTMConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or LSTMConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        cfg = self.config
        self.feature_map = nn.Sequential(
            nn.Linear(obs_dim, cfg.input_size),
            nn.LayerNorm(cfg.input_size, elementwise_affine=False),
        )
        self.preprocessor = nn.Sequential(
            nn.Linear(cfg.input_size, cfg.input_size),
            nn.LeakyReLU(inplace=True),
        )
        self.core = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
        )
        self.actor_body = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.actor_hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.actor_hidden_size, cfg.actor_hidden_size),
            nn.LeakyReLU(inplace=True),
        )
        self.actor_head = nn.Linear(cfg.actor_hidden_size, action_dim)

        self.reset_parameters()

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    def reset_parameters(self) -> None:
        self.feature_map.apply(orthogonal_init)
        self.preprocessor.apply(orthogonal_init)
        self.actor_body.apply(orthogonal_init)
        for name, param in self.core.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.normal_(self.actor_head.weight, std=0.01)
        nn.init.zeros_(self.actor_head.bias)

    def initial_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(
            2,
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
        )

    def _unpack_state(self, state: Tensor) -> tuple[Tensor, Tensor]:
        expected_ndim = 4
        if state.ndim != expected_ndim or state.shape[0] != 2:
            raise ValueError(
                "Expected LSTM state shape "
                f"[2, num_layers, batch, hidden_size], got {tuple(state.shape)}"
            )
        return state[0], state[1]

    def _pack_state(self, hidden: Tensor, cell: Tensor) -> Tensor:
        return torch.stack((hidden, cell), dim=0)

    def forward_sequence(
        self,
        obs: Tensor,
        hidden: Tensor,
        episode_starts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Run a [T, B, F] sequence with reset masks at episode boundaries."""
        if obs.ndim != 3:
            raise ValueError(f"Expected obs shape [T, B, F], got {tuple(obs.shape)}")

        x = self.preprocessor(self.feature_map(obs))
        h, c = self._unpack_state(hidden)
        outputs = []
        for step in range(x.shape[0]):
            if episode_starts[step].any():
                keep_mask = (~episode_starts[step]).to(x.dtype).view(1, -1, 1)
                h = h * keep_mask
                c = c * keep_mask
            step_output, (h, c) = self.core(x[step].unsqueeze(0), (h, c))
            outputs.append(step_output)

        features = torch.cat(outputs, dim=0)
        logits = self.actor_head(self.actor_body(features))
        return logits, self._pack_state(h, c)

    def act(
        self,
        obs: Tensor,
        hidden: Tensor,
        episode_starts: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, next_hidden = self.forward_sequence(
            obs.unsqueeze(0),
            hidden,
            episode_starts.unsqueeze(0),
        )
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, next_hidden, dist.entropy()

    def evaluate_actions(
        self,
        obs: Tensor,
        hidden: Tensor,
        episode_starts: Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        logits, next_hidden = self.forward_sequence(obs, hidden, episode_starts)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, next_hidden
