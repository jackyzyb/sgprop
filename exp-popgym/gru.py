
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
class GRUConfig:
    preprocessor_input_size: int = 128
    preprocessor_output_size: int = 128
    hidden_size: int = 256
    actor_hidden_size: int = 128
    num_recurrent_layers: int = 1

    def to_dict(self) -> Dict[str, int]:
        return {"model": "gru", **asdict(self)}


class GRUPolicy(nn.Module):
    """GRU recurrent policy matching the POPGym baseline actor layout."""

    def __init__(self, obs_dim: int, action_dim: int, config: GRUConfig | None = None):
        super().__init__()
        self.config = config or GRUConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        cfg = self.config
        self.feature_map = nn.Sequential(
            nn.Linear(obs_dim, cfg.preprocessor_input_size),
            nn.LayerNorm(cfg.preprocessor_input_size, elementwise_affine=False),
        )
        self.preprocessor = nn.Sequential(
            nn.Linear(cfg.preprocessor_input_size, cfg.preprocessor_output_size),
            nn.LeakyReLU(inplace=True),
        )
        self.core = nn.GRU(
            input_size=cfg.preprocessor_output_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_recurrent_layers,
        )
        self.actor_body = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.actor_hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.actor_hidden_size, cfg.actor_hidden_size),
            nn.LeakyReLU(inplace=True),
        )
        self.actor_head = nn.Linear(cfg.actor_hidden_size, action_dim)

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

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def num_recurrent_layers(self) -> int:
        return self.config.num_recurrent_layers

    def initial_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(
            self.num_recurrent_layers,
            batch_size,
            self.hidden_size,
            device=device,
        )

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
        outputs = []
        next_hidden = hidden
        for step in range(x.shape[0]):
            if episode_starts[step].any():
                keep_mask = (~episode_starts[step]).to(x.dtype).view(1, -1, 1)
                next_hidden = next_hidden * keep_mask
            step_output, next_hidden = self.core(x[step].unsqueeze(0), next_hidden)
            outputs.append(step_output)

        features = torch.cat(outputs, dim=0)
        logits = self.actor_head(self.actor_body(features))
        return logits, next_hidden

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
