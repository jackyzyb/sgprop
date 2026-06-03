
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@dataclass(frozen=True)
class SparseNetConfig:
    input_embedding_size: int = 128
    input_hidden_size: int = 128
    width: int = 64
    neuron_dim: int = 4
    num_parents: int = 4
    num_slots: int = 4
    actor_num_slots: int = 4
    num_input_neurons: int | None = None
    theta: float = 0.5
    theta_momentum: float = 1.0
    adaptive_theta: bool = False
    training_mode: str = "sgprop"
    sgprop_gradient_mode: str = "prediction"

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("width must be >= 1")
        if self.neuron_dim < 1:
            raise ValueError("neuron_dim must be >= 1")
        if self.num_parents < 1:
            raise ValueError("num_parents must be >= 1")
        if self.num_parents > self.width:
            raise ValueError("num_parents must be <= width")
        if self.num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        if self.actor_num_slots < 1:
            raise ValueError("actor_num_slots must be >= 1")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must be in [0, 1]")
        if not 0.0 <= self.theta_momentum <= 1.0:
            raise ValueError("theta_momentum must be in [0, 1]")
        if self.training_mode not in {"backprop", "sgprop"}:
            raise ValueError("training_mode must be 'backprop' or 'sgprop'")
        if self.sgprop_gradient_mode not in {"prediction", "positive_projection"}:
            raise ValueError(
                "sgprop_gradient_mode must be 'prediction' or 'positive_projection'"
            )
        if self.num_input_neurons is not None and self.num_input_neurons < 0:
            raise ValueError("num_input_neurons must be >= 0")

    def resolved_num_input_neurons(self) -> int:
        if self.num_input_neurons is None:
            return self.width
        return min(self.num_input_neurons, self.width)

    def to_dict(self) -> Dict[str, int | float | str | None]:
        return {
            "model": "sparsenet",
            **asdict(self),
            "num_input_neurons_resolved": self.resolved_num_input_neurons(),
        }


@dataclass
class SparseSequenceCache:
    logits: Tensor
    next_hidden: Tensor
    hidden_states: List[Tensor]
    sg_predictions: List[Tensor]
    neuron_inputs: List[Tensor]


class SparseNetPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: SparseNetConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SparseNetConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        cfg = self.config
        self.width = cfg.width
        self.neuron_dim = cfg.neuron_dim
        self.num_parents = cfg.num_parents
        self.num_slots = cfg.num_slots
        self.actor_num_slots = cfg.actor_num_slots
        self.num_input_neurons = cfg.resolved_num_input_neurons()
        self.state_dim = self.width * self.neuron_dim
        self.neuron_input_dim = self.num_parents * self.neuron_dim + cfg.input_hidden_size

        self.feature_map = nn.Sequential(
            nn.Linear(obs_dim, cfg.input_embedding_size),
            nn.LayerNorm(cfg.input_embedding_size, elementwise_affine=False),
        )
        self.preprocessor = nn.Sequential(
            nn.Linear(cfg.input_embedding_size, cfg.input_hidden_size),
            nn.LeakyReLU(inplace=True),
        )

        parent_idx = torch.stack(
            [
                torch.randperm(self.width)[: self.num_parents]
                for _ in range(self.width)
            ],
            dim=0,
        )
        self.register_buffer("parent_idx", parent_idx, persistent=True)
        input_mask = torch.zeros(self.width, 1)
        input_mask[: self.num_input_neurons] = 1.0
        self.register_buffer("input_mask", input_mask, persistent=True)
        self.register_buffer(
            "theta_state",
            torch.full((self.width,), float(cfg.theta)),
            persistent=True,
        )

        self.neuron_keys = nn.Parameter(
            torch.empty(self.width, self.num_slots, self.neuron_input_dim)
        )
        self.policy_A = nn.Parameter(
            torch.empty(self.width, self.num_slots, self.neuron_dim, self.neuron_input_dim)
        )
        self.policy_b = nn.Parameter(
            torch.empty(self.width, self.num_slots, self.neuron_dim)
        )
        self.sg_A = nn.Parameter(
            torch.empty(self.width, self.num_slots, self.neuron_dim, self.neuron_input_dim)
        )
        self.sg_b = nn.Parameter(
            torch.empty(self.width, self.num_slots, self.neuron_dim)
        )

        self.actor_keys = nn.Parameter(
            torch.empty(action_dim, self.actor_num_slots, self.state_dim)
        )
        self.actor_A = nn.Parameter(
            torch.empty(action_dim, self.actor_num_slots, self.state_dim)
        )
        self.actor_b = nn.Parameter(torch.empty(action_dim, self.actor_num_slots))

        self.hidden_logit_scale = 1.0 / math.sqrt(float(self.neuron_input_dim))
        self.actor_logit_scale = 1.0 / math.sqrt(float(self.state_dim))

        self.reset_parameters()

    @property
    def training_mode(self) -> str:
        return self.config.training_mode

    @property
    def theta(self) -> float:
        return self.config.theta

    @property
    def theta_momentum(self) -> float:
        return self.config.theta_momentum

    def effective_theta(self) -> Tensor:
        if self.training_mode == "backprop":
            return torch.ones_like(self.theta_state)
        return self.theta_state

    def set_theta(self, theta: Tensor) -> None:
        self.theta_state.copy_(theta)

    def sg_predictor_parameters(self) -> List[nn.Parameter]:
        return [self.sg_A, self.sg_b]

    def policy_parameters(self) -> List[nn.Parameter]:
        sg_param_ids = {id(param) for param in self.sg_predictor_parameters()}
        return [
            param
            for param in self.parameters()
            if id(param) not in sg_param_ids
        ]

    def reset_parameters(self) -> None:
        key_std = 1.0 / math.sqrt(float(self.neuron_input_dim))
        slot_std = 1.0 / math.sqrt(float(self.neuron_input_dim))
        actor_std = 1.0 / math.sqrt(float(self.state_dim))

        self.feature_map.apply(orthogonal_init)
        self.preprocessor.apply(orthogonal_init)
        nn.init.normal_(self.neuron_keys, mean=0.0, std=key_std)
        nn.init.normal_(self.policy_A, mean=0.0, std=slot_std)
        nn.init.normal_(self.policy_b, mean=0.0, std=slot_std)
        nn.init.zeros_(self.sg_A)
        nn.init.zeros_(self.sg_b)
        nn.init.normal_(self.actor_keys, mean=0.0, std=actor_std)
        nn.init.normal_(self.actor_A, mean=0.0, std=actor_std)
        nn.init.zeros_(self.actor_b)
        self.theta_state.fill_(float(self.config.theta))

    def initial_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(
            batch_size,
            self.width,
            self.neuron_dim,
            device=device,
        )

    def _embed_obs(self, obs: Tensor) -> Tensor:
        return self.preprocessor(self.feature_map(obs))

    def _build_neuron_input(self, hidden: Tensor, embed: Tensor) -> Tensor:
        parent_hidden = hidden[:, self.parent_idx, :]
        parent_flat = parent_hidden.reshape(hidden.shape[0], self.width, -1)
        masked_embed = embed.unsqueeze(1) * self.input_mask.unsqueeze(0)
        return torch.cat((parent_flat, masked_embed), dim=-1)

    def _routing_weights(self, neuron_input: Tensor) -> Tensor:
        logits = (
            torch.einsum("bwi,wmi->bwm", neuron_input, self.neuron_keys)
            * self.hidden_logit_scale
        )
        return torch.softmax(logits, dim=-1)

    def _policy_hidden(self, neuron_input: Tensor, weights: Tensor) -> Tensor:
        slot_outputs = torch.tanh(
            torch.einsum("bwi,wmni->bwmn", neuron_input, self.policy_A)
            + self.policy_b.unsqueeze(0)
        )
        return torch.einsum("bwm,bwmn->bwn", weights, slot_outputs)

    def _sg_prediction(self, neuron_input: Tensor, weights: Tensor) -> Tensor:
        detached_input = neuron_input.detach()
        detached_weights = weights.detach()
        slot_outputs = (
            torch.einsum("bwi,wmni->bwmn", detached_input, self.sg_A)
            + self.sg_b.unsqueeze(0)
        )
        return torch.einsum("bwm,bwmn->bwn", detached_weights, slot_outputs)

    def _actor_logits(self, hidden: Tensor) -> Tensor:
        flat = hidden.reshape(hidden.shape[0], self.state_dim)
        weights = torch.softmax(
            torch.einsum("bi,ami->bam", flat, self.actor_keys) * self.actor_logit_scale,
            dim=-1,
        )
        slot_logits = (
            torch.einsum("bi,ami->bam", flat, self.actor_A) + self.actor_b.unsqueeze(0)
        )
        return torch.einsum("bam,bam->ba", weights, slot_logits)

    def _step(
        self,
        obs: Tensor,
        hidden: Tensor,
        episode_starts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if episode_starts.any():
            hidden = hidden * (~episode_starts).to(hidden.dtype).view(-1, 1, 1)
        embed = self._embed_obs(obs)
        neuron_input = self._build_neuron_input(hidden, embed)
        weights = self._routing_weights(neuron_input)
        next_hidden = self._policy_hidden(neuron_input, weights)
        sg_prediction = self._sg_prediction(neuron_input, weights)
        return next_hidden, sg_prediction, neuron_input

    def forward_sequence(
        self,
        obs: Tensor,
        hidden: Tensor,
        episode_starts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        cache = self.forward_sequence_with_cache(obs, hidden, episode_starts)
        return cache.logits, cache.next_hidden

    def forward_sequence_with_cache(
        self,
        obs: Tensor,
        hidden: Tensor,
        episode_starts: Tensor,
    ) -> SparseSequenceCache:
        if obs.ndim != 3:
            raise ValueError(f"Expected obs shape [T, B, F], got {tuple(obs.shape)}")
        if hidden.ndim != 3:
            raise ValueError(
                f"Expected hidden shape [B, W, N], got {tuple(hidden.shape)}"
            )

        hidden_states: List[Tensor] = []
        sg_predictions: List[Tensor] = []
        neuron_inputs: List[Tensor] = []
        logits_list: List[Tensor] = []
        next_hidden = hidden

        for step in range(obs.shape[0]):
            next_hidden, sg_pred, neuron_input = self._step(
                obs=obs[step],
                hidden=next_hidden,
                episode_starts=episode_starts[step],
            )
            logits_list.append(self._actor_logits(next_hidden))
            hidden_states.append(next_hidden)
            sg_predictions.append(sg_pred)
            neuron_inputs.append(neuron_input)

        return SparseSequenceCache(
            logits=torch.stack(logits_list, dim=0),
            next_hidden=next_hidden,
            hidden_states=hidden_states,
            sg_predictions=sg_predictions,
            neuron_inputs=neuron_inputs,
        )

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
