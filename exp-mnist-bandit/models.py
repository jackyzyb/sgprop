import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def sparse_policy_parameter_count(
    n_dim: int,
    num_layers: int,
    width: int,
    num_slots: int,
    num_parents: int,
    num_actions: int,
) -> int:
    nd = n_dim * num_parents
    hidden_per_layer = width * num_slots * (nd + n_dim * nd + n_dim)
    action = num_actions * width * n_dim
    return num_layers * hidden_per_layer + action


def mlp_policy_parameter_count(
    input_dim: int,
    hidden_width: int,
    num_layers: int,
    num_actions: int,
) -> int:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    hidden = input_dim * hidden_width + hidden_width
    hidden += (num_layers - 1) * (hidden_width * hidden_width + hidden_width)
    output = hidden_width * num_actions + num_actions
    return hidden + output


class SparsePolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_dim: int,
        num_layers: int,
        width: int,
        num_slots: int,
        num_parents: int,
        num_actions: int,
        residual: bool,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if width < 1:
            raise ValueError("width must be >= 1")
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        if n_dim < 1:
            raise ValueError("n_dim must be >= 1")
        if num_actions < 2:
            raise ValueError("num_actions must be >= 2")
        if num_parents < 1:
            raise ValueError("num_parents must be >= 1")
        if num_parents > width:
            raise ValueError("num_parents must be <= width")

        self.input_dim = input_dim
        self.n_dim = n_dim
        self.num_layers = num_layers
        self.width = width
        self.num_slots = num_slots
        self.num_parents = num_parents
        self.num_actions = num_actions
        self.residual = residual

        self.nd = n_dim * num_parents
        self.hidden_logit_scale = 1.0 / math.sqrt(float(self.nd))
        self.action_logit_scale = 1.0 / math.sqrt(float(width * n_dim))

        # Layer 0 gathers nd raw pixels per neuron.
        first_layer_input_idx = torch.stack(
            [torch.randperm(input_dim)[: self.nd] for _ in range(width)],
            dim=0,
        )
        self.register_buffer("first_layer_input_idx", first_layer_input_idx, persistent=True)

        # Layers 1..L-1: per-child list of parent neuron indices.
        if num_layers > 1:
            parent_idx = torch.stack(
                [
                    torch.stack(
                        [torch.randperm(width)[:num_parents] for _ in range(width)],
                        dim=0,
                    )
                    for _ in range(num_layers - 1)
                ],
                dim=0,
            )
            self.register_buffer("parent_idx", parent_idx, persistent=True)
        else:
            self.register_buffer(
                "parent_idx",
                torch.zeros(0, width, num_parents, dtype=torch.long),
                persistent=True,
            )

        # parent_to_child[l, i, j] = 1 if neuron i at layer l is parent of neuron j at layer l+1.
        if num_layers > 1:
            parent_to_child = torch.zeros(num_layers - 1, width, width, dtype=torch.float32)
            for l in range(num_layers - 1):
                for child in range(width):
                    parent_to_child[l, self.parent_idx[l, child], child] = 1.0
            parent_child_count = parent_to_child.sum(dim=2).clamp_min(1.0)
            self.register_buffer("parent_to_child", parent_to_child, persistent=True)
            self.register_buffer("parent_child_count", parent_child_count, persistent=True)
        else:
            self.register_buffer(
                "parent_to_child",
                torch.zeros(0, width, width, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "parent_child_count",
                torch.zeros(0, width, dtype=torch.float32),
                persistent=True,
            )

        self.hidden_keys = nn.ParameterList(
            [nn.Parameter(torch.empty(width, num_slots, self.nd)) for _ in range(num_layers)]
        )
        self.hidden_A = nn.ParameterList(
            [nn.Parameter(torch.empty(width, num_slots, n_dim, self.nd)) for _ in range(num_layers)]
        )
        self.hidden_b = nn.ParameterList(
            [nn.Parameter(torch.empty(width, num_slots, n_dim)) for _ in range(num_layers)]
        )
        self.action_keys = nn.Parameter(torch.empty(num_actions, width * n_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        key_std = 1.0 / math.sqrt(float(self.nd))
        hidden_affine_std = 1.0 / math.sqrt(float(self.nd))
        action_std = 1.0 / math.sqrt(float(self.nd))

        for keys in self.hidden_keys:
            nn.init.normal_(keys, mean=0.0, std=key_std)
        for A in self.hidden_A:
            nn.init.normal_(A, mean=0.0, std=hidden_affine_std)
        for b in self.hidden_b:
            nn.init.normal_(b, mean=0.0, std=hidden_affine_std)
        nn.init.normal_(self.action_keys, mean=0.0, std=action_std)

    def shared_key_parameters(self) -> Iterable[nn.Parameter]:
        return list(self.hidden_keys) + [self.action_keys]

    def build_layer_input(self, x0: torch.Tensor, prev_hidden: torch.Tensor | None, layer_idx: int) -> torch.Tensor:
        if layer_idx == 0:
            return x0[:, self.first_layer_input_idx]
        if prev_hidden is None:
            raise ValueError("prev_hidden is required for layer_idx > 0")
        layer_parent_idx = self.parent_idx[layer_idx - 1]
        gathered = prev_hidden[:, layer_parent_idx, :]
        return gathered.reshape(gathered.shape[0], self.width, self.nd)

    def hidden_logits(self, layer_input: torch.Tensor, layer_idx: int, detach_keys: bool = False) -> torch.Tensor:
        keys = self.hidden_keys[layer_idx].detach() if detach_keys else self.hidden_keys[layer_idx]
        return torch.einsum("bwi,wmi->bwm", layer_input, keys) * self.hidden_logit_scale

    def hidden_output(
        self,
        layer_input: torch.Tensor,
        prev_hidden: torch.Tensor | None,
        layer_idx: int,
        detach_keys: bool = False,
        detach_outputs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.hidden_logits(layer_input, layer_idx=layer_idx, detach_keys=detach_keys)
        weights = torch.softmax(logits, dim=-1)

        A = self.hidden_A[layer_idx].detach() if detach_outputs else self.hidden_A[layer_idx]
        b = self.hidden_b[layer_idx].detach() if detach_outputs else self.hidden_b[layer_idx]
        # Slot outputs are affine in each neuron's current input, then constrained to [-1, 1].
        y = torch.tanh(torch.einsum("bwi,wmni->bwmn", layer_input, A) + b.unsqueeze(0))
        # y = torch.einsum("bwi,wmni->bwmn", layer_input, A) + b.unsqueeze(0)
        hidden = torch.einsum("bwm,bwmn->bwn", weights, y)

        if self.residual and layer_idx > 0:
            if prev_hidden is None:
                raise ValueError("prev_hidden is required for residual layer updates")
            hidden = torch.tanh(hidden) + prev_hidden
        return hidden, weights

    def forward_hidden_states(self, x0: torch.Tensor) -> List[torch.Tensor]:
        hidden_states: List[torch.Tensor] = []
        prev: torch.Tensor | None = None
        for layer_idx in range(self.num_layers):
            layer_input = self.build_layer_input(x0=x0, prev_hidden=prev, layer_idx=layer_idx)
            hidden, _ = self.hidden_output(
                layer_input=layer_input,
                prev_hidden=prev,
                layer_idx=layer_idx,
            )
            hidden_states.append(hidden)
            prev = hidden
        return hidden_states

    def action_logits(self, last_hidden: torch.Tensor, detach_keys: bool = False) -> torch.Tensor:
        action_keys = self.action_keys.detach() if detach_keys else self.action_keys
        flat = last_hidden.reshape(last_hidden.shape[0], self.width * self.n_dim)
        return (flat @ action_keys.T) * self.action_logit_scale

    def forward_logits(self, x0: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        hidden_states = self.forward_hidden_states(x0)
        logits = self.action_logits(hidden_states[-1])
        return hidden_states, logits

    def sample_action(
        self,
        x0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        hidden_states, logits = self.forward_logits(x0)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return action, logits, hidden_states


class MLPPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_width: int,
        num_layers: int,
        num_actions: int,
        residual: bool,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if hidden_width < 1:
            raise ValueError("hidden_width must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_actions < 2:
            raise ValueError("num_actions must be >= 2")

        self.input_dim = input_dim
        self.hidden_width = hidden_width
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.residual = residual

        layers: List[nn.Linear] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_width))
            in_dim = hidden_width
        self.hidden_layers = nn.ModuleList(layers)
        self.action_layer = nn.Linear(hidden_width, num_actions)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.hidden_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.action_layer.weight)
        nn.init.zeros_(self.action_layer.bias)

    def forward_hidden_states(self, x0: torch.Tensor) -> List[torch.Tensor]:
        hidden_states: List[torch.Tensor] = []
        h = x0
        for layer_idx, layer in enumerate(self.hidden_layers):
            prev_h = h
            h = torch.tanh(layer(h))
            if self.residual and layer_idx > 0:
                h = h + prev_h
            hidden_states.append(h)
        return hidden_states

    def action_logits(self, last_hidden: torch.Tensor) -> torch.Tensor:
        return self.action_layer(last_hidden)

    def forward_logits(self, x0: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        hidden_states = self.forward_hidden_states(x0)
        logits = self.action_logits(hidden_states[-1])
        return hidden_states, logits

    def sample_action(
        self,
        x0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        hidden_states, logits = self.forward_logits(x0)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return action, logits, hidden_states


class SparseSGFunctions(nn.Module):
    def __init__(
        self,
        num_layers: int,
        width: int,
        n_dim: int,
        num_slots: int,
        num_actions: int,
        hidden_input_dim: int,
        action_input_dim: int,
    ):
        super().__init__()
        if n_dim < 1:
            raise ValueError("n_dim must be >= 1")
        if hidden_input_dim < 1:
            raise ValueError("hidden_input_dim must be >= 1")
        if action_input_dim < 1:
            raise ValueError("action_input_dim must be >= 1")

        self.num_layers = num_layers
        self.width = width
        self.n_dim = n_dim
        self.num_slots = num_slots
        self.num_actions = num_actions
        self.hidden_input_dim = hidden_input_dim
        self.action_input_dim = action_input_dim

        self.hidden_A = nn.ParameterList(
            [nn.Parameter(torch.empty(width, num_slots, n_dim, hidden_input_dim)) for _ in range(num_layers)]
        )
        self.hidden_b = nn.ParameterList(
            [nn.Parameter(torch.empty(width, num_slots, n_dim)) for _ in range(num_layers)]
        )
        self.action_A = nn.Parameter(torch.empty(num_actions, action_input_dim, num_actions))
        self.action_b = nn.Parameter(torch.empty(num_actions, num_actions))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        hidden_std = 1.0 / math.sqrt(float(self.hidden_input_dim))
        action_std = 1.0 / math.sqrt(float(self.action_input_dim))
        for A in self.hidden_A:
            nn.init.normal_(A, mean=0.0, std=hidden_std)
        for b in self.hidden_b:
            nn.init.normal_(b, mean=0.0, std=hidden_std)
        nn.init.normal_(self.action_A, mean=0.0, std=action_std)
        nn.init.normal_(self.action_b, mean=0.0, std=action_std)

    def hidden_pred(
        self,
        policy: SparsePolicy,
        layer_input: torch.Tensor,
        layer_idx: int,
        detach_shared_keys: bool = False,
        detach_sg_params: bool = False,
    ) -> torch.Tensor:
        logits = policy.hidden_logits(layer_input, layer_idx=layer_idx, detach_keys=detach_shared_keys)
        weights = torch.softmax(logits, dim=-1)

        if layer_input.shape[-1] != self.hidden_input_dim:
            raise ValueError(
                f"Expected hidden input dim {self.hidden_input_dim}, got {layer_input.shape[-1]}"
            )
        A = self.hidden_A[layer_idx]
        b = self.hidden_b[layer_idx]
        if detach_sg_params:
            A = A.detach()
            b = b.detach()
        slot_preds = torch.einsum("bwi,wmni->bwmn", layer_input, A) + b.unsqueeze(0)
        return torch.einsum("bwm,bwmn->bwn", weights, slot_preds)

    def action_pred(
        self,
        policy: SparsePolicy,
        last_hidden: torch.Tensor,
        detach_shared_keys: bool = False,
        detach_sg_params: bool = False,
    ) -> torch.Tensor:
        logits = policy.action_logits(last_hidden, detach_keys=detach_shared_keys)
        weights = torch.softmax(logits, dim=-1)
        flat = last_hidden.reshape(last_hidden.shape[0], -1)
        if flat.shape[-1] != self.action_input_dim:
            raise ValueError(
                f"Expected action input dim {self.action_input_dim}, got {flat.shape[-1]}"
            )

        A = self.action_A.detach() if detach_sg_params else self.action_A
        b = self.action_b.detach() if detach_sg_params else self.action_b
        slot_preds = torch.einsum("bi,aij->baj", flat, A) + b.unsqueeze(0)
        return torch.einsum("ba,baj->bj", weights, slot_preds)
