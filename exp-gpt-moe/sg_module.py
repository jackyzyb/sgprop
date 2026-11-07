from math import log, sqrt

import torch
from torch import nn
from torch.nn import functional as F


class SyntheticGradientHook:
    """Capture, convexly mix, and score a synthetic gradient prediction."""

    def __init__(self, weight: float) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")

        self.weight = weight
        self.prediction: torch.Tensor | None = None
        self.target: torch.Tensor | None = None

    def mix_prediction(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.weight * target + (1.0 - self.weight) * prediction

    def register(self, output: torch.Tensor, prediction: torch.Tensor) -> None:
        self.prediction = prediction
        self.target = None
        hook_prediction = prediction.detach()

        def hook(gradient: torch.Tensor) -> torch.Tensor:
            self.target = gradient.detach()
            return self.mix_prediction(hook_prediction, gradient)

        output.register_hook(hook)

    def loss(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.prediction is None or self.target is None:
            raise RuntimeError("synthetic gradient target is unavailable")

        target = self.target.detach()
        loss = (self.prediction - target).square().mean()
        with torch.no_grad():
            cosine = F.cosine_similarity(self.prediction, target, dim=-1).mean()
            flat_target = target.reshape(-1, target.size(-1))
            gradient_variance = flat_target.var(dim=0, unbiased=False).mean()
            prediction_norm = self.prediction.norm(dim=-1).mean()
            target_norm = target.norm(dim=-1).mean()
            nonzero_target = target != 0
            same_sign = (self.prediction * target > 0) & nonzero_target
            same_sign_fraction = (
                same_sign.sum() / nonzero_target.sum().clamp_min(1)
            )

        return loss, {
            "cosine": cosine,
            "variance": gradient_variance,
            "prediction_norm": prediction_norm,
            "target_norm": target_norm,
            "same_sign_fraction": same_sign_fraction,
        }


class SGBlock(nn.Module):
    """Per-expert transform block with input-dependent routing."""

    GATE_TYPES = ("softmax", "topk_softmax")
    ACTIVATIONS = ("tanh", "relu", "gelu")

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        expert_dim: int,
        gate: str = "softmax",
        top_k: int = 2,
        temperature: float = 1.0,
        activation: str = "tanh",
        residual: bool = False,
        synthetic_grad: bool = True,
        synthetic_grad_lambda: float = 0.75,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if expert_dim <= 0:
            raise ValueError("expert_dim must be positive")
        if gate not in self.GATE_TYPES:
            raise ValueError(f"gate must be one of {self.GATE_TYPES}")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"activation must be one of {self.ACTIVATIONS}")
        if residual and input_dim != output_dim:
            raise ValueError("residual requires input_dim == output_dim")
        if not 0.0 <= synthetic_grad_lambda <= 1.0:
            raise ValueError("synthetic_grad_lambda must be in [0, 1]")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.gate = gate
        self.top_k = top_k
        self.temperature = temperature
        self.activation = activation
        self.residual = residual
        self.synthetic_grad = synthetic_grad

        self.input_weights = nn.Parameter(
            torch.empty(num_experts, expert_dim, input_dim)
        )
        self.input_biases = nn.Parameter(torch.empty(num_experts, expert_dim))
        self.keys = nn.Parameter(torch.empty(num_experts, input_dim))
        self.key_biases = nn.Parameter(torch.empty(num_experts))
        self.output_weights = nn.Parameter(
            torch.empty(num_experts, output_dim, expert_dim)
        )
        self.output_biases = nn.Parameter(torch.empty(num_experts, output_dim))
        self.synthetic_grad_lambda = synthetic_grad_lambda
        self.synthetic_gradient_hook = SyntheticGradientHook(synthetic_grad_lambda)
        if synthetic_grad:
            self.synthetic_grad_weights = nn.Parameter(
                torch.empty(num_experts, output_dim, expert_dim)
            )
            self.synthetic_grad_biases = nn.Parameter(
                torch.empty(num_experts, output_dim)
            )
        else:
            self.synthetic_grad_weights = None
            self.synthetic_grad_biases = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        input_bound = 1.0 / sqrt(self.input_dim)
        expert_bound = 1.0 / sqrt(self.expert_dim)
        nn.init.uniform_(self.input_weights, -input_bound, input_bound)
        nn.init.uniform_(self.input_biases, -input_bound, input_bound)
        nn.init.normal_(self.keys, std=input_bound)
        nn.init.zeros_(self.key_biases)
        nn.init.uniform_(self.output_weights, -expert_bound, expert_bound)
        nn.init.uniform_(self.output_biases, -expert_bound, expert_bound)
        if self.synthetic_grad:
            nn.init.zeros_(self.synthetic_grad_weights)
            nn.init.zeros_(self.synthetic_grad_biases)

    def routing(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = torch.einsum("bi,ei->be", x, self.keys)
        gate_logits = gate_logits + self.key_biases
        gate_logits = gate_logits / self.temperature

        if self.gate == "softmax":
            scores = gate_logits.softmax(dim=-1)
        elif self.gate == "topk_softmax":
            values, indices = gate_logits.topk(self.top_k, dim=-1)
            scores = torch.zeros_like(gate_logits).scatter(
                dim=-1,
                index=indices,
                src=values.softmax(dim=-1),
            )
        else:
            raise RuntimeError(f"unsupported gate: {self.gate}")

        return gate_logits, scores

    def routing_entropy(self, scores: torch.Tensor) -> torch.Tensor:
        if self.num_experts == 1:
            return scores.new_zeros(())
        probabilities = scores
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        return entropy.mean() / log(self.num_experts)

    def load_balance_loss(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = scores
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        mean_activation = probabilities.mean(dim=0)
        loss = mean_activation.square().sum() * self.num_experts
        return loss, mean_activation

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "tanh":
            return x.tanh()
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "gelu":
            return F.gelu(x)
        raise RuntimeError(f"unsupported activation: {self.activation}")

    def predict_synthetic_gradient(
        self, hidden: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        expert_predictions = torch.einsum(
            "bed,eod->beo", hidden.detach(), self.synthetic_grad_weights
        )
        expert_predictions = expert_predictions + self.synthetic_grad_biases
        return torch.einsum("be,beo->bo", scores.detach(), expert_predictions)

    def mix_synthetic_gradient(
        self, synthetic_grad: torch.Tensor, target_grad: torch.Tensor
    ) -> torch.Tensor:
        return self.synthetic_gradient_hook.mix_prediction(
            synthetic_grad,
            target_grad,
        )

    def register_synthetic_gradient_hook(
        self,
        output: torch.Tensor,
        prediction: torch.Tensor,
    ) -> None:
        self.synthetic_gradient_hook.register(output, prediction)

    def synthetic_gradient_parameters(self):
        if not self.synthetic_grad:
            return
        yield self.synthetic_grad_weights
        yield self.synthetic_grad_biases

    def synthetic_gradient_loss(
        self,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.synthetic_gradient_hook.loss()

    def forward_with_scores(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = torch.einsum("bi,edi->bed", x, self.input_weights)
        transformed = transformed + self.input_biases
        hidden = self.activate(transformed)
        _, scores = self.routing(x)
        expert_outputs = torch.einsum(
            "bed,eod->beo", hidden, self.output_weights
        )
        expert_outputs = expert_outputs + self.output_biases
        output = torch.einsum("be,beo->bo", scores, expert_outputs)

        if self.residual:
            output = output + x

        if self.synthetic_grad and torch.is_grad_enabled() and output.requires_grad:
            prediction = self.predict_synthetic_gradient(hidden, scores)
            self.register_synthetic_gradient_hook(output, prediction)

        return output, scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_scores(x)
        return output


class SGModule(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim: int = 256,
        num_layers: int = 1,
        num_experts: int = 32,
        expert_dim: int = 64,
        gate: str = "softmax",
        top_k: int = 8,
        temperature: float = 1.0,
        activation: str = "tanh",
        residual: bool = False,
        synthetic_grad: bool = True,
        synthetic_grad_lr: float = 1e-3,
        synthetic_grad_lambda: float = 0.75,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if synthetic_grad and synthetic_grad_lr <= 0:
            raise ValueError("synthetic_grad_lr must be positive")

        def make_block(
            block_input_dim: int,
            block_output_dim: int,
            block_residual: bool,
        ) -> SGBlock:
            return SGBlock(
                input_dim=block_input_dim,
                output_dim=block_output_dim,
                num_experts=num_experts,
                expert_dim=expert_dim,
                gate=gate,
                top_k=top_k,
                temperature=temperature,
                activation=activation,
                residual=block_residual,
                synthetic_grad=synthetic_grad,
                synthetic_grad_lambda=synthetic_grad_lambda,
            )

        if num_layers == 1:
            blocks = [make_block(input_dim, output_dim, residual)]
        else:
            blocks = [make_block(input_dim, hidden_dim, False)]
            blocks.extend(
                make_block(hidden_dim, hidden_dim, True)
                for _ in range(num_layers - 2)
            )
            blocks.append(make_block(hidden_dim, output_dim, False))

        self.blocks = nn.ModuleList(blocks)
        self._last_scores: list[torch.Tensor] | None = None
        self.synthetic_grad_optimizer = None
        if synthetic_grad:
            self.synthetic_grad_optimizer = torch.optim.Adam(
                self.synthetic_gradient_parameters(),
                lr=synthetic_grad_lr,
            )

    def synthetic_gradient_parameters(self):
        for block in self.blocks:
            yield from block.synthetic_gradient_parameters()

    def base_parameters(self):
        synthetic_grad_ids = {
            id(param) for param in self.synthetic_gradient_parameters()
        }
        for param in self.parameters():
            if id(param) not in synthetic_grad_ids:
                yield param

    def synthetic_gradient_loss(
        self,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses = []
        stats: dict[str, list[torch.Tensor]] = {}
        for block in self.blocks:
            if not block.synthetic_grad:
                continue
            loss, block_stats = block.synthetic_gradient_loss()
            losses.append(loss)
            for name, value in block_stats.items():
                stats.setdefault(name, []).append(value)

        if not losses:
            raise RuntimeError("synthetic gradients are disabled")

        return torch.stack(losses).mean(), {
            name: torch.stack(values).mean() for name, values in stats.items()
        }

    def update_sg(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.synthetic_grad_optimizer is None:
            raise RuntimeError("synthetic gradients are disabled")

        self.synthetic_grad_optimizer.zero_grad()
        loss, stats = self.synthetic_gradient_loss()
        loss.backward()
        self.synthetic_grad_optimizer.step()
        return loss, stats

    def routing_entropy(self, scores: list[torch.Tensor]) -> torch.Tensor:
        entropies = [
            block.routing_entropy(layer_scores)
            for block, layer_scores in zip(self.blocks, scores)
        ]
        return torch.stack(entropies).mean()

    def load_balance_loss(
        self, scores: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        losses = []
        mean_activations = []
        for block, layer_scores in zip(self.blocks, scores):
            loss, mean_activation = block.load_balance_loss(layer_scores)
            losses.append(loss)
            mean_activations.append(mean_activation)
        return torch.stack(losses).mean(), torch.stack(mean_activations)

    def routing_regularization_loss(
        self,
        routing_entropy_coef: float = 1.0,
        load_balance_coef: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._last_scores is None:
            raise RuntimeError("routing scores are unavailable")

        routing_entropy = self.routing_entropy(self._last_scores)
        load_balance_loss, expert_activation_means = self.load_balance_loss(
            self._last_scores
        )
        loss = (
            routing_entropy_coef * routing_entropy
            + load_balance_coef * load_balance_loss
        )
        return loss, {
            "routing_entropy": routing_entropy,
            "load_balance_loss": load_balance_loss,
            "expert_activation_mean": expert_activation_means.mean(),
        }

    def forward_with_scores(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        scores = []
        for block in self.blocks:
            x, layer_scores = block.forward_with_scores(x)
            scores.append(layer_scores)
        self._last_scores = scores
        return x, scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_scores(x)
        return output
