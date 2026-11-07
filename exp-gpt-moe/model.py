from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from sg_module import SGModule


@dataclass
class GPTConfig:
    """Configuration for the tiny causal GPT used in the experiment."""

    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.1
    # SG controls: use_sg toggles the replacement, sg_layers selects block indices.
    use_sg: bool = False
    sg_layers: tuple[int, ...] | None = None
    sg_synthetic_grad: bool = False
    sg_hidden_dim: int | None = None
    sg_num_layers: int = 2
    sg_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.sg_num_layers <= 0:
            raise ValueError("sg_num_layers must be positive")
        if self.sg_layers is not None:
            bad_layers = [idx for idx in self.sg_layers if not 0 <= idx < self.n_layer]
            if bad_layers:
                raise ValueError(f"sg_layers out of range: {bad_layers}")

        if self.sg_hidden_dim is None:
            self.sg_hidden_dim = 4 * self.d_model


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention for autoregressive language modeling."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=-1)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = q @ k.transpose(-2, -1) * (self.head_dim**-0.5)
        # Prevent each token from attending to future tokens.
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        att = att.masked_fill(mask == 0, torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class FFN(nn.Module):
    """Standard GPT feed-forward MLP sublayer."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SGFFN(nn.Module):
    """GPT FFN replacement backed by SGModule."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.module = SGModule(
            input_dim=config.d_model,
            output_dim=config.d_model,
            hidden_dim=config.sg_hidden_dim,
            num_layers=config.sg_num_layers,
            synthetic_grad=config.sg_synthetic_grad,
            **config.sg_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        # SGModule operates on rank-2 tensors, so tokens are treated as examples.
        y = self.module(x.reshape(batch_size * seq_len, channels))
        return y.reshape(batch_size, seq_len, channels)


class Block(nn.Module):
    """Transformer block with a swappable FFN sublayer."""

    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        # By default, every FFN is replaced. sg_layers enables location ablations.
        sg_layers = range(config.n_layer) if config.sg_layers is None else config.sg_layers
        use_sg_ffn = config.use_sg and layer_idx in sg_layers

        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = SGFFN(config) if use_sg_ffn else FFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Tiny GPT with optional synthetic gradients in SGModule FFNs."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.d_model),
                "wpe": nn.Embedding(config.block_size, config.d_model),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [Block(config, idx) for idx in range(config.n_layer)]
                ),
                "ln_f": nn.LayerNorm(config.d_model),
            }
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tie input and output token embeddings, following standard GPT practice.
        self.transformer["wte"].weight = self.lm_head.weight

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / (2 * config.n_layer) ** 0.5,
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def sg_modules(self) -> Iterable[SGModule]:
        """Iterate over SGModule instances inserted into transformer blocks."""

        for module in self.modules():
            if isinstance(module, SGFFN):
                yield module.module

    def synthetic_gradient_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters used only by SG synthetic-gradient predictors."""

        for module in self.sg_modules():
            yield from module.synthetic_gradient_parameters()

    def base_parameters(self) -> Iterable[nn.Parameter]:
        """Model parameters for the main optimizer, excluding SG predictors."""

        sg_param_ids = {id(param) for param in self.synthetic_gradient_parameters()}
        for param in self.parameters():
            if id(param) not in sg_param_ids:
                yield param

    def set_synthetic_gradient_training(self, enabled: bool) -> None:
        """Enable predictor graphs only during the SG phase."""

        for param in self.synthetic_gradient_parameters():
            param.requires_grad_(enabled)

    def update_sg_modules(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Update all enabled SG predictors after the main backward pass."""

        return self._update_synthetic_gradient_modules(
            self.sg_modules(),
            "no SG modules with enabled synthetic gradients found",
        )

    @staticmethod
    def _update_synthetic_gradient_modules(
        modules: Iterable,
        empty_error: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        losses = []
        stats: dict[str, list[torch.Tensor]] = {}
        for module in modules:
            if module.synthetic_grad_optimizer is None:
                continue
            loss, module_stats = module.update_sg()
            losses.append(loss.detach())
            for name, value in module_stats.items():
                stats.setdefault(name, []).append(value.detach())

        if not losses:
            raise RuntimeError(empty_error)

        return torch.stack(losses).mean(), {
            name: torch.stack(values).mean() for name, values in stats.items()
        }

    def sg_regularization_loss(
        self,
        routing_entropy_coef: float = 1.0,
        load_balance_coef: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Aggregate routing regularization terms from all SG modules."""

        losses = []
        stats: dict[str, list[torch.Tensor]] = {}
        for module in self.sg_modules():
            loss, module_stats = module.routing_regularization_loss(
                routing_entropy_coef=routing_entropy_coef,
                load_balance_coef=load_balance_coef,
            )
            losses.append(loss)
            for name, value in module_stats.items():
                stats.setdefault(name, []).append(value)

        if not losses:
            zero = next(self.parameters()).new_zeros(())
            return zero, {
                "routing_entropy": zero,
                "load_balance_loss": zero,
                "expert_activation_mean": zero,
            }

        return torch.stack(losses).mean(), {
            name: torch.stack(values).mean() for name, values in stats.items()
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )

        # Token and position embeddings form the transformer input stream.
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.transformer["wte"](input_ids) + self.transformer["wpe"](positions)
        x = self.transformer["drop"](x)
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # -100 labels are ignored, matching PyTorch language-modeling convention.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss
