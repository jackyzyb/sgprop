from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm


MASK_TOKEN = "<|mask|>"


@dataclass
class DataConfig:
    block_size: int = 256
    batch_size: int = 32
    mask_prob: float = 0.15
    target_perturb_prob: float = 0.15
    dataset_name: str = "roneneldan/TinyStories"
    train_split: str = "train"
    val_split: str = "validation"
    text_column: str = "text"
    seed: int = 1337
    max_train_examples: int | None = None
    max_val_examples: int | None = None
    hf_cache_dir: str = "data/hf_cache"
    token_cache_dir: str = "data/token_cache"
    use_token_cache: bool = True

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError("mask_prob must be in [0, 1]")
        if not 0.0 <= self.target_perturb_prob <= 1.0:
            raise ValueError("target_perturb_prob must be in [0, 1]")


def make_tokenizer(cache_dir: str | None = "data/hf_cache") -> Any:
    try:
        from transformers import GPT2TokenizerFast
    except ImportError as exc:
        raise ImportError("install transformers to create the GPT-2 tokenizer") from exc

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})
    return tokenizer


class TinyStoriesData:
    """Random chunk sampler for noisy TinyStories language modeling."""

    def __init__(self, config: DataConfig, tokenizer: Any) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.generator = torch.Generator().manual_seed(config.seed)
        self.mask_token_id = tokenizer.mask_token_id

        if self.mask_token_id is None:
            raise ValueError("tokenizer must define a mask token")

        self.train_tokens = self._load_tokens(
            split=config.train_split,
            max_examples=config.max_train_examples,
        )
        self.val_tokens = self._load_tokens(
            split=config.val_split,
            max_examples=config.max_val_examples,
        )

    def _load_tokens(self, split: str, max_examples: int | None) -> torch.Tensor:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("install datasets to load TinyStories") from exc

        cache_path = self._token_cache_path(split, max_examples)
        if cache_path is not None and cache_path.exists():
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        dataset = load_dataset(
            self.config.dataset_name,
            split=split,
            cache_dir=self.config.hf_cache_dir,
        )
        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        tokens: list[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        progress = tqdm(
            dataset,
            desc=f"tokenize {split}",
            dynamic_ncols=True,
            leave=False,
        )
        for example in progress:
            tokens.extend(self.tokenizer.encode(example[self.config.text_column]))
            tokens.append(eos_token_id)

        min_tokens = self.config.block_size + 1
        if len(tokens) < min_tokens:
            raise ValueError(
                f"split {split!r} has {len(tokens)} tokens, need at least {min_tokens}"
            )

        token_tensor = torch.tensor(tokens, dtype=torch.long)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(token_tensor, cache_path)

        return token_tensor

    def _token_cache_path(self, split: str, max_examples: int | None) -> Path | None:
        if not self.config.use_token_cache:
            return None

        key = {
            "dataset_name": self.config.dataset_name,
            "split": split,
            "text_column": self.config.text_column,
            "max_examples": max_examples,
            "tokenizer": getattr(self.tokenizer, "name_or_path", "unknown"),
            "vocab_size": len(self.tokenizer),
            "eos_token_id": self.tokenizer.eos_token_id,
            "mask_token_id": self.mask_token_id,
        }
        digest = sha1(repr(sorted(key.items())).encode("utf-8")).hexdigest()[:16]
        return Path(self.config.token_cache_dir) / f"{digest}.pt"

    def _sample_clean_batch(
        self,
        tokens: torch.Tensor,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(tokens) - self.config.block_size - 1
        starts = torch.randint(
            0,
            max_start + 1,
            (self.config.batch_size,),
            generator=self.generator,
        )
        offsets = torch.arange(self.config.block_size + 1)
        chunks = tokens[starts[:, None] + offsets[None, :]].to(device)
        return chunks[:, :-1], chunks[:, 1:]

    def corrupt_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.config.mask_prob == 0.0:
            return input_ids

        corrupted = input_ids.clone()
        mask = torch.rand_like(corrupted, dtype=torch.float) < self.config.mask_prob
        corrupted[mask] = self.mask_token_id
        return corrupted

    def perturb_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.config.target_perturb_prob == 0.0:
            return targets

        perturbed = targets.clone()
        mask = torch.rand_like(perturbed, dtype=torch.float) < self.config.target_perturb_prob
        random_targets = torch.randint(
            0,
            len(self.tokenizer),
            perturbed.shape,
            device=perturbed.device,
        )
        perturbed[mask] = random_targets[mask]
        return perturbed

    def batch(
        self,
        split: str,
        device: torch.device | str = "cpu",
        noisy: bool = True,
        perturb_targets: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            tokens = self.train_tokens
        elif split in {"val", "validation"}:
            tokens = self.val_tokens
        else:
            raise ValueError("split must be 'train', 'val', or 'validation'")

        input_ids, targets = self._sample_clean_batch(tokens, device)
        if noisy:
            input_ids = self.corrupt_inputs(input_ids)
        if perturb_targets:
            targets = self.perturb_targets(targets)
        return input_ids, targets

    def train_batch(
        self,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.batch("train", device=device, noisy=True, perturb_targets=True)

    def val_batch(
        self,
        device: torch.device | str = "cpu",
        noisy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.batch("val", device=device, noisy=noisy, perturb_targets=False)
