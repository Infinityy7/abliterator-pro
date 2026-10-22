"""Streaming activation capture that never retains full sequence caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .adapters.base import ModelAdapter
from .config import CaptureConfig
from .errors import ArchitectureError
from .runtime import require


@dataclass
class CaptureResult:
    means: dict[str, Any]
    counts: dict[str, int]
    prompt_count: int
    input_receipt: dict[str, Any]


class RunningMean:
    def __init__(self) -> None:
        self.total: Any | None = None
        self.count = 0

    def update(self, values: Any) -> None:
        torch = require("torch")
        if values.ndim != 2:
            raise ArchitectureError(
                f"Pooled activations must be [batch, hidden], got {values.shape}"
            )
        batch_sum = values.detach().to(device="cpu", dtype=torch.float32).sum(dim=0)
        self.total = batch_sum if self.total is None else self.total + batch_sum
        self.count += int(values.shape[0])

    def mean(self) -> Any:
        if self.total is None or self.count == 0:
            raise ArchitectureError("No activations were captured")
        return self.total / self.count


def _first_hidden(value: Any, hidden_size: int) -> Any | None:
    torch = require("torch")
    if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[-1] == hidden_size:
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_hidden(item, hidden_size)
            if found is not None:
                return found
    if isinstance(value, dict):
        for key in ("hidden_states", "last_hidden_state"):
            if key in value:
                found = _first_hidden(value[key], hidden_size)
                if found is not None:
                    return found
    return None


def pool_prompt_tokens(hidden: Any, attention_mask: Any, config: CaptureConfig) -> Any:
    torch = require("torch")
    mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
    if hidden.shape[:2] != mask.shape:
        raise ArchitectureError(
            f"Hidden states {tuple(hidden.shape)} do not align with mask {tuple(mask.shape)}"
        )
    if not bool(mask.any(dim=1).all()):
        raise ArchitectureError("Every prompt must contain at least one unmasked token")
    positions = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask)
    if config.pooling == "last":
        last = positions.masked_fill(~mask, -1).max(dim=1).values
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, last]
    if config.pooling == "mean":
        # Accumulate reduced-precision activations in FP32: their sum can
        # overflow FP16 even when the final mean is representable.
        values = hidden.float() if hidden.dtype in {torch.float16, torch.bfloat16} else hidden
        weights = mask.to(dtype=values.dtype).unsqueeze(-1)
        pooled_mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return pooled_mean.to(dtype=hidden.dtype)
    if config.pooling == "last-k":
        pooled: list[Any] = []
        for row in range(hidden.shape[0]):
            valid = positions[row][mask[row]]
            selected = valid[-config.last_k :]
            pooled.append(hidden[row, selected].mean(dim=0))
        return torch.stack(pooled)
    raise ArchitectureError(f"Unknown pooling mode {config.pooling!r}")


class ActivationCollector:
    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter

    def collect(self, prompts: list[str], config: CaptureConfig) -> CaptureResult:
        config.validate()
        if not prompts:
            raise ArchitectureError("At least one prompt is required for activation capture")
        torch = require("torch")
        layer_indices = (
            list(range(len(self.adapter.layers))) if config.layers is None else list(config.layers)
        )
        invalid = [
            index for index in layer_indices if index < 0 or index >= len(self.adapter.layers)
        ]
        if invalid:
            raise ArchitectureError(f"Capture layer indices out of range: {invalid}")
        accumulators = {f"layer.{index}.{config.point}": RunningMean() for index in layer_indices}
        tokenized_hashes: list[str] = []
        token_count = 0
        rows_at_max_length = 0

        for start in range(0, len(prompts), config.batch_size):
            batch = prompts[start : start + config.batch_size]
            encoded = self.adapter.tokenize(batch, max_length=config.max_length)
            mask = encoded.get("attention_mask")
            if mask is None:
                mask = torch.ones_like(encoded["input_ids"], dtype=torch.long)
            cpu_ids = encoded["input_ids"].detach().to(device="cpu")
            cpu_mask = mask.detach().to(device="cpu", dtype=torch.bool)
            for row in range(cpu_ids.shape[0]):
                token_ids = [int(value) for value in cpu_ids[row][cpu_mask[row]].tolist()]
                tokenized_hashes.append(
                    hashlib.sha256(
                        json.dumps(token_ids, separators=(",", ":")).encode("ascii")
                    ).hexdigest()
                )
                token_count += len(token_ids)
                rows_at_max_length += int(len(token_ids) == config.max_length)
            handles: list[Any] = []
            fired = {index: 0 for index in layer_indices}
            try:
                for index in layer_indices:
                    layer = self.adapter.layers[index]
                    key = f"layer.{index}.{config.point}"
                    if config.point == "resid_pre":

                        def pre_hook(
                            module: Any,
                            args: tuple[Any, ...],
                            kwargs: dict[str, Any],
                            *,
                            _index: int = index,
                            _key: str = key,
                            _mask: Any = mask,
                            _fired: dict[int, int] = fired,
                        ) -> None:
                            hidden = _first_hidden(args, self.adapter.hidden_size)
                            if hidden is None:
                                hidden = _first_hidden(kwargs, self.adapter.hidden_size)
                            if hidden is None:
                                raise ArchitectureError(
                                    f"No residual input found for layer {_index}"
                                )
                            accumulators[_key].update(pool_prompt_tokens(hidden, _mask, config))
                            _fired[_index] += 1

                        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
                    else:

                        def post_hook(
                            module: Any,
                            args: tuple[Any, ...],
                            kwargs: dict[str, Any],
                            output: Any,
                            *,
                            _index: int = index,
                            _key: str = key,
                            _mask: Any = mask,
                            _fired: dict[int, int] = fired,
                        ) -> None:
                            hidden = _first_hidden(output, self.adapter.hidden_size)
                            if hidden is None:
                                raise ArchitectureError(
                                    f"No residual output found for layer {_index}"
                                )
                            accumulators[_key].update(pool_prompt_tokens(hidden, _mask, config))
                            _fired[_index] += 1

                        handles.append(layer.register_forward_hook(post_hook, with_kwargs=True))
                with torch.inference_mode():
                    self.adapter.model(**encoded, use_cache=False, return_dict=True)
            finally:
                for handle in handles:
                    handle.remove()
            wrong = {index: count for index, count in fired.items() if count != 1}
            if wrong:
                raise ArchitectureError(
                    "Decoder-stack validation failed; expected each selected layer to execute "
                    f"once: {wrong}"
                )

        return CaptureResult(
            means={key: accumulator.mean() for key, accumulator in accumulators.items()},
            counts={key: accumulator.count for key, accumulator in accumulators.items()},
            prompt_count=len(prompts),
            input_receipt={
                "format": "token-id-hashes-v1",
                "rows": len(tokenized_hashes),
                "tokens": token_count,
                "rows_at_max_length": rows_at_max_length,
                "max_length": config.max_length,
                "combined_sha256": hashlib.sha256(
                    "\n".join(tokenized_hashes).encode("ascii")
                ).hexdigest(),
            },
        )
