from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from abliterator.adapters.base import ProjectionTarget
from abliterator.adapters.huggingface import HuggingFaceAdapter
from abliterator.capture import ActivationCollector, pool_prompt_tokens
from abliterator.config import ArchitectureSpec, CaptureConfig, ModelLoadConfig
from abliterator.directions import (
    direction_leakage,
    estimate_directions,
    orthonormalize_directions,
    project_activations,
)
from abliterator.errors import ArchitectureError, ConfigurationError, QuantizationError
from abliterator.interventions import ResidualProjector, WeightAblator
from abliterator.quantization import describe_module_quantization


def test_direction_estimation_and_activation_projection() -> None:
    target = {"layer.1.resid_pre": torch.tensor([2.0, 0.0, 0.0])}
    contrast = {"layer.1.resid_pre": torch.tensor([0.0, 0.0, 0.0])}
    direction = estimate_directions(target, contrast)["layer.1.resid_pre"]
    activation = torch.tensor([[[3.0, 4.0, 0.0]]])
    projected = project_activations(activation, direction)
    torch.testing.assert_close(projected, torch.tensor([[[0.0, 4.0, 0.0]]]))


def test_rank_revealing_basis_handles_a_zero_leading_column() -> None:
    matrix = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    basis = orthonormalize_directions(matrix)
    expected = torch.tensor([1.0, 1.0, 0.0]) / 2**0.5
    assert basis.shape == (3, 1)
    torch.testing.assert_close(torch.abs(basis[:, 0]), expected)


def test_direction_basis_is_invariant_to_column_magnitude() -> None:
    matrix = torch.tensor([[1.0e9, 0.0], [0.0, 1.0], [0.0, 0.0]])
    basis = orthonormalize_directions(matrix)
    projector = basis @ basis.transpose(0, 1)
    torch.testing.assert_close(projector, torch.diag(torch.tensor([1.0, 1.0, 0.0])))


class FakeAttention(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.o_proj = torch.nn.Linear(hidden, hidden, bias=False)


class FakeMlp(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.down_proj = torch.nn.Linear(hidden * 2, hidden, bias=False)


class FakeLinearAttention(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.out_proj = torch.nn.Linear(hidden, hidden, bias=False)


class FakeExperts(torch.nn.Module):
    def __init__(self, experts: int, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.down_proj = torch.nn.Parameter(torch.randn(experts, hidden, intermediate))


class FakeBlock(torch.nn.Module):
    def __init__(self, hidden: int, experts: bool = False) -> None:
        super().__init__()
        self.self_attn = FakeAttention(hidden)
        self.mlp = FakeMlp(hidden)
        if experts:
            self.mlp.experts = FakeExperts(3, hidden, hidden * 2)

    def forward(self, hidden_states: torch.Tensor, **_: object) -> tuple[torch.Tensor]:
        return (hidden_states,)


class FakeCustomBlock(FakeBlock):
    def __init__(self, hidden: int) -> None:
        super().__init__(hidden)
        self.custom_attention = torch.nn.Linear(hidden, hidden, bias=False)
        self.custom_mlp = torch.nn.Linear(hidden * 2, hidden, bias=False)


class FakeQwenNextBlock(torch.nn.Module):
    def __init__(self, hidden: int, layer_type: str) -> None:
        super().__init__()
        if layer_type == "linear_attention":
            self.linear_attn = FakeLinearAttention(hidden)
        elif layer_type == "full_attention":
            self.self_attn = FakeAttention(hidden)
        else:
            raise ValueError(layer_type)
        self.mlp = FakeMlp(hidden)

    def forward(self, hidden_states: torch.Tensor, **_: object) -> tuple[torch.Tensor]:
        return (hidden_states,)


class FakeModel(torch.nn.Module):
    def __init__(self, experts: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3",
            hidden_size=4,
            num_hidden_layers=2,
            architectures=["FakeForCausalLM"],
        )
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([FakeBlock(4), FakeBlock(4, experts=experts)])
        self.embed = torch.nn.Embedding(16, 4)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(last_hidden_state=hidden)


class FakeCustomModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.model.layers = torch.nn.ModuleList([FakeCustomBlock(4), FakeCustomBlock(4)])


class FakeQwenNextModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        layer_types = ["linear_attention", "full_attention"]
        self.config = SimpleNamespace(
            model_type="qwen3_next",
            hidden_size=4,
            num_hidden_layers=2,
            layer_types=layer_types,
            architectures=["FakeQwen3NextForCausalLM"],
        )
        self.model.layers = torch.nn.ModuleList(
            [FakeQwenNextBlock(4, layer_type) for layer_type in layer_types]
        )


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    init_kwargs: dict[str, str] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        thinking: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        assert not enable_thinking
        assert not thinking
        return f"user {messages[0]['content']} assistant"

    def __call__(self, prompts: list[str], **_: object) -> dict[str, torch.Tensor]:
        lengths = [max(1, min(4, len(prompt.split()))) for prompt in prompts]
        width = max(lengths)
        ids = torch.zeros((len(prompts), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, length in enumerate(lengths):
            ids[row, -length:] = torch.arange(1, length + 1)
            mask[row, -length:] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_huggingface_layout_discovers_dense_and_packed_expert_targets() -> None:
    adapter = HuggingFaceAdapter(FakeModel(experts=True), FakeTokenizer())
    targets = adapter.projection_targets()
    assert sum(target.kind == "attention" for target in targets) == 2
    assert sum(target.kind == "mlp" for target in targets) == 2
    experts = [target for target in targets if target.kind == "expert"]
    assert len(experts) == 1
    assert experts[0].shape == (3, 4, 8)
    assert experts[0].output_axis == 1


def test_explicit_layer_path_is_exclusive_and_fails_closed() -> None:
    model = FakeModel()
    with pytest.raises(ArchitectureError, match="does not exist"):
        HuggingFaceAdapter(
            model,
            FakeTokenizer(),
            ArchitectureSpec(layer_path="decoder.layers"),
        )


def test_declared_decoder_depth_must_match_discovered_stack() -> None:
    model = FakeModel()
    model.config.num_hidden_layers = 3
    with pytest.raises(ArchitectureError, match="declares 3 decoder layers"):
        HuggingFaceAdapter(model, FakeTokenizer())
    with pytest.raises(ArchitectureError, match="configured path.*contains 2"):
        HuggingFaceAdapter(
            model,
            FakeTokenizer(),
            ArchitectureSpec(layer_path="model.layers"),
        )


def test_known_layout_requires_sink_coverage_in_every_selected_layer() -> None:
    model = FakeModel()
    del model.model.layers[1].self_attn.o_proj
    adapter = HuggingFaceAdapter(model, FakeTokenizer())
    with pytest.raises(ArchitectureError, match=r"attention sink missing in layers \[1\]"):
        adapter.projection_targets()
    with pytest.raises(ArchitectureError, match="not a non-empty layer sequence"):
        HuggingFaceAdapter(
            model,
            FakeTokenizer(),
            ArchitectureSpec(layer_path="embed"),
        )


def test_custom_projection_patterns_are_exclusive_per_category() -> None:
    adapter = HuggingFaceAdapter(
        FakeCustomModel(),
        FakeTokenizer(),
        ArchitectureSpec(
            attention_output_patterns=("custom_attention",),
            mlp_output_patterns=("custom_mlp",),
        ),
    )
    targets = adapter.projection_targets()
    assert len(targets) == 4
    assert {target.name.rsplit(".", 1)[-1] for target in targets} == {
        "custom_attention",
        "custom_mlp",
    }

    attention_only = HuggingFaceAdapter(
        FakeCustomModel(),
        FakeTokenizer(),
        ArchitectureSpec(attention_output_patterns=("custom_attention",)),
    ).projection_targets()
    assert sum(target.name.endswith("custom_attention") for target in attention_only) == 2
    assert sum(target.name.endswith("mlp.down_proj") for target in attention_only) == 2
    assert not any(target.name.endswith("self_attn.o_proj") for target in attention_only)

    expert_model = FakeModel(experts=True)
    for layer in expert_model.model.layers:
        layer.custom_mlp = torch.nn.Linear(8, 4, bias=False)
    mlp_only = HuggingFaceAdapter(
        expert_model,
        FakeTokenizer(),
        ArchitectureSpec(mlp_output_patterns=("custom_mlp",)),
    ).projection_targets()
    assert sum(target.name.endswith("custom_mlp") for target in mlp_only) == 2
    assert not any(target.kind == "expert" for target in mlp_only)
    assert not any(target.name.endswith("mlp.down_proj") for target in mlp_only)


def test_qwen3_next_discovers_and_validates_each_attention_layer_type() -> None:
    model = FakeQwenNextModel()
    adapter = HuggingFaceAdapter(model, FakeTokenizer())
    targets = adapter.projection_targets(kinds=("attention",))
    assert [target.name.rsplit(".layers.", 1)[-1] for target in targets] == [
        "0.linear_attn.out_proj",
        "1.self_attn.o_proj",
    ]

    model.config.layer_types[1] = "linear_attention"
    mismatched = HuggingFaceAdapter(model, FakeTokenizer())
    with pytest.raises(ArchitectureError, match="expected output projection"):
        mismatched.projection_targets(kinds=("attention",))


@pytest.mark.parametrize(
    ("tokenizer", "tokenizer_revision", "expected_revision"),
    [
        (None, None, "model-commit"),
        ("organization/other-tokenizer", None, None),
        ("organization/other-tokenizer", "tokenizer-commit", "tokenizer-commit"),
    ],
)
def test_tokenizer_revision_is_independent_for_a_separate_repository(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: str | None,
    tokenizer_revision: str | None,
    expected_revision: str | None,
) -> None:
    tokenizer_calls: list[tuple[str, dict[str, object]]] = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(identifier: str, **kwargs: object) -> FakeTokenizer:
            tokenizer_calls.append((identifier, kwargs))
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(identifier: str, **kwargs: object) -> FakeModel:
            assert identifier == "organization/model"
            assert kwargs["revision"] == "model-commit"
            return FakeModel()

    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeAutoTokenizer,
        AutoModelForCausalLM=FakeAutoModel,
    )
    monkeypatch.setattr(
        "abliterator.adapters.huggingface.require",
        lambda name: fake_transformers if name == "transformers" else torch,
    )

    HuggingFaceAdapter.from_pretrained(
        ModelLoadConfig(
            model="organization/model",
            revision="model-commit",
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
            dtype="float32",
            device_map=None,
        )
    )
    assert tokenizer_calls[0][1]["revision"] == expected_revision


def test_image_text_loader_uses_and_preserves_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = SimpleNamespace(tokenizer=FakeTokenizer())

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(identifier: str, **kwargs: object) -> object:
            assert identifier == "organization/model"
            assert kwargs["revision"] == "model-commit"
            return processor

    class FakeAutoImageModel:
        @staticmethod
        def from_pretrained(identifier: str, **_: object) -> FakeModel:
            assert identifier == "organization/model"
            return FakeModel()

    class UnexpectedAutoTokenizer:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> object:
            raise AssertionError("processor tokenizer should be reused")

    fake_transformers = SimpleNamespace(
        AutoProcessor=FakeAutoProcessor,
        AutoTokenizer=UnexpectedAutoTokenizer,
        AutoModelForImageTextToText=FakeAutoImageModel,
    )
    monkeypatch.setattr(
        "abliterator.adapters.huggingface.require",
        lambda name: fake_transformers if name == "transformers" else torch,
    )
    adapter = HuggingFaceAdapter.from_pretrained(
        ModelLoadConfig(
            model="organization/model",
            revision="model-commit",
            model_class="image-text-to-text",
            dtype="float32",
            device_map=None,
        )
    )
    assert adapter.tokenizer is processor.tokenizer
    assert adapter.serialization_components == (processor,)


def test_image_text_loader_uses_causal_auto_mapping_when_image_mapping_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = SimpleNamespace(tokenizer=FakeTokenizer())
    calls: list[str] = []

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> object:
            return processor

    class UnsupportedImageMapping:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> object:
            calls.append("image")
            raise ValueError("Unrecognized configuration class KimiK25Config")

    class RegisteredCausalMapping:
        @staticmethod
        def from_pretrained(*_: object, **__: object) -> FakeModel:
            calls.append("causal")
            return FakeModel()

    fake_transformers = SimpleNamespace(
        AutoProcessor=FakeAutoProcessor,
        AutoTokenizer=object(),
        AutoModelForImageTextToText=UnsupportedImageMapping,
        AutoModelForCausalLM=RegisteredCausalMapping,
    )
    monkeypatch.setattr(
        "abliterator.adapters.huggingface.require",
        lambda name: fake_transformers if name == "transformers" else torch,
    )
    HuggingFaceAdapter.from_pretrained(
        ModelLoadConfig(
            model="organization/kimi",
            model_class="image-text-to-text",
            dtype="float32",
            device_map=None,
        )
    )
    assert calls == ["image", "causal"]


def test_streaming_capture_fires_each_layer_once_and_counts_rows() -> None:
    adapter = HuggingFaceAdapter(FakeModel(), FakeTokenizer())
    result = ActivationCollector(adapter).collect(
        ["one", "one two three"],
        CaptureConfig(layers=(0, 1), batch_size=2, pooling="last"),
    )
    assert result.prompt_count == 2
    assert result.counts == {"layer.0.resid_pre": 2, "layer.1.resid_pre": 2}
    assert all(mean.shape == (4,) for mean in result.means.values())
    assert result.input_receipt["rows"] == 2
    assert len(result.input_receipt["combined_sha256"]) == 64


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_mean_pooling_does_not_overflow_a_representable_mean(dtype: torch.dtype) -> None:
    hidden = torch.full((2, 1024, 4), 128.0, dtype=dtype)
    mask = torch.ones((2, 1024), dtype=torch.long)

    pooled = pool_prompt_tokens(hidden, mask, CaptureConfig(pooling="mean"))

    assert pooled.dtype == hidden.dtype
    assert pooled.device == hidden.device
    assert bool(torch.isfinite(pooled).all())
    torch.testing.assert_close(pooled, torch.full((2, 4), 128.0, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_mean_pooling_excludes_left_and_right_padding(dtype: torch.dtype) -> None:
    value = torch.tensor([128.0, -128.0, 16.0, -16.0], dtype=dtype)
    hidden = value.expand(2, 1024, 4).clone()
    mask = torch.zeros((2, 1024), dtype=torch.bool)
    mask[0, -800:] = True
    mask[1, :700] = True
    hidden[~mask] = torch.finfo(dtype).max
    original = hidden.clone()

    pooled = pool_prompt_tokens(hidden, mask, CaptureConfig(pooling="mean"))

    torch.testing.assert_close(pooled, value.expand(2, 4))
    torch.testing.assert_close(hidden, original)


def test_mean_pooling_rejects_an_empty_prompt_mask() -> None:
    with pytest.raises(ArchitectureError, match="at least one unmasked token"):
        pool_prompt_tokens(
            torch.ones((1, 4, 2), dtype=torch.float16),
            torch.zeros((1, 4), dtype=torch.long),
            CaptureConfig(pooling="mean"),
        )


def test_streaming_mean_capture_keeps_finite_fp16_activations() -> None:
    class LongTokenizer(FakeTokenizer):
        def __call__(self, prompts: list[str], **_: object) -> dict[str, torch.Tensor]:
            ids = torch.ones((len(prompts), 1024), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    model = FakeModel().half()
    with torch.no_grad():
        model.embed.weight.fill_(128.0)
    adapter = HuggingFaceAdapter(model, LongTokenizer())

    result = ActivationCollector(adapter).collect(
        ["first", "second"],
        CaptureConfig(layers=(0, 1), batch_size=2, pooling="mean"),
    )

    assert result.counts == {"layer.0.resid_pre": 2, "layer.1.resid_pre": 2}
    for mean in result.means.values():
        torch.testing.assert_close(mean, torch.full((4,), 128.0))


def test_residual_projector_is_reversible() -> None:
    model = FakeModel()
    adapter = HuggingFaceAdapter(model, FakeTokenizer())
    inputs = torch.tensor([[1, 2]])
    baseline = model(inputs).last_hidden_state
    with ResidualProjector(
        adapter,
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        layers=(0,),
        point="resid_pre",
    ):
        modified = model(inputs).last_hidden_state
    restored = model(inputs).last_hidden_state
    assert torch.allclose(modified[..., 0], torch.zeros_like(modified[..., 0]))
    torch.testing.assert_close(restored, baseline)


def test_residual_projector_cleans_up_when_enter_fails() -> None:
    model = FakeModel()
    adapter = HuggingFaceAdapter(model, FakeTokenizer())
    with (
        pytest.raises(ConfigurationError, match="No direction"),
        ResidualProjector(
            adapter,
            {0: torch.tensor([1.0, 0.0, 0.0, 0.0])},
            layers=(0, 1),
        ),
    ):
        pass
    assert not model.model.layers[0]._forward_pre_hooks
    assert not model.model.layers[1]._forward_pre_hooks


@pytest.mark.parametrize("strength", [-0.1, float("nan"), 1.1])
def test_residual_projector_rejects_unsafe_strength(strength: float) -> None:
    adapter = HuggingFaceAdapter(FakeModel(), FakeTokenizer())
    with pytest.raises(ConfigurationError, match="strength"):
        ResidualProjector(
            adapter,
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            strength=strength,
        )


class TargetAdapter:
    hidden_size = 4
    layers = [object()]

    def __init__(self, targets: list[ProjectionTarget]) -> None:
        self.targets = targets

    def projection_targets(self, **_: object) -> list[ProjectionTarget]:
        return self.targets


def test_weight_ablation_handles_linear_and_three_dimensional_experts() -> None:
    dense = torch.nn.Parameter(torch.randn(4, 7))
    experts = torch.nn.Parameter(torch.randn(3, 4, 7))
    targets = [
        ProjectionTarget("dense", 0, "mlp", dense, 0),
        ProjectionTarget("experts", 0, "expert", experts, 1),
    ]
    adapter = TargetAdapter(targets)
    direction = torch.tensor([1.0, 2.0, -1.0, 0.5])
    before = [direction_leakage(target.weight, direction, target.output_axis) for target in targets]
    records = WeightAblator(adapter).apply(direction, chunk_columns=3)
    after = [direction_leakage(target.weight, direction, target.output_axis) for target in targets]
    assert all(value > 1e-4 for value in before)
    assert all(value < 1e-6 for value in after)
    assert len(records) == 2


def test_weight_ablation_can_remove_direction_from_input_embeddings() -> None:
    model = FakeModel()
    adapter = HuggingFaceAdapter(model, FakeTokenizer())
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    before = direction_leakage(model.embed.weight, direction, output_axis=1)

    records = WeightAblator(adapter).apply(
        direction,
        include_input_embeddings=True,
        chunk_columns=8,
    )

    after = direction_leakage(model.embed.weight, direction, output_axis=1)
    embedding_records = [record for record in records if record.kind == "embedding"]
    assert before > 1e-4
    assert after < 1e-6
    assert len(embedding_records) == 1
    assert embedding_records[0].output_axis == 1


def test_strength_zero_preserves_weights() -> None:
    weight = torch.nn.Parameter(torch.randn(4, 5))
    original = weight.detach().clone()
    adapter = TargetAdapter([ProjectionTarget("dense", 0, "attention", weight, 0)])
    WeightAblator(adapter).apply(torch.tensor([1.0, 0.0, 0.0, 0.0]), strength=0.0)
    torch.testing.assert_close(weight, original)


def test_permanent_ablation_cannot_be_applied_twice_to_one_instance() -> None:
    weight = torch.nn.Parameter(torch.randn(4, 5))
    adapter = TargetAdapter([ProjectionTarget("dense", 0, "attention", weight, 0)])
    ablator = WeightAblator(adapter)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    ablator.apply(direction, strength=0.5)
    with pytest.raises(ArchitectureError, match="already entered permanent mutation"):
        ablator.apply(direction, strength=0.5)


def test_packed_target_is_rejected_before_mutation() -> None:
    weight = torch.nn.Parameter(torch.zeros((4, 5), dtype=torch.uint8), requires_grad=False)
    target = ProjectionTarget(
        "packed", 0, "attention", weight, 0, packed=True, quantization="bnb-fp4"
    )
    with pytest.raises(QuantizationError, match="packed low-bit"):
        WeightAblator(TargetAdapter([target])).apply(torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_scaled_and_packed_modules_are_detected_fail_closed() -> None:
    class ScaledLinear(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(4, 4, bias=False)
            self.weight_scale_inv = torch.ones(1)

    scaled = ScaledLinear()
    quantization, packed = describe_module_quantization(scaled, scaled.weight)
    assert packed
    assert quantization.startswith("scaled-")

    class PackedLinear(torch.nn.Module):
        out_features = 4
        weight = None

        def __init__(self) -> None:
            super().__init__()
            self.weight_packed = torch.zeros((4, 2), dtype=torch.uint8)

    packed_module = PackedLinear()
    _, packed = describe_module_quantization(packed_module, packed_module.weight_packed)
    assert packed


def test_float8_storage_is_never_treated_as_dense() -> None:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("Torch build has no float8 dtype")
    weight = torch.zeros((4, 4), dtype=dtype)
    quantization, packed = describe_module_quantization(None, weight)
    assert packed
    assert "float8" in quantization


def test_preflight_failure_does_not_mutate_an_earlier_target() -> None:
    first = torch.nn.Parameter(torch.randn(4, 5))
    second = torch.nn.Parameter(torch.randn(4, 5))
    second.data[0, 0] = float("nan")
    original = first.detach().clone()
    adapter = TargetAdapter(
        [
            ProjectionTarget("first", 0, "attention", first, 0),
            ProjectionTarget("second", 0, "mlp", second, 0),
        ]
    )
    with pytest.raises(ArchitectureError, match="NaN or infinity"):
        WeightAblator(adapter).apply(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(first, original)
