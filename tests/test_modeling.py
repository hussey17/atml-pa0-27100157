import pytest
import torch

from resnet152_pa0.modeling import (
    build_resnet152,
    configure_trainable_layers,
    count_parameters,
    disable_skip_connections,
)


def test_classifier_matches_cifar10() -> None:
    model = build_resnet152(pretrained=False)
    assert model.fc.in_features == 2048
    assert model.fc.out_features == 10


@pytest.mark.parametrize("mode", ["head", "final_block", "full"])
def test_trainable_modes(mode: str) -> None:
    model = build_resnet152(pretrained=False)
    configure_trainable_layers(model, mode)
    counts = count_parameters(model)
    assert counts["trainable"] > 0
    assert counts["trainable"] == counts["total"] if mode == "full" else counts["trainable"] < counts["total"]
    assert all(parameter.requires_grad for parameter in model.fc.parameters())
    if mode == "head":
        assert not any(parameter.requires_grad for parameter in model.layer4.parameters())
    if mode == "final_block":
        assert all(parameter.requires_grad for parameter in model.layer4.parameters())


def test_skip_ablation_changes_block_output_and_preserves_shape() -> None:
    torch.manual_seed(7)
    model = build_resnet152(pretrained=False).eval()
    block = model.layer2[1]
    x = torch.randn(1, 512, 8, 8)
    with torch.inference_mode():
        baseline = block(x)
    disable_skip_connections(model, ["layer2.1"])
    with torch.inference_mode():
        ablated = block(x)
    assert baseline.shape == ablated.shape
    assert not torch.allclose(baseline, ablated)


def test_transition_block_is_rejected() -> None:
    model = build_resnet152(pretrained=False)
    with pytest.raises(ValueError, match="transition block"):
        disable_skip_connections(model, ["layer2.0"])

