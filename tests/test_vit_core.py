import torch

from vit_pa0.masking import mask_pixel_patches, patch_indices
from vit_pa0.modeling import cls_patch_attention, pool_tokens


def test_pool_tokens() -> None:
    tokens = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    cls, mean = pool_tokens(tokens)
    assert torch.equal(cls, tokens[:, 0])
    assert torch.allclose(mean, tokens[:, 1:].mean(dim=1))


def test_cls_patch_attention_shape_and_values() -> None:
    attention = torch.zeros(2, 3, 5, 5)
    attention[:, :, 0, 1:] = torch.arange(4, dtype=torch.float32)
    mean_map, heads = cls_patch_attention((attention,))
    assert mean_map.shape == (2, 2, 2)
    assert heads.shape == (2, 3, 2, 2)
    assert torch.equal(mean_map[0].flatten(), torch.arange(4, dtype=torch.float32))


def test_patch_indices_are_reproducible_and_centered() -> None:
    first = patch_indices(4, 0.25, "random", torch.Generator().manual_seed(5))
    second = patch_indices(4, 0.25, "random", torch.Generator().manual_seed(5))
    assert torch.equal(first, second)
    centered = set(patch_indices(4, 0.25, "center").tolist())
    assert centered == {5, 6, 9, 10}


def test_masking_changes_exact_number_of_patches() -> None:
    pixels = torch.ones(2, 3, 8, 8)
    masked = mask_pixel_patches(pixels, patch_size=2, fraction=0.25, mode="center")
    zero_pixels_per_image = int((masked[0, 0] == 0).sum())
    assert zero_pixels_per_image == 4 * 2 * 2
    assert torch.equal(masked[0], masked[1])
