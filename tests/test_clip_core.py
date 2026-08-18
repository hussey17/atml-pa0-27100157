import numpy as np
import pytest
import torch

from clip_pa0.alignment import fit_orthogonal_alignment, save_alignment_visualization
from clip_pa0.config import ClipExperimentConfig
from clip_pa0.data import balanced_indices
from clip_pa0.evaluation import classification_metrics
from clip_pa0.modality import modality_gap_statistics
from clip_pa0.prompts import STL10_CLASSES, l2_normalize, prompts_for_class


def test_prompt_strategies_include_articles_and_ensemble() -> None:
    assert prompts_for_class("cat", "plain") == ["cat"]
    assert prompts_for_class("cat", "photo") == ["a photo of a cat."]
    assert prompts_for_class("airplane", "photo") == ["a photo of an airplane."]
    assert len(prompts_for_class("dog", "descriptive")) >= 3


def test_l2_normalize_produces_unit_rows() -> None:
    values = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
    assert torch.allclose(l2_normalize(values).norm(dim=1), torch.ones(2))


def test_balanced_indices_are_reproducible_and_balanced() -> None:
    labels = np.repeat(np.arange(10), 20)
    first = balanced_indices(labels, 53, seed=9)
    second = balanced_indices(labels, 53, seed=9)
    counts = np.bincount(labels[first], minlength=10)
    assert first == second
    assert len(first) == len(set(first)) == 53
    assert counts.max() - counts.min() <= 1


def test_classification_metrics_on_exact_prototypes() -> None:
    prototypes = torch.eye(10)
    labels = torch.arange(10)
    metrics = classification_metrics(prototypes, labels, prototypes, STL10_CLASSES)
    assert metrics["accuracy"] == 1.0
    assert metrics["correct"] == 10
    assert np.array(metrics["confusion_matrix"]).trace() == 10


def test_procrustes_recovers_known_orthogonal_map() -> None:
    generator = np.random.default_rng(4)
    images = generator.normal(size=(80, 12))
    known_rotation, _ = np.linalg.qr(generator.normal(size=(12, 12)))
    texts = images @ known_rotation
    estimated_rotation, diagnostics = fit_orthogonal_alignment(images, texts)
    assert np.allclose(images @ estimated_rotation, texts, atol=1e-9)
    assert diagnostics["residual_frobenius_after"] < 1e-8
    assert diagnostics["orthogonality_error_frobenius"] < 1e-8


def test_modality_statistics_measure_pairwise_gap() -> None:
    images = np.eye(3)
    texts = images.copy()
    statistics = modality_gap_statistics(images, texts)
    assert statistics["centroid_euclidean_distance"] == 0.0
    assert statistics["mean_paired_euclidean_distance"] == 0.0
    assert statistics["mean_paired_cosine_similarity"] == pytest.approx(1.0)


def test_config_enforces_manual_gap_sample_range() -> None:
    config = ClipExperimentConfig(gap_samples=49)
    with pytest.raises(ValueError, match="between 50 and 100"):
        config.validate()


def test_alignment_visualization_creates_output_directory(tmp_path) -> None:
    generator = np.random.default_rng(7)
    images = generator.normal(size=(20, 5))
    texts = images + 0.01 * generator.normal(size=(20, 5))
    rotation, _ = fit_orthogonal_alignment(images, texts)
    output_dir = tmp_path / "nested" / "alignment"
    save_alignment_visualization(
        images,
        images @ rotation,
        texts,
        np.repeat(np.arange(10), 2),
        output_dir,
        seed=7,
        perplexity=5,
    )
    assert (output_dir / "alignment_tsne.png").is_file()
    assert (output_dir / "alignment_tsne_coordinates.npz").is_file()
