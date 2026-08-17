import csv

from resnet152_pa0.reporting import summarize_runs


def test_summarize_runs(tmp_path) -> None:
    run_dir = tmp_path / "baseline_head"
    run_dir.mkdir()
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch", "train_loss", "train_accuracy", "val_loss",
                "val_accuracy", "train_seconds", "val_seconds",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 1, "train_loss": 2.0, "train_accuracy": 0.3,
                "val_loss": 1.8, "val_accuracy": 0.4,
                "train_seconds": 5.0, "val_seconds": 1.0,
            }
        )
    summary_path, figure_path = summarize_runs(tmp_path)
    assert summary_path.exists()
    assert figure_path.exists()
    assert "baseline_head" in summary_path.read_text(encoding="utf-8")
