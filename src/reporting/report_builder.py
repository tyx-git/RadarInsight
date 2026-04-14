import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plot_training_curves(history: list[dict[str, Any]], output_path: Path, title: str) -> None:
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_acc = [item["train_accuracy"] for item in history]
    val_acc = [item["val_accuracy"] for item in history]
    has_f1 = all("train_macro_f1" in item and "val_macro_f1" in item for item in history)
    train_f1 = [item.get("train_macro_f1", 0.0) for item in history]
    val_f1 = [item.get("val_macro_f1", 0.0) for item in history]

    num_axes = 3 if has_f1 else 2
    figure, axes = plt.subplots(1, num_axes, figsize=(6 * num_axes, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].text(epochs[-1], train_loss[-1], f"{train_loss[-1]:.2f}", fontsize=8, ha="left", va="bottom")
    axes[0].text(epochs[-1], val_loss[-1], f"{val_loss[-1]:.2f}", fontsize=8, ha="left", va="top")

    axes[1].plot(epochs, train_acc, marker="o", label="Train Accuracy")
    axes[1].plot(epochs, val_acc, marker="o", label="Val Accuracy")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].text(epochs[-1], train_acc[-1], f"{train_acc[-1]:.2f}", fontsize=8, ha="left", va="bottom")
    axes[1].text(epochs[-1], val_acc[-1], f"{val_acc[-1]:.2f}", fontsize=8, ha="left", va="top")

    if has_f1:
        axes[2].plot(epochs, train_f1, marker="o", label="Train Macro F1")
        axes[2].plot(epochs, val_f1, marker="o", label="Val Macro F1")
        axes[2].set_title(f"{title} - Macro F1")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Macro F1")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        axes[2].text(epochs[-1], train_f1[-1], f"{train_f1[-1]:.2f}", fontsize=8, ha="left", va="bottom")
        axes[2].text(epochs[-1], val_f1[-1], f"{val_f1[-1]:.2f}", fontsize=8, ha="left", va="top")

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_confusion_matrix(matrix: np.ndarray, labels: list[str], output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(int(matrix[row, col])), ha="center", va="center", color="black")

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_per_class_metric(per_class: dict[str, dict[str, float | int]], output_path: Path, title: str) -> None:
    labels = list(per_class.keys())
    f1_scores = [float(per_class[label]["f1"]) for label in labels]
    precisions = [float(per_class[label]["precision"]) for label in labels]
    recalls = [float(per_class[label]["recall"]) for label in labels]

    x_axis = np.arange(len(labels))
    width = 0.25

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(x_axis - width, precisions, width=width, label="Precision")
    axis.bar(x_axis, recalls, width=width, label="Recall")
    axis.bar(x_axis + width, f1_scores, width=width, label="F1")
    axis.set_title(title)
    axis.set_xticks(x_axis)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    for index, value in enumerate(precisions):
        axis.text(index - width, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for index, value in enumerate(recalls):
        axis.text(index, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for index, value in enumerate(f1_scores):
        axis.text(index + width, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_overall_metrics(metrics: dict[str, float], output_path: Path, title: str) -> None:
    labels = list(metrics.keys())
    values = [float(metrics[label]) for label in labels]
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"][: len(labels)])
    axis.set_title(title)
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_real_report(artifact_dir: str, report_root: str = "reports/real") -> dict[str, Any]:
    artifact_path = Path(artifact_dir)
    history = _read_json(artifact_path / "training_history.json")
    metadata = _read_json(artifact_path / "training_metadata.json")
    evaluation = _read_json(artifact_path / "evaluation_report.json") if (artifact_path / "evaluation_report.json").exists() else None

    report_dir = _reset_dir(Path(report_root) / artifact_path.name)
    _plot_training_curves(history, report_dir / "training_curves.png", title="Real Model")

    labels = list(metadata.get("labels", {}).values())
    if evaluation is not None:
        confusion_matrix = np.asarray(evaluation["confusion_matrix"], dtype=np.int64)
        _plot_confusion_matrix(confusion_matrix, labels, report_dir / "confusion_matrix.png", title="Real Evaluation Confusion Matrix")
        _plot_per_class_metric(evaluation["per_class"], report_dir / "per_class_metrics.png", title="Real Evaluation Per-Class Metrics")
        _plot_overall_metrics(
            {
                "Accuracy": float(evaluation.get("accuracy", 0.0)),
                "Macro F1": float(evaluation.get("macro_f1", 0.0)),
                "Macro Recall": float(evaluation.get("macro_recall", 0.0)),
                "Macro Prec": float(evaluation.get("macro_precision", 0.0)),
            },
            report_dir / "overall_metrics.png",
            title="Real Evaluation Overall Metrics",
        )
        session_level = evaluation.get("session_level")
        if session_level is not None:
            session_confusion_matrix = np.asarray(session_level["confusion_matrix"], dtype=np.int64)
            _plot_confusion_matrix(session_confusion_matrix, labels, report_dir / "session_confusion_matrix.png", title="Session-Level Evaluation Confusion Matrix")
            _plot_per_class_metric(session_level["per_class"], report_dir / "session_per_class_metrics.png", title="Session-Level Evaluation Per-Class Metrics")
            _plot_overall_metrics(
                {
                    "Accuracy": float(session_level.get("accuracy", 0.0)),
                    "Macro F1": float(session_level.get("macro_f1", 0.0)),
                    "Macro Recall": float(session_level.get("macro_recall", 0.0)),
                    "Macro Prec": float(session_level.get("macro_precision", 0.0)),
                },
                report_dir / "session_overall_metrics.png",
                title="Session-Level Overall Metrics",
            )

    result = {
        "report_dir": str(report_dir),
        "training_curves": str(report_dir / "training_curves.png"),
    }
    if evaluation is not None:
        result["confusion_matrix"] = str(report_dir / "confusion_matrix.png")
        result["per_class_metrics"] = str(report_dir / "per_class_metrics.png")
        result["overall_metrics"] = str(report_dir / "overall_metrics.png")
        if evaluation.get("session_level") is not None:
            result["session_confusion_matrix"] = str(report_dir / "session_confusion_matrix.png")
            result["session_per_class_metrics"] = str(report_dir / "session_per_class_metrics.png")
            result["session_overall_metrics"] = str(report_dir / "session_overall_metrics.png")
    return result
