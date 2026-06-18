import torch


def _masked_predictions(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    predictions = logits.argmax(dim=-1)
    return predictions[mask], labels[mask]


def accuracy_score(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute node-classification accuracy on the selected mask."""

    predictions, masked_labels = _masked_predictions(logits, labels, mask)
    if masked_labels.numel() == 0:
        return 0.0
    correct = int((predictions == masked_labels).sum().item())
    total = int(masked_labels.numel())
    return correct / total


def macro_f1_score(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute macro F1 without adding a hard dependency to sklearn at runtime."""

    predictions, masked_labels = _masked_predictions(logits, labels, mask)
    if masked_labels.numel() == 0:
        return 0.0

    num_classes = int(logits.size(-1))
    f1_values: list[float] = []
    for class_id in range(num_classes):
        pred_positive = predictions == class_id
        true_positive_label = masked_labels == class_id
        true_positive = (pred_positive & true_positive_label).sum().item()
        false_positive = (pred_positive & ~true_positive_label).sum().item()
        false_negative = (~pred_positive & true_positive_label).sum().item()
        denominator = (2 * true_positive) + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return float(sum(f1_values) / len(f1_values))
