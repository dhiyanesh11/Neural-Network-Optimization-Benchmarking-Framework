"""Registry to inference smoke test for the next pipeline module."""

import torch

from nnbench.registry import get_dataloader, get_model
from nnbench.seed import set_seed


def test_model_and_loader_complete_forward_and_accuracy() -> None:
    set_seed(42)
    model = get_model("mobilenet_v2").eval()
    loader = get_dataloader("cifar10", "test", batch_size=4, synthetic=True, num_samples=8)
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images)
            correct += int((outputs.argmax(dim=1) == labels).sum())
            total += labels.numel()
    accuracy = correct / total
    assert total == 8
    assert 0.0 <= accuracy <= 1.0
