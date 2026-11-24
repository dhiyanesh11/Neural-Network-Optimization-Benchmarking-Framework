"""Train and save a CIFAR-10 baseline checkpoint with a matching accuracy record."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from nnbench.registry import get_dataloader, get_model
from nnbench.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mobilenet_v2", "resnet18"], default="mobilenet_v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if absent")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_dataloader("cifar10", "train", args.batch_size, download=args.download)
    val_loader = get_dataloader("cifar10", "test", args.batch_size, download=args.download)
    model = get_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(model(images), labels).backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in val_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.numel()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / f"{args.model}_cifar10.pt"
    torch.save(model.cpu().state_dict(), checkpoint)
    (args.output / f"{args.model}_cifar10_metrics.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "dataset": "cifar10",
                "accuracy": correct / total,
                "epochs": args.epochs,
                "seed": args.seed,
                "checkpoint": str(checkpoint),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
