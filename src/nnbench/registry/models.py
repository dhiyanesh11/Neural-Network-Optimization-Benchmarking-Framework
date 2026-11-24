"""Factories for the supported image and text classification models."""

from collections.abc import Callable

import torch
from torch import nn


class _TinyDistilBert(nn.Module):
    """Adapt a DistilBERT classifier to return a plain logits tensor."""

    def __init__(self, num_labels: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from transformers import DistilBertConfig, DistilBertForSequenceClassification
        except ImportError as exc:
            raise ImportError(
                "The DistilBERT model requires the 'nlp' extra: " "pip install -e '.[nlp]'"
            ) from exc

        if pretrained:
            self.model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert/distilbert-base-uncased", num_labels=num_labels
            )
        else:
            config = DistilBertConfig(
                vocab_size=30522,
                max_position_embeddings=128,
                n_layers=2,
                n_heads=4,
                dim=128,
                hidden_dim=512,
                num_labels=num_labels,
            )
            self.model = DistilBertForSequenceClassification(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def _mobilenet_v2(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    try:
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    except ImportError as exc:
        raise ImportError("Image models require torchvision; install the 'vision' extra.") from exc
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    if num_classes != 1000:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise ImportError("Image models require torchvision; install the 'vision' extra.") from exc
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    # CIFAR-sized inputs work better without the ImageNet stem's early downsampling.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


MODEL_FACTORIES: dict[str, Callable[..., nn.Module]] = {
    "mobilenet_v2": _mobilenet_v2,
    "resnet18": _resnet18,
    "distilbert_sst2": lambda num_classes=2, pretrained=False: _TinyDistilBert(
        num_labels=num_classes, pretrained=pretrained
    ),
}


def list_models() -> list[str]:
    """Return registered model names."""
    return sorted(MODEL_FACTORIES)


def get_model(
    name: str,
    *,
    pretrained: bool = False,
    num_classes: int | None = None,
) -> nn.Module:
    """Construct a registered model, optionally loading public pretrained weights."""
    try:
        factory = MODEL_FACTORIES[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{name}'. Registered models: {', '.join(list_models())}"
        ) from exc
    if num_classes is None:
        num_classes = 2 if name.lower() == "distilbert_sst2" else 10
    return factory(num_classes=num_classes, pretrained=pretrained)
