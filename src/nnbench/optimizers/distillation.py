"""Teacher-student knowledge distillation optimizer."""

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from nnbench.optimizers.base import BaseOptimizer
from nnbench.runner.benchmark import _batch_parts, _forward


class KnowledgeDistillationOptimizer(BaseOptimizer):
    """Fine-tune a student copy using teacher soft targets and label loss."""

    name = "knowledge_distillation"

    def __init__(
        self,
        teacher: nn.Module | None = None,
        train_loader: Any = None,
        epochs: int = 1,
        temperature: float = 2.0,
        alpha: float = 0.5,
        learning_rate: float = 1e-4,
    ) -> None:
        self.teacher = teacher
        self.train_loader = train_loader
        self.epochs = epochs
        self.temperature = temperature
        self.alpha = alpha
        self.learning_rate = learning_rate

    def apply(self, model: nn.Module, calib_loader: Any = None) -> nn.Module:
        loader = self.train_loader or calib_loader
        if self.teacher is None or loader is None:
            raise ValueError("Knowledge distillation requires a teacher model and training loader.")
        if self.epochs < 1 or self.temperature <= 0 or not 0 <= self.alpha <= 1:
            raise ValueError("epochs, temperature, or alpha is outside its valid range.")
        student = self.clone(model)
        teacher = deepcopy(self.teacher).eval()
        device = next(student.parameters()).device
        teacher = teacher.to(device)
        optimizer = torch.optim.AdamW(student.parameters(), lr=self.learning_rate)
        cross_entropy = nn.CrossEntropyLoss()
        student.train()
        for _ in range(self.epochs):
            for batch in loader:
                inputs, labels = _batch_parts(batch, device)
                optimizer.zero_grad(set_to_none=True)
                student_logits = _forward(student, inputs)
                with torch.no_grad():
                    teacher_logits = _forward(teacher, inputs)
                soft_loss = (
                    nn.functional.kl_div(
                        nn.functional.log_softmax(student_logits / self.temperature, dim=-1),
                        nn.functional.softmax(teacher_logits / self.temperature, dim=-1),
                        reduction="batchmean",
                    )
                    * self.temperature**2
                )
                loss = self.alpha * soft_loss + (1 - self.alpha) * cross_entropy(
                    student_logits, labels
                )
                loss.backward()
                optimizer.step()
        return student.eval()
