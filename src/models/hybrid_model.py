"""
Hybrid Quantum-Classical Model.
Combines a classical CNN (ResNet) with our quantum layer for medical image classification.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.models.quantum_layer import build_quantum_layer, N_QUBITS


class HybridModel(nn.Module):
    """
    Architecture:
      Image -> ResNet18 (feature extractor) -> Linear -> Quantum Layer -> Classifier -> Output
    """

    def __init__(self, n_classes: int = 2, n_quantum_layers: int = 2):
        super().__init__()

        # 1) Classical part: pretrained ResNet18 as a feature extractor
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze the CNN weights (we use it only to extract features — faster training)
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Replace ResNet's final layer to output N_QUBITS features (to feed the quantum layer)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(in_features, N_QUBITS)

        # 2) Quantum part: our quantum layer
        self.quantum = build_quantum_layer(n_layers=n_quantum_layers)

        # 3) Final classifier: small network for more learning power
        self.classifier = nn.Sequential(
            nn.Linear(N_QUBITS, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        # Pass image through CNN -> get N_QUBITS features
        x = self.cnn(x)

        # Scale features into a good range for quantum encoding (0 to pi)
        x = torch.tanh(x) * torch.pi

        # Pass through the quantum layer
        x = self.quantum(x)

        # Final classification
        x = self.classifier(x)
        return x


# Quick test: run this file directly to check the full model works
if __name__ == "__main__":
    model = HybridModel(n_classes=2)

    # Create a fake "image" batch: 1 image, 3 color channels, 224x224 pixels
    fake_image = torch.randn(1, 3, 224, 224)

    output = model(fake_image)

    print("✅ Hybrid model works!")
    print("Input shape :", list(fake_image.shape))
    print("Output shape:", list(output.shape), "(2 class scores)")
    print("Output      :", output.tolist())