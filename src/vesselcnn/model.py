"""VesselCNN: a compact 3D CNN for intracranial vessel-segment classification.

Designed from scratch by Habib Ur Rehman for VesselMNIST3D (MedMNIST v2):
binary classification of 28x28x28 voxelized vessel segments as
  0 vessel    (healthy segment)
  1 aneurysm  (segment containing an intracranial aneurysm)

The volumes are sparse: only ~1.9% of voxels are non-zero, because a
vessel segment is a thin tube inside its bounding cube. The architecture
therefore uses 3x3x3 convolutions with max-pooling (max-pooling preserves
bright thin structures that average-pooling would wash out) and a global
average pooling head instead of a flattened MLP head (a flatten head on
7x7x7x64 features would add ~1.4M parameters against only 1,335 training
volumes).

Labels (MedMNIST v2 / VesselMNIST3D official order):
  0 vessel
  1 aneurysm
"""
import torch
import torch.nn as nn


class VesselCNN(nn.Module):
    """3 conv blocks (16 -> 32 -> 64 channels) + GAP classifier head.

    ~105k trainable parameters. Input: (N, 1, 28, 28, 28) float tensor,
    normalized with the VesselMNIST3D train-split mean/std (see
    vesselcnn.infer). The design follows the same compact conv-block
    family as the sibling SonoCNN / PathoCNN releases, re-derived for
    single-channel volumetric input and a binary head.
    """

    def __init__(self, in_ch=1, num_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(dropout * 0.5),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(2), nn.Dropout3d(dropout * 0.5),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        # 28^3 -> 14^3 -> 7^3 -> GAP -> 64 features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model():
    return VesselCNN()
