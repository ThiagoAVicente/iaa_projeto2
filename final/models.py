"""Model definitions.

Three architectures sharing a 2-class classification head:
  * SmallCNN          - shallow baseline, native 32x32 input
  * ResNet18 scratch  - CIFAR-style first block, random weights, 32x32
  * ResNet18 pretrained - ImageNet weights, finetuned at 224x224
"""

import torch.nn as nn
import torchvision.models as tvm

NUM_CLASSES = 2


class SmallCNN(nn.Module):
    """3-block CNN: 32x32x3 -> 2 logits.

    Each block: Conv3x3 -> BN -> ReLU -> MaxPool2.
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_cnn():
    return SmallCNN()


def build_resnet_scratch():
    m = tvm.resnet18(weights=None)
    # CIFAR-style first conv keeps 32x32 spatial resolution through the stem.
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def build_resnet_pretrained():
    m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


BUILDERS = {
    "cnn": build_cnn,
    "resnet_scratch": build_resnet_scratch,
    "resnet_pretrained": build_resnet_pretrained,
}
