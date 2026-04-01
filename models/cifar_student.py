import torch.nn as nn

from models.graph import GraphModule


class CifarGraphNet(GraphModule):
    """
    Small CNN for CIFAR-10 (3x32x32) built as a sequential GraphModule.
    Conv/head layout is fixed; structural mutations (Phase 1 ops) target Linear nodes.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.add_node("conv1", nn.Conv2d(3, 32, 3, padding=1))
        self.add_node("bn1", nn.BatchNorm2d(32))
        self.add_node("relu1", nn.ReLU())
        self.add_node("pool1", nn.MaxPool2d(2))

        self.add_node("conv2", nn.Conv2d(32, 64, 3, padding=1))
        self.add_node("bn2", nn.BatchNorm2d(64))
        self.add_node("relu2", nn.ReLU())
        self.add_node("pool2", nn.MaxPool2d(2))

        self.add_node("conv3", nn.Conv2d(64, 128, 3, padding=1))
        self.add_node("bn3", nn.BatchNorm2d(128))
        self.add_node("relu3", nn.ReLU())
        self.add_node("pool3", nn.MaxPool2d(2))

        self.add_node("flatten", nn.Flatten())
        # 128 * 4 * 4 = 2048
        self.add_node("fc1", nn.Linear(2048, 256))
        self.add_node("relu_fc", nn.ReLU())
        self.add_node("fc2", nn.Linear(256, num_classes))
