import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class TorchvisionResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = resnet50(weights=weights)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.base = base
        self.feature_channels = [256, 512, 1024, 2048]
        self.out_dim = 2048

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        out1 = self.base.layer1(x)
        out2 = self.base.layer2(out1)
        out3 = self.base.layer3(out2)
        out4 = self.base.layer4(out3)

        pooled = self.base.avgpool(out4)
        outf = pooled.view(pooled.size(0), -1)
        logits = self.base.fc(outf)
        return logits, [out1, out2, out3, out4], outf
