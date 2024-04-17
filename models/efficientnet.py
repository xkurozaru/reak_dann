import torch.nn as nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class EfficientNetV2(nn.Module):
    def __init__(self, num_class: int, pretrained: bool = True):
        super(EfficientNetV2, self).__init__()
        if pretrained:
            self.efficientnet = efficientnet_v2_s(EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        else:
            self.efficientnet = efficientnet_v2_s(EfficientNet_V2_S_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(
            self.efficientnet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.final_in_features = self.efficientnet.classifier[1].in_features

        self.class_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.final_in_features, num_class),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.class_classifier(x)
        return x
