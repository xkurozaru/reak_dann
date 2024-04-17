import torch.nn as nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from .function import GradientReversalLayer


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

        self.domain_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.final_in_features, 32),
            nn.Mish(),
            nn.Linear(32, 1),
        )

    def forward(self, inputs, alpha=1.0):
        feats = self.feature_extractor(inputs)
        class_preds = self.class_classifier(feats)
        reversal_feats = GradientReversalLayer.apply(feats, alpha)
        domain_preds = self.domain_classifier(reversal_feats)
        return class_preds, domain_preds.view(-1)


class DCNN(nn.Module):
    def __init__(self, num_class: int, pretrained: bool = True):
        super(DCNN, self).__init__()
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

        self.domain_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.final_in_features, 32),
            nn.BatchNorm1d(32),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, inputs):
        feats = self.feature_extractor(inputs)
        class_preds = self.class_classifier(feats)
        domain_preds = self.domain_classifier(feats)
        return class_preds, domain_preds.view(-1)
