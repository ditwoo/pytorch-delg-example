import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from genet import genet_small, genet_normal, genet_large
from efficientnet_pytorch import EfficientNet


class EfficientNetEncoder(nn.Module):
    def __init__(self, base: str):
        super().__init__()
        self.base = EfficientNet.from_pretrained(base)

    def out_channels(self):
        return self.base._conv_head.out_channels

    def in_features(self):
        return self.base._fc.in_features

    def forward(self, batch):
        x = self.base.extract_features(batch)
        return x


class GENetEncoder(nn.Module):
    def __init__(self, base: str, pretrain_dir: str = None):
        super().__init__()
        args = {}
        if pretrain_dir is not None:
            args["pretrained"] = True
            args["root"] = pretrain_dir
        if base == "small":
            self.base = genet_small(**args)
        elif base == "normal":
            self.base = genet_normal(**args)
        elif base == "large":
            self.base = genet_large(**args)

    def out_channels(self):
        return self.base.module_list[-2].netblock.num_features

    def in_features(self):
        return self.base.fc_linear.in_features

    def forward(self, batch):
        x = self.base.extract_features(batch)
        return x


def gem(x: torch.tensor, p: int = 3, eps: float = 1e-6) -> torch.tensor:
    """Generalized Mean Pooling.
    Args:
        x (torch.tensor): input features,
            expected shapes - BxCxHxW
        p (int, optional): normalization degree.
            Defaults is `3`.
        eps (float, optional): minimum value to use in x.
            Defaults is `1e-6`.
    Returns:
        tensor with shapes - BxCx1x1
    """
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    """Generalized Mean Pooling.
    Paper: https://arxiv.org/pdf/1711.02512.
    """

    def __init__(self, p: int = 3, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, features):
        return gem(features, self.p, self.eps)


class ArcFace(nn.Module):
    """Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf)"""

    def __init__(self, in_features, out_features, s=64.0, m=0.50, eps=1e-6):
        """
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
        """
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.threshold = math.pi - self.m

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        one_hot = torch.zeros(cos_theta.size()).to(input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        selected = torch.where(
            theta > self.threshold, torch.zeros_like(one_hot), one_hot
        ).bool()

        output = torch.cos(torch.where(selected, theta + self.m, theta))
        output *= self.s
        return output


class EncoderGlobalFeatures(nn.Module):
    def __init__(self, encoder, emb_dim=512, num_classes: int = 1):
        super().__init__()
        self.encoder = encoder
        self.pool = GeM(3, 1e-8)
        self.linear = nn.Linear(encoder.in_features(), emb_dim)
        self.head = ArcFace(emb_dim, num_classes, s=1.31, m=0.15)

    def forward(self, images, labels=None):
        x = self.encoder(images)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.linear(x)
        if labels is None:
            return x
        x = self.head(x, labels)
        return x


class Attention(nn.Module):
    def __init__(self, in_channels, hidden_filters=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_filters, 1)
        self.bn = nn.BatchNorm2d(hidden_filters)
        self.conv2 = nn.Conv2d(hidden_filters, 1, 1)

    def forward(self, features):
        x = F.relu(self.bn(self.conv1(features)))

        score = self.conv2(x)
        probability = F.softplus(score)
        return probability, score


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, num_channels=128):
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, num_channels, 1)
        self.decoder = nn.Conv2d(num_channels, in_channels, 1)

    def forward(self, features):
        x = self.encoder(features)
        x = F.relu(self.decoder(x))
        return x


class EncoderLocalFeatures(nn.Module):
    def __init__(self, encoder, num_classes=1):
        super().__init__()
        self.encoder = encoder
        encoder_channels = encoder.out_channels()

        self.autoencoder = AutoEncoder(encoder_channels)

        attention_filters = 512
        self.attention = Attention(encoder_channels, attention_filters)
        self.classifier = nn.Linear(encoder_channels, num_classes)

    def forward(self, features):
        with torch.no_grad():
            embedding = self.encoder(features)

        reconstruction = self.autoencoder(embedding)

        probability, score = self.attention(reconstruction)

        x = (reconstruction * probability).sum((2, 3))  # .squeeze(-1)
        cls = self.classifier(x)

        return embedding, reconstruction, cls, probability
