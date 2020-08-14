import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClsHead(nn.Module):
    def __init__(self,
                 inplanes,
                 num_anchors,
                 num_classes,
                 num_layers=4,
                 prior=0.01):
        super(ClsHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(inplanes,
                      num_anchors * num_classes,
                      kernel_size=3,
                      stride=1,
                      padding=1))
        self.cls_head = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = prior
        b = -math.log((1 - prior) / prior)
        self.cls_head[-1].bias.data.fill_(b)

    def forward(self, x):
        x = self.cls_head(x)
        x = self.sigmoid(x)

        return x


class RegHead(nn.Module):
    def __init__(self, inplanes, num_anchors, num_layers=4):
        super(RegHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(inplanes,
                      num_anchors * 4,
                      kernel_size=3,
                      stride=1,
                      padding=1))

        self.reg_head = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.reg_head(x)

        return x


if __name__ == '__main__':
    image_h, image_w = 640, 640
    from fpn import FPN
    fpn_model = FPN(512, 1024, 2048, 256)
    C3, C4, C5 = torch.randn(3, 512, 80, 80), torch.randn(3, 1024, 40,
                                                          40), torch.randn(
                                                              3, 2048, 20, 20)
    features = fpn_model([C3, C4, C5])

    print("1111", features[0].shape)

    cls_model = ClsHead(256, 9, 80)
    reg_model = RegHead(256, 9)

    cls_output = cls_model(features[0])
    reg_output = reg_model(features[0])

    print("2222", cls_output.shape, reg_output.shape)
