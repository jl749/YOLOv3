"""
Implementation of yolov3 architecture
"""
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn

""" 
Tuple is structured by (filters, kernel_size, stride) 
Every conv same padding.
"""
config = [
    # input = (BATCH_SIZE, 3, 416, 416)
    (32, 3, 1),              # output = (32, 416, 416)
    (64, 3, 2),              # (64, 208, 208)
    ["ResidualBlock", 1],    # (64, 208, 208)
    (128, 3, 2),             # (128, 104, 104)
    ["ResidualBlock", 2],    # (128, 104, 104)
    (256, 3, 2),             # (256, 52, 52)
    ["ResidualBlock", 8],    # (256, 52, 52)   ### CONCAT later(after Upsample2)
    (512, 3, 2),             # (512, 26, 26)
    ["ResidualBlock", 8],    # (512, 26, 26)   ### CONCAT later(after Upsample1)
    (1024, 3, 2),            # (1024, 13, 13)
    ["ResidualBlock", 4],    # (1024, 13, 13)
    # Darknet-53 -- feature extractor (encoding)

    (512, 1, 1),             # (512, 13, 13)
    (1024, 3, 1),            # (1024, 13, 13)
    "ScalePrediction",       # (1024, 13, 13)  -->  (512, 13, 13)  -->  (512, 13, 13) result1
    # (pass info for the next result, don't just waste it might be useful)
    (256, 1, 1),             # (256, 13, 13)
    "UpSample",              # (256, 26, 26)   ### CONCAT dim=1 (channels)
    (256, 1, 1),             # (256, 26, 26)
    (512, 3, 1),             # (512, 26, 26)
    "ScalePrediction",       # (512, 26, 26)  -->  (256, 26, 26)  --> (256, 26, 26) result2
    (128, 1, 1),             # (128, 26, 26)
    "UpSample",              # (128, 52, 52)   ### CONCAT dim=1 (channels)
    (128, 1, 1),             # (128, 52, 52)
    (256, 3, 1),             # (256, 52, 52)
    "ScalePrediction",       # (256,52, 52)  -->  (128, 52, 52)  -->  (128, 52, 52) result3
]  # 53 + 53 = 106 layers


class CNNBlock(nn.Module):
    def __init__(self, c1, c2, bn_act=True, **kwargs):  # batch_norm and activation
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, c1, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_repeats):
            self.layers += [nn.Sequential(
                CNNBlock(c1, c1 // 2, kernel_size=1),
                CNNBlock(c1 // 2, c1, kernel_size=3, padding=1),
            )]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, c1, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(c1, 2*c1, kernel_size=3, padding=1),
            # 3 output anchors, each (obj_prob, x, y, w, h)
            CNNBlock(2*c1, (num_classes+5)*3, bn_act=False, kernel_size=1),  # (output channel = (C+5)*3)
        )
        self.C = num_classes

    def forward(self, x):
        return (
            self.pred(x)
                .reshape(x.shape[0], 3, self.C + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2)  # (BATCH_SIZE, anchors=3, S, S, 5+C)
        )


class YOLOv3(LightningModule):
    def __init__(self, c1=3, num_classes=20):  # PASCAL_VOC=20, COCO=80
        super().__init__()
        self.c1 = c1
        self.C = num_classes
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each ScalePrediction
        route_connections = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                # print('###')
                # print(layer.__class__.__name__, '--------', list(x.shape))
                continue

            x = layer(x)
            # print(layer.__class__.__name__, '--------', list(x.shape))
            # if i == 10:  # end of Darknet-53
            #     print('\n')

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def training_step(self, batch, batch_idx):
        """
        return loss
        :param batch:
        :param batch_idx:
        :return:
        """
        pass  # return loss

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        c1 = self.c1

        for layer in config:
            if isinstance(layer, tuple):  # Conv Layer
                c2, k, s = layer
                layers.append(
                    CNNBlock(c1, c2, kernel_size=k, stride=s, padding=1 if k == 3 else 0)  # Conv + BN + LeakyReLU
                )
                c1 = c2  # Conv output channels --> next input
            elif isinstance(layer, list):  # ResidualBlock
                num_repeats = layer[1]
                layers.append(ResidualBlock(c1, num_repeats=num_repeats))
            elif isinstance(layer, str):  # ConvolutionalSet + ScalePrediction
                if layer == "ScalePrediction":
                    layers += [
                        # === ConvolutionalSet === #
                        ResidualBlock(c1, use_residual=False, num_repeats=1),  # bottleneck
                        CNNBlock(c1, c1 // 2, kernel_size=1),  # 1x1 kernel that reduces channels by half
                        # ======================== #
                        ScalePrediction(c1 // 2, num_classes=self.C)
                    ]
                    c1 = c1 // 2  # ConvolutionalSet output --> next input
                elif layer == "UpSample":
                    layers.append(nn.Upsample(scale_factor=2))  # mode={nearest(DEFAULT) OR bilinear}
                    c1 = c1 * 3  # concatenate right after the Upsample
                    # --> x = torch.cat([x, route_connections[-1]], dim=1)
                    # 128 --> 128+256, 256 --> 256+512

        return layers

    def configure_optimizers(self):
        """
        The LightningModule is subclassing Module --> you can access its children parameters directly with self.parameters().
        :return:
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)  # self.parameters() == model.parameters()


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    dummy_input = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(dummy_input)

    print(len(out))
    print(out[0].shape, out[1].shape, out[2].shape)
    assert model(dummy_input)[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5)
    assert model(dummy_input)[1].shape == (2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5)
    assert model(dummy_input)[2].shape == (2, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8, num_classes + 5)
    print("Success!")
