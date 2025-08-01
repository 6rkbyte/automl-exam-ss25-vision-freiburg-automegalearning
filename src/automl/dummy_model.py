from torch import nn
from torchvision.models import get_model


class DummyNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        num_neurons = 128
        layers = [nn.Flatten(),
                  nn.Linear(input_size, num_neurons),
                  nn.ReLU(),
                  nn.Linear(num_neurons, output_size)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def ConvBlock(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0)):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(num_features=in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.Dropout(p=0.8),
    )

class CNN(nn.Module):
    def __init__(self, input_shape=None, num_classes=17):
        super().__init__()

        self.type = "mobilenet"

        if self.type == "mobilenet":
            # load pretrained model
            mobilenet = get_model("mobilenet_v3_small", weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1")

            # separate layers
            backbone, _avgpool, head = mobilenet.children()
            backbone_layers = list(backbone.children())

            # remove some later layers
            # (c h w) = (40 14 14)
            backbone_reduced = nn.Sequential(
                *backbone_layers[:6],
                # nn.Sequential(
                #     nn.AdaptiveAvgPool2d((5, 5)),
                #     nn.Flatten()
                # )
            )
            self.backbone = backbone_reduced

            # trainable convolution
            self.block1 = ConvBlock(40, 120)
            
            # avarage pooling
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )

            # trainable FC head
            in_feat = 120 * 1 * 1
            self.fc = nn.Linear(in_feat, num_classes)

            # freeze backbone
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images):
        # images: (b, c, h, w)
        batch_size, channels, h, w = images.shape
        
        # if 1 channel, repeat the torch tensor
        if channels < 3:
            images = images.repeat(1, 3, 1, 1)
        
        if self.type == "mobilenet":
            op1 = self.backbone(images) # (b, c, h, w) -> (b, 40, 14, 14)
            op2 = self.block1(op1)
            x = self.head(op2)
            x = self.fc(x) # (b, 512) -> (b, 1)

        return x