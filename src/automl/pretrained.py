from torch import nn
from torchvision.models import get_model


class Mobilenet(nn.Module):
    def __init__(self, input_shape=None, num_classes=17):
        super().__init__()

        self.type = "mobilenet"

        if self.type == "mobilenet":
            # load pretrained model
            mobilenet = get_model("mobilenet_v3_small", weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1")

            # separate layers
            backbone, _avgpool, _head = mobilenet.children()
            backbone_layers = list(backbone.children())

            ## layer 5: (c h w) = (40 14 14)
            # backbone_reduced = nn.Sequential(
            #     *backbone_layers[:],
            # )
            
            self.backbone = backbone
            
            # trainable linear head
            head_in = 576
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(head_in, 128),
                nn.Hardswish(),
                nn.Dropout(p=0.2),
                nn.Linear(128, num_classes)
            )


            if False:
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
        
    def train_backbone(self, is_trainable):
        for param in self.backbone.parameters():
            param.requires_grad = is_trainable

    def forward(self, images):
        # images: (b, c, h, w)
        batch_size, channels, h, w = images.shape
        
        # if 1 channel, repeat the torch tensor
        if channels < 3:
            images = images.repeat(1, 3, 1, 1)
        
        if self.type == "mobilenet":
            ## trainable conv
            # op1 = self.backbone(images) # (b, c, h, w) -> (b, 40, 14, 14)
            # op2 = self.block1(op1)
            # x = self.head(op2)
            # x = self.fc(x) # (b, 512) -> (b, 1)

            x = self.backbone(images)
            x = self.head(x)

        return x