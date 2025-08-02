from torch import nn
from torchvision.models import get_model
from torchvision import transforms
import timm

class Vit(nn.Module):
    def __init__(self, input_shape=None, num_classes=17):
        super().__init__()

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.head = nn.Sequential(
            nn.Linear(self.vit.head.in_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, images):
        # images: (b, c, h, w)
        batch_size, channels, h, w = images.shape
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        images = transform(images) # (c, h, w)
        
        # if 1 channel, repeat the torch tensor
        if channels < 3:
            images = images.repeat(1, 3, 1, 1)

        x = self.vit(images) # (b, c, h, w) -> (b, 768) -> (b, num_classes)
        return x