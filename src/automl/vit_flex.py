from torch import nn
from torchvision.models import get_model
from torchvision import transforms
import timm

class VitFlex(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, params: dict):
        super().__init__()
        
        # Get architecture params
        n_layers = params["linear_n_layers"]
        linear_size_scale = params["linear_size_scale"]

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Fully connected layers
        head = []
        in_features = self.vit.head.in_features
        for i in range(n_layers): # 1 or 4 layers
            out_features = 64 * linear_size_scale # 64 or 64*8 
            head += [
                nn.Linear(in_features, out_features),
                nn.Hardswish(),
                nn.Dropout(p=0.1),
            ]
            in_features = out_features
        
        # output layer
        head += [
            nn.Linear(out_features, num_classes)
        ] 
        
        self.vit.head = nn.Sequential(*head)

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













class CNN_flex(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, params: dict):
        super().__init__()
        self.layers = nn.ModuleList()

        # Get architecture params
        n_layers = params["linear_n_layers"]
        conv_n_layers = params["conv_n_layers"]

        # Initial conv layers
        channels = in_channels
        for _ in range(conv_n_layers):
            self.layers.append(nn.Conv2d(channels, channels * 3, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.GELU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            channels *= 3

        self.layers.append(nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.AdaptiveMaxPool2d((12, 12)))
        self.layers.append(nn.Flatten())
        channels *= 2
        in_features = 12 * 12 * channels

        # Fully connected layers
        for i in range(n_layers):
            out_features = params[f"n_units_l{i}"]
            #dropout = params[f"dropout_l{i}"]
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
            #self.layers.append(nn.Dropout(dropout))
            in_features = out_features

        # Output layer
        self.layers.append(nn.Linear(in_features, num_classes))
        self.layers.append(nn.LogSoftmax(dim=1))

        # Wrap in a Sequential
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)