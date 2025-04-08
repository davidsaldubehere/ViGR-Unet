import torch
import torch.nn as nn
import torch.nn.functional as F

#removed one depth level since the 256 x 256 is too small

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4):
        super().__init__()
        
        # Contracting Path (3 levels)
        self.conv1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottom (changed from conv5 to conv4)
        self.conv4 = DoubleConv(256, 512)
        
        # Expansive Path (3 levels)
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(256 + 256, 256)  # 256 (from up5) + 256 (from conv3)
        
        self.up6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(128 + 128, 128)  # 128 (from up6) + 128 (from conv2)
        
        self.up7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(64 + 64, 64)     # 64 (from up7) + 64 (from conv1)
        
        # Final 1x1 convolution remains the same
        self.conv8 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        inputSize = x.size()
        
        # Contracting Path
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        # Bottom
        conv4 = self.conv4(pool3)
        
        # Expansive Path
        up5 = self.up5(conv4)
        # Crop conv3 to match up5
        crop3 = self.crop(conv3, up5)
        merge5 = torch.cat([crop3, up5], dim=1)
        conv5 = self.conv5(merge5)
        
        up6 = self.up6(conv5)
        # Crop conv2 to match up6
        crop2 = self.crop(conv2, up6)
        merge6 = torch.cat([crop2, up6], dim=1)
        conv6 = self.conv6(merge6)
        
        up7 = self.up7(conv6)
        # Crop conv1 to match up7
        crop1 = self.crop(conv1, up7)
        merge7 = torch.cat([crop1, up7], dim=1)
        conv7 = self.conv7(merge7)
        
        # Final 1x1 convolution with upsampling to original size
        out = F.interpolate(conv7, size=inputSize[2:], mode='bilinear', align_corners=True)
        out = self.conv8(out)
        
        return out

    def crop(self, tensor, target_tensor):
        target_height, target_width = target_tensor.size()[2:]
        tensor_height, tensor_width = tensor.size()[2:]
        
        delta_height = tensor_height - target_height
        delta_width = tensor_width - target_width
        
        crop_top = delta_height // 2
        crop_bottom = delta_height - crop_top
        crop_left = delta_width // 2
        crop_right = delta_width - crop_left
        
        return tensor[:, :, crop_top:tensor_height-crop_bottom, crop_left:tensor_width-crop_right]
