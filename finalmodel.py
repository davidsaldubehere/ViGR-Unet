import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from timm.models.layers import DropPath

import random
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GraphRefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch, n_classes, drop_path=0.1):
        super(GraphRefUnet, self).__init__()

        # Add drop path for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Replace conv0 with graph conv
        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)
        
        # Replace conv1 block
        self.conv1 = DynamicGraphConvBlock(inc_ch, K=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        # Replace conv2 block
        self.conv2 = DynamicGraphConvBlock(64, K=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        # Replace conv3 block
        self.conv3 = DynamicGraphConvBlock(64, K=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        # Replace conv4 block
        self.conv4 = DynamicGraphConvBlock(64, K=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        # Replace conv5 block
        self.conv5 = DynamicGraphConvBlock(64, K=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)
        self.dp_d4 = DropPath(drop_path * 0.8) if drop_path > 0. else nn.Identity()

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)
        self.dp_d3 = DropPath(drop_path * 0.6) if drop_path > 0. else nn.Identity()

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)
        self.dp_d2 = DropPath(drop_path * 0.4) if drop_path > 0. else nn.Identity()

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)
        self.dp_d1 = DropPath(drop_path * 0.2) if drop_path > 0. else nn.Identity()

        self.conv_d0 = nn.Conv2d(64,n_classes,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self,x):
        # The forward pass remains the same as RefUnet but with drop path added
        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.dp_d4(self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1)))))
        hx = self.upscore2(d4)

        d3 = self.dp_d3(self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1)))))
        hx = self.upscore2(d3)

        d2 = self.dp_d2(self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1)))))
        hx = self.upscore2(d2)

        d1 = self.dp_d1(self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1)))))

        residual = self.conv_d0(d1)

        return x + self.drop_path(residual)
    
class Stem(nn.Module):
    def __init__(self, input_dim=1, output_dim=64, drop_path=0.05):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(output_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.act(self.bn(self.conv(x))))
    

class DepthWiseSeparable(nn.Module):
    def __init__(self, in_dim, kernel, expansion=4):
        super().__init__()

        self.pw1 = nn.Conv2d(in_dim, in_dim * 4, 1) # kernel size = 1
        self.norm1 = nn.BatchNorm2d(in_dim * 4)
        self.act1 = nn.GELU()
        
        self.dw = nn.Conv2d(in_dim * 4, in_dim * 4, kernel_size=kernel, stride=1, padding=1, groups=in_dim * 4) # kernel size = 3
        self.norm2 = nn.BatchNorm2d(in_dim * 4)
        self.act2 = nn.GELU()
        
        self.pw2 = nn.Conv2d(in_dim * 4, in_dim, 1)
        self.norm3 = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        x = self.pw1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.dw(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        x = self.pw2(x)
        x = self.norm3(x)
        return x

    
class InvertedResidual(nn.Module):
    def __init__(self, dim, kernel, expansion_ratio=4., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.dws = DepthWiseSeparable(in_dim=dim, kernel=kernel, expansion=expansion_ratio)
        self.dropout = nn.Dropout(drop) if drop > 0. else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.dropout(self.dws(x)))
        else:
            x = x + self.drop_path(self.dropout(self.dws(x)))
        return x
   

class DynamicMRConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.K = K
        self.mean = 0
        self.std = 0
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_j = x - x

        # get an estimate of the mean distance by computing the distance of points b/w quadrants. This is for efficiency to minimize computations.
        x_rolled = torch.cat([x[:, :, -H//2:, :], x[:, :, :-H//2, :]], dim=2)
        x_rolled = torch.cat([x_rolled[:, :, :, -W//2:], x_rolled[:, :, :, :-W//2]], dim=3)

        # Norm, Euclidean Distance
        norm = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True)

        self.mean = torch.mean(norm, dim=[2,3], keepdim=True)
        self.std = torch.std(norm, dim=[2,3], keepdim=True)

        for i in range(0, H, self.K):
            x_rolled = torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)

            dist = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True)

            # Got 83.86%
            mask = torch.where(dist < self.mean - self.std, 1, 0)

            x_rolled_and_masked = (x_rolled - x) * mask
            x_j = torch.max(x_j, x_rolled_and_masked)

        for j in range(0, W, self.K):
            x_rolled = torch.cat([x[:, :, :, -j:], x[:, :, :, :-j]], dim=3)

            dist = torch.norm((x - x_rolled), p=1, dim=1, keepdim=True)

            mask = torch.where(dist < self.mean - self.std, 1, 0)

            x_rolled_and_masked = (x_rolled - x) * mask
            x_j = torch.max(x_j, x_rolled_and_masked)
                 
        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class ConditionalPositionEncoding(nn.Module):
    """
    Implementation of conditional positional encoding. For more details refer to paper: 
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_
    """
    def __init__(self, in_channels, kernel_size, drop_path=0.1):
        super().__init__()
        self.pe = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=in_channels
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.pe(x))
        return x


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, K, drop_path=0.1):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.cpe = ConditionalPositionEncoding(in_channels, kernel_size=7, drop_path=drop_path)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DynamicMRConv4d(in_channels * 2, in_channels, K=self.K)  
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )  # out_channels back to 1x}
        self.dropout = nn.Dropout(0.1)  # Add dropout
       
    def forward(self, x):
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)

        return x

    
class DynamicGraphConvBlock(nn.Module):
    def __init__(self, in_dim, drop_path=0., K=2, use_layer_scale=True, layer_scale_init_value=1e-5, dropout=0.1):
        super().__init__()
        
        self.mixer = Grapher(in_dim, K, drop_path=drop_path*0.5)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),  # Add dropout in FFN
            nn.Conv2d(in_dim * 4, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(in_dim), requires_grad=True) 
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(in_dim), requires_grad=True) 
        
    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(x))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(x))
        else:
            x = x + self.drop_path(self.mixer(x))
            x = x + self.drop_path(self.ffn(x))
        return x


class Downsample(nn.Module):
    """ 
    Convolution-based downsample
    """
    def __init__(self, in_dim, out_dim, drop_path=0.05):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        x = self.drop_path(self.conv(x))
        return x


class Backbone(torch.nn.Module):
    def __init__(self, blocks, channels, kernels, stride,
                 act_func, dropout=0.1, drop_path=0.3, emb_dims=512,
                 K=2, distillation=True, num_classes=4,
                 out_indices=None, deep=True):
        super(Backbone, self).__init__()
        self.deep = deep
        self.CatChannels = channels[0]
        self.CatBlocks = 4
        self.UpChannels = self.CatChannels * self.CatBlocks
        
        # Add dropout to segmentation heads
        self.dropout = nn.Dropout(dropout)
        self.segmentation_head = SegmentationHead(self.UpChannels, num_classes, dropout)
        
        # Create seg heads for the intermediate decoders with dropout
        self.deep_segmentation_head1 = SegmentationHead(self.UpChannels, num_classes, dropout)
        self.deep_segmentation_head2 = SegmentationHead(self.UpChannels, num_classes, dropout)
        self.deep_segmentation_head3 = SegmentationHead(channels[-1], num_classes, dropout)

        # Pass drop_path to GraphRefUnet
        self.ref = GraphRefUnet(num_classes, 64, num_classes, drop_path=drop_path * 0.5)

        self.distillation = distillation
        self.out_indices = out_indices        
        n_blocks = sum([sum(x) for x in blocks])
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]  # stochastic depth decay rule 
        dpr_idx = 0

        # Pass drop_path to Stem
        self.stem = Stem(input_dim=1, output_dim=channels[0], drop_path=drop_path * 0.05)

        # Add drop path to decoder blocks
        self.dp_hd3 = DropPath(drop_path * 0.5) if drop_path > 0. else nn.Identity()
        self.dp_hd2 = DropPath(drop_path * 0.4) if drop_path > 0. else nn.Identity()
        self.dp_hd1 = DropPath(drop_path * 0.3) if drop_path > 0. else nn.Identity()

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(channels[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)
        self.h1_PT_hd3_dropout = nn.Dropout(dropout)  # Add dropout

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(channels[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)
        self.h2_PT_hd3_dropout = nn.Dropout(dropout)  # Add dropout

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(channels[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)
        self.h3_Cat_hd3_dropout = nn.Dropout(dropout)  # Add dropout

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(channels[3], self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)
        self.hd4_UT_hd3_dropout = nn.Dropout(dropout)  # Add dropout

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(channels[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)
        self.h1_PT_hd2_dropout = nn.Dropout(dropout)  # Add dropout

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(channels[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)
        self.h2_Cat_hd2_dropout = nn.Dropout(dropout)  # Add dropout

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)
        self.hd3_UT_hd2_dropout = nn.Dropout(dropout)  # Add dropout

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(channels[3], self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)
        self.hd4_UT_hd2_dropout = nn.Dropout(dropout)  # Add dropout

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(channels[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)
        self.h1_Cat_hd1_dropout = nn.Dropout(dropout)  # Add dropout

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)
        self.hd2_UT_hd1_dropout = nn.Dropout(dropout)  # Add dropout

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)
        self.hd3_UT_hd1_dropout = nn.Dropout(dropout)  # Add dropout

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(channels[3], self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)
        self.hd4_UT_hd1_dropout = nn.Dropout(dropout)  # Add dropout

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.backbone = []
        for i in range(len(blocks)):
            stage = []
            local_stages = blocks[i][0]
            global_stages = blocks[i][1]
            if i > 0:
                # Pass drop_path to Downsample
                stage.append(Downsample(channels[i-1], channels[i], drop_path=drop_path * 0.1))
            for _ in range(local_stages):
                # Add dropout to InvertedResidual
                stage.append(InvertedResidual(dim=channels[i], kernel=3, expansion_ratio=4, drop=dropout, drop_path=dpr[dpr_idx]))
                dpr_idx += 1
            for _ in range(global_stages):
                # Pass dropout to DynamicGraphConvBlock
                stage.append(DynamicGraphConvBlock(channels[i], drop_path=dpr[dpr_idx], K=K[i], dropout=dropout))
                dpr_idx += 1
            self.backbone.append(nn.Sequential(*stage))
            
        self.backbone = nn.Sequential(*self.backbone)

        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def forward(self, inputs):
        x = self.stem(inputs)
        encoder_features = []
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in self.out_indices:
                encoder_features.append(x)

        h1 = encoder_features[0]
        h2 = encoder_features[1]
        h3 = encoder_features[2]
        h4 = encoder_features[3]

        ## -------------Decoder-------------
        hd4 = h4

        # Add dropout in decoder branches
        h1_PT_hd3 = self.h1_PT_hd3_dropout(self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1)))))
        h2_PT_hd3 = self.h2_PT_hd3_dropout(self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2)))))
        h3_Cat_hd3 = self.h3_Cat_hd3_dropout(self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3))))
        hd4_UT_hd3 = self.hd4_UT_hd3_dropout(self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4)))))
        
        # Add drop path to decoder fusion
        hd3 = self.dp_hd3(self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3), 1))))) # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_dropout(self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1)))))
        h2_Cat_hd2 = self.h2_Cat_hd2_dropout(self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2))))
        hd3_UT_hd2 = self.hd3_UT_hd2_dropout(self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3)))))
        hd4_UT_hd2 = self.hd4_UT_hd2_dropout(self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4)))))
        
        # Add drop path to decoder fusion
        hd2 = self.dp_hd2(self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2), 1))))) # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_dropout(self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1))))
        hd2_UT_hd1 = self.hd2_UT_hd1_dropout(self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2)))))
        hd3_UT_hd1 = self.hd3_UT_hd1_dropout(self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3)))))
        hd4_UT_hd1 = self.hd4_UT_hd1_dropout(self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4)))))
        
        # Add drop path to decoder fusion
        hd1 = self.dp_hd1(self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1), 1))))) # hd1->320*320*UpChannels

        # Add dropout before segmentation heads
        upsampled_out = nn.functional.interpolate(hd1, size=inputs.shape[2:], mode='bilinear', align_corners=False)
        segmentation_map = self.segmentation_head(self.dropout(upsampled_out))
        refined_segmentation_map = self.ref(segmentation_map)

        # Deep supervision outputs with dropout
        if self.deep:
            upsampled_out1 = nn.functional.interpolate(hd2, size=inputs.shape[2:], mode='bilinear', align_corners=False)
            upsampled_out2 = nn.functional.interpolate(hd3, size=inputs.shape[2:], mode='bilinear', align_corners=False)
            upsampled_out3 = nn.functional.interpolate(hd4, size=inputs.shape[2:], mode='bilinear', align_corners=False)
            
            # Add dropout before heads
            out1 = self.deep_segmentation_head1(self.dropout(upsampled_out1))
            out2 = self.deep_segmentation_head2(self.dropout(upsampled_out2))
            out3 = self.deep_segmentation_head3(self.dropout(upsampled_out3))

            return [refined_segmentation_map, out1, out2, out3]
            
        else:    
            return refined_segmentation_map


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)  # Add dropout
        self.conv2 = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x)
        return x

def greedyvig_b_feat(deep=True, pretrained=True, dropout=0.5, drop_path=0.3):
     model = Backbone(blocks=[[4,4], [4,4], [12,4], [3,3]],
                     channels=[64, 128, 256, 512],
                     kernels=3,
                     stride=1,
                     act_func='gelu',
                     dropout=dropout,  # Use the provided dropout value
                     drop_path=drop_path,  # Use the provided drop_path value
                     emb_dims=768,
                     K=[8, 4, 2, 1],
                     distillation=True,
                     num_classes=4,
                     out_indices=[0, 1, 2, 3],
                     deep=deep,
                    )
     return model
