from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision

class ResNet50TP(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50TP, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))

        # collect individual frame features and global video features by avg pooling
        spatial_out = self.base(x)
        avg_spatial_out = F.avg_pool2d(spatial_out, spatial_out.size()[2:])
        individual_img_features = avg_spatial_out.view(b,t,-1)
        individual_features_permuted = individual_img_features.permute(0,2,1)
        video_features = F.avg_pool1d(individual_features_permuted, t)
        video_features = video_features.view(b, self.feat_dim)

        if not self.training:
            return video_features, individual_img_features
        
        y = self.classifier(video_features)
        return y, video_features, individual_img_features
