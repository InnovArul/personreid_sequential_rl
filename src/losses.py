from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['CrossEntropyLabelSmooth', 'TripletLoss', 'CenterLoss']

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = Variable(targets, requires_grad=False)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class CosegLoss(nn.Module):
    """Cosegmentation loss

    Args:
        num_classes (int): number of classes.
        seq_len (int): number of image frames per sequence
    """
    def __init__(self):
        super(CosegLoss, self).__init__()

    def mean_l2(self, vec1, vec2):
        return torch.mean(torch.pow(vec1 - vec2, 2))

    def mask_loss(self, fg1, bg1, fg2, bg2):
        fg_mean_diff = self.mean_l2(fg1, fg2)
        fg_bg_mean_diff = (self.mean_l2(fg1, bg1) + self.mean_l2(fg2, bg2)) / 2
        current_diff_bt_diff = fg_bg_mean_diff - fg_mean_diff
        return -torch.log(torch.sigmoid(current_diff_bt_diff))
        
    def forward(self, fg_features, bg_features):
        """
        Args:
            inputs: fg features, bg features of size (batch * seq_len, feat_dim)
        """
        num_persons, num_frames, feat_dim = fg_features.shape
        total_loss = []
    
        for b in range(num_persons):
            current_sequence_loss = []
            for i in range(num_frames):
                fg_i = fg_features[b][i]
                bg_i = bg_features[b][i]

                for j in range(num_frames):
                    fg_j = fg_features[b][j]
                    bg_j = bg_features[b][j]
                    current_sequence_loss.append(self.mask_loss(fg_i, bg_i, fg_j, bg_j))

            total_loss.append(torch.stack(current_sequence_loss).sum())
    
        loss = torch.stack(total_loss).sum()
        # print('non eff', loss)

        # efficient computations
        fg_norm = torch.sum(torch.pow(fg_features, 2), dim=2, keepdim=True)
        repeated_fg_norm = fg_norm.repeat(1, 1, num_frames)

        bg_norm = torch.sum(torch.pow(bg_features, 2), dim=2, keepdim=True)

        fg_mutual_distance = torch.abs(repeated_fg_norm + 
                                       repeated_fg_norm.transpose(1,2) - 
                                       2 * (fg_features @ fg_features.transpose(1,2)))

        fg_bg_pairwise_distance = torch.abs(fg_norm + bg_norm - 2 * torch.sum((fg_features * bg_features), dim=2, keepdim=True))
        # print(fg_bg_pairwise_distance.shape)
        fg_bg_pairwise_distance = fg_bg_pairwise_distance.repeat(1, 1, num_frames)
        # print(fg_bg_pairwise_distance.shape)
        fg_bg_pairwise_distance = fg_bg_pairwise_distance + fg_bg_pairwise_distance.transpose(1,2)
        # print(fg_bg_pairwise_distance.shape)

        # normalize
        fg_mutual_distance = fg_mutual_distance / (feat_dim)
        fg_bg_pairwise_distance = fg_bg_pairwise_distance / (2 * feat_dim)

        log_loss = -torch.log(torch.sigmoid(fg_bg_pairwise_distance - fg_mutual_distance))
        log_loss = log_loss.sum()
        # print('eff', log_loss)

        return log_loss 

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class TripletLoss_WeightedNorm(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, feat_dim, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.feat_dim = feat_dim
        self.weights = nn.Parameter(torch.randn(self.feat_dim))
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n, seq_len, feat_dim = inputs.shape
        assert feat_dim == self.feat_dim
        weights = torch.sqrt(self.weights[None, None, :]).repeat(n, seq_len, 1)
        inputs = inputs * weights

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        classes = Variable(classes)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

if __name__ == '__main__':
    pass