import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossEntropy(nn.Module):
    def __init__(self, hyper_params):
        super(MaskedCrossEntropy, self).__init__()

        class_weight = hyper_params['class_weight']
        self.long_length = hyper_params['long_length']
        self.additional_weight = torch.FloatTensor(class_weight).view(-1, 1, 1).to(hyper_params['device']) - 1.0
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred: torch.Tensor, label: torch.LongTensor, mask: torch.BoolTensor):
        loss = self.cross_entropy(pred, label)
        m = mask.shape[1]
        
        if self.long_length:
            base_weight = torch.ones(label.shape, dtype=torch.float, device=label.device)
            trunc_mat = torch.zeros([m, m], dtype=torch.float)
            for kk in range(self.long_length):
                if kk != 0:
                    trunc_mat = trunc_mat + \
                        torch.diag(torch.ones(m - kk), kk) + torch.diag(torch.ones(m - kk), -kk)
                else:
                    trunc_mat = trunc_mat + torch.diag(torch.ones(m - kk), kk)
            trunc_mat = 1.0 - trunc_mat.unsqueeze(0).to(label.device)
            class_weight = self.additional_weight.expand(10, m, m)
            weight_mask = class_weight.gather(0, label)
            weight_mask = base_weight + weight_mask * trunc_mat
            mask = weight_mask * mask
            
        loss = loss * mask
        loss = torch.sum(loss) / (m ** 2)
        
        return loss


class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma = 2.0, eps = 1e-6):
        super(MaskedFocalLoss, self).__init__()
        self.alpha = torch.FloatTensor(alpha)
        self.gamma = gamma
        self.eps = eps
        self.cross_entropy = nn.NLLLoss(reduction = 'none')
        
    def forward(self, res, gt, mask):
        # SoftMax
        res = F.softmax(res, dim=1)
        res = torch.log(res + self.eps)
        alpha = self.alpha.to(res.device)
        loss = self.cross_entropy(res, gt.long())
        pt = torch.exp(- loss)
        loss = loss * mask
        alpha = alpha.gather(0, gt.view(-1)).reshape(gt.shape)
        loss = loss * alpha
        loss = loss * torch.pow(1 - pt, self.gamma)
        sample_loss = loss.sum(dim = [1, 2]) / mask.sum(dim = [1, 2])
        mean_batch_loss = torch.mean(sample_loss)
        return mean_batch_loss
