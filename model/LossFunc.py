import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossEntropy(nn.Module):
    """
        Masked cross entropy loss function with long-term unbalanced weight.
        class_weight: List[float]
        long_length: None or int
    """
    def __init__(self, hyper_params):
        super(MaskedCrossEntropy, self).__init__()

        class_weight = hyper_params['class_weight']
        self.long_length = hyper_params['long_length']
        self.additional_weight = torch.FloatTensor(class_weight).view(-1, 1, 1).to(hyper_params['device']) - 1.0
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred: torch.Tensor, label: torch.LongTensor, mask: torch.BoolTensor):
        """
            pred: 1 x 10 x L x L
            label: 1 x L x L
            mask: 1 x L x L
            out: 1
        """
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
    """
        Masked focal loss function with long-term unbalanced weight.
        class_weight: List[float]
        gamma: float
        eps: float
    """
    def __init__(self, class_weight, gamma = 2.0, eps = 1e-6):
        super(MaskedFocalLoss, self).__init__()
        self.class_weight = torch.FloatTensor(class_weight)
        self.gamma = gamma
        self.eps = eps
        self.cross_entropy = nn.NLLLoss(reduction = 'none')
        
    def forward(self, pred: torch.FloatTensor, label: torch.LongTensor, mask: torch.BoolTensor):
        """
            pred: 1 x 10 x L x L
            label: 1 x L x L
            mask: 1 x L x L
            out: 1
        """
        pred = torch.softmax(pred, dim=1)
        pred = torch.log(pred + self.eps)

        class_weight = self.class_weight.to(pred.device)
        loss = self.cross_entropy(pred, label.long())
        pt = torch.exp(-loss)
        loss = loss * mask
        class_weight = class_weight.gather(0, label.view(-1)).reshape(label.shape)
        loss = loss * class_weight

        loss = loss * torch.pow(1 - pt, self.gamma)
        loss = loss.sum(dim = [1, 2]) / mask.sum(dim = [1, 2])
        loss = torch.mean(loss)

        return loss
