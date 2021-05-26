import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossEntropy(nn.Module):
    def __init__(self, hyper_params):
        super(MaskedCrossEntropy, self).__init__()

        class_weight = hyper_params['class_weight']
        self.class_weight = torch.FloatTensor(class_weight).view(-1, 1, 1).to(hyper_params['device'])
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred: torch.Tensor, label: torch.LongTensor, mask: torch.BoolTensor):
        loss = self.cross_entropy(pred, label)
        m = mask.shape[1]
        class_weight = self.class_weight.expand(10, m, m)
        weight_mask = class_weight.gather(0, label)
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
