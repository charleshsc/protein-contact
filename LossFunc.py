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
