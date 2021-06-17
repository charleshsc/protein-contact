import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class foolDiscriminator(nn.Module):
    def __init__(self):
        super(foolDiscriminator, self).__init__()
    def forward(self, prediction, mask):
        ones = torch.ones_like(prediction) - 1e-6
        zeros = torch.zeros_like(prediction) + 1e-6
        prediction = torch.where(prediction>=1, ones, prediction)
        prediction = torch.where(prediction<=0, zeros, prediction)
        loss = -torch.log(prediction)
        loss = loss * mask
        loss = loss.sum(dim=[2,3]) / mask.sum(dim=[1,2])
        return torch.mean(loss)

class focal_loss(nn.Module):
    def __init__(self, alpha, gamma):
        super(focal_loss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self, result, gt, mask):
        if len(gt.shape) == 3:
            gt = F.one_hot(gt, num_classes = 10).permute(0, 3, 1, 2).type(torch.float)
        ones = torch.ones_like(result) - 1e-6
        zeros = torch.zeros_like(result) + 1e-6
        result = torch.where(result>=1, ones, result)
        result = torch.where(result<=0, zeros, result)
        loss = self._alpha * torch.pow((1-result),self._gamma)*gt* torch.log(result) + (1- self._alpha)* torch.pow(result, self._gamma) * (1 - gt) * torch.log(1-result)
        if torch.sum(torch.isnan(loss)) > 0:
            raise Exception("Invalid Non")

        loss = loss.contiguous()
        loss = -loss
        loss = loss * mask
        loss = loss.sum(dim=[2, 3]) / mask.sum(dim=[1, 2])
        loss = torch.mean(loss)
        return loss

class symmetrical_loss(nn.Module):
    def __init__(self):
        super(symmetrical_loss, self).__init__()

        self.mse = nn.MSELoss(reduction='none')

    def forward(self, result, mask):
        result_T = result.permute(0,1,3,2)
        loss = -self.mse(result, result_T)
        loss = loss * mask
        loss = loss.sum(dim=[2, 3]) / mask.sum(dim=[1, 2])
        return torch.mean(loss)

class MaskCrossEntropy(nn.Module):
    def __init__(self):
        super(MaskCrossEntropy, self).__init__()

        self.weight = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0.4])
        if torch.cuda.is_available():
            self.weight = self.weight.cuda()
        self.CrossEntropy = nn.CrossEntropyLoss(weight=self.weight,reduction='none')

    def forward(self, result, gt, mask):
        loss = self.CrossEntropy(result, gt)
        loss = loss * mask
        loss = loss.sum(dim=[1,2]) / mask.sum(dim=[1,2])
        return torch.mean(loss)

class MaskNLLLoss(nn.Module):
    def __init__(self):
        super(MaskNLLLoss, self).__init__()

        self.weight = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0.4])
        if torch.cuda.is_available():
            self.weight = self.weight.cuda()
        self.NLLLoss = nn.NLLLoss(weight=self.weight,reduction='none')

    def forward(self, result, gt, mask):
        loss = self.NLLLoss(result, gt)
        loss = loss * mask
        loss = loss.sum(dim=[1,2]) / mask.sum(dim=[1,2])
        return torch.mean(loss)

class GeneratorLoss(nn.Module):
    def __init__(self, alpha = 0.25, beta = 1, gamma = 2, lamda = 1):
        super(GeneratorLoss, self).__init__()
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._lambda = lamda

        self.L_G_adv = foolDiscriminator()
        self.L_G_F = focal_loss(self._alpha,self._gamma)
        self.L_G_S = symmetrical_loss()
        self.maskCrossEntropy = MaskCrossEntropy()
        self.masknllloss = MaskNLLLoss()

    def forward(self, prediction, result, gt, mask):
        loss1 = self.L_G_adv(prediction,mask)
        loss2 = self.L_G_F(result, gt, mask)
        loss3 = self.L_G_S(result, mask)
        # loss4 = self.maskCrossEntropy(result,gt,mask)
        loss = self._lambda * loss1 + self._beta * loss2 + loss3

        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, res, gt, mask):
        if len(gt.shape) == 3:
            gt = F.one_hot(gt, num_classes=10).permute(0, 3, 1, 2).type(torch.float)
        loss = self.bce_loss(res,gt)
        loss = loss * mask
        loss = loss.sum(dim=[2, 3]) / mask.sum(dim=[1, 2])
        mean_loss = torch.mean(loss)
        return mean_loss