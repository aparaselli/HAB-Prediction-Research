import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        alpha: weight for the positive class (set <0.5 to up-weight negatives)
        gamma: focusing parameter (higher => more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B,) or (B,1) for binary BCEWithLogits
        # targets: (B,)  with 0/1
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)                # p_t = sigmoid(logits) if target=1, else 1−sigmoid
        loss = self.alpha * (1 - p_t)**self.gamma * bce
        
        if self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()
        else:
            return loss
