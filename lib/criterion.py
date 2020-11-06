import torch
import torch.nn.functional as F
import torch.nn as nn

class ReliableCrossEntropyLoss(nn.Module):
  def __init__(self):
    super(ReliableCrossEntropyLoss, self).__init__()

  def forward(self, x, y, rlb):
    batch_size  = x.shape[0]

    preds = F.softmax(x, 1)
    self_preds = preds.gather(1, y.view(-1, 1)).view(-1)

    #if rlb.features.sum() == 0:
    if rlb.features is None:
      reliable_preds_1 = reliable_preds_2 = 0
    else:
      reliables_1, reliables_2 = rlb.reliablity(y)
      reliable_preds_1 = torch.mul(preds, reliables_1).sum(1)
      reliable_preds_2 = torch.mul(preds, reliables_2).sum(1) / (reliables_2.sum(1) + 1e-6)

    loss = -1 * torch.log(self_preds + reliable_preds_1 + reliable_preds_2).sum(0)

    return loss / batch_size
