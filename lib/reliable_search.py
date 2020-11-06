import torch
import torch.nn as nn
import torch.nn.functional as F

class ReliableSearch(nn.Module):
  def __init__(self, nsamples, nfeatures, threshold_1, threshold_2, batch_size=128):
    super(ReliableSearch, self).__init__()
    self.samples_num = nsamples
    self.threshold_1 = threshold_1
    self.threshold_2 = threshold_2
    self.batch_size = batch_size

    #self.register_buffer('features', torch.zeros(nsamples, nfeatures))
    self.features = None
    self.register_buffer('position', torch.arange(nsamples).long())


  def update(self, memory, labels):
    with torch.no_grad():
      batch_size = self.batch_size
      self.features = memory
      corrects_1 = corrects_2 = 0.0
      num_reliable_1 = num_reliable_2 = 0.0

      for start in range(0, self.samples_num, batch_size):
        end = start + batch_size
        end = min(end, self.samples_num)

        sims = torch.mm(self.features[start:end], self.features.t())

        sims.scatter_(1, self.position[start:end].view(-1, 1), -1)
        reliable_matrix_1 = (sims > self.threshold_1).float()
        num_reliable_1 += reliable_matrix_1.sum().item()

        sims -= (sims > self.threshold_1).float()
        reliable_matrix_2 = (sims > self.threshold_2).float()
        num_reliable_2 += reliable_matrix_2.sum().item()

        reliable_indices_1 = reliable_matrix_1.nonzero()
        centers_1 = reliable_indices_1[:,0]+start
        neighbors_1 = reliable_indices_1[:,1]
        center_labels_1 = labels.index_select(0, centers_1.view(-1))
        neighbour_labels_1 = labels.index_select(0, neighbors_1.view(-1))
        corrects_1 += torch.eq(center_labels_1, neighbour_labels_1).sum().item()

        reliable_indices_2 = reliable_matrix_2.nonzero()
        centers_2 = reliable_indices_2[:,0]+start
        neighbors_2 = reliable_indices_2[:,1]
        center_labels_2 = labels.index_select(0, centers_2.view(-1))
        neighbour_labels_2 = labels.index_select(0, neighbors_2.view(-1))
        corrects_2 += torch.eq(center_labels_2, neighbour_labels_2).sum().item()

        del sims
        del reliable_matrix_1, reliable_indices_1, centers_1, neighbors_1, center_labels_1, neighbour_labels_1
        del reliable_matrix_2, reliable_indices_2, centers_2, neighbors_2, center_labels_2, neighbour_labels_2

      return num_reliable_1/float(self.samples_num), corrects_1/float(num_reliable_1), num_reliable_2/float(self.samples_num), corrects_2/float(num_reliable_2)


  def reliablity(self, index):
    batch = self.features.index_select(0, index)
    sims = torch.mm(batch, self.features.t())

    sims.scatter_(1, index.view(-1, 1), -1)
    reliable_matrix_1 = (sims > self.threshold_1).float()

    sims -= (sims > self.threshold_1).float()
    reliable_matrix_2 = (sims > self.threshold_2).float()

    return reliable_matrix_1, reliable_matrix_2
