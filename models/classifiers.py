import math
import torch
from torch import nn
from torch.autograd import Function

class FeatMemoProduct(Function):
  @staticmethod
  def forward(self, feature, index, memory, temperature, momentum):
    batch_size = feature.size(0)

    output = torch.mm(feature.data, memory.t())
    output.div_(temperature)
    self.save_for_backward(feature, index, memory, temperature, momentum)

    return output

  @staticmethod
  def backward(self, grad_output):
    feature, index, memory, temperature, momentum = self.saved_tensors
    batch_size = grad_output.size(0)
    # add temperature
    grad_output.data.div_(temperature)
    # gradient of linear
    grad_input = torch.mm(grad_output.data, memory)
    grad_input.resize_as_(feature)
    # update the memory
    weight_pos = memory.index_select(0, index.data.view(-1)).resize_as_(feature)
    weight_pos.mul_(momentum)
    weight_pos.add_(torch.mul(feature.data, 1-momentum))
    w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
    updated_weight = weight_pos.div(w_norm)
    memory.index_copy_(0, index, updated_weight)

    return grad_input, None, None, None, None, None

class NonParametricClassifier(nn.Module):
  def __init__(self, num_features, num_classes, temperature=0.1, momentum=0.5):
    super(NonParametricClassifier, self).__init__()

    self.register_buffer('temperature', torch.tensor(temperature))
    self.register_buffer('momentum', torch.tensor(momentum))
    self.register_buffer('memory', torch.zeros(num_classes, num_features))

  def forward(self, feature, index):
    output = FeatMemoProduct.apply(feature, index, self.memory, self.temperature, self.momentum)
    return output

class FCClassifier(nn.Module):
  def __init__(self, num_features, num_classes):
    super(FCClassifier, self).__init__()

    self.fc = nn.Linear(num_features, num_classes)

  def forward(self, input):
    output = self.fc(input)
    return output
