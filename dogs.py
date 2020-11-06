import sys
import os
import argparse
import numpy as np
import copy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models.resnet_32 import ResNet18
from models.classifiers import NonParametricClassifier

from lib.protocols import compute_memory, kNN
from lib.criterion import Criterion
from lib.reliable_search import ReliableSearch
from lib.utils import AverageMeter, adjust_learning_rate, StepPolicy

from datasets import dogs
from config import cfg

# basic args for dataset
cfg.data_root = 'data/dogs'

# args for image preprocessing
#cfg.size = (32, 32)
cfg.size = 32
cfg.resize = 32
cfg.scale = (0.2, 1.)
cfg.ratio = (0.75, 1.333333)
cfg.colorjitter = (0.4, 0.4, 0.4, 0.4)
cfg.random_grayscale = 0.2
cfg.means = (0.4914, 0.4822, 0.4465)
cfg.stds = (0.2023, 0.1994, 0.2010)

# args for cuda
cfg.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'

#---------------------------------------------------------------Setup--------------------------------------------------------------
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
np.random.seed(cfg.seed)

if not os.path.exists(cfg.sess_dir):
  os.makedirs(cfg.sess_dir)

logger = open(os.path.join(cfg.sess_dir, 'log.txt'), 'a')

#------------------------------------------------------------DataLoader------------------------------------------------------------
traintransform = transforms.Compose([transforms.RandomResizedCrop(size=cfg.size, scale=cfg.scale, ratio=cfg.ratio),
                                     transforms.ColorJitter(*cfg.colorjitter),
                                     transforms.RandomGrayscale(p=cfg.random_grayscale),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(cfg.means, cfg.stds)])
testtransform = transforms.Compose([transforms.Resize(size=cfg.resize),
                                    transforms.CenterCrop(cfg.size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(cfg.means, cfg.stds)])

trainset = dogs.CropDogs(root=cfg.data_root, train=True, download=True, transform=traintransform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers_num)
testset = dogs.CropDogs(root=cfg.data_root, train=False, download=True, transform=testtransform)
testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers_num)

trainlabels = torch.tensor(trainset.targets).long().to(cfg.device)
ntrain, ntest = len(trainset), len(testset)

#-------------------------------------------------------------Network--------------------------------------------------------------
net = ResNet18(cfg.num_features).to(cfg.device)
npc = NonParametricClassifier(cfg.num_features, ntrain, cfg.npc_temperature, cfg.npc_momentum).to(cfg.device)
rlb = ReliableSearch(ntrain, cfg.num_features, cfg.threshold_1, cfg.threshold_2, cfg.batch_size).to(cfg.device)
criterion = Criterion().to(cfg.device)
if len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) > 1:
  net = torch.nn.DataParallel(net)


#-------------------------------------------------------------Learning--------------------------------------------------------------
optimizer = torch.optim.SGD(params=net.parameters(), lr=cfg.lr_base, momentum=cfg.momentum, dampening=0, weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
lr_handler = StepPolicy(cfg.lr_base, cfg.lr_decay_offset, cfg.lr_decay_step, cfg.lr_decay_rate)

#-------------------------------------------------------------Training--------------------------------------------------------------
def train(epoch, net, trainloader, optimizer, npc, criterion, rlb, lr):
  train_loss = AverageMeter()
  net.train()
  adjust_learning_rate(optimizer, lr)
  for (inputs, _, indexes) in trainloader:
    optimizer.zero_grad()
    inputs, indexes = inputs.to(cfg.device), indexes.to(cfg.device)

    features = net(inputs)
    outputs = npc(features, indexes)
    loss = criterion(outputs, indexes, rlb)

    loss.backward()
    train_loss.update(loss.item(), inputs.size(0))

    optimizer.step()
  return train_loss.avg

#---------------------------------------------------------------Main----------------------------------------------------------------
best_acc = 0.0
best_net_wts = copy.deepcopy(net.state_dict())
for round in range(cfg.max_round+1):
  if round > 0:
    num_reliable_1, consistency_1, num_reliable_2, consistency_2 = rlb.update(memory, trainlabels)
    log = 'Round [%04d] - reliable1: %.12f, reliable2: %.12f, consistency1: %.12f, consistency2: %.12f'%(round, num_reliable_1, num_reliable_2, consistency_1, consistency_2)
    print(log)
    logger.write(log+'\n')
    sys.stdout.flush()
    #logger.flush()

  if round == cfg.max_round:
    break

  epoch =  0
  lr = cfg.lr_base
  while lr > 0 and epoch < cfg.max_epoch:
    lr = lr_handler.update(epoch)
    loss = train(epoch, net, trainloader, optimizer, npc, criterion, rlb, lr)
    memory, _ = compute_memory(net, trainloader, testloader.dataset.transform, cfg.device)
    acc = kNN(net, memory, trainloader, trainlabels, testloader, 200, cfg.npc_temperature, cfg.device)
    if acc > best_acc:
      best_acc = acc
      best_net_wts = copy.deepcopy(net.state_dict())
    epoch += 1
    log = '[%04d-%04d]\tloss:%2.12f acc:%2.12f best:%2.12f'%(round+1, epoch, loss, acc, best_acc)
    print(log)
    logger.write(log+'\n')
    sys.stdout.flush()
    #logger.flush()
  round += 1
torch.save(best_net_wts, os.path.join(cfg.sess_dir, 'checkpoint.dict'))
