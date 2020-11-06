import argparse
import torch

parser = argparse.ArgumentParser()

# args for training
parser.add_argument('--max-round', default=10, type=int, help='max iteration, including initialisation one. (default: 10)')
parser.add_argument('--max-epoch', default=2, type=int, help='max epoch per round. (default: 250)')
parser.add_argument('--batch-size', default=128, type=int, help='batch_size. (default: 128)')
parser.add_argument('--lr-base', default=0.03, type=float, help='base learning rate for step learning policy. (default: 0.03)')
parser.add_argument('--lr-decay-offset', default=80, type=int, help='learning rate will start to decay at which step. (default: 80)')
parser.add_argument('--lr-decay-step', default=40, type=int, help='learning rate will decay at every n round. (default: 40)')
parser.add_argument('--lr-decay-rate', default=0.1, type=float, help='learning rate will decay at every n round. (default: 0.1)')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay for SGD. (default: 5e-4)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD. (default: 0.9)')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov for SGD. (default: True)')

# args for CNN model
parser.add_argument('--num-features', default=128, type=int, help='length of CNN feature. (default: 128)')

# args for non-parametric classifier
parser.add_argument('--npc-temperature', default=0.1, type=float, help='temperature for non-parametric classifier. (default: 0.1)')
parser.add_argument('--npc-momentum', default=0.5, type=float, help='memory update rate for non-parametric classifier. (default: 0.5)')

# args for reliable discovery
parser.add_argument('--threshold-1', default=0.85, type=float, help='size of ANs. (default: 0.85)')
parser.add_argument('--threshold-2', default=0.65, type=float, help='size of ANs. (default: 0.65)')

# args for dataloader
parser.add_argument('--workers-num', default=4, type=int, help='number of workers being used to load data. (default: 4)')

# other args
parser.add_argument('--sess-dir', default='sessions', type=str, help='directory to store session. (default: sessions)')
parser.add_argument('--seed', default=999, type=int, help='session random seed. (default: 999)')

cfg = parser.parse_args()
