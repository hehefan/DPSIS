import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def recompute_memory(net, lemniscate, trainloader, testloader, batch_size, num_workers):
    net.eval()

    trainFeatures = lemniscate.memory.detach().t()

    transform_bak = trainloader.dataset.transform
    trainloader.dataset.transform = testloader.dataset.transform
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size, shuffle=False, num_workers=num_workers)
    for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
        targets = targets.cuda(async=True)
        batchSize = inputs.size(0)
        features = net(inputs)
        trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.data.t()

    trainloader.dataset.transform = transform_bak

    net.train()

    return trainFeatures.t()
