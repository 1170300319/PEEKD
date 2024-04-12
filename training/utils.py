# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
import torch


def get_uncertenty(p, y):
    entropy = [torch.max(p[i]).cpu().numpy() if y[i]==torch.argmax(p[i]) else 0 for i in range(p.size(0))]
    return sum(entropy)


def get_weights(a, b):
    s = torch.exp(a)+torch.exp(b)
    return a/s, b/s


def random_swap_token(inputs):
    aug_inputs = inputs.copy()
    attention_mask = aug_inputs['attention_mask']
    each_lens = [sum(each) for each in attention_mask.cpu().numpy()]
    for i in range(len(each_lens)):
        ps1, ps2 = random.randint(2, each_lens[i])-1, random.randint(2, each_lens[i])-1
        t = aug_inputs['input_ids'][i][ps1]
        aug_inputs['input_ids'][i][ps1] = aug_inputs['input_ids'][i][ps2]
        aug_inputs['input_ids'][i][ps2] = torch.tensor(t)
    return aug_inputs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    a = {'input_ids': torch.tensor([[   0,  133,  470,    1,    1,    1],
        [   0,  500, 5914,  1,    1,    1],
        [   0,  133,  232,  1,    1,    1],
        [   0, 4993,    5,  532,    4,    2],
        [   0,  133, 1537,  1,    1,    1],
        [   0,  387, 2154,  1,    1,    1]]),
        'attention_mask': torch.tensor([[1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 0, 0, 0]]),
        }
    print(a)
    random_swap_token(a)
    print(a)
