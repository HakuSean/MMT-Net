import csv
import numpy as np
import logging
import os
import sys
import time
import torch

from collections import Counter
from collections import OrderedDict

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


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]

def calculate_accuracy(outputs, targets):
    # only used for torch arrays
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

def fuse_2d(outputs, thresh=0.05):
    # fuse results from frames into a volume-level score
    binary = torch.stack((outputs[:,0], outputs[:,-1]), axis=1)
    weights = torch.Tensor([[1. if score[1] > score[0] else 0.] for score in binary])
    if binary.is_cuda:
        weights = weights.cuda()

    if weights.sum().item() / len(outputs) >= thresh:
        return (weights * binary).mean(axis=0).unsqueeze(dim=0)
    else:
        return binary.mean(axis=0).unsqueeze(dim=0)


# from TSN: to get topk, for multi-class classification
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def f1_score(outputs, targets, compute=1, delta=1e-11):
    true_sum = np.sum(targets)
    pred_sum = np.sum(outputs)
    tp = np.sum(np.bitwise_and(outputs, targets))
    fp = pred_sum - tp
    fn = true_sum - tp
    tn = len(targets) - true_sum - fp

    # if concerned label is 0 instead of 1
    if not compute:
        tn, tp = tp, tn
        fn, fp = fp, fn

    #define the precision recall and F1-score
    precision = float(tp)/(tp + fp + delta)
    recall = float(tp)/(tp + fn + delta) # is also sensitivity
    specificity = float(tn) / ( tn + fp + delta) # if tn + fp = tp + fn, then spec = 2* acc - precision
    F1 = 2.0*precision*recall/(precision+recall+delta) 
    print('tp: {}, tn: {}, fp: {}, fn: {}'.format(tp,tn,fp,fn))
    return precision, recall, F1, specificity


def create_logger(log_path, name, runtime=True):
    # define format
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # define logger
    clogger = logging.getLogger(name)
    clogger.setLevel(logging.INFO)

    # add file handler
    if runtime:
        handler = logging.FileHandler(os.path.join(log_path, '{}_{}.log'.format(name, time.strftime('%b%d-%H%M')))) 
    else:
        handler = logging.FileHandler(os.path.join(log_path, '{}.log'.format(name)))        
    handler.setFormatter(formatter)
    clogger.addHandler(handler)

    # print to stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger

# vote for cases with more that one scans
# Two mode: majority vote, average vote
def case_vote(datalist, gt, score, thresh, majority=True):
    results = OrderedDict()
    gts = list()
    for idx, line in enumerate(open(datalist, 'r')):
        name = line.split(' ', 1)[0]
        if not name in results: 
            results[name] = [score[idx]]
            gts.append(gt[idx])
        else:
            results[name].append(score[idx])

        idx += 1

    case_gt = np.array(gts)

    # majority vote: score uses the average of majority results
    if majority:
        case_pred = list()
        case_score = list()
        # using Counter to count the most common labels. 
        # If 0 and 1 has the same count, use 1 (By default in Counter, it uses the smaller number)
        for s in results.values():
            pred = (np.array(s) > thresh).astype(np.int)
            cnts = Counter(pred).most_common()
            label = 1 if len(cnts) > 1 and cnts[0][1] == cnts[1][1] else cnts[0][0]

            # append label and score
            case_pred.append(label)
            case_score.append(
                ((pred ^ (1 - label)) * s).sum() / (pred ^ (1 - label)).sum()
            )

        case_pred = np.array(case_pred)
        case_score = np.array(case_score)
    # average vote: score uses the average of all scans, and the label is 1 when score > 0
    else:
        case_score = np.array([sum(i)/len(i) for i in results.values()])
        case_pred = np.array([int(i > thresh) for i in case_score])

    return case_gt, case_pred, case_score


# ----------------------------------------------------------------
# deserted -------------------------------------------------------
# ----------------------------------------------------------------
class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

    def info(self, printout):
        print(printout)
        self.log_file.write(printout+'\n')