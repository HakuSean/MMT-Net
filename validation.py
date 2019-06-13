import torch
from torch.autograd import Variable
import time
import sys
import ipdb

from utils import AverageMeter, calculate_accuracy, f1_score


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets[0].unsqueeze(0).cuda()

            outputs = model(inputs).mean(dim=0, keepdim=True)

            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            acc = calculate_accuracy(outputs, targets)
            accuracies.update(acc, inputs.size(0))

            if i % (opt.print_freq) == 0:
                logger.info(('Test Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {3} ({acc.avg:.3f})'.format(epoch,
                       i + 1, len(data_loader), outputs.argmax().item(), 
                       batch_time=batch_time, data_time=data_time,
                       loss=losses, acc=accuracies)))

    logger.info(('Testing Results: Acc {acc.avg:.3f} Loss {loss.avg:.5f}'
          .format(acc=accuracies, loss=losses)))

    return losses.avg, accuracies.avg


def val_analysis(data_names, data_loader, model, criterion, opt, logger, concern_label=1):

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    bad_index = []
    outputs_score = []
    outputs_label = []
    targets_label = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader): # each time only read one instance
            data_time.update(time.time() - end_time)

            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets[0].unsqueeze(0).cuda()

            outputs = model(inputs).mean(dim=0, keepdim=True)

            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            acc = calculate_accuracy(outputs, targets)
            accuracies.update(acc, inputs.size(0))

            print('Epoch: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {2} ({acc.avg:.3f})'.format(
                      i + 1,
                      len(data_loader),
                      outputs.argmax().item(), 
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))
            # ipdb.set_trace()

            outputs_label.append(outputs.argmax().item())
            targets_label.append(targets.item())
            outputs_score.append([round(i.item(), 4) for i in outputs.cpu()[0]])

            # select out the bad
            if not outputs_label[-1] == targets_label[-1]:
                bad_index.append(i)

        precision, recall, F1, specificity = f1_score(outputs_label, targets_label, compute=concern_label)

    logger.log({'loss': losses.avg, 'acc': accuracies.avg})
    logger.info('Precision (tp/ tp+fp): {:.3f}\t'
                'Recall (tp / tp+fn): {:.3f}\t'
                'F1 (2pr/p+r): {:.3f}\t'
                'Sensitivity (tp / tp+fn):{:.3f}\t'
                'Specificity (tn / tn+fp):{:.3f}'.format(
                    precision,
                    recall,
                    F1,
                    recall,
                    specificity))
    return bad_index, outputs_score
