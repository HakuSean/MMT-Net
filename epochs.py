import torch
from torch.nn.utils import clip_grad_norm

import time
import os
import ipdb

from utils import AverageMeter, calculate_accuracy, f1_score


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt, logger):
    print('train at epoch {}'.format(epoch))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # for tsn
    if opt.model_type == 'tsn':
        if opt.no_partialbn:
            model.module.partialBN(False)
        else:
            model.module.partialBN(True)

    model.train()

    end_time = time.time()
    for iteration, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if torch.cuda.is_available():
            targets = targets.cuda()
            inputs = inputs.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()

        # from TSN: clip gradients
        if opt.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), opt.clip_gradient)
            if total_norm > opt.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, opt.clip_gradient / total_norm))

        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if iteration % opt.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch, iteration + 1, len(data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc=accuracies, 
                   lr=optimizer.param_groups[-1]['lr'])))

    if epoch % opt.eval_freq == 0:
        save_file_path = os.path.join(opt.result_path, opt.dataset + opt.tag, 'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)


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
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
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
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch,
                       i + 1, len(data_loader), 
                       batch_time=batch_time, data_time=data_time,
                       loss=losses, acc=accuracies)))

    logger.info(('Testing Results: Acc {acc.avg:.3f} Loss {loss.avg:.5f}'
          .format(acc=accuracies, loss=losses)))

    return losses.avg, accuracies.avg


def evaluate_model(data_loader, model, opt, logger, concern_label=1):
    # batch_size should always be one.

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    outputs_score = []
    outputs_label = []
    targets_label = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader): # each time only read one instance
            data_time.update(time.time() - end_time)

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets[0].unsqueeze(0).cuda()

            outputs = model(inputs)
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            acc = calculate_accuracy(outputs, targets)
            accuracies.update(acc, inputs.size(0))

            logger.info(
                'Case: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Acc {2} ({acc.avg:.3f})'.format(
                i + 1,
                len(data_loader),
                outputs.argmax().item(),
                batch_time=batch_time,
                data_time=data_time,
                acc=accuracies))

            outputs_label.append(outputs.argmax().item())
            targets_label.append(targets.item())
            outputs_score.append([round(s.item(), 4) for s in outputs.cpu()[0]])

            # # select out the bad
            # if not outputs_label[-1] == targets_label[-1]:
            #     bad_index.append(i)

        precision, recall, F1, specificity = f1_score(outputs_label, targets_label, compute=concern_label)

    logger.info('Model {}-{}: Precision (tp/tp+fp):\t{:.3f}'.format(opt.arch, opt.model_type, precision))
    logger.info('Model {}-{}: Recall (tp/tp+fn):\t{:.3f}'.format(opt.arch, opt.model_type, recall))
    logger.info('Model {}-{}: F1-measure (2pr/p+r):\t{:.3f}'.format(opt.arch, opt.model_type, F1))
    logger.info('Model {}-{}: Sensitivity (tp/tp+fn):\t{:.3f}'.format(opt.arch, opt.model_type, recall))
    logger.info('Model {}-{}: Specificity (tn/tn+fp):\t{:.3f}'.format(opt.arch, opt.model_type, specificity))
    logger.info('Model {}-{}: Accuracy (tn+tp/all):\t{acc.avg:.3f}\n'.format(opt.arch, opt.model_type, acc=accuracies))

    return outputs_score


def predict(data_loader, model, opt, concern_label=1):
    # batch_size must be one.

    model.eval()

    outputs_score = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader): # each time only read one instance
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets[0].unsqueeze(0).cuda()

            outputs = model(inputs)
            # pred = outputs.argmax().item()
            
            outputs_score.append([round(s.item(), 4) for s in outputs.cpu()[0]])

    return outputs_score