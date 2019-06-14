import torch
from torch.nn.utils import clip_grad_norm

import time
import os
import sys
import logging

from utils import AverageMeter, calculate_accuracy


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
        save_file_path = os.path.join(opt.result_path, opt.dataset + opt.tag, 
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
