import torch
from torch.nn.utils import clip_grad_norm

import time
import os
import ipdb

from utils import AverageMeter, calculate_accuracy, f1_score, fuse_2d
import numpy as np
from sklearn.metrics import average_precision_score

from apex import amp

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt, logger):
    print('train at epoch {}'.format(epoch))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

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
        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()

        # print logits for debugging
        neg = 0
        pos = 0
        for i in range(len(targets)):
            if targets[i].cpu().numpy()[1]:
                if pos < 2:
                    print('positive logits:', outputs[i].data.cpu(), targets[i].data.cpu(), criterion(outputs[i], targets[i]).cpu().item())
                    pos += 1
            elif neg < 2:
                print('\nnegative logits:', outputs[i].data.cpu(), targets[i].data.cpu(), criterion(outputs[i], targets[i]).cpu().item())
                neg += 1

        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()

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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, iteration + 1, len(data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   lr=optimizer.param_groups[-1]['lr'])))

def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end_time = time.time()

    all_outputs = list()
    all_targets = list()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs.view((-1,) + inputs.shape[-4:]))
            outputs = outputs.view(len(targets), -1, outputs.shape[-1]).mean(dim=1)  # max(dim=1)[0] or mean(dim=1)

            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            if i % (opt.print_freq) == 0:
                logger.info(('Test Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch,
                       i + 1, len(data_loader), 
                       batch_time=batch_time, data_time=data_time,
                       loss=losses)))

    mAP = average_precision_score(np.array(all_targets), np.array(all_outputs))
    hemo_ap = average_precision_score(np.array(all_targets)[:, 1], np.array(all_outputs)[:, 1])
    logger.info(('Testing Results: mAP {0:.4f} ap for hemorrhage {1:.4f} Loss {loss.avg:.5f}'
          .format(mAP, hemo_ap, loss=losses)))

    return losses.avg, mAP


def evaluate_model(data_loader, model, criterion, opt, logger, concern_label=1):
    # batch_size should always be one.

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

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

            # for 3D models, use FiveCrop and mean on each
            # The outputs are only fc outputs (without softmax)
            if not opt.model_type == '2d':
                outputs = model(inputs.view((-1,) + inputs.shape[-4:]))
                outputs = outputs.view(len(targets), -1, outputs.shape[-1]).mean(dim=1)  # max(dim=1)[0] or mean(dim=1)
            else: # for 2d: output should be fused into one result
                outputs = model(inputs.squeeze())
                outputs = fuse_2d(outputs, thresh=0.025)

            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            gt = int(targets[0][1].item())
            pred = 1 if outputs[0][1].item() > 0 else 0

            logger.info(
                'Case: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {2:4f} ({losses.avg:.3f})\t'
                'Acc {3} (gt {4}, score {5:.4f})'.format(
                i + 1,
                len(data_loader),
                loss.item(),
                int(pred == gt), gt, outputs[0][1].item(),
                batch_time=batch_time,
                data_time=data_time,
                losses=losses))

            # get hemorrage results
            outputs_label.append(pred)
            targets_label.append(int(targets[0][1].item()))
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

            if opt.model_type == 'tsn':
                outputs = model(inputs.view((-1,) + inputs.shape[-3:]))
                outputs = outputs.view(len(targets), -1, outputs.shape[-1]).mean(dim=1)  # max(dim=1)[0] or mean(dim=1)
            elif opt.model_type == '3d':
                outputs = model(inputs.view((-1,) + inputs.shape[-4:]))
                outputs = outputs.view(len(targets), -1, outputs.shape[-1]).mean(dim=1)  # max(dim=1)[0] or mean(dim=1)
            else: # for 2d: output should be fused into one result
                outputs = model(inputs.squeeze())
                outputs = fuse_2d(outputs, thresh=0.025)

            outputs_score.append([round(s.item(), 4) for s in outputs.cpu()[0]])

    return outputs_score