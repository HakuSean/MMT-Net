import torch
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F

import time
import os
import ipdb

from utils import AverageMeter, calculate_accuracy, f1_score, fuse_2d, grad_cam, show_cam_on_image, ModelOutputs
import numpy as np
from sklearn.metrics import average_precision_score
from utils import DiceLoss

from apex import amp

dice_loss = DiceLoss()

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

    end_time = time.time()
    model.train()

    for iteration, (inputs, targets, _) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if torch.cuda.is_available():
            targets = targets.cuda()
            if opt.model_type == 'mmt':
                # use masks as labels: all, internal, external
                masks = F.interpolate(inputs[1].view((-1, opt.n_channels) + inputs[1].size()[-2:]).cuda(), size=[inputs[1].size()[-1]//32, inputs[1].size()[-1]//32])
                inputs = inputs[0].cuda()
            elif opt.model_type == 'mtsn':
                inputs = inputs[0].cuda()
            else:
                inputs = inputs.cuda()
        
        if opt.model_type == 'mmt':
            outputs, branch_out = model(inputs)
            loss = criterion(outputs, targets) + dice_loss(branch_out[0], masks[:, 1]) + dice_loss(branch_out[1], masks[:, 2])
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # update with hard examples when learning rate is small
        if epoch > 100: #opt.n_epochs * 0.5 - 1:
            large_loss = F.threshold(loss.mean(axis=1), 0.8, 0., inplace=True)
            small_loss = F.threshold(loss.mean(axis=1), 0.2, 0., inplace=True)
            if (large_loss > 0).any():
                back_loss = large_loss.sum() / (large_loss > 0).sum()
                # print((back_loss > 0).sum().cpu().item() / opt.batch_size)
            elif (small_loss > 0).any():
                back_loss = small_loss.sum() / (small_loss > 0).sum()
                # print((back_loss > 0).sum().cpu().item() / opt.batch_size)
            else:
                back_loss = loss.mean()
        else:
            back_loss = loss.mean()

        losses.update(back_loss.item(), inputs.size(0))

        with amp.scale_loss(back_loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # loss.backward()

        # # print logits for debugging
        # neg = 0
        # pos = 0
        # for i in range(len(targets)):
        #     if (not targets[i].shape and targets[i]==1) or (targets[i].shape and targets[i].cpu().numpy()[opt.concern_label]):
        #         if pos < 2:
        #             print('positive logits:', outputs[i].data.cpu(), targets[i].data.cpu(), loss[i].mean().item())
        #             pos += 1
        #     elif neg < 2:
        #         print('\nnegative logits:', outputs[i].data.cpu(), targets[i].data.cpu(), loss[i].mean().item())
        #         neg += 1

        # from TSN: clip gradients
        if opt.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), opt.clip_gradient)
            if total_norm > opt.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, opt.clip_gradient / total_norm))

        optimizer.step()
        optimizer.zero_grad()

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
        for i, (inputs, targets, _) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if torch.cuda.is_available():
                targets = targets.cuda()
                if opt.model_type == 'mmt':
                    # use masks as labels: all, internal, external
                    masks = F.interpolate(inputs[1].view((-1, opt.n_channels) + inputs[1].size()[-2:]).cuda(), size=[inputs[1].size()[-1]//32, inputs[1].size()[-1]//32])
                    inputs = inputs[0].cuda()
                elif opt.model_type == 'mtsn':
                    inputs = inputs[0].cuda()
                else:
                    inputs = inputs.cuda()
            
            if opt.model_type == 'mmt':
                outputs, branch_out = model(inputs)
                loss = criterion(outputs, targets) + dice_loss(branch_out[0], masks[:, 1]) + dice_loss(branch_out[1], masks[:, 2])
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            outputs = outputs.view(len(targets), -1, outputs.shape[-1]).mean(dim=1)  # max(dim=1)[0] or mean(dim=1)

            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            losses.update(loss.mean().item(), inputs.size(0))

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

    if not targets[0].shape:
        acc = calculate_accuracy(torch.tensor(all_outputs), torch.tensor(all_targets))
        logger.info(('Testing Results: Acc {acc:.3f} Loss {loss.avg:.5f}'
              .format(acc=acc, loss=losses)))
        return losses.avg, acc
    
    else:
        mAP = average_precision_score(np.array(all_targets), np.array(all_outputs)) # only consider 5 subclasses
        hemo_ap = average_precision_score(np.array(all_targets)[:, opt.concern_label], np.array(all_outputs)[:, opt.concern_label])
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
        for i, (inputs, targets, path) in enumerate(data_loader): # each time only read one instance

            data_time.update(time.time() - end_time)

            if torch.cuda.is_available():
                targets = targets.cuda()
                if opt.model_type == 'mmt':
                    # use masks as labels: all, internal, external
                    masks = inputs[1].view((-1, opt.n_channels) + inputs[1].size()[-2:]).cuda()
                    inputs = inputs[0].cuda()
                elif opt.model_type == 'mtsn':
                    inputs = inputs[0].cuda()
                else:
                    inputs = inputs.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            gt = int(targets[0][opt.concern_label].item())
            pred = 1 if outputs[0][opt.concern_label].item() > 0 else 0

            logger.info(
                'Case: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {2:4f} ({losses.avg:.3f})\t'
                'Acc {3} (gt {4}, score {5:.4f})\t'
                'Name {6}'.format(
                i + 1,
                len(data_loader),
                loss.item(),
                int(pred == gt), gt, outputs[0][opt.concern_label].item(),
                path[0],
                batch_time=batch_time,
                data_time=data_time,
                losses=losses))

            # get hemorrage results
            outputs_label.append(pred)
            targets_label.append(int(targets[0][opt.concern_label].item()))
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


def predict(data_loader, model, norm_method, target_layer_names, concern_label=1):
    # batch_size must be one.

    model.eval()
    if not getattr(model, 'module', None): # consider with/without dataparallel
        extractor = ModelOutputs(model, target_layer_names)
    else:
        extractor = ModelOutputs(model.module, target_layer_names)

    outputs_score = list()
    masks = list()

    for i, (inputs, targets, _) in enumerate(data_loader): # each time only read one instance
        if torch.cuda.is_available():
            inputs = inputs.squeeze().cuda()
            targets = targets[0].unsqueeze(0).cuda()
        
        # extract masks and scores
        # masks: list of masks, len(masks)=30, masks[0].shape = (224, 224)
        # scores: a tensor of (1, 8) or (1, 7), depend on no_postop
        mask, score = grad_cam(target_layer_names, extractor, norm_method(inputs), index=concern_label)

        outputs_score.append([round(s.item(), 4) for s in score.cpu()[0]])
        masks.append(mask)

        # release cuda memory
        torch.cuda.empty_cache()
        del mask, score, inputs

    return np.array(masks), outputs_score