'''
Main training file for CTBinaryTriage. After first prepare labels and nifti, use main.py to train the model.
The hyper-parameters are in opts.py. Here is an example.

2019-06-18: use two kinds of results storage. For testing, use only the saved 'best_loss.pth' or 'best_acc.pth' according to validation. The other one is simlpy used for tracking, which could be removed once training is done. (in train.py)

'''

import os
import sys
import json
import numpy as np
import time
from math import sqrt

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms

from ctdataset import CTDataSet
from epochs import train_epoch, val_epoch
from opts import parse_opts
from model3d import generate_3d
from tsnmodel import generate_tsn
from mmt import generate_mmt
from spatial_transforms import *
from temporal_transforms import *
from utils import *

import apex
from apex import amp
amp.register_float_function(torch, 'sigmoid')

if __name__ == '__main__':
    # set this to avoid loading memory problems
    # torch.backends.cudnn.enabled = False 
    args = parse_opts()

    # set gpus
    if args.gpus: # if none, use all
        gpus = ','.join(args.gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    start = time.time()

    # input label files
    train_list = os.path.join(args.annotation_path + args.dataset, 'training_' + args.split + '.txt')
    val_list = os.path.join(args.annotation_path + args.dataset, 'validation_' + args.split + '.txt')

    # set directory to save logs and training outputs
    outpath = os.path.join(args.result_path, args.dataset + '_split' + args.split + '_' + args.tag)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # set fusion_type. Default: no attention
    args.fusion_type = 'att' if args.attention_size else args.fusion_type

    # set name of logs
    train_logger = create_logger(outpath, 'train')
    val_logger = create_logger(outpath, 'val')
    with open(os.path.join(outpath, 'args_{}.json'.format(time.strftime('%b%d-%H%M'))), 'w') as arg_file:
        json.dump(vars(args), arg_file)

    # save and print args
    train_logger.info(args)

    torch.manual_seed(args.manual_seed)
    print('Initial Definition time: {}'.format(time.time() - start))

    # -----------------------------------
    # --- prepare model -----------------
    # -----------------------------------
    if 'inception' in args.model:
        args.arch = args.model
    elif args.model == 'svm':
        print('Please use train_svm.py')
        sys.exit()
    else: # ResNet related models
        args.arch = '{}-{}'.format(args.model, args.model_depth)

    if args.model_type == '3d':
        model, parameters = generate_3d(args)
    elif args.model_type == 'tsn':
        model, parameters = generate_tsn(args)
    elif args.model_type == 'mmt' :
        model, parameters = generate_mmt(args)
    elif args.model_type == '2d':
        model = generate_2d(args)
    else:
        raise ValueError("Unknown model type")

    # -----------------------------------
    # --- prepare loss function ---------
    # -----------------------------------
    if args.loss_type == 'nll':
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss_type == 'weighted':
        weight_tensor = torch.ones(args.n_classes, dtype=torch.float)
        weight_tensor[args.concern_label] = 2
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor, reduction='none')
        # criterion = BCEWithLogitsWeightedLoss(args.n_classes, class_weight=weight_tensor)
    elif args.loss_type == 'focal':
        weight_tensor = torch.ones(args.n_classes, dtype=torch.float)
        weight_tensor[args.concern_label] = 2
        criterion = MultiLabelFocalLoss(args.n_classes, alpha=weight_tensor)
    elif args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        raise ValueError("Unknown loss type")

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    print('Model & Loss Definition time: {}'.format(time.time() - start))

    # ------------------------------------------
    # --- prepare transformation (train) -------
    # ------------------------------------------
    if not args.pretrain_path or args.pretrain_path == 'None' or args.pretrain_path == 'False':
        norm_method = GroupNormalize([0.] * args.n_channels, [1.] * args.n_channels)
        model.input_mean = [0.] * args.n_channels
        model.input_std = [1.] * args.n_channels
    else:
        # prepare normalization method
        if args.no_mean_norm and not args.std_norm:
            norm_method = GroupNormalize([0.], [1.])
        elif not args.std_norm:
            norm_method = GroupNormalize(model.input_mean, [1.]) # by default
        else:
            norm_method = GroupNormalize(model.input_mean, model.input_std) # the model is already wrapped by DataParalell

    # prepare for value range
    if args.input_format == 'jpg':
        norm_value = 255.0
    elif args.input_format in ['nifti', 'nii', 'nii.gz']:
        norm_value = 1.0
    elif args.input_format in ['dicom', 'dcm']:
        norm_value = None # will be dealt in ToTorchTensor
    else:
        raise ValueError("Unknown input format type.")

    # prepare for crop
    assert args.spatial_crop in ['random', 'five', 'center', 'resize']
    if args.spatial_crop == 'resize':
        crop_method = GroupRandomResizedCrop(args.sample_size, scale=(0.7, 1.0))
    elif args.spatial_crop == 'random':
        crop_method = GroupRandomCrop(args.sample_size)
    elif args.spatial_crop == 'five':
        crop_method = GroupFiveCrop(args.sample_size)
    elif args.spatial_crop == 'center':
        crop_method = GroupCenterCrop(args.sample_size)

    # define spatial and temporal transform
    if args.model_type == 'mmt':
        compose = MaskCompose
    else:
        compose = transforms.Compose

    spatial_transform = compose([
        # GroupResize(args.sample_size if 'inception' in args.model and args.sample_size >= 300 else 512),
        crop_method,
        GroupRandomRotation(30, p=0.5),
        GroupRandomHorizontalFlip(),
        GroupRandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.5),
        ToTorchTensor(args.model_type, norm=norm_value, caffe_pretrain=args.arch == 'bninception'),
        norm_method,
    ])

    assert args.temporal_crop in ['segment', 'jump', 'step', 'center']
    if args.temporal_crop == 'segment':
        temporal_transform = TemporalSegmentCrop(args.n_slices, args.sample_thickness)
    elif args.temporal_crop == 'jump':
        temporal_transform = TemporalJumpCrop(args.n_slices, args.sample_thickness)
    elif args.temporal_crop == 'step':
        temporal_transform = TemporalStepCrop(args.n_slices, args.sample_step, args.sample_thickness)
    elif args.temporal_crop == 'center':
        temporal_transform = TemporalStepCrop(args.n_slices, args.sample_step, args.sample_thickness, test=True)

    print('Transformation Definition time: {}'.format(time.time() - start))

    training_data = CTDataSet(train_list, args, spatial_transform, temporal_transform)

    print('Dataset Definition time: {}'.format(time.time() - start))

    # prepare for sampler (making a list is quite slow)
    if args.sampler == 'sqrt':
        weights = [1./sqrt(training_data.class_count[x.label]) for x in training_data.ct_list]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(training_data.ct_list))
    elif args.sampler == 'weighted':
        weights = [1./training_data.class_count[x.label] for x in training_data.ct_list]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(training_data.ct_list))
    else:
        sampler = None

    print('Sampler Definition time: {}'.format(time.time() - start))

    # define train_loader
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=not sampler,
        num_workers=args.n_threads,
        pin_memory=True,
        sampler=sampler
        )

    print('Train Loader time {}'.format(time.time() - start))

    # -----------------------------------
    # --- prepare optimizer -------------
    # -----------------------------------

    if args.nesterov:
        dampening = 0
    else:
        dampening = args.dampening

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            parameters,
            lr=args.learning_rate,
            momentum=args.momentum,
            dampening=dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov)
    elif args.optimizer == 'adadelta':
        optimizer = optim.adadelta(
            parameters,
            weight_decay=args.weight_decay)            
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            amsgrad=args.nesterov)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.adadelta(
            parameters,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            centered=args.nesterov)
    else:
        raise ValueError("Unknown optimizer type")

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = nn.DataParallel(model)

    # -----------------------------------------
    # --- prepare dataset (validation) --------
    # -----------------------------------------
    # val_spatial_transform = transforms.Compose([
    #     GroupResize(args.sample_size if args.model_type == 'tsn' and args.sample_size >= 300 else 512),
    #     GroupCenterCrop(crop_size),
    #     ToTorchTensor(args.model_type, norm=norm_value, caffe_pretrain=args.arch == 'bninception'),
    #     norm_method, 
    # ])

    val_temporal_transform = TemporalSegmentCrop(args.n_slices, args.sample_thickness, test=True)
    validation_data = CTDataSet(val_list, args, spatial_transform, val_temporal_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=True)

    # -----------------------------------------
    # --- prepare pretrain/resume model -------
    # -----------------------------------------
    # load model states if it is not imagenet
    if os.path.isfile(args.pretrain_path):
        print('loading pretrained model {}'.format(args.pretrain_path))
        checkpoint = torch.load(args.pretrain_path)
        model.load_state_dict(checkpoint['state_dict'])
        best_loss, best_acc = checkpoint.get('best', [float('inf'), 0.])
        print('best loss', best_loss, 'best mAP', best_acc)

    # load from previous stage
    elif args.resume_path:
        print('loading checkpoint {}'.format(args.resume_path))
        checkpoint = torch.load(args.resume_path)
        assert args.arch == checkpoint['arch']

        args.begin_epoch = checkpoint['epoch']

        try: # consider case with different param_groups
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            model.load_state_dict(checkpoint['state_dict'], strict=False)

        best_loss, best_acc = checkpoint.get('best', [float('inf'), 0.])
        print('best loss', best_loss, 'best mAP', best_acc)
    else:    
        print('=> Initial Validation')
        best_loss, best_acc = float('inf'), 0.
        # best_loss, best_acc = val_epoch(args.begin_epoch, val_loader, model, criterion, args, val_logger)

    # =========================================
    # --- Start training / validation ---------
    # =========================================
    # print(model.module.feat2att.weight)

    print('=> Start Training')
    for epoch in range(args.begin_epoch, args.n_epochs):

        train_epoch(epoch, train_loader, model, criterion, optimizer, args, train_logger)
        # print(model.module.feat2att.weight)

        if epoch % args.eval_freq == 0 or epoch == args.n_epochs - 1:

            val_loss, val_acc = val_epoch(epoch, val_loader, model, criterion, args, val_logger)

            # save two models according to loss or accuracy
            if val_loss < best_loss:
                print('Val loss decreases at epoch {}.'.format(epoch))
                save_file_path = os.path.join(outpath, 'best_loss.pth')
                best_loss = val_loss
            elif val_acc > best_acc:
                print('Val accuracy increases at epoch {}.'.format(epoch))
                best_acc = val_acc
                save_file_path = os.path.join(outpath, 'best_acc.pth')
            else:
                # save checkpoint when nothing happens
                print('Val accuracy/loss remains or gets worse at epoch {}.'.format(epoch))
                save_file_path = os.path.join(outpath, 'save_{}.pth'.format(epoch))

            states = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best': [best_loss, best_acc],
                'args': args # used for ensemble predictions
            }
            torch.save(states, save_file_path)

        # scheduler.step(validation_loss)
        learning_rate_steps(optimizer, epoch, args)

