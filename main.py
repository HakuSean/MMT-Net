'''
Main training file for CTBinaryTriage. After first prepare labels and nifti, use main.py to train the model.
The hyper-parameters are in opts.py. Here is an example.






'''

import os
import sys
import json
import numpy as np
import ipdb
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms

from opts import parse_opts
from 3dmodel import generate_model
from tsnmodel import generate_tsn

from utils import *

from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate
from train import train_epoch
from validation import val_epoch
import test
import time

if __name__ == '__main__':
    # set this to avoid loading memory problems
    # torch.backends.cudnn.enabled = False 
    args = parse_opts()

    if args.model == 'bninception':
        args.arch = args.model
    elif args.model == 'svm':
        print('Please use train_svm.py')
        sys.exit()
    else: # ResNet related models
        args.arch = '{}-{}'.format(args.model, args.model_depth)

    args.mean = get_mean(args.norm_value, dataset=args.mean_dataset)
    args.std = get_std(args.norm_value, dataset=args.mean_dataset)

    # set directory to save logs and training outputs
    if args.tag:
        args.tag = '_' + args.tag
    if not os.path.exists(os.path.join(args.result_path, args.dataset + args.tag)):
        os.makedirs(os.path.join(args.result_path, args.dataset + args.tag))
    
    # set name of logs
    with open(os.path.join(args.result_path, args.dataset + args.tag, 'args_{}.json'.format(time.strftime('%b%d-%H%M'))), 'w') as arg_file:
        json.dump(vars(args), arg_file)

    torch.manual_seed(args.manual_seed)


    # -----------------------------------
    # --- prepare model -----------------
    # -----------------------------------
    if args.model_type == '3d':
        model, parameters = generate_3dmodel(args)
    elif args.model_type == 'tsn':
        model, parameters= generate_tsn(args)
    elif args.model_type == '2d':
        model = generate_2dmodel(args)
    else:
        raise ValueError("Unknown model type")

    print(model)

    # -----------------------------------
    # --- prepare loss function ---------
    # -----------------------------------
    if args.loss_type == 'nll':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == 'weighted':
        if args.n_classes == 3:
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor((1.2, 1.1, 1.7)))
        elif args.n_classes == 2:
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor((3., 1.))) # 0-hemo, 1-non, 0 is more penaltied because of fewer data. Or want the error in 0 is severe than error in 1, which will incur more false positive.
    elif args.loss_type == 'focal':
        criterion = FocalLoss(args.n_classes)
    else:
        raise ValueError("Unknown loss type")

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # -----------------------------------
    # --- prepare transformation --------
    # -----------------------------------

    # prepare normalization method
    if args.no_mean_norm and not args.std_norm:
        norm_method = Normalize(0., 1.)
    elif not args.std_norm:
        norm_method = Normalize(args.mean, 1.) # by default
    else:
        norm_method = Normalize(args.mean, 0.227)

    # prepare for crop
    assert args.train_crop in ['random', 'corner', 'center']
    if args.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(args.scales, args.sample_size)
    elif args.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(args.scales, args.sample_size)
    elif args.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            args.scales, args.sample_size, crop_positions=['c'])

    spatial_transform = Compose([
        crop_method,
        # CenterCrop(args.sample_size),
        GroupRandomRotation((30)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.5], [0.227]),
    ])
    # temporal_transform = TemporalRandomCrop(args.sample_duration)
    temporal_transform = TemporalRandomStep(args.sample_duration, args.sample_step)
    target_transform = ClassLabel()
    training_data = get_training_set(args, spatial_transform,
                                     temporal_transform, target_transform)

    # ipdb.set_trace()

    if args.sampler:
        weights = [1./training_data.count[label] for _, label in training_data]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(training_data))
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=not sampler,
        num_workers=args.n_threads,
        pin_memory=True,
        sampler=sampler
        )

    train_logger = Logger(
        os.path.join(args.result_path, args.dataset+args.tag, 'train_{}.log'.format(time.strftime('%b%d-%H%M'))),
        ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(
        os.path.join(args.result_path, args.dataset+args.tag, 'train_batch_{}.log'.format(time.strftime('%b%d-%H%M'))),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

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


            # scheduler = lr_scheduler.ReduceLROnPlateau(
            #     optimizer, 'min', patience=args.lr_patience)

    # prepare for validation
    if not args.no_val:
        spatial_transform = Compose([
            CenterCrop(args.sample_size),
            ToTensor(),
            Normalize([0.5], [0.227]) 
            # transforms.Normalize(0.5, 0.227),
        ])
        # temporal_transform = LoopPadding(args.sample_duration)
        temporal_transform = TemporalStepCrop(args.sample_duration, args.sample_step)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            args, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=args.n_val_samples,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(args.result_path, args.dataset+args.tag, 'val_{}.log'.format(time.strftime('%b%d-%H%M'))), ['epoch', 'loss', 'acc'])

    # load from previous stage
    if args.resume_path:
        print('loading checkpoint {}'.format(args.resume_path))
        checkpoint = torch.load(args.resume_path)
        assert args.arch == checkpoint['arch']

        args.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not args.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    # start running
    print('run')
    best_val = float('inf')
    for i in range(args.begin_epoch, args.n_epochs + 1):
        # ipdb.set_trace()
        if not args.no_train:

            train_epoch(i, train_loader, model, criterion, optimizer, args,
                        train_logger, train_batch_logger)
        if not args.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, args,
                                        val_logger)

        if not args.no_train and not args.no_val:
            # scheduler.step(validation_loss)
            adjust_learning_rate(optimizer, i, args)

        if i % args.checkpoint == 0 and validation_loss < best_val:
            best_val = validation_loss

            save_file_path = os.path.join(args.result_path, args.dataset+args.tag, 
                                      'best_{}.pth'.format(i))
            states = {
                'epoch': i + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)




    # do some test
    if args.test:
        spatial_transform = Compose([
            Scale(int(args.sample_size / args.scale_in_test)),
            CornerCrop(args.sample_size, args.crop_position_in_test),
            ToTensor(args.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(args.sample_duration)
        target_transform = VideoID()

        test_data = get_test_set(args, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True)
        test.test(test_loader, model, args, test_data.class_names)
