'''
python3 test.py ct40k_0514_bin_hemorrhage --model_depth 18 --n_test_samples 1 --sample_duration 20 --n_classes 2 --sample_size 384 --resume_path ./results/ct40k_0514_bin_hemorrhage_b4_s384_d20_randomcrop_flip_rot30/best_20.pth 

'''

import os
import sys
import numpy as np
import time
import pandas as pd

import torch
from torch import nn
from torchvision import transforms

from utils import *
from ctdataset import CTDataSet
from epochs import predict
from opts import parse_opts
from model3d import generate_3d
from tsnmodel import generate_tsn
from spatial_transforms import *
from temporal_transforms import *

if __name__ == '__main__':

    # ===================================
    # --- Initialization ----------------
    # ===================================
    args = parse_opts()
    start = time.time()

    if args.score_weights is None:
        score_weights = [1] * len(args.test_models)
    else:
        score_weights = args.score_weights
        if len(score_weights) != len(args.test_models):
            raise ValueError("Only {} weight specifed for a total of {} models".format(len(score_weights), len(args.test_models)))

    # set gpus
    if args.gpus: # if none, use all
        gpus = ','.join(args.gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # input label files or one single case
    if os.path.exists(os.path.join(args.annotation_path, args.dataset, 'validation.txt')):
        test_list = os.path.join(args.annotation_path, args.dataset, 'validation.txt')
    elif ' ' in args.dataset or os.path.exists(args.dataset):
        test_list = args.dataset # one single case
    else:
        raise ValueError("Input should be a list file or a \"path frames label\" combination.")

    # set directory to save logs and training outputs
    if args.tag:
        args.tag = '_' + args.tag

    outpath = os.path.join(args.result_path, 'test')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    # set name of logs
    test_logger = create_logger(outpath, args.dataset.split('/')[-1] + args.tag, runtime=False)
    test_logger.info('Using model {}'.format(args.test_models))
    test_logger.info('Using weights {}'.format(score_weights))

    test_logger.info('Initial Definition time: {}'.format(time.time() - start))

    score_lists = list()

    # prepare for value range
    if args.input_format == 'jpg':
        norm_value = 255.0
    elif args.input_format in ['nifti', 'nii', 'nii.gz']:
        norm_value = 1.0
    elif args.input_format in ['dicom', 'dcm']:
        norm_value = None # will be dealt in ToTorchTensor
    else:
        raise ValueError("Unknown input format type.")

    # ===================================
    # --- Each model --------------------
    # ===================================
    
    for checkpoint in args.test_models:
        start = time.time()
        # -----------------------------------
        # --- Prepare model -----------------
        # -----------------------------------

        # load from previous stage
        
        print('loading checkpoint {}'.format(checkpoint))
        checkpoint = torch.load(checkpoint)
        snap_opts = checkpoint['args']
        snap_opts.pretrain_path = ''
        snap_opts.input_format = args.input_format
        arch = checkpoint['arch']

        # load model
        if snap_opts.model_type == '3d':
            model, parameters = generate_3d(snap_opts)
        elif snap_opts.model_type == 'tsn':
            model, parameters = generate_tsn(snap_opts)
        elif snap_opts.model_type == '2d':
            model = generate_2d(snap_opts)
        else:
            raise ValueError("Unknown model type")

        # load model states
        model.load_state_dict(checkpoint['state_dict'])

        # -----------------------------------
        # --- Prepare data ------------------
        # -----------------------------------

        # prepare normalization method
        if snap_opts.no_mean_norm and not snap_opts.std_norm:
            norm_method = GroupNormalize([0.], [1.])
        elif not snap_opts.std_norm:
            norm_method = GroupNormalize(model.module.input_mean, [1.]) # by default
        else:
            norm_method = GroupNormalize(model.module.input_mean, model.module.input_std) # the model is already wrapped by DataParalell

        # get input_size other wise use the sample_size
        crop_size = getattr(model.module, 'input_size', snap_opts.sample_size)

        spatial_transform = transforms.Compose([
            # GroupResize(384 if args.input_format == 'nifti' else 512),
            GroupCenterCrop(crop_size),
            ToTorchTensor(snap_opts.model_type, norm=norm_value, caffe_pretrain=snap_opts.arch == 'BNInception'),
            norm_method, 
        ])
        temporal_transform = TemporalSegmentCrop(snap_opts.n_slices, snap_opts.sample_thickness, test=True)

        test_data = CTDataSet(test_list, snap_opts, spatial_transform, temporal_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True)

        print('Preparation time for {}-{} is {}'.format(arch, snap_opts.model_type, time.time() - start))
        
        scores = predict(test_loader, model, snap_opts, concern_label=args.concern_label)
        score_lists.append(np.array(scores))

        print('Prediction time for {}-{} is {}'.format(arch, snap_opts.model_type, time.time() - start))

    # -----------------------------------
    # --- Post-process Score ------------
    # -----------------------------------

    # score_aggregation
    final_scores = np.zeros_like(scores)
    for s, w in zip(score_lists, score_weights):
        final_scores += w * softmax(s) # should use softmax, so the score values in different models should be comparable 

    # print and save predictions
    pred_labels = (final_scores[:, 1] >= args.threshold).astype(int)
    test_logger.info('Use >={} for non-hemorrhage (label 1).'.format(args.threshold))

    for i, label in enumerate(pred_labels):
        file = test_data.ct_list[i].path

        test_logger.info(
            'Case: [{0}/{1}]\t'
            'Name: {2:13s} '
            'Pred: {3}'.format(
            i + 1, 
            len(test_data), 
            file.split('/')[-1], 
            label))

        # if not label == args.concern_label:  # i.e. non-hemorrhage
        #     test_logger.info('There is no hemorrhage in the case by prediction.')
        #     os.system('rm -r {}'.format(file))
