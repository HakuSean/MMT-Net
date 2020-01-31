'''
python3 test.py ct40k_0514_bin_hemorrhage --model_depth 18 --n_test_samples 1 --sample_duration 20 --n_classes 2 --sample_size 384 --resume_path ./results/ct40k_0514_bin_hemorrhage_b4_s384_d20_randomcrop_flip_rot30/best_20.pth 

'''

import os
import sys
import numpy as np
import time

import torch
from torch import nn
from torchvision import transforms

from model2d import generate_2d
from model3d import generate_3d
from tsnmodel import generate_tsn

from utils import *
from ctdataset import CTDataSet
from epochs import predict
from opts import parse_opts

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
    mask_lists = None

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
        # -----------------------------------
        # --- Prepare model -----------------
        # -----------------------------------

        # load from previous stage        
        print('loading checkpoint {}'.format(checkpoint))
        checkpoint = torch.load(checkpoint)

        snap_opts = checkpoint.get('args', args)
        snap_opts.input_format = args.input_format
        snap_opts.modality = getattr(args, 'modality', 'soft')
        snap_opts.arch = arch = checkpoint.get('arch', args.model)
        snap_opts.no_postop = getattr(snap_opts, 'no_postop', args.no_postop)
        snap_opts.pretrain_path = False # do not have to use pretrained in evaluation/test
        target_layer_names = ['layer4']

        # load model
        if snap_opts.model_type == '3d':
            model, _ = generate_3d(snap_opts)
        elif snap_opts.model_type == 'tsn':
            model, _ = generate_tsn(snap_opts)
        else: # training using a different model and test here
            model, snap_opts = generate_2d(snap_opts)

        # load model states
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])

        # -----------------------------------
        # --- Prepare data ------------------
        # -----------------------------------

        # prepare spatial_transform for 3d/tsn/2d accordingly
        if snap_opts.model_type == '3d' or snap_opts.model_type == 'tsn':
            if snap_opts.no_mean_norm and not snap_opts.std_norm:
                norm_method = GroupNormalize([0.], [1.], cuda=torch.cuda.is_available())
            elif not snap_opts.std_norm:
                norm_method = GroupNormalize(model.module.input_mean, [1.], cuda=torch.cuda.is_available()) # by default
            else:
                norm_method = GroupNormalize(model.module.input_mean, model.module.input_std, cuda=torch.cuda.is_available()) # the model is already wrapped by DataParalell

            # get input_size other wise use the sample_size
            crop_size = getattr(model.module, 'input_size', snap_opts.sample_size) 
        else:
            norm_method = GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], cuda=torch.cuda.is_available())
            crop_size = snap_opts.sample_size

        # define spatial/temporal transforms
        spatial_transform = transforms.Compose([
            GroupResize(crop_size),
            ToTorchTensor(snap_opts.model_type, norm=norm_value, caffe_pretrain=snap_opts.arch == 'BNInception'),
        ])
                  
        temporal_transform = TemporalSegmentCrop(snap_opts.n_slices, snap_opts.sample_thickness, test=True)

        # define dataset and dataloader
        test_data = CTDataSet(test_list, snap_opts, spatial_transform, temporal_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True)

        # print('Preparation time for {}-{} is {}'.format(arch, snap_opts.model_type, time.time() - start))
        masks, scores = predict(test_loader, model, norm_method, target_layer_names, concern_label=args.concern_label)

        score_lists.append(np.array(scores))
        mask_lists = np.array(masks) if mask_lists is None else mask_lists + np.array(masks)

        # print('Prediction time for {}-{} is {}'.format(arch, snap_opts.model_type, time.time() - start))

    # -----------------------------------
    # --- Post-process Score ------------
    # -----------------------------------

    # score_aggregation
    HEMO, POST = 1, 2
    final_scores = np.zeros_like(scores)

    if args.fusion_type == 'avg':
        for s, w in zip(score_lists, score_weights):
            final_scores += w / (1 + np.exp(-s)) # should use sigmoid, so the score values in different models should be comparable
        args.threshold = args.threshold * len(score_weights)
    elif args.fusion_type == 'max':
        for score_id, score_value in enumerate(zip(*score_lists)):
            final_scores[score_id] = 1/ (1+ np.exp(-np.array(score_value).max(axis=0)))

    if args.no_postop:
        pred_labels = np.array([int(i[HEMO] >= args.threshold) for i in final_scores])
    else:
        pred_labels = np.array([int(i[HEMO] >= args.threshold and i[HEMO] - i[POST] >= 0.2) for i in final_scores])

    for i, ((img, _, path), mask, label) in enumerate(zip(test_loader, mask_lists, pred_labels)):

        test_logger.info(
            'Case: [{0}/{1}]\t'
            'Name: {2:13s} '
            'Pred: {3}'.format(
            i + 1, 
            len(test_data), 
            path[0], 
            label))

        if label:
            # prepare image for printout:
            show_cam_on_image(img.squeeze().permute((0, 2, 3, 1)).numpy(), mask / len(args.test_models), os.path.join(outpath,path[0]))

    print('Prediction time per case is {}'.format((time.time() - start) / len(test_data)))
    
        # if not label == args.concern_label:  # i.e. non-hemorrhage
        #     test_logger.info('There is no hemorrhage in the case by prediction.')
        #     os.system('rm -r {}'.format(file))
