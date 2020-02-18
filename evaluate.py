'''
python3 evaluate.py ct40k_0514_bin_hemorrhage --model_depth 18 --n_test_samples 1 --sample_duration 20 --n_classes 2 --sample_size 384 --resume_path ./results/ct40k_0514_bin_hemorrhage_b4_s384_d20_randomcrop_flip_rot30/best_20.pth 

'''

import os
import sys
import numpy as np
import time

import torch
from torch import nn
from torchvision import transforms
from sklearn.metrics import confusion_matrix, roc_auc_score

from utils import *
from ctdataset import CTDataSet
from epochs import evaluate_model
from opts import parse_opts

from model2d import generate_2d
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

    # set tags to distinguish different times of running
    if args.tag:
        args.tag = '_' + args.tag

    # input label files or one single case
    if os.path.exists(os.path.join(args.annotation_path, 'validation_' + args.dataset + '.txt')):
        eval_list = os.path.join(args.annotation_path, 'validation_' + args.dataset + '.txt')
        outpath = os.path.join(args.result_path, 'split_' + args.dataset + args.tag)

    elif ' ' in args.dataset or os.path.exists(args.dataset):
        eval_list = args.dataset # one single case
        folder = args.dataset.rsplit('/', 1)[-1].replace('.txt', '')
        outpath = os.path.join(args.result_path, folder + args.tag)
    else:
        raise ValueError("Input should be a list file or a \"path frames label\" combination.")

    # set directory to save logs and training outputs
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    # set name of logs
    eval_logger = create_logger(outpath, 'eval')
    eval_logger.info('Using model {}'.format(args.test_models))
    eval_logger.info('Using weights {}'.format(score_weights))

    eval_logger.info('Initial Definition time: {}'.format(time.time() - start))

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
        score_file = '_'.join(checkpoint.rsplit('/', 2)[-2:]).replace('.pth', '')
        checkpoint = torch.load(checkpoint)

        snap_opts = checkpoint.get('args', args)
        snap_opts.input_format = args.input_format
        snap_opts.modality = getattr(args, 'modality', 'soft')
        snap_opts.arch = arch = checkpoint.get('arch', args.model)
        snap_opts.no_postop = getattr(snap_opts, 'no_postop', args.no_postop)
        snap_opts.pretrain_path = False # do not have to use pretrained in evaluation/test

        # load model
        if snap_opts.model_type == '3d':
            model, _ = generate_3d(snap_opts)
        elif snap_opts.model_type == 'tsn':
            model, _ = generate_tsn(snap_opts)
        else:
            model, snap_opts = generate_2d(snap_opts)

        # load model states
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])

        # -----------------------------------
        # --- Prepare data ------------------
        # -----------------------------------

        # prepare normalization method
        if snap_opts.model_type == '3d' or snap_opts.model_type == 'tsn':
            if snap_opts.no_mean_norm and not snap_opts.std_norm:
                norm_method = GroupNormalize([0.], [1.])
            elif not snap_opts.std_norm:
                norm_method = GroupNormalize(model.module.input_mean, [1.]) # by default
            else:
                norm_method = GroupNormalize(model.module.input_mean, model.module.input_std) # the model is already wrapped by DataParalell

            # get input_size other wise use the sample_size
            crop_size = getattr(model.module, 'input_size', snap_opts.sample_size)

            spatial_transform = transforms.Compose([
                # GroupResize(snap_opts.sample_size if 'inception' in snap_opts.model and snap_opts.sample_size >= 300 else 512),
                GroupResize(crop_size),
                # GroupFiveCrop(crop_size)
                ToTorchTensor(snap_opts.model_type, norm=norm_value, caffe_pretrain=snap_opts.arch == 'bninception'),
                norm_method,
                ])
        else:
            spatial_transform = transforms.Compose([
                GroupResize(snap_opts.sample_size),
                GroupCenterCrop(snap_opts.sample_size),
                ToTorchTensor(snap_opts.model_type),
                GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        temporal_transform = TemporalSegmentCrop(snap_opts.n_slices, snap_opts.sample_thickness, test=True)

        eval_data = CTDataSet(eval_list, snap_opts, spatial_transform, temporal_transform)

        eval_loader = torch.utils.data.DataLoader(
            eval_data,
            batch_size=1,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True)


        # load scores if predicted:
        if os.path.isfile(os.path.join(outpath, 'pred_{}.npy'.format(score_file))):
            print('pred_{}.npy has already been predicted'.format(score_file))
            scores = np.load(os.path.join(outpath, 'pred_{}.npy'.format(score_file)))
            score_lists.append(np.array(scores))
            continue

        # -----------------------------------
        # --- prepare criterion -------------
        # -----------------------------------
        criterion = nn.BCEWithLogitsLoss()

        if torch.cuda.is_available():
            criterion = criterion.cuda()

        print('Preparation time for {}-{} is {}'.format(arch, snap_opts.model_type, time.time() - start))
        
        scores = evaluate_model(eval_loader, model, criterion, snap_opts, eval_logger, concern_label=args.concern_label)
        score_lists.append(np.array(scores))

        # save scores for later usage
        np.save(os.path.join(outpath, 'pred_{}.npy'.format(score_file)), np.array(scores))

        print('Prediction time for {}-{} is {}'.format(arch, snap_opts.model_type, time.time() - start))

    # -----------------------------------
    # --- Post-process Score ------------
    # -----------------------------------
    # score_aggregation
    HEMO, POST = 1, 2
    final_scores = np.zeros_like(scores)

    if args.fusion_type == 'avg':
        for s, w in zip(score_lists, score_weights):
            final_scores += w / (1 + np.exp(-s)) # should use softmax, so the score values in different models should be comparable
        args.threshold = args.threshold * len(score_weights)
    elif args.fusion_type == 'max':
        for score_id, score_value in enumerate(zip(*score_lists)):
            final_scores[score_id] = 1/ (1+ np.exp(-np.array(score_value).max(axis=0)))

    # specific for rsna and xnat
    if not args.subset:
        index = list(range(len(eval_data)))
    elif args.subset == 'rsna':
        index = [i for i in range(len(eval_data)) if 'ID_' in eval_data.ct_list[i].path]
    elif args.subset == 'xnat':
        index = [i for i in range(len(eval_data)) if 'ct_soft' in eval_data.ct_list[i].path]

    # calculate accuracy
    if args.no_postop:
        ground_truth = np.array([int(record.label[HEMO]) for record in eval_data.ct_list])[index]
    else:
        ground_truth = np.array([int(record.label[HEMO] and not record.label[POST]) for record in eval_data.ct_list])[index]

    eval_logger.info('Use >={} for label 1 (usually hemorrhage).'.format(args.threshold))

    # max vote for cq500 (in total 490 cases):
    if 'cq500' in args.dataset:
        ground_truth, pred_labels, pred_scores = case_vote(args.dataset, ground_truth, final_scores[index, HEMO], args.threshold, majority=True)
    else:
        if args.no_postop:
            pred_scores = final_scores[index, HEMO]
            pred_labels = np.array([int(i[HEMO] >= args.threshold) for i in final_scores])
        else:
            pred_scores = final_scores[index, HEMO] - final_scores[index, POST]
            pred_labels = np.array([int(i[HEMO] >= args.threshold and i[HEMO] - i[POST] >= 0.2) for i in final_scores])

    acc = (ground_truth == pred_labels).sum() / len(ground_truth)
    
    measures = f1_score(pred_labels, ground_truth, compute=args.concern_label)

    if args.subset:
        eval_logger.info('Result for dataset {}'.format(args.subset))

    eval_logger.info('Final Precision (tp/tp+fp):\t{:.3f}'.format(measures[0]))
    eval_logger.info('Final Recall (tp/tp+fn):\t{:.3f}'.format(measures[1]))
    eval_logger.info('Final F1-measure (2pr/p+r):\t{:.3f}'.format(measures[2]))
    eval_logger.info('Final Sensitivity (tp/tp+fn):\t{:.3f}'.format(measures[1]))
    eval_logger.info('Final Specificity (tn/tn+fp):\t{:.3f}'.format(measures[3]))
    eval_logger.info('Final Accuracy (tn+tp/all):\t{:.03f}%'.format(acc * 100))
    eval_logger.info("AUC Score (Test): %f" % roc_auc_score(ground_truth, pred_scores))

    # -----------------------------------
    # --- Analysis Results --------------
    # -----------------------------------

    # prepare false_alarm and missed cases
    if args.analysis:
        false_alarm = ['line\tid\t\tscores (hemo, postop)']
        missed = ['line\tid\t\tscores (hemo, postop)']

        for i in range(len(ground_truth)):
            if not ground_truth[i] and pred_labels[i]:
                false_alarm.append('{}\t{}\t[{:.4f} {:.4f}]'.format(
                    i+1, eval_data.ct_list[i].path.split('/')[-1], final_scores[i][HEMO], final_scores[i][POST]))

            if ground_truth[i] and not pred_labels[i]:
                missed.append('{}\t{}\t[{:.4f} {:.4f}]'.format(
                    i+1, eval_data.ct_list[i].path.split('/')[-1], final_scores[i][HEMO], final_scores[i][POST]))

        # print
        eval_logger.info('\nFalse alarms: ')
        for i in false_alarm:
            eval_logger.info(i)

        eval_logger.info('\nMissed cases: ')
        for i in missed:
            eval_logger.info(i)
