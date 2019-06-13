'''
python3 ensemble.py ct40k_0514_bin_hemorrhage --model_depth 18 --n_val_samples 1 --sample_duration 20 --n_classes 2 --sample_size 384 --resume_path ./results/ct40k_0514_bin_hemorrhage_b4_s384_d20_randomcrop_flip_rot30/best_20.pth 

'''

import os
import sys
import json
import numpy as np
import ipdb
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms

from opts import parse_opts
from spatial_transforms import *
from temporal_transforms import *
from utils import Logger, adjust_learning_rate, f1_score
from validation import val_analysis
import time
import sqlite3
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    opt = parse_opts()
    opt.no_train = True
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    if not opt.resume_path:
        print('Please indicate a resume_path for testing')
        sys.exit()

    # opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    # opt.std = get_std(opt.norm_value, dataset=opt.mean_dataset)

    # check the input options
    print(opt)

    # store logs
    opt.tag = os.path.dirname(opt.resume_path)

    torch.manual_seed(opt.manual_seed)

    # prepare model
    model, parameters = generate_model(opt)
    # print(model)

    # prepare loss function
    if opt.loss_type == 'nll':
        criterion = nn.CrossEntropyLoss()
    elif opt.loss_type == 'weighted':
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor((0.2, 0.1, 0.7)))
    elif opt.loss_type == 'focal':
        criterion = FocalLoss(opt.n_classes)
    else:
        raise ValueError("Unknown loss type")
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # prepare for validation
    spatial_transform = Compose([
        CenterCrop(opt.sample_size),
        ToTensor(),
        Normalize([0.5], [0.227]) 
        # transforms.Normalize(0.5, 0.227),
    ])
    # temporal_transform = LoopPadding(opt.sample_duration)
    temporal_transform = TemporalStepCrop(opt.sample_duration, opt.sample_step)
    target_transform = ClassLabel()
    validation_data = get_validation_set(
        opt, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=1, # would want to anaylysis for each case
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    val_logger = Logger(
        os.path.join(opt.tag, 'analysis_{}.log'.format(time.strftime('%b%d-%H'))), ['loss', 'acc'])

    # get names
    val_names = [i['video'].split('/')[-1].replace('_', '.') for i in validation_data.data]

    # load from previous stage
    print('loading checkpoint {}'.format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    assert opt.arch == checkpoint['arch']

    opt.begin_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

    # start running
    print('Start testing')

    _, scores = val_analysis(val_names, val_loader, model, criterion, opt, val_logger, concern_label=0)

    # combine the scores with TSN
    tsn = np.load(opt.second)['scores']
    labels = np.load(opt.second)['labels']

    if opt.score_weights is None:
        score_weights = [1] * 2
    else:
        score_weights = opt.score_weights
        if len(score_weights) != 2:
            raise ValueError("Only {} weight specifed for a total of {} score files"
                             .format(len(score_weights), len(score_npz_files)))

    # score_aggregation
    score_lists = [np.array(scores)]
    score_lists.append(tsn)
    final_scores = np.zeros_like(scores)
    for s, w in zip(score_lists, score_weights):
        final_scores += w * softmax(s)

    # accuracy
    acc = mean_class_accuracy(final_scores, labels)
    pred = np.argmax(final_scores, axis=1)
    measures = f1_score(pred, labels, compute=0)

    # prepare false_alarm and missed cases
    false_alarm = ['line\tid\tscores\n']
    missed = ['line\tid\tscores\n']
    for i in range(len(labels)):
        if labels[i] and not pred[i]: # 1 = non, 0 = hemo
            false_alarm.append('{}\t{}\t{}\n'.format(
                i+1, validation_data.data[i]['video'].split('/')[-1], final_scores[i]))

        if not labels[i] and pred[i]:
            missed.append('{}\t{}\t{}\n'.format(
                i+1, validation_data.data[i]['video'].split('/')[-1], final_scores[i]))
    # print
    print('False alarms: ')
    for i in false_alarm:
        print(i)

    print('Missed cases: ')
    for i in missed:
        print(i)
        
    print('Final accuracy {:.03f}%'.format(acc * 100))
    print('Precision (tp/tp+fp) = {:.03f}, '
        'Recall/Sensitivity (tp/tp+fn) = {:.03f}, '
        'F1 (2pr/p+r) = {:.03f}, '
        'Specificity (tn/tn+fp) = {:.03f}, '
        'Accuracy (Specificity+Sensitivity/2) = {:.03f}'.format(
            measures[0], measures[1], measures[2], measures[3],
            (measures[1]+measures[3])/2))



    # # do some test
    # if args.test:
    #     spatial_transform = Compose([
    #         Scale(int(args.sample_size / args.scale_in_test)),
    #         CornerCrop(args.sample_size, args.crop_position_in_test),
    #         ToTensor(args.norm_value), norm_method
    #     ])
    #     temporal_transform = LoopPadding(args.sample_duration)
    #     target_transform = VideoID()

    #     test_data = get_test_set(args, spatial_transform, temporal_transform,
    #                              target_transform)
    #     test_loader = torch.utils.data.DataLoader(
    #         test_data,
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         num_workers=args.n_threads,
    #         pin_memory=True)
    #     test.test(test_loader, model, args, test_data.class_names)