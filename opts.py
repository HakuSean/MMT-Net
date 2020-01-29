import argparse

def parse_opts():
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of Binary Classification for CT Images")

    # ---------------------------------------------
    # -- Input and Output -------------------------
    # ---------------------------------------------
    parser.add_argument('dataset', type=str, default='0',
                        help='The number of split with in the 5_fold model. OR use single case for testing')
    # parser.add_argument('--ct_path', type=str, default='/home/dongang/Documents/crcp/data/', 
    #                     help='Directory path of dataset')
    parser.add_argument('--annotation_path', type=str, default='../DataPreparation/labels/hemorrhage/5_fold',
                        help='Annotation file path')
    parser.add_argument('--result_path', type=str, default='./results',
                        help='Result directory path. ')
    parser.add_argument('--tag', type=str, default='',
                        help='Tags to distinguish from other logs/saved models. This also saves the hdf5 files for svm.')

    parser.add_argument('--input_format', type=str, default='dcm', choices=['jpg', 'dicom', 'dcm', 'nii', 'nifti', 'nii.gz'],
                        help='Input format of the images.')    
    parser.add_argument('--n_channels', type=int, default=3,
                        help='Number of channels for CT. Two channels are soft and bone. Three channels are for different windows')

    parser.add_argument('--print_freq', type=int, default=10, 
                        help='print frequency on iterations (default: 10)')
    parser.add_argument('--eval_freq', type=int, default=5, 
                        help='evaluation frequency on training epochs (default: 5)')

    # ---------------------------------------------
    # -- Hardware Control -------------------------
    # ---------------------------------------------
    parser.add_argument('--gpus', nargs='+', type=str, default=None,
                        help='Index of gpus used in training. None means all.')
    parser.add_argument('--n_threads', type=int, default=8,
                        help='Number of threads for multi-thread loading')

    # ---------------------------------------------
    # -- Transformation ---------------------------
    # --------------------------------------------- 
    parser.add_argument('--sample_size', type=int, default=384,
                        help='Height and width of inputs')
    parser.add_argument('--no_mean_norm', action='store_true', default=False,
                        help='If true, inputs are not normalized by mean.')
    parser.add_argument('--std_norm', action='store_true', default=False,
                        help='If true, inputs are normalized by standard deviation.')
    parser.add_argument('--norm_value', type=int, default=1,
                        help= 'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    # parser.add_argument('--initial_scale', type=float, default=1.0,
    #                     help='Initial scale for multiscale cropping')
    # parser.add_argument('--n_scales', type=int, default=1,
    #                     help='Number of scales for multiscale cropping')
    # parser.add_argument('--scale_step', type=float, default=0.84089641525,
    #                     help='Scale step for multiscale cropping')
    parser.add_argument('--spatial_crop', '--sc', type=str, default='resize', choices=['random', 'five', 'center', 'resize'],
                        help= 'Spatial cropping method in training. random is uniform. five is selection from 4 corners and 1 center.  (random | five | center | resize)')
    parser.add_argument('--temporal_crop', '--tc', type=str, default='segment', choices=['segment', 'jump', 'step', 'center'],
                        help= 'Temporal cropping method in training. segment is to select slices within each segment; jump is purely random crop; step is to select slices every several slices; center is a special step where select the center several slices.  (segment | jump | step | center)')

    parser.add_argument('--n_slices', type=int, default=30,
                        help='Temporal duration of inputs. Used as num_segments for segmental input, and as the size for step input or random slices input.')
    parser.add_argument('--sample_step', type=int, default=5,
                        help='Temporal step of inputs')
    parser.add_argument('--sample_thickness', type=int, default=1,
                        help='Select multiple slices at one sampling point.')
    parser.add_argument('--registration', type=int, default=0,
                        help='Degrees in random registration. Default: 0 (no registration)')

    parser.add_argument('--sampler', type=str, default='', choices=['weighted', 'sqrt', ''],
                        help='Use sampler to deal with unbalanced data, default as None.')
    parser.add_argument('--modality', type=str, default=None, choices=['soft', 'bone', 'both'],
                        help='Processed modality for dicom.')

    # ---------------------------------------------
    # -- Model Structure --------------------------
    # ---------------------------------------------   
    parser.add_argument('--model', type=str, default='resnet',
                        help='(resnet | preresnet | wideresnet | resnext | densenet | BNInception | svm')
    parser.add_argument('--model_depth', type=int, default=18,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--model_type', type=str, default='3d', choices=['3d', 'tsn', '2d'],
                        help='Type of model. When 2d and tsn, model will use pretrained weights from ImageNet.')
    parser.add_argument('--resnet_shortcut', type=str, default='B',
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', type=int, default=2, 
                        help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', type=int, default=32,
                        help='ResNeXt cardinality')
    parser.add_argument('--n_classes', type=int, default=8,
                        help='Number of classes 3 (critical: 0, non-critical: 1, normal: 2), other wise: 0: concern, 1: non')
    parser.add_argument('--attention_size', type=int, default=0,
                        help='Hidden layer size of attention.')
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='Whether to use SE block in training. Default: False. Suggested: use SE block in finetuning.')

    # ---------------------------------------------
    # -- Model Initialization ---------------------
    # --------------------------------------------- 
    parser.add_argument('--pretrain_path', type=str, default='imagenet',
                        help='Use the pretrained weights from ImageNet.')
    parser.add_argument('--dropout', type=float, default=0.7,
                        help='When training from pretrained, add dropout previous to an extra linear layer.')
    parser.add_argument('--resume_path', type=str, default='',
                        help='Use the saved .pth of previous training for finetuning')
    parser.add_argument('--ft_begin_index', type=int, default=0,
                        help='Begin block index of fine-tuning')
    parser.add_argument('--manual_seed', type=int, default=1, 
                        help='Manually set random seed for training') 

    # ---------------------------------------------
    # -- Training Setting -------------------------
    # ---------------------------------------------
    parser.add_argument('--loss_type', type=str, default='nll', choices=['nll', 'focal', 'weighted', 'ce'],
                        help='Loss function used for training.')
    parser.add_argument('--learning_rate','--lr', type=float, default=0.01,
                        help= 'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--lr_steps', type=int, nargs="+", default=[20, 40], 
                        help='epochs to decay learning rate by 10. If lr_steps == [0], then use inv set up.')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Use momentum for optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, 
                        help='Weight Decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adadelta', 'adam', 'rmsprop'],
                        help='optimizer used in training')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='Special choices in different optimizers. default as nesterov for sgd, and can also represent for amsgrad in adam and centered in rmsprop.')
    parser.add_argument('--dampening', default=0.9, type=float, 
                        help='dampening of SGD')

    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch Size')
    parser.add_argument( '--n_epochs', type=int, default=100, 
                        help='Number of total epochs to run')
    parser.add_argument('--begin_epoch', type=int, default=0,
                        help= 'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                        help='gradient norm clipping (default: disabled)')
    parser.add_argument('--no_partialbn', '--pb', default=False, action="store_true",
                        help='Whether to use partialbn for training (default: yes)')
    parser.add_argument('--no_postop', action='store_true', default=False,
                        help='Do not include postop cases in training/prediction.')

    # ---------------------------------------------
    # -- Val and Test -----------------------------
    # ---------------------------------------------
    # parser.add_argument('--n_val_samples', type=int, default=1,
    #                     help='Number of validation samples for each activity')
    # parser.add_argument('--no_softmax_in_test', action='store_true', default=False,
    #                     help='If true, output for each clip/network is not normalized using softmax.')
    parser.add_argument('--analysis', action='store_true', default=False,                help='Analysis bad results.')
    parser.add_argument('--score_weights', type=float, nargs='+', default=None,
                        help='score weights for fusion')
    parser.add_argument('--test_models', type=str, nargs='+', default=None,
                        help='The models to be used for testing, should be saved model files, i.e. .pth or .tar .')
    parser.add_argument('--concern_label', type=int, default=1,
                        help='The label for positive class.')
    parser.add_argument('--fusion_type', type=str, default='avg', choices=['avg', 'max', 'topk', 'att'],
                        help='Fusion method for prediction scores in different frames/clips.')
    parser.add_argument('--threshold', '--th', type=float, default=0.5,
                        help='Threshold used for accuracy. Only used in Test.')
    parser.add_argument('--subset', type=str, default=None, choices=['imed', 'rsna'],
                        help='Select to print out results for specific subset. (Default: both)')

    args = parser.parse_args()

    # no_postop = args.n_classes == 7 and vice versa
    if args.no_postop and args.n_classes >= 7:
        args.n_classes = 7

    if args.n_classes and not args.no_postop:
        args.no_postop = True

    return args
