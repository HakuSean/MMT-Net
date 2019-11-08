import torch
from torch import nn
from torchvision.models import resnet, densenet
import os

import torchvision
if int(torchvision.__version__.split('.')[1]) >= 4:
    from torchvision.models import resnext101_32x8d, resnext50_32x4d

from models_2d import se_resnet50, se_resnet101, se_resnext50_32x4d, se_resnext101_32x4d


def generate_2d(opt):
    assert opt.model in [
        'resnet', 'resnext', 'densenet'
    ]

    # get model
    if opt.model == 'resnet':
        assert opt.model_depth in [18, 34, 50, 101, 152]
        model = get_resnet(opt)

    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101]
        model = get_resnext(opt)

    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 161, 169, 201]
        model = get_densenet(opt)

    print(("""
Initializing 2D-Net with base model: {}{}
2D-Net Configurations:
    crop_size:          {}
        """.format(opt.model, opt.model_depth, opt.sample_size)))

    # first make cuda, since pretrained model are on cuda
    model = nn.DataParallel(model).cuda()

    return model, opt


def get_resnet(opt):
    model_name = getattr(resnet, '{}{}'.format(opt.model, opt.model_depth))
    model = model_name()
    model.fc = nn.Linear(model.fc.in_features, opt.n_classes, bias=True)

    return model

def get_resnext(opt):
    if opt.model_depth == 50:
        if opt.use_se:
            model_name = se_resnext50_32x4d
        else:
            model_name = resnext50_32x4d
    elif opt.model_depth == 101:
        if opt.use_se:
            model_name = se_resnext101_32x4d
        else:
            model_name = resnext101_32x8d

    model = model_name(pretrained=False)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(model.last_linear.in_features, opt.n_classes) # original no bias=True
    return model

    
def get_densenet(opt):
    model_name = getattr(densenet, '{}{}'.format(opt.model, opt.model_depth))
    return model_name(
        sample_size=opt.sample_size,
        sample_duration=opt.n_slices,
        )


def get_fine_tuning_parameters(model, ft_begin_index, is_densenet=False):
    # here the model is DataParallel object, should add module to get attributes from original model

    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []

    # if ft_begin_index > 0, then ignore the top several 'layers' 
    # and also ignore the first conv1 and bn1.
    for i in range(ft_begin_index, 5): # collect the names of layers
        if is_densenet:
            ft_module_names.append('denseblock{}'.format(i))
            ft_module_names.append('transition{}'.format(i))
        else:
            ft_module_names.append('layer{}'.format(i))

    if is_densenet:
        ft_module_names.append('norm5')
        ft_module_names.append('classifier')
    else:
        ft_module_names.append('fc')          

    parameters = []
    for k, v in model.named_parameters():
        # for attention parameters:
        if 'feat2att.weight' in k or 'alpha_net.weight' in k:
            parameters.append({'params': v, 'lr_mult':5 }) # give a larger rate on attention layers
        elif 'feat2att.bias' in k or 'alpha_net.bias' in k:
            parameters.append({'params': v, 'lr_mult':10 })
        else:

            # for the rest parameters:        
            for ft_module in ft_module_names: 
                if ft_module in k: # k looks like : layer1.bn1.weight, so 'in' is correct
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0}) # do not update the weights

    return parameters


def construct_2d_model(model, opt):
    # modify the convolution layers
    # Torch models are usually defined in a hierarchical way.
    # nn.modules.children() return all sub modules in a DFS manner
    modules = list(model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d), list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    params = [x.clone() for x in conv_layer.parameters()] # len=1 when there is no bias, otherwise the length should be 2.
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1 * opt.n_channels, ) + kernel_size[2:] # add amone different lists, in that case, kernel_size[0] and kernel_size[2:] are kept unchanged. Only change kernel_size[1]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

    new_conv = nn.Conv2d(opt.n_channels, conv_layer.out_channels,
                         conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                         bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

    # replace the first convlution layer
    setattr(container, layer_name, new_conv)
    return model
