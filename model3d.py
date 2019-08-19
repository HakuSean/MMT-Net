import torch
from torch import nn
import ipdb

from models_3d import resnet, pre_act_resnet, wide_resnet, resnext, densenet

def generate_3d(opt):
    assert opt.model in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    # get model
    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        model = get_resnet(opt)

    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]
        model = get_wideresnet(opt)

    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]
        model = get_resnext(opt)

    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]
        model = get_preresnet(opt)

    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]
        model = get_densenet(opt)

    print(("""
Initializing 3D-Net with base model: {}{}
3D-Net Configurations:
    num_segments:       {}
    channels:           {}
    crop_size:          {}
        """.format(opt.model, opt.model_depth, opt.n_slices, opt.n_channels, opt.sample_size)))

    # first make cuda, since pretrained model are on cuda
    model = nn.DataParallel(model).cuda()

    # load pretrained parameters
    if opt.pretrain_path:
        print('Use pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        assert opt.arch == pretrain['arch']

        model.load_state_dict(pretrain['state_dict'], strict=False)

    model = construct_3d_model(model, opt)

    if opt.model == 'densenet':
        model.module.classifier = nn.Linear(
            model.module.classifier.in_features, opt.n_classes).cuda()
    else:
        model.module.fc = nn.Linear(model.module.fc.in_features,
                                    opt.n_classes).cuda()

    parameters = get_fine_tuning_parameters(model, opt.ft_begin_index, is_densenet= opt.model == 'densenet')

    return model, parameters


def get_resnet(opt):
    model_name = getattr(resnet, '{}{}'.format(opt.model, opt.model_depth))
    return model_name(
        shortcut_type=opt.resnet_shortcut,
        sample_size=opt.sample_size,
        sample_duration=opt.n_slices,
        attention_size=getattr(opt, 'attention_size', 0),
        )

def get_wideresnet(opt):
    return wide_resnet.resnet50(
        shortcut_type=opt.resnet_shortcut,
        sample_size=opt.sample_size,
        sample_duration=opt.n_slices,
        attention_size=getattr(opt, 'attention_size', 0),
        k=opt.wide_resnet_k,
        )

def get_resnext(opt):
    model_name = getattr(resnext, '{}{}'.format(opt.model, opt.model_depth))
    return model_name(
        shortcut_type=opt.resnet_shortcut,
        sample_size=opt.sample_size,
        sample_duration=opt.n_slices,
        attention_size=getattr(opt, 'attention_size', 0),
        cardinality=opt.resnext_cardinality,
        )

def get_preresnet(opt):
    model_name = getattr(pre_act_resnet, '{}{}'.format(opt.model, opt.model_depth))
    return model_name(
        shortcut_type=opt.resnet_shortcut,
        sample_size=opt.sample_size,
        sample_duration=opt.n_slices,
        attention_size=getattr(opt, 'attention_size', 0),
        )

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


def construct_3d_model(model, opt):
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

    new_conv = nn.Conv3d(opt.n_channels, conv_layer.out_channels,
                         conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                         bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

    # replace the first convlution layer
    setattr(container, layer_name, new_conv)
    return model

# class CT3D(nn.Module):
#     def __init__(self, num_class, num_segments,
#                  base_model='resnet101', channels=1,
#                  dropout=0.8, pretrained=True,
#                  partial_bn=False):
#         super(CT3D, self).__init__()
#         self.channels = channels
#         self.num_segments = num_segments
#         self.dropout = dropout
#         self.pretrained = pretrained

#         # prepare model
#         self._prepare_base_model(base_model)

#         feature_dim = self._prepare_ct3d(num_class) # initialize the last layer

#         print("Converting the ImageNet model to a CT3D init model")
#         self.base_model = self._construct_3d_model(self.base_model)
#         print("Done. CT3D model ready...")

#         # special operations
#         # self.consensus = ConsensusModule(consensus_type)
#         self.consensus = consensus_type

#         if not self.before_softmax:
#             self.softmax = nn.Softmax()

#         self._enable_pbn = partial_bn

#         if partial_bn:
#             self.partialBN(True)

#     def _prepare_ct3d(self, num_class):
#         feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
#         if self.dropout == 0:
#             setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
#             self.new_fc = None
#         else:
#             setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
#             self.new_fc = nn.Linear(feature_dim, num_class)

#         std = 0.001
#         if self.new_fc is None:
#             normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
#             constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
#         else:
#             normal_(self.new_fc.weight, 0, std)
#             constant_(self.new_fc.bias, 0)
#         return feature_dim

#     # change the original input str to the real models
#     def _prepare_base_model(self, base_model):

#         if 'resnet' in base_model or 'vgg' in base_model:
#             self.base_model = getattr(torchvision.models, base_model)(self.pretrained)
#             self.base_model.last_layer_name = 'fc'
#             self.input_size = 224
#             # self.input_mean = [0.485, 0.456, 0.406]
#             # self.input_std = [0.229, 0.224, 0.225]
#             self.input_mean = [0.5]
#             self.input_std = [0.226]

#         elif base_model == 'BNInception':
#             self.base_model = getattr(tf_model_zoo, base_model)(self.pretrained)
#             self.base_model.last_layer_name = 'last_linear'
#             self.input_size = 224
#             # self.input_mean = [104, 117, 128]
#             self.input_std = [1]
#             self.input_mean = [128]

#         elif 'inception' in base_model:
#             self.base_model = getattr(torchvision.models, base_model)(self.pretrained)
#             self.base_model.last_layer_name = 'classif'
#             self.input_size = 299
#             self.input_mean = [0.5]
#             self.input_std = [0.5]
#         else:
#             raise ValueError('Unknown base model: {}'.format(base_model))

#     def train(self, mode=True):
#         """
#         Override the default train() to freeze the BN parameters
#         :return:
#         """
#         super(CT3D, self).train(mode)
#         count = 0
#         if self._enable_pbn:
#             print("Freezing BatchNorm3D except the first one.")
#             for m in self.base_model.modules():
#                 if isinstance(m, nn.BatchNorm3d):
#                     count += 1
#                     if count >= (2 if self._enable_pbn else 1):
#                         m.eval()

#                         # shutdown update in frozen mode
#                         m.weight.requires_grad = False
#                         m.bias.requires_grad = False

#     def partialBN(self, enable):
#         self._enable_pbn = enable

#     # different lr_mult and decay for different layers
#     def get_optim_policies(self):
#         first_conv_weight = []
#         first_conv_bias = []
#         normal_weight = []
#         normal_bias = []
#         bn = []

#         conv_cnt = 0
#         bn_cnt = 0
#         for m in self.modules():
#             if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv1d):
#                 ps = list(m.parameters())
#                 conv_cnt += 1
#                 if conv_cnt == 1:
#                     first_conv_weight.append(ps[0])
#                     if len(ps) == 2:
#                         first_conv_bias.append(ps[1])
#                 else:
#                     normal_weight.append(ps[0])
#                     if len(ps) == 2:
#                         normal_bias.append(ps[1])
#             elif isinstance(m, torch.nn.Linear):
#                 ps = list(m.parameters())
#                 normal_weight.append(ps[0])
#                 if len(ps) == 2:
#                     normal_bias.append(ps[1])
                  
#             elif isinstance(m, torch.nn.BatchNorm1d):
#                 bn.extend(list(m.parameters()))
#             elif isinstance(m, torch.nn.BatchNorm3d):
#                 bn_cnt += 1
#                 # later BN's are frozen
#                 if not self._enable_pbn or bn_cnt == 1:
#                     bn.extend(list(m.parameters()))
#             elif len(m._modules) == 0:
#                 if len(list(m.parameters())) > 0:
#                     raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

#         return [
#             {'params': first_conv_weight, 'lr_mult': 5, 'decay_mult': 1,
#              'name': "first_conv_weight"},
#             {'params': first_conv_bias, 'lr_mult': 10, 'decay_mult': 0,
#              'name': "first_conv_bias"},
#             {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
#              'name': "normal_weight"},
#             {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
#              'name': "normal_bias"},
#             {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
#              'name': "BN scale/shift"},
#         ]

#     def forward(self, input):
#         # sample_len = 2 * self.new_length # here 2 means flow_x and flow_y, 3 for RGB

#         base_out = self.base_model(input.view((-1, self.channels) + input.size()[-2:]))

#         if self.dropout > 0:
#             base_out = self.new_fc(base_out)

#         if not self.before_softmax:
#             base_out = self.softmax(base_out)
#         if self.reshape:
#             base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

#         # output = self.consensus(base_out)
#         if self.consensus == 'avg':
#             output = base_out.mean(dim=1, keepdim=True)
#         elif self.consensus == 'max':
#             output = base_out.max(dim=1, keepdim=True)[0]
#         else:
#             output = base_out

#         return output.squeeze(1)

#     def _construct_3d_model(self, base_model):
#         # modify the convolution layers
#         # Torch models are usually defined in a hierarchical way.
#         # nn.modules.children() return all sub modules in a DFS manner
#         modules = list(self.base_model.modules())
#         first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d), list(range(len(modules)))))[0]
#         conv_layer = modules[first_conv_idx]
#         container = modules[first_conv_idx - 1]

#         # modify parameters, assume the first blob contains the convolution kernels
#         params = [x.clone() for x in conv_layer.parameters()] # len=1 when there is no bias, otherwise the length should be 2.
#         kernel_size = params[0].size()
#         new_kernel_size = kernel_size[:1] + (1 * self.channels, ) + kernel_size[2:] # add amone different lists, in that case, kernel_size[0] and kernel_size[2:] are kept unchanged. Only change kernel_size[1]
#         new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

#         new_conv = nn.Conv3d(1 * self.channels, conv_layer.out_channels,
#                              conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
#                              bias=True if len(params) == 2 else False)
#         new_conv.weight.data = new_kernels
#         if len(params) == 2:
#             new_conv.bias.data = params[1].data # add bias if neccessary
#         layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

#         # replace the first convlution layer
#         setattr(container, layer_name, new_conv)
#         return base_model

#     @property
#     def crop_size(self):
#         return self.input_size # the crop size is the input size of the struture

#     @property
#     def scale_size(self):
#         return self.input_size * 256 // 224