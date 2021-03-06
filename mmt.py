'''
2019-05-31: new_length -> channels. Reason: each segmentation only sample one slice. Even the average within the segment, input should be the averaged value instead of more than one slices. However, channel can be different, because bones are needed.

'''

import torch
from torch import nn
import torchvision
from torch.nn.init import normal_, constant_
from torch.nn import functional as F

import models_2d

def generate_mmt(args):
    base_name = args.arch.replace('-', '')
    model = CMMT(args.n_classes, args.n_slices,
                base_model=base_name, 
                channels=args.n_channels, 
                fusion_type=args.fusion_type if not args.model_type == 'chmmt' else None, 
                dropout=args.dropout, 
                pretrained=args.pretrain_path,
                partial_bn=not args.no_partialbn,
                attention_size=getattr(args, 'attention_size', 0),
                use_se=getattr(args, 'use_se', False))
    policies = model.get_optim_policies()
    # for group in policies:
    #     print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
    #         group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if not args.sample_size:
        args.sample_size = model.input_size
    
    # print only in training
    if args.pretrain_path:
        print(("""
Initializing CMMT with base model: {}.
CMMT Configurations:
    num_segments:       {}
    channels:           {}
    crop_size:          {}
    dropout_ratio:      {}
        """.format(base_name, args.n_slices, args.n_channels, args.sample_size, args.dropout)))
    
    model = model.cuda()

    return model, policies

class CMMT(nn.Module):
    def __init__(self, num_class, num_segments,
                 base_model='resnet101', channels=1,
                 fusion_type='avg', before_softmax=True,
                 dropout=0.8, pretrained=True,
                 partial_bn=True, attention_size=0, use_se=False):
        super(CMMT, self).__init__()
        self.channels = channels
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.reshape = not fusion_type == 'att' and fusion_type is not None
        self.dropout = dropout
        self.pretrained = pretrained
        if not before_softmax and fusion_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if use_se:
            base_model = 'se_' + base_model
        if 'resnext' in base_model:
            base_model = base_model + '_32x4d' 

        # prepare model
        self._prepare_base_model(base_model)

        # self.consensus = ConsensusModule(fusion_type)
        self.consensus = fusion_type
        self.attention_size = attention_size

        feature_dim = self._prepare_cmmt(num_class) # initialize the last layer

        if fusion_type is None:
            print("Converting the ImageNet model to a CMMT init model")
            self.base_model = self._construct_ct_model(self.base_model)
        print("Done. CMMT model ready...")

        # special operations
        self.softmax = nn.Softmax(dim=1)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_cmmt(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        std_linear = 0.001
        std_conv = 0.01

        # initialize FC for attention
        if self.consensus == 'att':
            self.feat2att = nn.Linear(feature_dim, self.attention_size)
            self.alpha_net = nn.Linear(self.attention_size, 1)
            normal_(self.feat2att.weight, 0, std_linear)
            constant_(self.feat2att.bias, 0)
            normal_(self.alpha_net.weight, 0, std_linear)
            constant_(self.alpha_net.bias, 0)

        # update avg_pool between layer4 and last_linear 
        self.base_model.avg_pool = nn.AdaptiveAvgPool2d(1)

        # update last_linear weights
        self.last_linear_1 = nn.Linear(2048, 2)
        self.last_linear_2 = nn.Linear(2048, 2)
        self.last_linear_0 = nn.Linear(2048, num_class - 4)
        normal_(self.last_linear_1.weight, 0, std_linear)
        constant_(self.last_linear_1.bias, 0)
        normal_(self.last_linear_2.weight, 0, std_linear)
        constant_(self.last_linear_2.bias, 0)
        normal_(self.last_linear_0.weight, 0, std_linear)
        constant_(self.last_linear_0.bias, 0)

        # add branch bottle necks for message passing
        self.branch1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(2048),
            # nn.ReLU(inplace=True)
            )
        self.branch2 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(2048),
            # nn.ReLU(inplace=True)
            )
        normal_(self.branch1[0].weight, 0, std_conv)
        constant_(self.branch1[0].bias, 0)
        normal_(self.branch2[0].weight, 0, std_conv)
        constant_(self.branch2[0].bias, 0)

        # add conv1x1 layers to build mask output
        self.mask_out1 = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
            )
        self.mask_out2 = nn.Sequential(
            nn.Conv2d(2048, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
            )
        normal_(self.mask_out1[0].weight, 0, std_conv)
        constant_(self.branch1[0].bias, 0)
        normal_(self.mask_out2[0].weight, 0, std_conv)
        constant_(self.branch2[0].bias, 0)

        return feature_dim

    # change the original input str to the real models
    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(self.pretrained)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            # self.input_mean = [0.485, 0.456, 0.406]
            # self.input_std = [0.229, 0.224, 0.225]
            self.input_mean = [0.5]
            self.input_std = [0.226]

        elif 'resnext' in base_model or base_model == 'bninception':
            self.base_model = getattr(models_2d, base_model)(pretrained=self.pretrained, use_branch=True)
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = getattr(self.base_model, 'input_size', [3, 224, 224])[-1]
            self.input_mean = getattr(self.base_model, 'mean', [0.485, 0.456, 0.406])
            self.input_std = getattr(self.base_model, 'std', [0.229, 0.224, 0.225])
            if self.channels == 1:
                self.input_mean = [sum(self.input_mean)/len(self.input_mean)]
                self.input_std = [sum(self.input_std)/len(self.input_std)]

        elif 'inception' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(self.pretrained)
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(CMMT, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval().half()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    # different lr_mult and decay for different layers
    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bottleneck_weight = []
        last_linear_weight = []
        last_linear_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                elif m.kernel_size == (1, 1):
                    bottleneck_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if m.out_features <= 3:
                    last_linear_weight.append(ps[0])
                    if len(ps) == 2:
                        last_linear_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))


        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 1, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 1, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': bottleneck_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "bottleneck_weight"},
            {'params': last_linear_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "last_linear_weight"},
            {'params': last_linear_bias, 'lr_mult': 1, 'decay_mult': 0,
             'name': "last_linear_bias"},
        ]

        # return [
        #     {'params': first_conv_weight, 'lr_mult': 5, 'decay_mult': 1,
        #      'name': "first_conv_weight"},
        #     {'params': first_conv_bias, 'lr_mult': 10, 'decay_mult': 0,
        #      'name': "first_conv_bias"},
        #     {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
        #      'name': "normal_weight"},
        #     {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
        #      'name': "normal_bias"},
        #     {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
        #      'name': "BN scale/shift"},
        #     {'params': bottleneck_weight, 'lr_mult': 10, 'decay_mult': 1,
        #      'name': "bottleneck_weight"},
        #     {'params': last_linear_weight, 'lr_mult': 2, 'decay_mult': 1,
        #      'name': "last_linear_weight"},
        #     {'params': last_linear_bias, 'lr_mult': 2, 'decay_mult': 0,
        #      'name': "last_linear_bias"},
        # ]

    def forward(self, input):
        # sample_len = 2 * self.new_length # here 2 means flow_x and flow_y, 3 for RGB
        if self.consensus is None:
            branch_out = self.base_model(input.view((-1, self.channels * self.num_segments) + input.size()[-2:]))
        else:
            branch_out = self.base_model(input.view((-1, self.channels) + input.size()[-2:]))
        message = [self.branch1(branch_out[0]), self.branch2(branch_out[1])]
        mask_out = [self.mask_out1(branch_out[0]), self.mask_out2(branch_out[1])]

        # after message passing
        internal_out = self.base_model.logits(F.relu(message[1] + branch_out[0]))
        internal_out = self.last_linear_1(internal_out)
        external_out = self.base_model.logits(F.relu(message[0] + branch_out[1]))
        external_out = self.last_linear_2(external_out)
        total_out = self.base_model.logits(F.relu(message[0] + message[1]))
        total_out = self.last_linear_0(total_out)

        if self.consensus == 'att':
            base_out = self.attention_net(base_out)

        # if self.dropout > 0:
        #     base_out = self.new_fc(base_out)

        # if not self.before_softmax:
        #     base_out = self.softmax(base_out)

        if self.reshape:
            internal_out = internal_out.view((-1, self.num_segments) + internal_out.size()[1:])
            external_out = external_out.view((-1, self.num_segments) + external_out.size()[1:])
            total_out = total_out.view((-1, self.num_segments) + total_out.size()[1:])
            base_out = torch.cat((total_out, internal_out, external_out), 2).cuda()
        else:
            base_out = torch.cat((total_out, internal_out, external_out), 1).cuda()
            # print(base_out.shape)

        # output = self.consensus(base_out)
        if self.consensus == 'avg':
            output = base_out.mean(dim=1)
        elif self.consensus == 'max':
            output = base_out.max(dim=1, keepdim=True)[0]
        else:
            output = base_out

        return output, ((mask_out[0] > 0).to(torch.float), (mask_out[1] > 0).to(torch.float))

    # def attention_net(self, base_out):
    #     att = self.feat2att(base_out) # batch*segments, attention_size
    #     att = self.alpha_net(torch.tanh(att)).view(-1, self.num_segments) # batch, segments

    #     alphas = self.softmax(att).unsqueeze(-1) # dim means which dim adds up to 1

    #     state = base_out.view((-1, self.num_segments) + base_out.size()[1:])
    #     #print(state.size()) = (batch_size, segments, feature_dim)

    #     attn_output = torch.sum(state * alphas, 1)
    #     #print(attn_output.size()) = (batch_size, feature_dim)

    #     return attn_output


    def _construct_ct_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()] # len=1 when there is no bias, otherwise the length should be 2.
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (self.num_segments * self.channels, ) + kernel_size[2:] # add amone different lists, in that case, kernel_size[0] and kernel_size[2:] are kept unchanged. Only change kernel_size[1]
        # new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_kernels = params[0].data.repeat(1, self.num_segments, 1, 1).contiguous()

        new_conv = nn.Conv2d(self.num_segments * self.channels, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size # the crop size is the input size of the struture

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    # keep this function, but may not be useful
    def get_augmentation(self):
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]), ])
