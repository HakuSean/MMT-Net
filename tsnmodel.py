'''
2019-05-31: new_length -> channels. Reason: each segmentation only sample one slice. Even the average within the segment, input should be the averaged value instead of more than one slices. However, channel can be different, because bones are needed.

'''

import torch
from torch import nn
import torchvision
from torch.nn.init import normal_, constant_

import models_2d

def generate_tsn(args):
    model = CTSN(args.n_classes, args.n_slices, 
                base_model=args.arch.replace('-', ''), 
                channels=args.n_channels, 
                fusion_type=args.fusion_type, dropout=args.dropout, 
                pretrained=not args.pretrain_path == '',
                partial_bn=not args.no_partialbn,
                attention_size=getattr(args, 'attention_size', 0),
                use_se=getattr(args, 'use_se', False))
    policies = model.get_optim_policies()
    # for group in policies:
    #     print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
    #         group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    print(("""
Initializing CTSN with base model: {}.
CTSN Configurations:
    num_segments:       {}
    channels:           {}
    crop_size:          {}
    dropout_ratio:      {}
        """.format(args.arch, args.n_slices, args.n_channels, model.input_size, args.dropout)))
    
    model = nn.DataParallel(model).cuda()

    return model, policies

class CTSN(nn.Module):
    def __init__(self, num_class, num_segments,
                 base_model='resnet101', channels=1,
                 fusion_type='avg', before_softmax=True,
                 dropout=0.8, pretrained=True,
                 partial_bn=True, attention_size=0, use_se=False):
        super(CTSN, self).__init__()
        self.channels = channels
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.reshape = not fusion_type == 'att'
        self.dropout = dropout
        self.pretrained = pretrained
        self.use_se = use_se
        if not before_softmax and fusion_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        # prepare model
        self._prepare_base_model(base_model)

        # self.consensus = ConsensusModule(fusion_type)
        self.consensus = fusion_type
        self.attention_size = attention_size

        feature_dim = self._prepare_ctsn(num_class) # initialize the last layer

        print("Converting the ImageNet model to a CTSN init model")
        self.base_model = self._construct_ct_model(self.base_model)
        print("Done. CTSN model ready...")

        # special operations
        self.softmax = nn.Softmax(dim=1)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_ctsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        std = 0.001

        # initialize FC for attention
        if self.consensus == 'att':
            self.feat2att = nn.Linear(feature_dim, self.attention_size)
            self.alpha_net = nn.Linear(self.attention_size, 1)
            normal_(self.feat2att.weight, 0, std)
            constant_(self.feat2att.bias, 0)
            normal_(self.alpha_net.weight, 0, std)
            constant_(self.alpha_net.bias, 0)

        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)
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

        elif base_model == 'BNInception':
            self.base_model = getattr(models_2d, base_model)(pretrained=self.pretrained)
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 224
            # self.input_mean = [104, 117, 128]
            self.input_std = [1]
            self.input_mean = [128]

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
        super(CTSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

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
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
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
            {'params': first_conv_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        # sample_len = 2 * self.new_length # here 2 means flow_x and flow_y, 3 for RGB

        base_out = self.base_model(input.view((-1, self.channels) + input.size()[-2:]))

        if self.consensus == 'att':
            base_out = self.attention_net(base_out)

        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        # output = self.consensus(base_out)
        if self.consensus == 'avg':
            output = base_out.mean(dim=1, keepdim=True)
        elif self.consensus == 'max':
            output = base_out.max(dim=1, keepdim=True)[0]
        else:
            output = base_out

        return output.squeeze(1)

    def attention_net(self, base_out):
        att = self.feat2att(base_out) # batch*segments, attention_size
        att = self.alpha_net(torch.tanh(att)).view(-1, self.num_segments) # batch, segments

        alphas = self.softmax(att).unsqueeze(-1) # dim means which dim adds up to 1

        state = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        #print(state.size()) = (batch_size, segments, feature_dim)

        attn_output = torch.sum(state * alphas, 1)
        #print(attn_output.size()) = (batch_size, feature_dim)

        return attn_output


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
        new_kernel_size = kernel_size[:1] + (1 * self.channels, ) + kernel_size[2:] # add amone different lists, in that case, kernel_size[0] and kernel_size[2:] are kept unchanged. Only change kernel_size[1]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(1 * self.channels, conv_layer.out_channels,
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
