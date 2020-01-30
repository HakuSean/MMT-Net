'''
This part is used for visualization using grad-cam.

The codes are modified based on the following repositories:
    https://github.com/jacobgil/pytorch-grad-cam
    https://github.com/utkuozbulak/pytorch-cnn-visualizations
    https://github.com/kazuto1011/grad-cam-pytorch

Modifications are made mainly for 3D inputs.

'''

import torch
import cv2
import sys
import numpy as np
import os

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers 
    
    This could be treated as a forward function with hook.
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        for name, module in self.model._modules.items():
            x = module(x) # compute this layer
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.base_model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.new_fc(output)
        return target_activations, output


def grad_cam(model, target_layer_names, extractor, input, index=1):
    features, output = extractor(input)
    if index == None:
        index = np.argmax(output.cpu().data.numpy())

    cam_volume = list()

    # consider each slice:
    for idx, slice in enumerate(output):

        # define the class to print
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = torch.tensor(one_hot, requires_grad = True)
        one_hot = torch.sum(one_hot.cuda() * slice) # only consider the output based on the one-hot vector

        model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = extractor.get_gradients()[-1][idx].cpu().data.numpy()

        target = features[-1][idx]
        target = target.cpu().data.numpy()

        weights = np.mean(grads_val, axis = (1, 2))
        cam = np.zeros(target.shape[1: ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        # upsample image to correct size
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = np.clip(cam, 0.1, 10)
        cam = (cam - 0.1) / np.max(cam)

        cam_volume.append(cam)

    # normalize for the whole volume:
    # import ipdb
    # ipdb.set_trace()
    # cam_volume = np.clip(cam_volume, 0.01, 1.5)
    # cam_volume = cam_volume / np.max(cam_volume)

    return cam_volume, output.mean(dim=0, keepdim=True)

def show_cam_on_image(imgs, masks, outpath):

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for idx, (img, mask) in enumerate(zip(imgs, masks)):

        heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cv2.imwrite(os.path.join(outpath, "cam{:03d}.jpg".format(idx)), np.uint8(255 * cam))
        cv2.imwrite(os.path.join(outpath, "img{:03d}.jpg".format(idx)), np.uint8(255 * img))


# img = cv2.imread(args.image_path, 1)
# img = np.float32(cv2.resize(img, (224, 224))) / 255
# input = preprocess_image(img)

# model = models.vgg19(pretrained=True)
# target_layer_names = ["35"]
# extractor = ModelOutputs(model, target_layer_names)

# if args.use_cuda:
#     model = model.cuda()
#     input = input.cuda()

# mask = grad_cam(model, target_layer_names, extractor, input, args.index)

# show_cam_on_image(img, mask)
