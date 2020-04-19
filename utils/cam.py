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
        self.gradients.append(grad.cpu())

    def __call__(self, x):
        outputs = []
        for name, module in self.model._modules.items():
            x = module(x.squeeze()) # compute this layer
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
        # output = self.model.new_fc(output)
        return target_activations, output


def grad_cam(target_layer_names, extractor, input, index=1):
    features, output = extractor(input)
    if index == None:
        index = np.argmax(output.cpu().data.numpy())

    one_hot = torch.zeros_like(output)
    one_hot[:, index] = 1
    one_hot = torch.sum(one_hot * output) # only consider the output based on the one-hot vector

    extractor.model.zero_grad()
    one_hot.backward(retain_graph=True)

    grads_val = extractor.get_gradients()[-1].cuda()

    target = features[-1]
    weights = torch.mean(grads_val, axis = (2, 3))
    cam_volume = torch.zeros((target.size()[0],) + target.size()[2:])

    for i, (w, cam) in enumerate(zip(weights, target)):
        cam_volume[i] = torch.matmul(cam.permute((1, 2, 0)), w).squeeze()

    # upsample image to correct size/value
    cam_volume = np.maximum(cam_volume.cpu().data.numpy(), 0) # only keep values >= 0

    return cam_volume, output.mean(dim=0, keepdim=True)


def show_cam_on_image(imgs, masks, outpath):

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    max_value = np.max(masks)

    for idx, (img, mask) in enumerate(zip(imgs, masks)):
        h, w, _ = img.shape
        cam = cv2.resize(mask, (h, w))
        cam = np.clip(cam, 0.1, max_value)
        cam = np.clip((cam - 0.1) / (max_value * 0.9), 0, 1)

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cv2.imwrite(os.path.join(outpath, "cam{:03d}.jpg".format(idx)), np.uint8(255 * cam))
        cv2.imwrite(os.path.join(outpath, "img{:03d}.jpg".format(idx)), np.uint8(255 * img))

