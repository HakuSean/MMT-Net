'''
The operations in this script deal with spatial transformation.
For each group of images, if there is any random operations, use the same random value for all.

'''
import random
import math
import numbers
import collections
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
import cv2

class MaskCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class GroupRandomBrightnessContrast(object):
    """Randomly change brightness and contrast of the input image.
    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.prob = p

    def __call__(self, img_group, mask_group=None):
        # sometimes do nothing
        if random.random() > self.prob:
            if mask_group is not None:
                return img_group, mask_group
            else:
                return img_group
        else:
            alpha = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
            beta = 0.0 + random.uniform(-self.brightness_limit, self.brightness_limit)

            dtype = np.dtype("uint8")
            max_value = 255 # max value for uint8

            lut = np.arange(0, max_value + 1).astype("float32")

            if alpha != 1:
                lut *= alpha
            if beta != 0:
                lut += beta * max_value

            lut = np.clip(lut, 0, max_value).astype(dtype)

            out_images = list()

            for img in img_group:
                out_images.append(Image.fromarray(cv2.LUT(np.array(img), lut)))

            # output
            if mask_group is not None:
                return out_images, mask_group
            else:
                return out_images

# different from RandomCrop: use the same random number for the group
class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group, mask_group=None):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()
        out_masks = list() 

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        if mask_group is not None:
            for img, mask in zip(img_group, mask_group):
                assert(img.size[0] == w and img.size[1] == h) # in case the image shapes dont match
                if w == tw and h == th:
                    out_images.append(img)
                    out_masks.append(mask)
                else:
                    out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
                    out_masks.append(mask.crop((x1, y1, x1 + tw, y1 + th)))
            return out_images, out_masks

        else:
            for img in img_group:
                assert(img.size[0] == w and img.size[1] == h) # in case the image shapes dont match
                if w == tw and h == th:
                    out_images.append(img)
                else:
                    out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

            return out_images

# different from RandomCrop: Including random resize function
class GroupRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.worker = torchvision.transforms.Resize((size, size))

    def __call__(self, img_group, mask_group=None):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        out_images = list()
        out_masks = list()

        i, j, h, w = self.get_params(img_group[0], self.scale, self.ratio)

        if mask_group is not None:
            for img, mask in zip(img_group, mask_group):
                out_images.append(self.worker(img.crop((i, j, i + w, j + h))))
                out_masks.append(self.worker(mask.crop((i, j, i + w, j + h))))

            return out_images, out_masks

        else:
            for img in img_group:
                out_images.append(self.worker(img.crop((i, j, i + w, j + h))))

            return out_images

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = img.size
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w



class GroupFiveCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.FiveCrop(size)

    def __call__(self, img_group, mask_group=None):
        if mask_group is not None:
            return [self.worker(img) for img in img_group], [self.worker(img) for img in mask_group]
        else:
            return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group, mask_group=None):
        if mask_group is not None:
            return [self.worker(img) for img in img_group], [self.worker(img) for img in mask_group]
        else:
            return [self.worker(img) for img in img_group]

# different from RandomHorizentalFlip: flip for the whole group according to probability
class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5

    The parameter is_flow only influence this operation. It is not useful in Medical images.
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, mask_group=None, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping

            if mask_group is not None:
                mask_mirror = [mask.transpose(Image.FLIP_LEFT_RIGHT) for mask in mask_group]
                return ret, mask_mirror
            else:
                return ret
        else:
            if mask_group is not None:
                return img_group, mask_group
            else:
                return img_group

# different from RandomRotation: rotate for the whole group according to probability
class GroupRandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
    """

    def __init__(self, degrees, p=0.5):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.prob = p

    def __call__(self, img_group, mask_group=None):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        if random.random() > self.prob:
            out_images = img_group
            out_masks = mask_group
        else:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            out_images = [img.rotate(angle) for img in img_group]
            if mask_group is not None:
                out_masks = [img.rotate(angle) for img in mask_group]

        if mask_group is not None:
            return out_images, out_masks
        else:
            return out_images

# put images within one group into one numpy.ndarray
class ToTorchTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    
    For reference:   
        For RGB images, C is the RGB three channles multiplying number of segments.
        However, this is not useful in CT images.

    The main difference between 2D TSN and 3D ResNet inputs is the final reshape operation.
    """

    def __init__(self, model_type='3d', norm=None, caffe_pretrain=False):
        self.model_type = model_type
        self.norm = norm
        self.caffe_pretrain = caffe_pretrain

    def __call__(self, pic, mask=None):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """

        # step 1: from pic to torch.Tensor
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        elif isinstance(pic, list):
            # handle image group, ie list(Image):
            if isinstance(pic[0], Image.Image):
                img = self.imgs2tensor(pic) # output size Nx512x512
            else: # i.e. used FiveCrop
                img = torch.stack([self.imgs2tensor(p) for p in pic], dim=1) # first to 5x224x224, then stack in dim 1 to 5xNx224x224
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()

        # step 2: transpose for 3d
        if self.model_type == '3d':
            img = img.unsqueeze(img.dim() - 3) # add one dim for real channel, the previous num_slices will become one input for 3D Net

        # step 3: form 0-255 or 0-1
        # if norm == 1, means do nothing for normalization
        if not self.norm: # for dicom, each case specifically
            img = img.float().div(255.0) # 255 because using windows to cut images into around 255.
            # else:
            #     img = img.float().sub(img.min()).div(img.max() - img.min())
        elif not self.norm == 1.0: # for jpg, norm = 255
            img = img.float().div(self.norm)
        else:
            img = img.float()

        if self.caffe_pretrain:
            img =  img.mul(255.0)
        
        if mask is not None:
            return img, self.imgs2tensor(mask)#, is_mask=True)
        else:
            return img

    def imgs2tensor(self, imgs, is_mask=False):
        '''Transfer image to torch tensor, i.e. ToTensor
        '''
        if is_mask:
            img_size = np.array(imgs[0]).shape[0]
            imgs = [x.resize((img_size//32, img_size//32), Image.ANTIALIAS) for x in imgs]
            
        img = np.concatenate([np.expand_dims(x, 0) for x in imgs], axis=0) # N x 512 x 512 x 3
        img = torch.from_numpy(img).permute(0, 3, 1, 2).contiguous() # N x 3 x 512 x512

        return img


# add channel-wise duplication in original Normalize
class GroupNormalize(object):
    def __init__(self, mean, std, cuda=False):
        self.mean = torch.Tensor(mean) 
        self.std = torch.Tensor(std) 
        if len(mean) > 1:
            self.mean = self.mean.view(-1, 1, 1)
            self.std = self.std.view(-1, 1, 1)
        if cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def __call__(self, tensor, mask_tensor=None):
        # tensor.size()[0] is the real channel, len(self.mean) is the channel of one single picture.
        # For TSN, the division is the total slice number. For 3D, the division should always be one.

        new_tensor = torch.zeros_like(tensor)
        for i in range(len(tensor)):
            new_tensor[i] += tensor[i].sub(self.mean).div(self.std)

        if mask_tensor is not None:
            return new_tensor, mask_tensor.float() #.sub_(0.5).div_(0.25)
        else:
            return new_tensor

# different from Resize: flip for the whole group according to probability
class GroupResize(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) 
            and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group, mask_group=None):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """

        if isinstance(self.size, int):
            w, h = img_group[0].size

            if (w <= h and w == self.size) or (h <= w and h == self.size):
                if mask_group is not None:
                    return img_group, mask_group
                else:
                    return img_group

            if h < w:
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)

            out_images = [img.resize((ow, oh), self.interpolation) for img in img_group]

            if mask_group is not None:
                return out_images, [img.resize((ow, oh), self.interpolation) for img in mask_group]
            else:
                return out_images
        else:
            out_images = [img.resize(self.size, self.interpolation) for img in img_group]
            if mask_group is not None:
                return out_images, [img.resize(self.size, self.interpolation) for img in mask_group]
            else:
                return out_images
