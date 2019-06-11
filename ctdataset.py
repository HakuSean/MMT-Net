import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import random

import nibabel as nib
import SimpleITK as sitk

class CTRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_slices(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class CTDataSet(data.Dataset):
    def __init__(self, list_file,
                 sample_thickness=1, 
                 image_type='jpg',
                 spatial_transform=None, temporal_transform=None,
                 registration=False):

        self.list_file = list_file # input file list, use space as delimiter: path, num_slices, label (0, 1, 2, ...)
        # self.num_segments = num_segments # used in TemporalSegmentCrop
        self.sample_thickness = sample_thickness # whether to consider multiple slices at each sampling point

        # self.sample_step = sample_step # used in TemporalStepCrop and TemporalRandomStep
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.num_classes = 2 # dataset feature useful for model
        self.image_type = image_type
        self.registration = registration # default is False, can be the registration method

        if image_type == 'jpg':
            self.image_tmpl = 'img{}_{:05d}.jpg'

        self._parse_list()

    def _load_volume(self, record):
        '''
        input:
            CTRecord of the study.
            indices: indices of the frames selected from temporal_transform

        output: list of imgs, length depends on self.num_channels
            only x_img: default output, for soft tissue
            y_img: if considering bone, the output should be length=2.
        
        Operations:
            if thickness > 1: output the average image of multiple consecutive images
            if image_type is dicom: need to conduct slope and intercept to all images

        Usage:
            1. if indices is None, then load all images and output.
            2. if indices is not None, which means the input size of the network is given,
                then load all images to memory and select values from the np.array.

        '''
        if self.image_type == 'jpg':
            # read all images
            volume = np.array([np.array(Image.open(os.path.join(record.directory, self.image_tmpl.format(0, i)))) for i in range(record.num_slices)])

        elif self.image_type in set(['dcm', 'dicom']):
            # need to read dicoms and conduct windows, this is mainly designed for testing

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(next(os.walk(record.directory, topdown=False))[0])

            # for windows, needs to read one image and get metadata.
            # 0028,1050 -- Window Center
            # 0028,1051 -- Window Width
            # 0028,1052 -- Rescale Intercept
            # 0028,1053 -- Rescale Slope
            single_reader = sitk.ImageFileReader()
            single_reader.SetFileName(dicom_names[0])
            winCenter = int(single_reader.GetMetaData('0028|1050').split('\\')[0])
            winWidth = int(single_reader.GetMetaData('0028|1051').split('\\')[0])
            rescaleIntercept = int(single_reader.GetMetaData('0028|1052'))
            rescaleSlope = int(single_reader.GetMetaData('0028|1053'))
 
            # change image pixel values
            yMin = winCenter - 0.5 * winWidth
            yMax = winCenter + 0.5 * winWidth

            # read in all images
            reader.SetFileNames(dicom_names)
            volume = sitk.GetArrayFromImage(reader.Execute())
            volume = np.clip(imgs * rescaleSlope - rescaleIntercept, yMin, yMax)

        elif self.image_type in set(['nifti', 'nii', 'nii.gz']):
            reader = sitk.ReadImage(record.directory+'.nii.gz') # directory is actually the file name of nii's
            volume = sitk.GetArrayFromImage(reader)

        return volume # numpy.ndarray

    def _load_images(self, volume, indices):
        samples = list()
        for idx in indices:
            imgs = volume[idx: idx + self.sample_thickness]
            samples.append(Image.fromarray(imgs.mean(axis=0).astype('float32')))
        return samples # list(Images)

        # if not second_offset:
        #     y_img = Image.open(os.path.join(directory, self.image_tmpl.format(1, idx))).convert('L')
        # else:
        #     y_img = Image.open(os.path.join(directory, self.image_tmpl.format(0, idx + second_offset))).convert('L')

        # return [x_img, y_img]

    def _parse_list(self):
        self.ct_list = list()
        self.class_count = {i: 0 for i in range(self.num_classes)} # prepare for weighted sampler
        for x in open(self.list_file):
            row = x.strip().split(' ') # three components: path, frames, label

            # num_classes:
            if int(row[-1]) >= self.num_classes:
                for i in range(self.num_classes, int(row[-1]) + 1):
                    self.class_count[i] = 0
                self.num_classes = int(row[-1]) + 1

            self.ct_list.append(CTRecord(row))
            self.class_count[int(row[-1])] += 1


    def __getitem__(self, index):
        record = self.ct_list[index]
        segment_indices = self.temporal_transform(record)
        
        volume = self._load_volume(record)
        if self.registration:
            volume = self.registration(volume)
            
        images = self._load_images(volume, segment_indices)

        return self.transform(images), record.label

    def __len__(self):
        return len(self.ct_list)
