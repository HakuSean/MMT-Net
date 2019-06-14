import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import random

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
                 input_format='jpg',
                 spatial_transform=None, temporal_transform=None,
                 registration=False):

        self.list_file = list_file # input file list, use space as delimiter: path, num_slices, label (0, 1, 2, ...)
        # self.num_segments = num_segments # used in TemporalSegmentCrop
        self.sample_thickness = sample_thickness # whether to consider multiple slices at each sampling point

        # self.sample_step = sample_step # used in TemporalStepCrop and TemporalRandomStep
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.num_classes = 2 # dataset feature useful for model
        self.input_format = input_format
        self.registration = registration

        if input_format == 'jpg':
            self.image_tmpl = 'img{}_{:05d}.jpg'

        self._parse_list()
        self.volumes = dict()

        # for idx, record in enumerate(self.ct_list)
        #     self.volumes[idx] = self._load_volumes(record)
        #     # print loading situation
        #     if not idx % 100:
        #         print('Load data [{}/{}]'.format(idx, len(self.ct_list)))

    def _load_volumes(self, record):
        '''
        Load all volumes into memory, waited for 3d rotation and save time for DataLoader

        input:
            directory of the study, or file name of nifti.

        output: 
            the list of volumes which in the dataset.
        
        Operations:
            if input_format is dicom: need to conduct slope and intercept to all images
        '''
        if self.input_format == 'jpg':
            imgs = np.array([np.array(Image.open(os.path.join(record.path, self.image_tmpl.format(0, i)))) for i in range(1, record.num_slices)])

        elif self.input_format in set(['dcm', 'dicom']):
            # need to read dicoms and conduct windows, this is mainly designed for testing
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(next(os.walk(record.path, topdown=False))[0])

            # for windows, needs to read one image and get metadata.
            # 0028,1050 -- Window Center
            # 0028,1051 -- Window Width
            # 0028,1052 -- Rescale Intercept
            # 0028,1053 -- Rescale Slope
            single_reader = sitk.ImageFileReader()
            single_reader.SetFileName(dicom_names[1])
            winCenter = int(single_reader.GetMetaData('0028|1050').split('\\')[0])
            winWidth = int(single_reader.GetMetaData('0028|1051').split('\\')[0])
            rescaleIntercept = int(single_reader.GetMetaData('0028|1052'))
            rescaleSlope = int(single_reader.GetMetaData('0028|1053'))
 
             # change image pixel values
            yMin = winCenter - 0.5 * winWidth
            yMax = winCenter + 0.5 * winWidth

            reader.SetFileNames(dicom_names[1:]) # ignore the first slice
            imgs = sitk.GetArrayFromImage(reader.Execute())

            # conduct rescale and window
            imgs = np.clip(imgs * rescaleSlope - rescaleIntercept, yMin, yMax)

        elif self.input_format in set(['nifti', 'nii', 'nii.gz']):
            reader = sitk.ReadImage(record.path+'.nii.gz') # record.path is actually the file name of nii's
            imgs = sitk.GetArrayFromImage(reader)

        return imgs

    def _load_images(self, volume, indices=None):
        '''
        Load images according to the selected indices.

        input:
            volume which is in the memory.
            indices: indices of the frames selected from temporal_transform

        output: list of imgs, length depends on self.num_channels
            only x_img: default output, for soft tissue
            y_img: if considering bone, the output should be length=2.
        
        Operations:
            if thickness > 1: output the average image of multiple consecutive images
        '''
        samples = list()
        for idx in indices:
            mean_img = volume[idx:idx + self.sample_thickness].mean(axis=0)
            samples.append(Image.fromarray(mean_img.astype('float32')))
            
        return samples

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

        if index in self.volumes:
            vol = self.volumes[index]
        else:
            vol = self._load_volumes(record)
            self.volumes[index] = vol

        if self.registration:
            vol = self.registration(vol)

        segment_indices = self.temporal_transform(record)
        images = self._load_images(vol, segment_indices)

        return self.spatial_transform(images), record.label

    def __len__(self):
        return len(self.ct_list)
