import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import random

import SimpleITK as sitk

class CTRecord(object):
    '''
    Input: 
        row: should be a list of strings, at least one component
    '''
    def __init__(self, row):
        self._data = row
        self._num_slices = int(row[1]) if len(row) == 3 else -1

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[-1]) if len(self._data) > 1 else 0

    @property
    def num_slices(self):
        return self._num_slices

    @num_slices.setter
    def num_slices(self, value):
        if self._num_slices == -1 and value > 0:
            self._num_slices = value


class CTDataSet(data.Dataset):
    def __init__(self, list_file, opts,
                 spatial_transform=None, temporal_transform=None):

        self.list_file = list_file # input file list, use space as delimiter: path, num_slices, label (0, 1, 2, ...)
        # self.num_segments = num_segments # used in TemporalSegmentCrop
        self.sample_thickness = opts.sample_thickness# whether to consider multiple slices at each sampling point

        # self.sample_step = sample_step # used in TemporalStepCrop and TemporalRandomStep
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.num_classes = opts.n_classes
        self.input_format = opts.input_format 
        self.registration = opts.registration
        self.modality = opts.modality
        self.model_type = opts.model_type # used for the different preprocessing methods for 2D model

        if os.path.isfile(list_file):
            self._parse_list()
        else:
            self._parse_case()

        # self.volumes = dict()

        # for idx, record in enumerate(self.ct_list)
        #     self.volumes[idx] = self._load_volumes(record)
        #     # print loading situation
        #     if not idx % 100:
        #         print('Load data [{}/{}]'.format(idx, len(self.ct_list)))

    def _load_images(self, record, indices=None):
        '''
        input:
            directory of the study, or file name of nifti.
            indices: indices of the frames selected from temporal_transform

        output: list of imgs, length depends on self.num_channels
            only x_img: default output, for soft tissue
            y_img: if considering bone, the output should be length=2.
        
        Operations:
            if thickness > 1: output the average image of multiple consecutive images
            if input_format is dicom: need to conduct slope and intercept to all images
        '''
        samples = list()
        directory = record.path

        if self.input_format == 'jpg':
            # first get all file names under folder
            jpg_names = os.listdir(directory)
            jpg_names.sort()

            # record slice number
            if record.num_slices == -1:
                record.num_slices = len(jpg_names)

            # select slices from jpg files
            indices = self.temporal_transform(record)

            for idx in indices:
                imgs = np.array([np.array(Image.open(os.path.join(directory, jpg_names[idx + i]))) for i in range(self.sample_thickness)])
                samples.append(Image.fromarray(imgs.mean(axis=0).astype('float32')))

        elif self.input_format in set(['dcm', 'dicom']):
            # need to read dicoms and conduct windows, this is mainly designed for testing
            reader = sitk.ImageSeriesReader()
            reader.MetaDataDictionaryArrayUpdateOn()
            dicom_names = reader.GetGDCMSeriesFileNames(next(os.walk(directory, topdown=False))[0])

            # record slice number
            if record.num_slices == -1 or not record.num_slices == len(dicom_names):
                record.num_slices = len(dicom_names)
            
            # add an option for 2D slice prediction
            if self.model_type == '2d':
                reader.SetFileNames(dicom_names)
                imgs = sitk.GetArrayFromImage(reader.Execute())

                _, h, w = imgs.shape
                windows = [(20, 60), (0, 100), (40, 80)]

                for frame in imgs:
                    output = np.zeros((h, w, 3))
                    for idx, win in enumerate(windows):
                        sub_img = np.clip(frame, win[0], win[1])
                        output[:, :, idx] = sub_img

                    samples.append(Image.fromarray(output.astype(np.uint8)))

            else:
                # select dicom slices from original folder
                indices = self.temporal_transform(record)

                for idx in indices:
                    reader.SetFileNames(dicom_names[idx:idx + self.sample_thickness])
                    try:
                        imgs = sitk.GetArrayFromImage(reader.Execute())
                    except RuntimeError:
                        print('Slice {} is beyond length of {}.'.format(directory, idx))
                        break

                    # Get default window info. Rescale intercept has been considered.
                    # 0028,1050 -- Window Center
                    # 0028,1051 -- Window Width
                    winCenter = float(reader.GetMetaData(0, '0028|1050').split('\\')[0])
                    winWidth = float(reader.GetMetaData(0, '0028|1051').split('\\')[0])
                    
                    if self.modality == 'soft':
                        winCenter = 40 if winCenter > 250 else winCenter
                        winWidth = 90 if winWidth > 1000 else winWidth
                    elif self.modality == 'bone':
                        winCenter = 500 if winCenter < 250 else winCenter
                        winWidth = 2000 if winWidth < 1000 else winWidth
                    elif self.modality:
                        raise ValueError('The input modality is unknown.')
                    
                    # change image pixel values
                    yMin = int(winCenter - 0.5 * winWidth)
                    yMax = int(winCenter + 0.5 * winWidth)

                    # conduct rescale and window
                    imgs = np.clip(imgs, yMin, yMax)
                    samples.append(Image.fromarray(imgs.mean(axis=0).astype('float32')))

        elif self.input_format in set(['nifti', 'nii', 'nii.gz']):
            reader = sitk.ReadImage(directory + '.' + self.input_format) # directory is actually the file name of nii's
            volume = sitk.GetArrayFromImage(reader)

            # record slice number
            if record.num_slices == -1:
                record.num_slices = reader.GetSize()[-1]

            # select slices from original nifti file
            indices = self.temporal_transform(record)

            for idx in indices:
                imgs = volume[range(idx, idx + self.sample_thickness)]
                samples.append(Image.fromarray(imgs.mean(axis=0).astype('float32')))

        return record, samples


    def _parse_list(self):
        self.ct_list = list()
        self.class_count = {i: 0 for i in range(self.num_classes)} # prepare for weighted sampler

        max_split = 2 # at most three components

        # first decide the max_split
        for x in open(self.list_file):
            if max_split == 0:
                break

            # three components: path, frames, label
            row = x.strip().split(' - ')[-1].rsplit(' ', max_split) # deal with logs
            if len(row) <= max_split:
                max_split = len(row) - 1
            elif not row[-2].isdigit() and not row[-2] == '-1':
                if not row[-1].isdigit() and not row[-1] == '-1':
                    max_split = 0
                else:
                    max_split = 1

        # start make row
        for x in open(self.list_file):
            row = x.strip().split(' - ')[-1].rsplit(' ', max_split) # deal with logs

            # skip cases that does not exist
            if not os.path.exists(row[0] + '.' + self.input_format) and not os.path.exists(row[0]):
                print(row[0], 'does not exist...')
                continue

            self.ct_list.append(CTRecord(row))
            self.class_count[self.ct_list[-1].label] += 1

    def _parse_case(self):
        row = self.list_file.strip().split(' - ')[-1].split(' ')
        self.ct_list = [CTRecord(row)]


    def __getitem__(self, index):
        record = self.ct_list[index]

        if record.num_slices == -1: # update record for the first time input
            self.ct_list[index], images = self._load_images(record)
        else:
            _, images = self._load_images(record)

        return self.spatial_transform(images), record.label


    def __len__(self):
        return len(self.ct_list)


# ---------------------------------
# ---- Deserted -------------------
# ---------------------------------
# These three functions are not used because of the lower speed.

# def _load_volumes(self, record):
#     '''
#     Load all volumes into memory, waited for 3d rotation and save time for DataLoader

#     input:
#         directory of the study, or file name of nifti.

#     output: 
#         the list of volumes which in the dataset.
    
#     Operations:
#         if input_format is dicom: need to conduct slope and intercept to all images
#     '''
#     if self.input_format == 'jpg':
#         imgs = np.array([np.array(Image.open(os.path.join(record.path, self.image_tmpl.format(0, i)))) for i in range(1, record.num_slices)])

#     elif self.input_format in set(['dcm', 'dicom']):
#         # need to read dicoms and conduct windows, this is mainly designed for testing
#         reader = sitk.ImageSeriesReader()
#         dicom_names = reader.GetGDCMSeriesFileNames(next(os.walk(record.path, topdown=False))[0])

#         # for windows, needs to read one image and get metadata.
#         # 0028,1050 -- Window Center
#         # 0028,1051 -- Window Width
#         # 0028,1052 -- Rescale Intercept
#         # 0028,1053 -- Rescale Slope
#         single_reader = sitk.ImageFileReader()
#         single_reader.SetFileName(dicom_names[1])
#         winCenter = int(single_reader.GetMetaData('0028|1050').split('\\')[0])
#         winWidth = int(single_reader.GetMetaData('0028|1051').split('\\')[0])
#         rescaleIntercept = int(single_reader.GetMetaData('0028|1052'))
#         rescaleSlope = int(single_reader.GetMetaData('0028|1053'))

#          # change image pixel values
#         yMin = winCenter - 0.5 * winWidth
#         yMax = winCenter + 0.5 * winWidth

#         reader.SetFileNames(dicom_names[1:]) # ignore the first slice
#         imgs = sitk.GetArrayFromImage(reader.Execute())

#         # conduct rescale and window
#         imgs = np.clip(imgs * rescaleSlope - rescaleIntercept, yMin, yMax)

#     elif self.input_format in set(['nifti', 'nii', 'nii.gz']):
#         reader = sitk.ReadImage(record.path+'.nii.gz') # record.path is actually the file name of nii's
#         imgs = sitk.GetArrayFromImage(reader)

#     return imgs

# def _load_images(self, volume, indices=None):
#     '''
#     Load images according to the selected indices.

#     input:
#         volume which is in the memory.
#         indices: indices of the frames selected from temporal_transform

#     output: list of imgs, length depends on self.num_channels
#         only x_img: default output, for soft tissue
#         y_img: if considering bone, the output should be length=2.
    
#     Operations:
#         if thickness > 1: output the average image of multiple consecutive images
#     '''
#     samples = list()
#     for idx in indices:
#         mean_img = volume[idx:idx + self.sample_thickness].mean(axis=0)
#         samples.append(Image.fromarray(mean_img.astype('float32')))
        
#     return samples

# def _getitem_volume(self, index):
#     record = self.ct_list[index]

#     if index in self.volumes:
#         vol = self.volumes[index]
#     else:
#         vol = self._load_volumes(record)
#         self.volumes[index] = vol

#     if self.registration:
#         vol = self.registration(vol)

#     segment_indices = self.temporal_transform(record)
#     images = self._load_images(vol, segment_indices)

#     return self.spatial_transform(images), record.label