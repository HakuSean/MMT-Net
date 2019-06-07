'''
This is called temporal transform because the operations are derived from video processing methods.
In essence, it is about picking up slices from the original CT volume.

JumpCrop: randomly select slices among the volume.
SegmentCrop: ideas from TSN, cut the volume into segments, randomly pick one slice in each segment
StepCrop: randomly select some slices make the input, the steps between either two slices are same
CenterCrop: select slices in the middle of the volume, with specific step length. This is equal to StepCrop(test=True)

Common Args:
    sample_size: length of output list
    sample_thickness: average for multiple slices.
    record: contains num_slices, label and path to file. Refer to CTRecord in ctdataset.py.

Outputs:
    offsets: namely the numpy.array for all selected slice indices.

'''

import random
import math
import numpy as np
from numpy.random import randint
from numpy.random import choice

class TemporalJumpCrop(object):
    """Temporally crop from random locations.

    If the number of frames is less than the size,
    select all locations and randomly add several others.

    This is designed only for training.

    Args:
        sample_size (int): Desired output size of the crop.
        sample_thickness (int): average over multiple frames.

    """

    def __init__(self, sample_size, sample_thickness):
        self.sample_size = sample_size
        self.sample_thickness = sample_thickness

    def __call__(self, record):
        """
        Args:
            record: path, num_slices, label
        Returns:
            list (or numpy.array): Cropped frame indices.
        """
        num_candidates = record.num_slices - self.sample_thickness + 1
        
        offsets = np.sort(randint(num_candidates, size=self.sample_size))

        return offsets


class TemporalSegmentCrop(object):
    """Temporally crop the given frame indices into segments.

    If duration of each segment is small, 
    then randomly select each frame without replacement.
    This seems like an unreasonable setting.
    
    If the number of frames is less than the segments, 
    then use all slices and several duplicates.

    Random select for training, uniformly select for val and test.

    Args:
        sample_size (int): Desired output size of the crop, i.e. sample_size
        sample_thickness (int): average over multiple frames.
        test (bool): whether or not random select frame from the segment
    """

    def __init__(self, sample_size, sample_thickness, test=False):
        self.sample_size = sample_size
        self.sample_thickness = sample_thickness
        self.test = test

    def __call__(self, record):
        """
        Args:
            record: path, num_slices, label
        Returns:
            list (or numpy.array): Cropped frame indices.
        """
        num_candidates = record.num_slices - self.sample_thickness + 1

        # for val and test, the frames are retrieved uniformly
        if self.test:
            tick = num_candidates / float(self.sample_size)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.sample_size)])
            return offsets # offset + 1 when idx starts with 1

        # when training with random
        average_duration = num_candidates // self.sample_size

        if average_duration > 0:
            offsets = np.multiply(list(range(self.sample_size)), average_duration) + randint(average_duration, size=self.sample_size) # the output is the array with the size of sample_size, random numbers.

        # elif record.num_slices > self.sample_size: # frames - thick + 1 < segment, i.e. the length of the ct is less than the total segment number.
        #     offsets = np.sort(randint(num_candidates, size=self.sample_size)) # sample segment within the segment, some are missing, but some are duplicate

            # offsets = np.sort(choice(record.num_slices - self.sample_thickness + 1, self.sample_size, replace=False)) # sample segment within the segment, some are missing

        else: # not enough for one length, so select all the original ct several times, and append with some repeats. This is very rare.
            slices = list(range(num_candidates)) * (self.sample_size // num_candidates)
            slices.extend(choice(num_candidates, self.sample_size % num_candidates))

            offsets = np.sort(slices)
        
        return offsets # offset + 1 when idx starts with 1


class TemporalStepCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    then use all slices and several duplicates.

    For training, use random setting. For testing, use central crop.

    Args:
        sample_size (int): Desired output size of the crop.
        sample_step (int): Step size between either two output slices.
        sample_thickness (int): average over multiple frames.
        test (bool): whether or not random select frame from the segment
    """

    def __init__(self, sample_size, sample_step, sample_thickness, test=False):
        self.sample_size = sample_size
        self.sample_step = sample_step
        self.sample_thickness = sample_thickness
        self.test = test

    def __call__(self, record):
        """
        Args:
            record: path, num_slices, label
        Returns:
            list (or numpy.array): Cropped frame indices.
        """
        num_candidates = record.num_slices - self.sample_thickness + 1

        step = min( (num_candidates - 1) // (self.sample_size - 1), self.sample_step)

        if step > 0: # num_candidates >= size
            if self.test:
                begin_index = (num_candidates - step * (self.sample_size - 1)) // 2
            else:
                begin_index = randint(0, num_candidates - step * (self.sample_size - 1) )

            offsets = np.multiply(list(range(self.sample_size)), step) + begin_index # each step add the begin_index as offset
        else: # num_candidates < size, i.e. not enough slices. Need to add some random repeats
            slices = list(range(num_candidates)) * (self.sample_size // num_candidates)
            slices.extend(choice(num_candidates, self.sample_size % num_candidates))

            offsets = np.sort(slices)

        return offsets


# --------------------------------------------------------------
# ------ Deserted ----------------------------------------------
# --------------------------------------------------------------


class LoopPadding(object):
    ''' Append from the beginning if not enough objects.
        This is suitable for action, but not suitable for brain.
    '''
    def __init__(self, size):
        self.sample_size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.sample_size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.

    This is not suitable for brain, because never have a prior knowledge 
    on where will the lesion happen.
    """

    def __init__(self, size):
        self.sample_size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.sample_size]

        for index in out:
            if len(out) >= self.sample_size:
                break
            out.append(index)

        return out


