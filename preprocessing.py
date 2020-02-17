'''
This script is used to extract mask from given dicom file

'''
import os
import SimpleITK as sitk
import cv2
from ctdataset import CTRecord
from opts import parse_opts as ParseOpts
import ipdb
import numpy as np

def ContinuousSeq(array, mean=True):
    # input: an array of 0 and 1.
    # output: 
    #   default: mean=True, i.e. middle point of continuous 1's
    #   mead = False: return the boundary of each sequence
    points = list()
    if not sum(array):
        return points

    last = 0
    for idx in range(len(array)):
        num = array[idx]
        # num == 0, last == 0: skip
        if not num and not last:
            last = num
            continue
        # num == 1, last == 0: start a new array
        elif num and not last:
            seq = [idx]
        # num == 1, last == 1: continue previous array
        elif num and last:
            if mean:
                seq.append(idx)
        # num == 1, until the last digit:
        elif num and idx == len(array)-1:
            if mean:
                points.append((sum(seq) + idx) // (len(seq) + 1))
            else:
                points.extend([seq[0], idx])
        # num == 0, last == 1: stop
        elif not num and last:
            if mean:
                points.append(sum(seq) // len(seq))
            else:
                points.extend([seq[0], idx - 1])

        last = num

    return points

def ContinuousValidSeq(array, img_array):
    # input: an array of 0 and 1.
    # output: 
    #   default: mean=True, i.e. middle point of continuous 1's
    #   mead = False: return the boundary of each sequence
    points = list()
    if not sum(array):
        return points

    last = 0
    for idx in range(len(array)):
        num = array[idx]
        # num == 0, last == 0: skip
        if not num and not last:
            last = num
            continue
        # num == 1, last == 0: start a new array
        elif num and not last:
            seq = [idx]
        # num == 1, last == 1: continue previous array
        elif num and last:
            seq.append(idx)
        # num == 1, until the last digit:
        elif num and idx == len(array)-1:
            if img_array[seq[0]-1] > 120:
                points.append((sum(seq) + idx) // (len(seq) + 1))
        # num == 0, last == 1: stop
        elif not num and last:
            if img_array[seq[0]-1] > 120 and img_array[seq[-1]+1] > 120:
                points.append(sum(seq) // len(seq))

        last = num

    return points

def CountSkull(img_line):
    # input: a line of original image
    # output: modified values
    # skull = (img_line > 100).astype('int')
    # if not skull.sum():
    #     return skull
    # else:
    #     boundary = ContinuousSeq(skull, mean=False)
    skull = (img_line > 200).astype('int')

    if skull.sum() == 0:
        return skull
    else:
        ones = np.where(skull)[0]
        boundary = (ones[0], ones[-1])

    # find the skull:
    inbrain = False
    last = img_line[boundary[0]-1] if boundary[0] > 0 else -1
    for idx, this in enumerate(img_line[boundary[0]:boundary[1]+1]):
        # hole left boundary
        if this < 0 and last >= 0 and last <= 100:
            skull[idx + boundary[0]] = 1
        # hole right boundary
        elif this <= 100 and this >= 0 and last < 0 and inbrain:
            skull[idx + boundary[0] -1] = 1

        last = this

    if this >= 0 and this <= 100:
        skull[-2:] = 1

    return skull


def CalMaskCV(np_img):
    # np_img: (axel, cor, sag), top-down, left-right shifted
    # 1. extract skull: make skull connect by dilation. 
    # 2. use skull as a boundary and set a mask. 
    # 3. find the correct grayscale levels for brain tissue.

    sag_masks = list()

    for i in range(np_img.shape[-1]):
        img = np_img[:, :, i]
        skull = np.zeros_like(img)
        
        # skip slice that does not contain bone
        if img.max() < 200:
            sag_masks.append(np.zeros_like(img))
            continue

        # get brain pixels
        brain = cv2.threshold(img, 95, 255, cv2.THRESH_TOZERO_INV)[1]
        brain = cv2.threshold(brain, 5, 100, cv2.THRESH_BINARY)[1].astype('uint8')

        # morphology on brain pixels
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        brain = cv2.morphologyEx(brain, cv2.MORPH_ERODE, erode_kernel)
        brain = cv2.morphologyEx(brain, cv2.MORPH_DILATE, dilate_kernel)

        # get brain contours and find the largest 
        _, contours, _ = cv2.findContours(brain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = np.array([cv2.contourArea(c) for c in contours])
        brain = cv2.drawContours(brain, contours, areas.argmax(), 255, -1)

        # get skull contour
        skull = cv2.threshold(img, 95, 255, cv2.THRESH_BINARY)[1].astype('uint8') # get all the bones
        skull = cv2.morphologyEx(skull, cv2.MORPH_DILATE, dilate_kernel)
        _, contours, _ = cv2.findContours(brain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = np.array([cv2.contourArea(c) for c in contours])
        brain = cv2.drawContours(brain, contours, areas.argmax(), 255, -1)

        # floodFill with mask
        #cv2.floodFill(brain, mask, (x, y), (color), (), (), cv2.FLOODFILL_MASK_ONLY)
        cv2.floodFill(copyImg, mask, (220, 250), (0, 255, 255), (100, 100, 100), (50, 50 ,50), cv.FLOODFILL_FIXED_RANGE)


        # # define close kernel (do not have to be square):
        # if i < 160 or i > 350:
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))# np.ones((5, 5), dtype='uint8')
        # else:
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))# np.ones((5, 5), dtype='uint8')

        # close_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (60, 60))# np.ones((5, 5), dtype='uint8')

        # # get skull and make skull connect
        # # method 1: cv2 threshold and close
        # # skull = cv2.inRange(img, 0, 100)
        # skull = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)[1].astype('uint8') # get all the bones
        # brain = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO_INV)[1]
        # brain = cv2.threshold(brain, 0, 100, cv2.THRESH_BINARY)[1].astype('uint8')

        # skull_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        # brain_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))

        # skull = cv2.morphologyEx(skull, cv2.MORPH_CLOSE, skull_kernel)
        # brain = cv2.morphologyEx(brain, cv2.MORPH_OPEN, brain_kernel)

        # # method 2: iterate each line
        # for iid, iline in enumerate(img):
        #     skull[iid] = CountSkull(iline)

        # get skull contour and make a mask
        # brain, contour, _ = cv2.findContours(skull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # whole = cv2.drawContours(brain, contour, -1, 100, -1)


        # # watershed
        # ret, markers = cv2.connectedComponents(brain)
        # markers = markers+1
        # markers[skull==255] = 0

        # ori = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)[1]
        # ori = cv2.threshold(ori, 200, 200, cv2.THRESH_TOZERO_INV)[1].astype('uint8')
        # ori = cv2.cvtColor(ori, cv2.COLOR_GRAY2BGR)
        # markers = cv2.watershed(ori, markers)
        # ori[markers==-1] = 255
        # brain = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)

        sag_masks.append(brain)



        # get brain and make a mask
        # brain = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO_INV)[1]
        # brain = cv2.threshold(brain, 0, 255, cv2.THRESH_BINARY)[1].astype('uint8')
        # brain = cv2.morphologyEx(brain, cv2.MORPH_ERODE, kernel)
        # brain = cv2.morphologyEx(brain, cv2.MORPH_DILATE, close_kernel)



        # # bilateral filter
        # ori = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)[1]
        # ori = cv2.threshold(ori, 200, 200, cv2.THRESH_TOZERO_INV)[1].astype('uint8')
        # ori = cv2.cvtColor(ori, cv2.COLOR_GRAY2BGR)
        # brain = cv2.bilateralFilter(ori, 0, 100, 15)
        # brain = cv2.cvtColor(brain, cv2.COLOR_BGR2GRAY)











        # # last line:
        # boundary = ContinuousSeq(skull[iid], mean=False)
        # if len(boundary) > 2:
        #     skull[iid][boundary[0]: boundary[-1]] = 1

        # skull = cv2.morphologyEx(skull, cv2.MORPH_CLOSE, kernel).astype('uint8')

        # if hierarchy[0].sum(0)[-1] == len(hierarchy[0]) * -1: # no one has child contour
        #     sag_masks.append(np.zeros_like(img))
        #     continue

        # useful = list()
        # for hi, hv in enumerate(hierarchy[0]):
        #     if not hv[3] == -1:
        #         useful.append(contour[hi])

        # if len(useful) > 1:
        #     brain_contour = max(useful, key = cv2.contourArea)
        # else:
        #     sag_masks.append(np.zeros_like(img))
        #     continue

        # # get largest area of connected brain
        # mask = np.zeros_like(img)
        # mask = cv2.fillPoly(mask, [brain_contour], 255)
        # sag_masks.append(mask)


        # cv2.imwrite(dicom.path + '/test.png', mask) 
        # cv2.imwrite(dicom.path + '/test_ori.png', img) 

        # ipdb.set_trace()

        # # 
        # sag_mask = sitk.GetArrayFromImage(sitk.ConnectedThreshold(sitk_img[i, :, :], seedList=sag_seeds, lower=0, upper=100))
        # sag_masks.append(cv2.morphologyEx(sag_mask, cv2.MORPH_OPEN, np.ones((7, 7), dtype='uint8')))

    masks = np.array(sag_masks)
    masks = sitk.GetImageFromArray(masks.transpose(1, 2, 0))
    return masks

def CalMaskSITK(sitk_img):
    # sitk_img: (sag, cor, axel), top-down, left-right shifted
    # first checkout 16 saggital view images:
    positions = list(range(112, 401, 16))
    # positions = range(512)
    sag_masks = list()
    mid_point = sitk_img.GetDepth()//2
    sag_seeds = [(mid_point - mid_point//10, 255), (mid_point + mid_point//10, 255), (mid_point, 255), \
             (mid_point - mid_point//10, 245), (mid_point + mid_point//10, 265)]
    for i in positions:
        sag_mask = sitk.GetArrayFromImage(sitk.ConnectedThreshold(sitk_img[i, :, :], seedList=sag_seeds, lower=-20, upper=100))
        sag_masks.append(cv2.morphologyEx(sag_mask, cv2.MORPH_OPEN, np.ones((7, 7), dtype='uint8')))

    masks = np.array(sag_masks)
    # masks = sitk.GetImageFromArray(masks.transpose(1, 2, 0))

    # from sag_masks, get new seeds:
    masks = list()
    for img_idx in range(sitk_img.GetDepth()):
        axial_seeds = list()
        for pi in range(len(positions)):
            points = ContinuousValidSeq(sag_masks[pi][img_idx], sitk.GetArrayFromImage(sitk_img[:, :, img_idx])[:, positions[pi]])
            # points = ContinuousSeq(sag_masks[pi][img_idx])            

            for point in points:
                axial_seeds.append((positions[pi], point))

        if len(axial_seeds) > 1:
            mask = sitk.GetArrayFromImage(sitk.ConnectedThreshold(sitk_img[:, :, img_idx], seedList=axial_seeds, lower=-10, upper=100))
            masks.append(mask)#cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), dtype='uint8')))
        else:
            masks.append(np.zeros((512, 512)))

    # make new vol
    masks = sitk.GetImageFromArray(np.array(masks))

    # masks.SetDirection(sitk_img.GetDirection())
    # masks.SetOrigin(sitk_img.GetOrigin()) 
    # masks.SetSpacing(sitk_img.GetSpacing())

    return masks

def ParseDicom(record):
    # need to read dicoms and conduct windows, this is mainly designed for testing
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    dicom_names = reader.GetGDCMSeriesFileNames(next(os.walk(record.path, topdown=False))[0])

    # record slice number
    if record.num_slices == -1 or not record.num_slices == len(dicom_names):
        record.num_slices = len(dicom_names)
    
    reader.SetFileNames(dicom_names)
    try:
        sitk_img = reader.Execute() # shape of [512, 512, X]
        np_img = sitk.GetArrayFromImage(sitk_img) # shape of [X, 512, 512]
    except RuntimeError:
        print('Slice {} is beyond length of {}.'.format(record.path, slice_idx))
    
    # _, h, w = imgs.shape
    # windows = [(40, 80), (80, 200), (40, 380)]
    # output = np.zeros((h, w, 3)) 

    # for frame in imgs:                    
    #     for win_idx, win in enumerate(windows):
    #         yMin = int(win[0] - 0.5 * win[1])
    #         yMax = int(win[0] + 0.5 * win[1])
    #         sub_img = np.clip(frame, yMin, yMax)
    #         output[:, :, win_idx] += (sub_img - yMin) / (yMax - yMin) * 255

    # samples.append(Image.fromarray((output/self.sample_thickness).astype(np.uint8)))
    return sitk_img, np_img

def ParseList(list_file):
    ct_list = list()

    # start make row
    for x in open(list_file):
        row = x.strip().rsplit(' ', 2) # deal with logs

        # skip cases that does not exist
        if not os.path.exists(row[0]):
            print(row[0], 'does not exist...')
            continue

        ct_list.append(CTRecord(row))

    return ct_list

def SaveImage(vol, file_name, output_format='nii.gz', norm=True):
    '''For a given sitk.image: N x W x H, generate image file with normalization to 0-1 (nii.gz)
    or 0-255 (jpg).

    Input:
        image: sitk.image
        file_name: /path/to/file. The file could be named as scanid of the file or 
                    patientid_scanid, which is used for M's review.
        output_format: jpg or nii.gz

    Output:
        store directly to file_name, no return.
    '''

    if output_format.lower() in ['nii', 'nii.gz', 'nifti', 'nifty']:
        if norm:
            sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(vol, outputMinimum=0, outputMaximum=1), sitk.sitkFloat32), '{0}.{1}'.format(file_name, output_format)) # otherwise, store as float64 but internal is float32
        else:
            sitk.WriteImage(vol, '{0}.{1}'.format(file_name, output_format))
    elif output_format.lower() in ['jpg', 'jpeg']:
        if not os.path.exists(file_name):
            os.makedirs(file_name)

        sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(vol, outputMinimum=0, outputMaximum=255), sitk.sitkUInt8), ['{0}/img_{1:05d}.{2}'.format(file_name, i, output_format) for i in range(vol.GetSize()[2])])
    elif output_format.lower() in ['tif', 'tiff']:
        if not os.path.exists(file_name):
            os.makedirs(file_name)

        sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(vol, outputMinimum=0, outputMaximum=1), sitk.sitkFloat32), ['{0}/img_{1:05d}.{2}}'.format(file_name, i, output_format) for i in range(vol.GetSize()[2])])


if __name__ == '__main__':
    args = ParseOpts()
    files = ParseList('./labels/{}.txt'.format(args.dataset))

    # check output path
    if args.dataset == 'rsna':
        out_path = '/raid/snac/crc/rsna/masks/'
    else:
        out_path = '/raid/snac/crc/xnat/masks/' 
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # go through all files
    for dicom in files:
        sitk_img, np_img = ParseDicom(dicom)
        mask = CalMaskCV(np_img)
        # mask = CalMaskSITK(sitk_img)

        # make mask the same direction/origin/spacing as the original image
        mask.SetDirection(sitk_img.GetDirection())
        mask.SetOrigin(sitk_img.GetOrigin()) 
        mask.SetSpacing(sitk_img.GetSpacing())
        sitk.WriteImage(mask, dicom.path + '/test_opencv.nii.gz')

        ipdb.set_trace()
        print('None')

    print('Finish')







