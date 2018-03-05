#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   18.09.2017
#-------------------------------------------------------------------------------
# This file is part of SSD-TensorFlow.
#
# SSD-TensorFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SSD-TensorFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SSD-Tensorflow.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

import cv2
import random

import numpy as np

from ssdutils import get_anchors_for_preset, get_preset_by_name, anchors2array
from ssdutils import box2array, compute_overlap, compute_location, anchors2array
from utils import Size, Sample, Point, Box, abs2prop, prop2abs
from math import sqrt

#-------------------------------------------------------------------------------
class Transform:
    def __init__(self, **kwargs):
        for arg, val in kwargs.items():
            setattr(self, arg, val)
        self.initialized = False

#-------------------------------------------------------------------------------
class ImageLoaderTransform(Transform):
    """
    Load and image from the file specified in the Sample object
    """
    def __call__(self, data, label, gt):
        return cv2.imread(gt.filename), label, gt

#-------------------------------------------------------------------------------
def process_overlap(overlap, box, anchor, matches, num_classes, vec):
    if overlap.idx in matches and matches[overlap.idx] >= overlap.score:
        return

    matches[overlap.idx] = overlap.score
    vec[overlap.idx, 0:num_classes+1] = 0
    vec[overlap.idx, box.labelid]     = 1
    vec[overlap.idx, num_classes+1:]  = compute_location(box, anchor)

#-------------------------------------------------------------------------------
class LabelCreatorTransform(Transform):
    """
    Create a label vector out of a ground trut sample
    Parameters: preset, num_classes
    """
    #---------------------------------------------------------------------------
    def initialize(self):
        self.anchors = get_anchors_for_preset(self.preset)
        self.vheight = len(self.anchors)
        self.vwidth = self.num_classes+5 # background class + location offsets
        self.img_size = Size(1000, 1000)
        self.anchors_arr = anchors2array(self.anchors, self.img_size)
        self.initialized = True

    #---------------------------------------------------------------------------
    def __call__(self, data, label, gt):
        #-----------------------------------------------------------------------
        # Initialize the data vector and other variables
        #-----------------------------------------------------------------------
        if not self.initialized:
            self.initialize()

        vec = np.zeros((self.vheight, self.vwidth), dtype=np.float32)

        #-----------------------------------------------------------------------
        # For every box compute the best match and all the matches above 0.5
        # Jaccard overlap
        #-----------------------------------------------------------------------
        overlaps = {}
        for box in gt.boxes:
            box_arr = box2array(box, self.img_size)
            overlaps[box] = compute_overlap(box_arr, self.anchors_arr, 0.5)

        #-----------------------------------------------------------------------
        # Set up the training vector resolving conflicts in favor of a better
        # match
        #-----------------------------------------------------------------------
        vec[:, self.num_classes]   = 1 # background class
        vec[:, self.num_classes+1] = 0 # x offset
        vec[:, self.num_classes+2] = 0 # y offset
        vec[:, self.num_classes+3] = 0 # log width scale
        vec[:, self.num_classes+4] = 0 # log height scale

        matches = {}
        for box in gt.boxes:
            for overlap in overlaps[box].good:
                anchor = self.anchors[overlap.idx]
                process_overlap(overlap, box, anchor, matches, self.num_classes, vec)

        matches = {}
        for box in gt.boxes:
            overlap = overlaps[box].best
            if not overlap:
                continue
            anchor  = self.anchors[overlap.idx]
            process_overlap(overlap, box, anchor, matches, self.num_classes, vec)

        return data, vec, gt

#-------------------------------------------------------------------------------
class ResizeTransform(Transform):
    """
    Resize an image
    Parameters: width, height, algorithms
    """
    def __call__(self, data, label, gt):
        alg = random.choice(self.algorithms)
        resized = cv2.resize(data, (self.width, self.height), interpolation=alg)
        return resized, label, gt

#-------------------------------------------------------------------------------
class RandomTransform(Transform):
    """
    Call another transform with a given probability
    Parameters: prob, transform
    """
    def __call__(self, data, label, gt):
        p = random.uniform(0, 1)
        if p < self.prob:
            return self.transform(data, label, gt)
        return data, label, gt

#-------------------------------------------------------------------------------
class BrightnessTransform(Transform):
    """
    Transform brightness
    Parameters: delta
    """
    def __call__(self, data, label, gt):
        data = data.astype(np.float32)
        delta = random.randint(-self.delta, self.delta)
        data += delta
        data[data>255] = 255
        data[data<0] = 0
        data = data.astype(np.uint8)
        return data, label, gt

#-------------------------------------------------------------------------------
class ContrastTransform(Transform):
    """
    Transform contrast
    Parameters: lower, upper
    """
    def __call__(self, data, label, gt):
        data = data.astype(np.float32)
        delta = random.uniform(self.lower, self.upper)
        data *= delta
        data[data>255] = 255
        data[data<0] = 0
        data = data.astype(np.uint8)
        return data, label, gt

#-------------------------------------------------------------------------------
class HueTransform(Transform):
    """
    Transform hue
    Parameters: delta
    """
    def __call__(self, data, label, gt):
        data = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
        data = data.astype(np.float32)
        delta = random.randint(-self.delta, self.delta)
        data[0] += delta
        data[0][data[0]>180] -= 180
        data[0][data[0]<0] +=180
        data = data.astype(np.uint8)
        data = cv2.cvtColor(data, cv2.COLOR_HSV2BGR)
        return data, label, gt

#-------------------------------------------------------------------------------
class SaturationTransform(Transform):
    """
    Transform hue
    Parameters: lower, upper
    """
    def __call__(self, data, label, gt):
        data = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
        data = data.astype(np.float32)
        delta = random.uniform(self.lower, self.upper)
        data[1] *= delta
        data[1][data[1]>255] = 255
        data[1][data[1]<0] = 0
        data = data.astype(np.uint8)
        data = cv2.cvtColor(data, cv2.COLOR_HSV2BGR)
        return data, label, gt

#-------------------------------------------------------------------------------
def transform_box(box, orig_size, new_size, h_off, w_off):
    #---------------------------------------------------------------------------
    # Compute the new coordinates of the box
    #---------------------------------------------------------------------------
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, orig_size)
    xmin += w_off
    xmax += w_off
    ymin += h_off
    ymax += h_off

    #---------------------------------------------------------------------------
    # Check if the center falls within the image
    #---------------------------------------------------------------------------
    width = xmax - xmin
    height = ymax - ymin
    new_cx = xmin + int(width/2)
    new_cy = ymin + int(height/2)
    if new_cx < 0 or new_cx >= new_size.w:
        return None
    if new_cy < 0 or new_cy >= new_size.h:
        return None

    center, size = abs2prop(xmin, xmax, ymin, ymax, new_size)
    return Box(box.label, box.labelid, center, size)

#-------------------------------------------------------------------------------
def transform_gt(gt, new_size, h_off, w_off):
    boxes = []
    for box in gt.boxes:
        box = transform_box(box, gt.imgsize, new_size, h_off, w_off)
        if box is None:
            continue
        boxes.append(box)
    return Sample(gt.filename, boxes, new_size)

#-------------------------------------------------------------------------------
class ExpandTransform(Transform):
    """
    Expand the image and fill the empty space with the mean value
    Parameters: max_ratio, mean_value
    """
    def __call__(self, data, label, gt):
        #-----------------------------------------------------------------------
        # Calculate sizes and offsets
        #-----------------------------------------------------------------------
        ratio = random.uniform(1, self.max_ratio)
        orig_size = gt.imgsize
        new_size = Size(int(orig_size.w*ratio), int(orig_size.h*ratio))
        h_off = random.randint(0, new_size.h-orig_size.h)
        w_off = random.randint(0, new_size.w-orig_size.w)

        #-----------------------------------------------------------------------
        # Create the new image and place the input image in it
        #-----------------------------------------------------------------------
        img = np.zeros((new_size.h, new_size.w, 3))
        img[:, :] = np.array(self.mean_value)
        img[h_off:h_off+orig_size.h, w_off:w_off+orig_size.w, :] = data

        #-----------------------------------------------------------------------
        # Transform the ground truth
        #-----------------------------------------------------------------------
        gt = transform_gt(gt, new_size, h_off, w_off)

        return img, label, gt

#-------------------------------------------------------------------------------
class SamplerTransform(Transform):
    """
    Sample a fraction of the image according to given parameters
    Params: min_scale, max_scale, min_aspect_ratio, max_aspect_ratio,
            min_jaccard_overlap
    """
    def __call__(self, data, label, gt):
        #-----------------------------------------------------------------------
        # Check whether to sample or not
        #-----------------------------------------------------------------------
        if not self.sample:
            return data, label, gt

        #-----------------------------------------------------------------------
        # Retry sampling a couple of times
        #-----------------------------------------------------------------------
        source_boxes = anchors2array(gt.boxes, gt.imgsize)
        box = None
        box_arr = None
        for _ in range(self.max_trials):
            #-------------------------------------------------------------------
            # Sample a bounding box
            #-------------------------------------------------------------------
            scale = random.uniform(self.min_scale, self.max_scale)
            aspect_ratio = random.uniform(self.min_aspect_ratio,
                                          self.max_aspect_ratio)

            # make sure width and height will not be larger than 1
            aspect_ratio = max(aspect_ratio, scale**2)
            aspect_ratio = min(aspect_ratio, 1/(scale**2))

            width = scale*sqrt(aspect_ratio)
            height = scale/sqrt(aspect_ratio)
            cx = 0.5*width + random.uniform(0, 1-width)
            cy = 0.5*height + random.uniform(0, 1-height)
            center = Point(cx, cy)
            size = Size(width, height)

            #-------------------------------------------------------------------
            # Check if the box satisfies the jaccard overlap constraint
            #-------------------------------------------------------------------
            box_arr = np.array(prop2abs(center, size, gt.imgsize))
            overlap = compute_overlap(box_arr, source_boxes, 0)
            if overlap.best and overlap.best.score >= self.min_jaccard_overlap:
                box = Box(None, None, center, size)
                break

        if box is None:
            return None

        #-----------------------------------------------------------------------
        # Crop the box and adjust the ground truth
        #-----------------------------------------------------------------------
        new_size = Size(box_arr[1]-box_arr[0], box_arr[3]-box_arr[2])
        w_off = -box_arr[0]
        h_off = -box_arr[2]
        data = data[box_arr[2]:box_arr[3], box_arr[0]:box_arr[1]]
        gt = transform_gt(gt, new_size, h_off, w_off)

        return data, label, gt

#-------------------------------------------------------------------------------
class SamplePickerTransform(Transform):
    """
    Run a bunch of sample transforms and return one of the produced samples
    Parameters: samplers
    """
    def __call__(self, data, label, gt):
        samples = []
        for sampler in self.samplers:
            sample = sampler(data, label, gt)
            if sample is not None:
                samples.append(sample)
        return random.choice(samples)

#-------------------------------------------------------------------------------
class HorizontalFlipTransform(Transform):
    """
    Horizontally flip the image
    """
    def __call__(self, data, label, gt):
        data = cv2.flip(data, 1)
        boxes = []
        for box in gt.boxes:
            center = Point(1-box.center.x, box.center.y)
            box = Box(box.label, box.labelid, center, box.size)
            boxes.append(box)
        gt = Sample(gt.filename, boxes, gt.imgsize)

        return data, label, gt
