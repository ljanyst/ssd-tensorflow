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
from ssdutils import box2array, compute_overlap, compute_location
from utils import Size

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
