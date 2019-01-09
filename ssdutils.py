#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   29.08.2017
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

import numpy as np

from utils import Size, Point, Overlap, Score, Box, prop2abs, normalize_box
from collections import namedtuple, defaultdict
from math import sqrt, log, exp

#-------------------------------------------------------------------------------
# Define the flavors of SSD that we're going to use and it's various properties.
# It's done so that we don't have to build the whole network in memory in order
# to pre-process the datasets.
#-------------------------------------------------------------------------------
SSDMap = namedtuple('SSDMap', ['size', 'scale', 'aspect_ratios'])
SSDPreset = namedtuple('SSDPreset', ['name', 'image_size', 'maps',
                                     'extra_scale', 'num_anchors'])

SSD_PRESETS = {
    'vgg300': SSDPreset(name = 'vgg300',
                        image_size = Size(300, 300),
                        maps = [
                            SSDMap(Size(38, 38), 0.1,   [2, 0.5]),
                            SSDMap(Size(19, 19), 0.2,   [2, 3, 0.5, 1./3.]),
                            SSDMap(Size(10, 10), 0.375, [2, 3, 0.5, 1./3.]),
                            SSDMap(Size( 5,  5), 0.55,  [2, 3, 0.5, 1./3.]),
                            SSDMap(Size( 3,  3), 0.725, [2, 0.5]),
                            SSDMap(Size( 1,  1), 0.9,   [2, 0.5])
                        ],
                        extra_scale = 1.075,
                        num_anchors = 8732),
    'vgg512': SSDPreset(name = 'vgg512',
                        image_size = Size(512, 512),
                        maps = [
                            SSDMap(Size(64, 64), 0.07, [2, 0.5]),
                            SSDMap(Size(32, 32), 0.15, [2, 3, 0.5, 1./3.]),
                            SSDMap(Size(16, 16), 0.3,  [2, 3, 0.5, 1./3.]),
                            SSDMap(Size( 8,  8), 0.45, [2, 3, 0.5, 1./3.]),
                            SSDMap(Size( 4,  4), 0.6,  [2, 3, 0.5, 1./3.]),
                            SSDMap(Size( 2,  2), 0.75, [2, 0.5]),
                            SSDMap(Size( 1,  1), 0.9,  [2, 0.5])
                        ],
                        extra_scale = 1.05,
                        num_anchors = 24564)
}

#-------------------------------------------------------------------------------
# Default box parameters both in terms proportional to image dimensions
#-------------------------------------------------------------------------------
Anchor = namedtuple('Anchor', ['center', 'size', 'x', 'y', 'scale', 'map'])

#-------------------------------------------------------------------------------
def get_preset_by_name(pname):
    if not pname in SSD_PRESETS:
        raise RuntimeError('No such preset: '+pname)
    return SSD_PRESETS[pname]

#-------------------------------------------------------------------------------
def get_anchors_for_preset(preset):
    """
    Compute the default (anchor) boxes for the given SSD preset
    """
    #---------------------------------------------------------------------------
    # Compute the width and heights of the anchor boxes for every scale
    #---------------------------------------------------------------------------
    box_sizes = []
    for i in range(len(preset.maps)):
        map_params = preset.maps[i]
        s = map_params.scale
        aspect_ratios = [1] + map_params.aspect_ratios
        aspect_ratios = list(map(lambda x: sqrt(x), aspect_ratios))

        sizes = []
        for ratio in aspect_ratios:
            w = s * ratio
            h = s / ratio
            sizes.append((w, h))
        if i < len(preset.maps)-1:
            s_prime = sqrt(s*preset.maps[i+1].scale)
        else:
            s_prime = sqrt(s*preset.extra_scale)
        sizes.append((s_prime, s_prime))
        box_sizes.append(sizes)

    #---------------------------------------------------------------------------
    # Compute the actual boxes for every scale and feature map
    #---------------------------------------------------------------------------
    anchors = []
    for k in range(len(preset.maps)):
        fk = preset.maps[k].size[0]
        s = preset.maps[k].scale
        for size in box_sizes[k]:
            for j in range(fk):
                y = (j+0.5)/float(fk)
                for i in range(fk):
                    x = (i+0.5)/float(fk)
                    box = Anchor(Point(x, y), Size(size[0], size[1]),
                                 i, j, s, k)
                    anchors.append(box)
    return anchors

#-------------------------------------------------------------------------------
def anchors2array(anchors, img_size):
    """
    Computes a numpy array out of absolute anchor params (img_size is needed
    as a reference)
    """
    arr = np.zeros((len(anchors), 4))
    for i in range(len(anchors)):
        anchor = anchors[i]
        xmin, xmax, ymin, ymax = prop2abs(anchor.center, anchor.size, img_size)
        arr[i] = np.array([xmin, xmax, ymin, ymax])
    return arr

#-------------------------------------------------------------------------------
def box2array(box, img_size):
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
    return np.array([xmin, xmax, ymin, ymax])

#-------------------------------------------------------------------------------
def jaccard_overlap(box_arr, anchors_arr):
    areaa = (anchors_arr[:, 1]-anchors_arr[:, 0]+1) * \
            (anchors_arr[:, 3]-anchors_arr[:, 2]+1)
    areab = (box_arr[1]-box_arr[0]+1) * (box_arr[3]-box_arr[2]+1)

    xxmin = np.maximum(box_arr[0], anchors_arr[:, 0])
    xxmax = np.minimum(box_arr[1], anchors_arr[:, 1])
    yymin = np.maximum(box_arr[2], anchors_arr[:, 2])
    yymax = np.minimum(box_arr[3], anchors_arr[:, 3])

    w = np.maximum(0, xxmax-xxmin+1)
    h = np.maximum(0, yymax-yymin+1)
    intersection = w*h
    union = areab+areaa-intersection
    return intersection/union

#-------------------------------------------------------------------------------
def compute_overlap(box_arr, anchors_arr, threshold):
    iou = jaccard_overlap(box_arr, anchors_arr)
    overlap = iou > threshold

    good_idxs = np.nonzero(overlap)[0]
    best_idx  = np.argmax(iou)
    best = None
    good = []

    best = Score(best_idx, iou[best_idx])

    for idx in good_idxs:
        good.append(Score(idx, iou[idx]))

    return Overlap(best, good)

#-------------------------------------------------------------------------------
def compute_location(box, anchor):
    arr = np.zeros((4))
    arr[0] = (box.center.x-anchor.center.x)/anchor.size.w*10
    arr[1] = (box.center.y-anchor.center.y)/anchor.size.h*10
    arr[2] = log(box.size.w/anchor.size.w)*5
    arr[3] = log(box.size.h/anchor.size.h)*5
    return arr

#-------------------------------------------------------------------------------
def decode_location(box, anchor):
    box[box > 100] = 100 # only happens early training

    x = box[0]/10 * anchor.size.w + anchor.center.x
    y = box[1]/10 * anchor.size.h + anchor.center.y
    w = exp(box[2]/5) * anchor.size.w
    h = exp(box[3]/5) * anchor.size.h
    return Point(x, y), Size(w, h)

#-------------------------------------------------------------------------------
def decode_boxes(pred, anchors, confidence_threshold = 0.01, lid2name = {},
                 detections_cap=200):
    """
    Decode boxes from the neural net predictions.
    Label names are decoded using the lid2name dictionary - the id to name
    translation is not done if the corresponding key does not exist.
    """

    #---------------------------------------------------------------------------
    # Find the detections
    #---------------------------------------------------------------------------
    num_classes = pred.shape[1]-4
    bg_class    = num_classes-1
    box_class   = np.argmax(pred[:, :num_classes-1], axis=1)
    confidence  = pred[np.arange(len(pred)), box_class]
    if detections_cap is not None:
        detections = np.argsort(confidence)[::-1][:detections_cap]
    else:
        detections = np.argsort(confidence)[::-1]

    #---------------------------------------------------------------------------
    # Decode coordinates of each box with confidence over a threshold
    #---------------------------------------------------------------------------
    boxes = []
    for idx in detections:
        confidence = pred[idx, box_class[idx]]
        if confidence < confidence_threshold:
            break

        center, size = decode_location(pred[idx, num_classes:], anchors[idx])
        cid = box_class[idx]
        cname = None
        if cid in lid2name:
            cname = lid2name[cid]
        det = (confidence, normalize_box(Box(cname, cid, center, size)))
        boxes.append(det)

    return boxes

#-------------------------------------------------------------------------------
def non_maximum_suppression(boxes, overlap_threshold):
    #---------------------------------------------------------------------------
    # Convert to absolute coordinates and to a more convenient format
    #---------------------------------------------------------------------------
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    conf = []
    img_size = Size(1000, 1000)

    for box in boxes:
        params = prop2abs(box[1].center, box[1].size, img_size)
        xmin.append(params[0])
        xmax.append(params[1])
        ymin.append(params[2])
        ymax.append(params[3])
        conf.append(box[0])

    xmin = np.array(xmin)
    xmax = np.array(xmax)
    ymin = np.array(ymin)
    ymax = np.array(ymax)
    conf = np.array(conf)

    #---------------------------------------------------------------------------
    # Compute the area of each box and sort the indices by confidence level
    # (lowest confidence first first).
    #---------------------------------------------------------------------------
    area = (xmax-xmin+1) * (ymax-ymin+1)
    idxs = np.argsort(conf)
    pick = []

    #---------------------------------------------------------------------------
    # Loop until we still have indices to process
    #---------------------------------------------------------------------------
    while len(idxs) > 0:
        #-----------------------------------------------------------------------
        # Grab the last index (ie. the most confident detection), remove it from
        # the list of indices to process, and put it on the list of picks
        #-----------------------------------------------------------------------
        last = idxs.shape[0]-1
        i    = idxs[last]
        idxs = np.delete(idxs, last)
        pick.append(i)
        suppress = []

        #-----------------------------------------------------------------------
        # Figure out the intersection with the remaining windows
        #-----------------------------------------------------------------------
        xxmin = np.maximum(xmin[i], xmin[idxs])
        xxmax = np.minimum(xmax[i], xmax[idxs])
        yymin = np.maximum(ymin[i], ymin[idxs])
        yymax = np.minimum(ymax[i], ymax[idxs])

        w = np.maximum(0, xxmax-xxmin+1)
        h = np.maximum(0, yymax-yymin+1)
        intersection = w*h

        #-----------------------------------------------------------------------
        # Compute IOU and suppress indices with IOU higher than a threshold
        #-----------------------------------------------------------------------
        union    = area[i]+area[idxs]-intersection
        iou      = intersection/union
        overlap  = iou > overlap_threshold
        suppress = np.nonzero(overlap)[0]
        idxs     = np.delete(idxs, suppress)

    #---------------------------------------------------------------------------
    # Return the selected boxes
    #---------------------------------------------------------------------------
    selected = []
    for i in pick:
        selected.append(boxes[i])

    return selected

#-------------------------------------------------------------------------------
def suppress_overlaps(boxes):
    class_boxes    = defaultdict(list)
    selected_boxes = []
    for box in boxes:
        class_boxes[box[1].labelid].append(box)

    for k, v in class_boxes.items():
        selected_boxes += non_maximum_suppression(v, 0.45)
    return selected_boxes
