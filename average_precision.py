#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   13.09.2017
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

from collections import defaultdict
from ssdutils import jaccard_overlap
from utils import Size, prop2abs

IMG_SIZE = Size(1000, 1000)

#-------------------------------------------------------------------------------
def APs2mAP(aps):
    """
    Take a mean of APs over all classes to compute mAP
    """
    num_classes = 0.
    sum_ap = 0.
    for _, v in aps.items():
        sum_ap += v
        num_classes += 1
    return sum_ap/num_classes

#-------------------------------------------------------------------------------
class APCalculator:
    """
    Compute average precision of object detection as used in PASCAL VOC
    Challenges. It is a peculiar measure because of the way it calculates the
    precision-recall curve. It's highly sensitive to the sorting order of the
    predictions in different images. Ie. the exact same resulting bounding
    boxes in all images may get different AP score depending on the way
    the boxes are sorted globally by confidence.
    Reference: http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
    Reference: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
    """
    #---------------------------------------------------------------------------
    def __init__(self, minoverlap=0.5):
        """
        Initialize the calculator.
        """
        self.minoverlap = minoverlap
        self.clear()

    #---------------------------------------------------------------------------
    def add_detections(self, gt_sample, boxes):
        """
        Add new detections to the calculator.
        :param gt_sample: ground truth sample
        :param boxes:     a list of (float, Box) tuples representing
                          detections and their confidences, the detections
                          must have a correctly set label
        """

        sample_id = len(self.gt_samples)
        self.gt_samples.append(gt_sample)

        for conf, box in boxes:
            arr = np.array(prop2abs(box.center, box.size, IMG_SIZE))
            self.det_params[box.label].append(arr)
            self.det_confidence[box.label].append(conf)
            self.det_sample_ids[box.label].append(sample_id)

    #---------------------------------------------------------------------------
    def compute_aps(self):
        """
        Compute the average precision per class as well as mAP.
        """

        #-----------------------------------------------------------------------
        # Split the ground truth samples by class and sample
        #-----------------------------------------------------------------------
        counts = defaultdict(lambda: 0)
        gt_map = defaultdict(dict)

        for sample_id, sample in enumerate(self.gt_samples):
            boxes_by_class = defaultdict(list)
            for box in sample.boxes:
                counts[box.label] += 1
                boxes_by_class[box.label].append(box)

            for k, v in boxes_by_class.items():
                arr = np.zeros((len(v), 4))
                match = np.zeros((len(v)), dtype=np.bool)
                for i, box in enumerate(v):
                    arr[i] = np.array(prop2abs(box.center, box.size, IMG_SIZE))
                gt_map[k][sample_id] = (arr, match)

        #-----------------------------------------------------------------------
        # Compare predictions to ground truth
        #-----------------------------------------------------------------------
        aps = {}
        for k in gt_map:
            #-------------------------------------------------------------------
            # Create numpy arrays of detection parameters and sort them
            # in descending order
            #-------------------------------------------------------------------
            params = np.array(self.det_params[k], dtype=np.float32)
            confs = np.array(self.det_confidence[k], dtype=np.float32)
            sample_ids = np.array(self.det_sample_ids[k], dtype=np.int)
            idxs_max = np.argsort(-confs)
            params = params[idxs_max]
            confs = confs[idxs_max]
            sample_ids = sample_ids[idxs_max]

            #-------------------------------------------------------------------
            # Loop over the detections and count true and false positives
            #-------------------------------------------------------------------
            tps = np.zeros((params.shape[0])) # true positives
            fps = np.zeros((params.shape[0])) # false positives
            for i in range(params.shape[0]):
                sample_id = sample_ids[i]
                box = params[i]

                #---------------------------------------------------------------
                # The image this detection comes from contains no objects of
                # of this class
                #---------------------------------------------------------------
                if not sample_id in gt_map[k]:
                    fps[i] = 1
                    continue

                #---------------------------------------------------------------
                # Compute the jaccard overlap and see if it's over the threshold
                #---------------------------------------------------------------
                gt = gt_map[k][sample_id][0]
                matched = gt_map[k][sample_id][1]

                iou = jaccard_overlap(box, gt)
                max_idx = np.argmax(iou)

                if iou[max_idx] < self.minoverlap:
                    fps[i] = 1
                    continue

                #---------------------------------------------------------------
                # Check if the max overlap ground truth box is already matched
                #---------------------------------------------------------------
                if matched[max_idx]:
                    fps[i] = 1
                    continue

                tps[i] = 1
                matched[max_idx] = True

            #-------------------------------------------------------------------
            # Compute the precision, recall
            #-------------------------------------------------------------------
            fps = np.cumsum(fps)
            tps = np.cumsum(tps)
            recall = tps/counts[k]
            prec = tps/(tps+fps)
            ap = 0
            for r_tilde in np.arange(0, 1.1, 0.1):
                prec_rec = prec[recall>=r_tilde]
                if len(prec_rec) > 0:
                    ap += np.amax(prec_rec)

            ap /= 11.
            aps[k] = ap

        return aps

    #---------------------------------------------------------------------------
    def clear(self):
        """
        Clear the current detection cache. Useful for restarting the calculation
        for a new batch of data.
        """
        self.det_params = defaultdict(list)
        self.det_confidence = defaultdict(list)
        self.det_sample_ids = defaultdict(list)
        self.gt_samples = []
