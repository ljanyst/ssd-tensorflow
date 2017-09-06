#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   30.08.2017
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
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

import lxml.etree
import random
import math
import cv2
import os

import numpy as np

from utils import Label, Box, Sample, Size
from utils import rgb2bgr, abs2prop
from glob import glob
from tqdm import tqdm

#-------------------------------------------------------------------------------
# Labels
#-------------------------------------------------------------------------------
label_defs = [
    Label('aeroplane',   rgb2bgr((0,     0,   0))),
    Label('bicycle',     rgb2bgr((111,  74,   0))),
    Label('bird',        rgb2bgr(( 81,   0,  81))),
    Label('boat',        rgb2bgr((128,  64, 128))),
    Label('bottle',      rgb2bgr((244,  35, 232))),
    Label('bus',         rgb2bgr((230, 150, 140))),
    Label('car',         rgb2bgr(( 70,  70,  70))),
    Label('cat',         rgb2bgr((102, 102, 156))),
    Label('chair',       rgb2bgr((190, 153, 153))),
    Label('cow',         rgb2bgr((150, 120,  90))),
    Label('diningtable', rgb2bgr((153, 153, 153))),
    Label('dog',         rgb2bgr((250, 170,  30))),
    Label('horse',       rgb2bgr((220, 220,   0))),
    Label('motorbike',   rgb2bgr((107, 142,  35))),
    Label('person',      rgb2bgr((152, 251, 152))),
    Label('pottedplant', rgb2bgr(( 70, 130, 180))),
    Label('sheep',       rgb2bgr((220,  20,  60))),
    Label('sofa',        rgb2bgr((  0,   0, 142))),
    Label('train',       rgb2bgr((  0,   0, 230))),
    Label('tvmonitor',   rgb2bgr((119,  11,  32)))]

#-------------------------------------------------------------------------------
class PascalVOC2007Source:
    #---------------------------------------------------------------------------
    def __init__(self, vocid='VOC2007'):
        self.num_classes   = len(label_defs)
        self.colors        = {l.name: l.color for l in label_defs}
        self.lid2name      = {i: l.name for i, l in enumerate(label_defs)}
        self.lname2id      = {l.name: i for i, l in enumerate(label_defs)}
        self.num_train     = 0
        self.num_valid     = 0
        self.num_test      = 0
        self.train_samples = []
        self.valid_samples = []
        self.test_samples  = []
        self.vocid         = vocid

    #---------------------------------------------------------------------------
    def __build_sample_list(self, root, dataset_name):
        """
        Build a list of samples for the VOC dataset (either trainval or test)
        """
        image_root  = root + '/JPEGImages/'
        annot_root  = root + '/Annotations/'
        annot_files = glob(annot_root + '/*xml')
        samples     = []

        #-----------------------------------------------------------------------
        # Process each annotated sample
        #-----------------------------------------------------------------------
        for fn in tqdm(annot_files, desc=dataset_name, unit='samples'):
            with open(fn, 'r') as f:
                doc      = lxml.etree.parse(f)
                filename = image_root+doc.xpath('/annotation/filename')[0].text

                #---------------------------------------------------------------
                # Get the file dimensions
                #---------------------------------------------------------------
                if not os.path.exists(filename):
                    continue

                img     = cv2.imread(filename)
                imgsize = Size(img.shape[1], img.shape[0])

                #---------------------------------------------------------------
                # Get boxes for all the objects
                #---------------------------------------------------------------
                boxes    = []
                objects  = doc.xpath('/annotation/object')
                for obj in objects:
                    #-----------------------------------------------------------
                    # Skip dificult detections if we have them labelled
                    #-----------------------------------------------------------
                    difficult = obj.xpath('difficult')
                    if difficult:
                        difficult = int(difficult[0].text)
                        if difficult:
                            continue

                    #-----------------------------------------------------------
                    # Get the properties of the box and convert them to the
                    # proportional terms
                    #-----------------------------------------------------------
                    label = obj.xpath('name')[0].text
                    xmin  = int(float(obj.xpath('bndbox/xmin')[0].text))
                    xmax  = int(float(obj.xpath('bndbox/xmax')[0].text))
                    ymin  = int(float(obj.xpath('bndbox/ymin')[0].text))
                    ymax  = int(float(obj.xpath('bndbox/ymax')[0].text))
                    center, size = abs2prop(xmin, xmax, ymin, ymax, imgsize)
                    box = Box(label, self.lname2id[label], center, size)
                    boxes.append(box)
                if not boxes:
                    continue
                sample = Sample(filename, boxes, imgsize)
                samples.append(sample)

        return samples

    #---------------------------------------------------------------------------
    def load_raw_data(self, data_dir, valid_fraction):
        """
        Load the raw data
        :param data_dir:       the directory where the dataset's file are stored
        :param valid_fraction: what franction of the dataset should be used
                               as a validation sample
        """
        trainval_root = data_dir + '/trainval/VOCdevkit/'+self.vocid
        test_root     = data_dir + '/test/VOCdevkit/'+self.vocid

        trainval_samples   = self.__build_sample_list(trainval_root, 'trainval')
        random.shuffle(trainval_samples)
        tvlen              = len(trainval_samples)
        self.test_samples  = self.__build_sample_list(test_root, 'test    ')
        self.valid_samples = trainval_samples[:int(valid_fraction*tvlen)]
        self.train_samples = trainval_samples[int(valid_fraction*tvlen):]

        if len(self.train_samples) == 0:
            raise RuntimeError('No training samples found in ' + data_dir)
        if len(self.valid_samples) == 0:
            raise RuntimeError('No validation samples found in ' + data_dir)
        if len(self.test_samples) == 0:
            raise RuntimeError('No testing samples found in ' + data_dir)

        self.num_train = len(self.train_samples)
        self.num_valid = len(self.valid_samples)
        self.num_test  = len(self.test_samples)

#-------------------------------------------------------------------------------
def get_source():
    return PascalVOC2007Source()
