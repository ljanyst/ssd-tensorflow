#!/usr/bin/env python
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

import argparse
import pickle
import sys
import cv2
import os

import numpy as np

from ssdutils import get_preset_by_name
from utils import load_data_source, str2bool, draw_box
from tqdm import tqdm

#-------------------------------------------------------------------------------
def annotate(data_dir, samples, colors, sample_name):
    """
    Draw the bounding boxes on the sample images
    :param data_dir: the directory where the dataset's files are stored
    :param samples:  samples to be processed
    :param colors:   a dictionary mapping class name to a BGR color tuple
    :param colors:   name of the sample
    """
    result_dir = data_dir+'/annotated/'+sample_name.strip()+'/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for sample in tqdm(samples, desc=sample_name, unit='samples'):
        img    = cv2.imread(sample.filename)
        basefn = os.path.basename(sample.filename)
        for box in sample.boxes:
            draw_box(img, box, colors[box.label])
        cv2.imwrite(result_dir+basefn, img)

#-------------------------------------------------------------------------------
def main():
    #---------------------------------------------------------------------------
    # Parse the commandline
    #---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Process a dataset for SSD')
    parser.add_argument('--data-source', default='pascal_voc',
                        help='data source')
    parser.add_argument('--data-dir', default='pascal-voc',
                        help='data directory')
    parser.add_argument('--validation-fraction', type=float, default=0.025,
                        help='fraction of the data to be used for validation')
    parser.add_argument('--annotate', type=str2bool, default='False',
                        help="Annotate the data samples")
    parser.add_argument('--compute-td', type=str2bool, default='True',
                        help="Compute training data")
    parser.add_argument('--preset', default='vgg300',
                        choices=['vgg300', 'vgg512'],
                        help="The neural network preset")
    args = parser.parse_args()

    print('[i] Data source:          ', args.data_source)
    print('[i] Data directory:       ', args.data_dir)
    print('[i] Validation fraction:  ', args.validation_fraction)
    print('[i] Annotate:             ', args.annotate)
    print('[i] Compute training data:', args.compute_td)
    print('[i] Preset:               ', args.preset)

    #---------------------------------------------------------------------------
    # Load the data source
    #---------------------------------------------------------------------------
    print('[i] Configuring the data source...')
    try:
        source = load_data_source(args.data_source)
        source.load_trainval_data(args.data_dir, args.validation_fraction)
        source.load_test_data(args.data_dir)
        print('[i] # training samples:   ', source.num_train)
        print('[i] # validation samples: ', source.num_valid)
        print('[i] # testing samples:    ', source.num_test)
        print('[i] # classes:            ', source.num_classes)
    except (ImportError, AttributeError, RuntimeError) as e:
        print('[!] Unable to load data source:', str(e))
        return 1

    #---------------------------------------------------------------------------
    # Annotate samples
    #---------------------------------------------------------------------------
    if args.annotate:
        print('[i] Annotating samples...')
        annotate(args.data_dir, source.train_samples, source.colors, 'train')
        annotate(args.data_dir, source.valid_samples, source.colors, 'valid')
        annotate(args.data_dir, source.test_samples,  source.colors, 'test ')

    #---------------------------------------------------------------------------
    # Compute the training data
    #---------------------------------------------------------------------------
    if args.compute_td:
        preset = get_preset_by_name(args.preset)
        with open(args.data_dir+'/train-samples.pkl', 'wb') as f:
            pickle.dump(source.train_samples, f)
        with open(args.data_dir+'/valid-samples.pkl', 'wb') as f:
            pickle.dump(source.valid_samples, f)

        with open(args.data_dir+'/training-data.pkl', 'wb') as f:
            data = {
                'preset':      preset,
                'num-classes': source.num_classes,
                'colors':      source.colors,
                'lid2name':    source.lid2name,
                'lname2id':    source.lname2id
            }
            pickle.dump(data, f)

    return 0

if __name__ == '__main__':
    sys.exit(main())
