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
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

import argparse
import sys
import cv2
import os

from utils import load_data_source, str2bool, prop2abs
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
            xmin, xmax, ymin, ymax = prop2abs(box.center, box.size,
                                              sample.imgsize)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                          colors[box.label], 2)
            cv2.rectangle(img, (xmin-1, ymin), (xmax, ymin-20),
                          colors[box.label], cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, box.label, (xmin+5, ymin-5), font, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(result_dir+basefn, img)

#-------------------------------------------------------------------------------
def main():
    #---------------------------------------------------------------------------
    # Parse the commandline
    #---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Train the SSD')
    parser.add_argument('--data-source', default='pascal_voc_2007',
                        help='data source')
    parser.add_argument('--data-dir', default='pascal-voc-2007',
                        help='data directory')
    parser.add_argument('--annotate', type=str2bool, default='False',
                        help="Annotate the date samples")
    args = parser.parse_args()

    print('[i] Data source:          ', args.data_source)
    print('[i] Data directory:       ', args.data_dir)
    print('[i] Annotate:             ', args.annotate)

    #---------------------------------------------------------------------------
    # Load the data source
    #---------------------------------------------------------------------------
    print('[i] Configuring data source...')
    try:
        source = load_data_source(args.data_source)
        source.load_raw_data(args.data_dir, 0.1)
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

    return 0

if __name__ == '__main__':
    sys.exit(main())
