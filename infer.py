#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   09.09.2017
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
import math
import sys
import cv2
import os

import tensorflow as tf
import numpy as np

from ssdutils import get_anchors_for_preset, decode_boxes, suppress_overlaps
from ssdvgg import SSDVGG
from utils import str2bool, load_data_source, draw_box
from tqdm import tqdm

#-------------------------------------------------------------------------------
def sample_generator(samples, image_size, batch_size):
    image_size = (image_size.w, image_size.h)
    for offset in range(0, len(samples), batch_size):
        files = samples[offset:offset+batch_size]
        images = []
        idxs   = []
        for i in range(len(files)):
            image_file = files[i]
            image = cv2.resize(cv2.imread(image_file), image_size)
            images.append(image.astype(np.float32))
            idxs.append(offset+i)
        yield np.array(images), idxs

#-------------------------------------------------------------------------------
def main():
    #---------------------------------------------------------------------------
    # Parse commandline
    #---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='SSD inference')
    parser.add_argument("files", nargs="*")
    parser.add_argument('--name', default='test',
                        help='project name')
    parser.add_argument('--checkpoint', type=int, default=-1,
                        help='checkpoint to restore; -1 is the most recent')
    parser.add_argument('--training-data',
                        default='pascal-voc-2007/training-data.pkl',
                        help='Information about parameters used for training')
    parser.add_argument('--output-dir', default='test-output',
                        help='directory for the resulting images')
    parser.add_argument('--annotate', type=str2bool, default='False',
                        help="Annotate the data samples")
    parser.add_argument('--dump-prediction', type=str2bool, default='False',
                        help="Dump raw predictions")
    parser.add_argument('--data-source', default=None,
                        help='Use test files from the data source')
    parser.add_argument('--data-dir', default='pascal-voc-2007',
                        help='Use test files from the data source')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    args = parser.parse_args()

    #---------------------------------------------------------------------------
    # Check if we can get the checkpoint
    #---------------------------------------------------------------------------
    state = tf.train.get_checkpoint_state(args.name)
    if state is None:
        print('[!] No network state found in ' + args.name)
        return 1

    try:
        checkpoint_file = state.all_model_checkpoint_paths[args.checkpoint]
    except IndexError:
        print('[!] Cannot find checkpoint ' + str(args.checkpoint_file))
        return 1

    metagraph_file = checkpoint_file + '.meta'

    if not os.path.exists(metagraph_file):
        print('[!] Cannot find metagraph ' + metagraph_file)
        return 1

    #---------------------------------------------------------------------------
    # Load the training data
    #---------------------------------------------------------------------------
    try:
        with open(args.training_data, 'rb') as f:
            data = pickle.load(f)
        preset = data['preset']
        colors = data['colors']
        lid2name = data['lid2name']
        image_size = preset.image_size
        anchors = get_anchors_for_preset(preset)
    except (FileNotFoundError, IOError, KeyError) as e:
        print('[!] Unable to load training data:', str(e))
        return 1

    #---------------------------------------------------------------------------
    # Load the data source if defined
    #---------------------------------------------------------------------------
    source = None
    if args.data_source:
        print('[i] Configuring the data source...')
        try:
            source = load_data_source(args.data_source)
            source.load_test_data(args.data_dir)
            print('[i] # testing samples:    ', source.num_test)
            print('[i] # classes:            ', source.num_classes)
        except (ImportError, AttributeError, RuntimeError) as e:
            print('[!] Unable to load data source:', str(e))
            return 1

    #---------------------------------------------------------------------------
    # Create a list of files to analyse and make sure that the output directory
    # exists
    #---------------------------------------------------------------------------
    files = []

    if source:
        for sample in source.test_samples:
            files.append(sample.filename)

    if args.annotate and not source:
        if args.files:
            files = args.files

        if not files:
            print('[!] No files specified')
            return 1

    files = list(filter(lambda x: os.path.exists(x), files))
    if files:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    #---------------------------------------------------------------------------
    # Print parameters
    #---------------------------------------------------------------------------
    print('[i] Project name:      ', args.name)
    print('[i] Network checkpoint:', checkpoint_file)
    print('[i] Metagraph file:    ', metagraph_file)
    print('[i] Training data:     ', args.training_data)
    print('[i] Image size:        ', image_size)
    print('[i] Number of files:   ', len(files))
    print('[i] Batch size:        ', args.batch_size)
    print('[i] Data source:       ', args.data_source)
    print('[i] Data directory:    ', args.data_dir)

    #---------------------------------------------------------------------------
    # Create the network
    #---------------------------------------------------------------------------
    with tf.Session() as sess:
        print('[i] Creating the model...')
        net = SSDVGG(sess)
        net.build_from_metagraph(metagraph_file, checkpoint_file)

        #-----------------------------------------------------------------------
        # Process the images
        #-----------------------------------------------------------------------
        generator = sample_generator(files, image_size, args.batch_size)
        n_sample_batches = int(math.ceil(len(files)/args.batch_size))
        description = '[i] Processing samples'

        for x, idxs in tqdm(generator, total=n_sample_batches,
                      desc=description, unit='batches'):
            feed = {net.image_input:  x,
                    net.keep_prob:    1}
            enc_boxes = sess.run(net.result, feed_dict=feed)

            #-------------------------------------------------------------------
            # Annotate the samples
            #-------------------------------------------------------------------
            if args.annotate:
                for i in range(enc_boxes.shape[0]):
                    boxes = decode_boxes(enc_boxes[i], anchors, 0.99, lid2name)
                    boxes = suppress_overlaps(boxes)
                    filename = files[idxs[i]]
                    img = cv2.imread(filename)
                    for box in boxes:
                        draw_box(img, box[1], colors[box[1].label])
                    fn = args.output_dir+'/'+os.path.basename(filename)
                    cv2.imwrite(fn, img)

                #---------------------------------------------------------------
                # Dump the predictions
                #---------------------------------------------------------------
                if args.dump_prediction:
                    for i in range(enc_boxes.shape[0]):
                        filename = files[idxs[i]]
                        base_name = os.path.basename(filename)
                        fn = args.output_dir+'/'+base_name+'.npy'
                        np.save(fn, enc_boxes[i])

    print('[i] All done.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
