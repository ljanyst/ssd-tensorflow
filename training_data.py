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

import pickle
import random
import math
import cv2
import os

import multiprocessing as mp
import numpy as np
import queue as q

from data_queue import DataQueue
from copy import copy

#-------------------------------------------------------------------------------
class TrainingData:
    #---------------------------------------------------------------------------
    def __init__(self, data_dir):
        #-----------------------------------------------------------------------
        # Read the dataset info
        #-----------------------------------------------------------------------
        try:
            with open(data_dir+'/training-data.pkl', 'rb') as f:
                data = pickle.load(f)
            with open(data_dir+'/train-samples.pkl', 'rb') as f:
                train_samples = pickle.load(f)
            with open(data_dir+'/valid-samples.pkl', 'rb') as f:
                valid_samples = pickle.load(f)
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(str(e))

        #-----------------------------------------------------------------------
        # Set the attributes up
        #-----------------------------------------------------------------------
        self.preset          = data['preset']
        self.num_classes     = data['num-classes']
        self.label_colors    = data['colors']
        self.lid2name        = data['lid2name']
        self.lname2id        = data['lname2id']
        self.train_generator = self.__batch_generator(train_samples)
        self.valid_generator = self.__batch_generator(valid_samples)
        self.num_train       = len(train_samples)
        self.num_valid       = len(valid_samples)
        self.train_samples   = list(map(lambda x: x[0], train_samples))
        self.valid_samples   = list(map(lambda x: x[0], valid_samples))

    #---------------------------------------------------------------------------
    def __batch_generator(self, sample_list_):
        image_size = (self.preset.image_size.w, self.preset.image_size.h)

        #-----------------------------------------------------------------------
        def process_samples(samples):
            images = []
            labels = []
            gt_boxes = []
            for s in samples:
                image_file = s[0].filename
                label_file = s[1]
                boxes = s[0].boxes

                image = cv2.resize(cv2.imread(image_file), image_size)
                label = np.load(label_file)

                images.append(image.astype(np.float32))
                labels.append(label)
                gt_boxes.append(boxes)

            images = np.array(images, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)
            return images, labels, gt_boxes

        #-----------------------------------------------------------------------
        def batch_producer(sample_queue, batch_queue):
            while True:
                #---------------------------------------------------------------
                # Process the sample
                #---------------------------------------------------------------
                try:
                    samples = sample_queue.get(timeout=1)
                except q.Empty:
                    break

                images, labels, gt_boxes = process_samples(samples)

                #---------------------------------------------------------------
                # Pad the result in the case where we don't have enough samples
                # to fill the entire batch
                #---------------------------------------------------------------
                if images.shape[0] < batch_queue.img_shape[0]:
                    images_norm = np.zeros(batch_queue.img_shape,
                                           dtype=np.float32)
                    labels_norm = np.zeros(batch_queue.label_shape,
                                           dtype=np.float32)
                    images_norm[:images.shape[0]] = images
                    labels_norm[:images.shape[0]] = labels
                    batch_queue.put(images_norm, labels_norm, gt_boxes)
                else:
                    batch_queue.put(images, labels, gt_boxes)

        #-----------------------------------------------------------------------
        def gen_batch(batch_size, num_workers=0):
            sample_list = copy(sample_list_)
            random.shuffle(sample_list)

            #-------------------------------------------------------------------
            # Set up the parallel generator
            #-------------------------------------------------------------------
            if num_workers > 0:
                #---------------------------------------------------------------
                # Set up the queues
                #---------------------------------------------------------------
                img_template = np.zeros((batch_size, self.preset.image_size.h,
                                         self.preset.image_size.w, 3),
                                        dtype=np.float32)
                label_template = np.zeros((batch_size, self.preset.num_anchors,
                                           self.num_classes+5),
                                          dtype=np.float32)
                max_size = num_workers*2
                n_batches = int(math.ceil(len(sample_list_)/batch_size))
                sample_queue = mp.Queue(n_batches)
                batch_queue = DataQueue(img_template, label_template, max_size)

                #---------------------------------------------------------------
                # Set up the workers, set the environment variable to make sure
                # that the internal state of tensorflow is not messed up
                #---------------------------------------------------------------
                workers = []
                os.environ['CUDA_VISIBLE_DEVICES'] = ""
                for i in range(num_workers):
                    args = (sample_queue, batch_queue)
                    w = mp.Process(target=batch_producer, args=args)
                    workers.append(w)
                    w.start()
                del os.environ['CUDA_VISIBLE_DEVICES']

                #---------------------------------------------------------------
                # Fill the sample queue with data
                #---------------------------------------------------------------
                for offset in range(0, len(sample_list), batch_size):
                    samples = sample_list[offset:offset+batch_size]
                    sample_queue.put(samples)

                #---------------------------------------------------------------
                # Return the data
                #---------------------------------------------------------------
                for offset in range(0, len(sample_list), batch_size):
                    images, labels, gt_boxes = batch_queue.get()
                    num_items = len(gt_boxes)
                    yield images[:num_items], labels[:num_items], gt_boxes

                #---------------------------------------------------------------
                # Join the workers
                #---------------------------------------------------------------
                for w in workers:
                    w.join()

            #-------------------------------------------------------------------
            # Return a serial generator
            #-------------------------------------------------------------------
            else:
                for offset in range(0, len(sample_list), batch_size):
                    samples = sample_list[offset:offset+batch_size]
                    images, labels, gt_boxes = process_samples(samples)
                    yield images, labels, gt_boxes

        return gen_batch
