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
import cv2

import numpy as np

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

    #---------------------------------------------------------------------------
    def __batch_generator(self, sample_list):
        image_size = (self.preset.image_size.w, self.preset.image_size.h)
        sample_list = list(enumerate(sample_list))
        def gen_batch(batch_size):
            random.shuffle(sample_list)
            for offset in range(0, len(sample_list), batch_size):
                samples = sample_list[offset:offset+batch_size]
                images = []
                labels = []
                sample_ids = []

                for s in samples:
                    sample_id  = s[0]
                    image_file = s[1][0].filename
                    label_file = s[1][1]

                    image = cv2.resize(cv2.imread(image_file), image_size)
                    label = np.load(label_file)

                    images.append(image.astype(np.float32))
                    labels.append(label)

                    sample_ids.append(sample_id)

                yield np.array(images), np.array(labels), sample_ids

        return gen_batch
