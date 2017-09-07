#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   27.08.2017
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

import zipfile
import shutil
import os

import tensorflow as tf

from urllib.request import urlretrieve
from tqdm import tqdm

#-------------------------------------------------------------------------------
class DLProgress(tqdm):
    last_block = 0

    #---------------------------------------------------------------------------
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

#-------------------------------------------------------------------------------
def conv_map(x, size, shape, stride, name, padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable("weights",
                            shape=[shape, shape, x.get_shape()[3], size],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(size), name='biases')
        x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

#-------------------------------------------------------------------------------
def classifier(x, size, mapsize, name):
    with tf.variable_scope(name):
        w = tf.get_variable("weights",
                            shape=[3, 3, x.get_shape()[3], size],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(size), name='biases')
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
    return tf.reshape(x, [-1, mapsize.w*mapsize.h, size])


#-------------------------------------------------------------------------------
class SSDVGG:
    #---------------------------------------------------------------------------
    def __init__(self, session, preset):
        self.session = session
        self.preset  = preset
        self.__built = False

    #---------------------------------------------------------------------------
    def build_from_vgg(self, vgg_dir, num_classes, progress_hook):
        """
        Build the model for training based on a pre-define vgg16 model.
        :param vgg_dir:       directory where the vgg model should be stored
        :param num_classes:   number of classes
        :param progress_hook: a hook to show download progress of vgg16;
                              the value may be a callable for urlretrieve
                              or string "tqdm"
        """
        self.num_classes = num_classes
        self.__download_vgg(vgg_dir, progress_hook)
        self.__load_vgg(vgg_dir)
        self.__build_vgg_mods()
        self.__build_ssd_layers()
        self.__select_feature_maps()
        self.__build_classifiers()
        self.__built = True

    #---------------------------------------------------------------------------
    def __download_vgg(self, vgg_dir, progress_hook):
        #-----------------------------------------------------------------------
        # Check if the model needs to be downloaded
        #-----------------------------------------------------------------------
        vgg_archive = 'vgg.zip'
        vgg_files   = [
            vgg_dir + '/variables/variables.data-00000-of-00001',
            vgg_dir + '/variables/variables.index',
            vgg_dir + '/saved_model.pb']

        missing_vgg_files = [vgg_file for vgg_file in vgg_files \
                             if not os.path.exists(vgg_file)]

        if missing_vgg_files:
            if os.path.exists(vgg_dir):
                shutil.rmtree(vgg_dir)
            os.makedirs(vgg_dir)

            #-------------------------------------------------------------------
            # Download vgg
            #-------------------------------------------------------------------
            url = 'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip'
            if not os.path.exists(vgg_archive):
                if callable(progress_hook):
                    urlretrieve(url, vgg_archive, progress_hook)
                else:
                    with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
                        urlretrieve(url, vgg_archive, pbar.hook)

            #-------------------------------------------------------------------
            # Extract vgg
            #-------------------------------------------------------------------
            zip_archive = zipfile.ZipFile(vgg_archive, 'r')
            zip_archive.extractall(vgg_dir)
            zip_archive.close()

    #---------------------------------------------------------------------------
    def __load_vgg(self, vgg_dir):
        sess = self.session
        graph = tf.saved_model.loader.load(sess, ['vgg16'], vgg_dir+'/vgg')
        self.image_input = sess.graph.get_tensor_by_name('image_input:0')
        self.keep_prob   = sess.graph.get_tensor_by_name('keep_prob:0')
        self.vgg_conv4_3 = sess.graph.get_tensor_by_name('conv4_3/Relu:0')
        self.vgg_conv5_3 = sess.graph.get_tensor_by_name('conv5_3/Relu:0')
        self.vgg_fc6_w   = sess.graph.get_tensor_by_name('fc6/weights:0')
        self.vgg_fc6_b   = sess.graph.get_tensor_by_name('fc6/biases:0')
        self.vgg_fc7_w   = sess.graph.get_tensor_by_name('fc7/weights:0')
        self.vgg_fc7_b   = sess.graph.get_tensor_by_name('fc7/biases:0')

    #---------------------------------------------------------------------------
    def __build_vgg_mods(self):
        self.mod_pool5 = tf.nn.max_pool(self.vgg_conv5_3, ksize=[1, 3, 3, 1],
                                        strides=[1, 1, 1, 1], padding='SAME',
                                        name='mod_pool5')

        with tf.variable_scope('mod_conv6'):
            x = tf.nn.conv2d(self.mod_pool5, self.vgg_fc6_w,
                             strides=[1, 1, 1, 1], padding='SAME')
            x = tf.nn.bias_add(x, self.vgg_fc6_b)
            self.mod_conv6 = tf.nn.relu(x)

        with tf.variable_scope('mod_conv7'):
            x = tf.nn.conv2d(self.mod_conv6, self.vgg_fc7_w,
                             strides=[1, 1, 1, 1], padding='SAME')
            x = tf.nn.bias_add(x, self.vgg_fc7_b)
            self.mod_conv7 = tf.nn.relu(x)

    #---------------------------------------------------------------------------
    def __build_ssd_layers(self):
        self.ssd_conv8_1  = conv_map(self.mod_conv7,    256, 1, 1, 'conv8_1')
        self.ssd_conv8_2  = conv_map(self.ssd_conv8_1,  512, 3, 2, 'conv8_2')
        self.ssd_conv9_1  = conv_map(self.ssd_conv8_2,  128, 1, 1, 'conv9_1')
        self.ssd_conv9_2  = conv_map(self.ssd_conv9_1,  256, 3, 2, 'conv9_2')
        self.ssd_conv10_1 = conv_map(self.ssd_conv9_2,  128, 1, 1, 'conv10_1')
        self.ssd_conv10_2 = conv_map(self.ssd_conv10_1, 256, 3, 1, 'conv10_2', 'VALID')
        self.ssd_conv11_1 = conv_map(self.ssd_conv10_2, 128, 1, 1, 'conv11_1')
        self.ssd_conv11_2 = conv_map(self.ssd_conv11_1, 256, 3, 1, 'conv11_2', 'VALID')

    #---------------------------------------------------------------------------
    def __select_feature_maps(self):
        self.__maps = [
            self.vgg_conv4_3,
            self.mod_conv7,
            self.ssd_conv8_2,
            self.ssd_conv9_2,
            self.ssd_conv10_2,
            self.ssd_conv11_2]

    #---------------------------------------------------------------------------
    def __build_classifiers(self):
        self.__classifiers = []
        size = self.num_classes+5
        for i in range(len(self.__maps)):
            fmap     = self.__maps[i]
            map_size = self.preset.map_sizes[i]
            for j in range(5):
                name    = 'classifier{}_{}'.format(i, j)
                clsfier = classifier(fmap, size, map_size, name)
                self.__classifiers.append(clsfier)
            if i < len(self.__maps)-1:
                name    = 'classifier{}_{}'.format(i, 6)
                clsfier = classifier(fmap, size, map_size, name)
                self.__classifiers.append(clsfier)
        output = tf.concat(self.__classifiers, axis=1)
        self.classifier = output[:,:,:self.num_classes+1]
        self.locator    = output[:,:,self.num_classes+1:]
