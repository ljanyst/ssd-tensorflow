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
import cv2

import tensorflow as tf
import numpy as np

from collections import namedtuple

#-------------------------------------------------------------------------------
def initialize_uninitialized_variables(sess):
    """
    Only initialize the weights that have not yet been initialized by other
    means, such as importing a metagraph and a checkpoint. It's useful when
    extending an existing model.
    """
    uninit_vars    = []
    uninit_tensors = []
    for var in tf.global_variables():
        uninit_vars.append(var)
        uninit_tensors.append(tf.is_variable_initialized(var))
    uninit_bools = sess.run(uninit_tensors)
    uninit = zip(uninit_bools, uninit_vars)
    uninit = [var for init, var in uninit if not init]
    sess.run(tf.variables_initializer(uninit))

#-------------------------------------------------------------------------------
def load_data_source(data_source):
    """
    Load a data source given it's name
    """
    source_module = __import__('source_'+data_source)
    get_source    = getattr(source_module, 'get_source')
    return get_source()

#-------------------------------------------------------------------------------
def rgb2bgr(tpl):
    """
    Convert RGB color tuple to BGR
    """
    return (tpl[2], tpl[1], tpl[0])

#-------------------------------------------------------------------------------
Label   = namedtuple('Label',   ['name', 'color'])
Size    = namedtuple('Size',    ['w', 'h'])
Point   = namedtuple('Point',   ['x', 'y'])
Sample  = namedtuple('Sample',  ['filename', 'boxes', 'imgsize'])
Box     = namedtuple('Box',     ['label', 'labelid', 'center', 'size'])
Score   = namedtuple('Score',   ['idx', 'score'])
Overlap = namedtuple('Overlap', ['best', 'good'])

#-------------------------------------------------------------------------------
def str2bool(v):
    """
    Convert a string to a boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#-------------------------------------------------------------------------------
def abs2prop(xmin, xmax, ymin, ymax, imgsize):
    """
    Convert the absolute min-max box bound to proportional center-width bounds
    """
    width   = float(xmax-xmin)
    height  = float(ymax-ymin)
    cx      = float(xmin)+width/2
    cy      = float(ymin)+height/2
    width  /= imgsize.w
    height /= imgsize.h
    cx     /= imgsize.w
    cy     /= imgsize.h
    return Point(cx, cy), Size(width, height)

#-------------------------------------------------------------------------------
def prop2abs(center, size, imgsize):
    """
    Convert proportional center-width bounds to absolute min-max bounds
    """
    width2  = size.w*imgsize.w/2
    height2 = size.h*imgsize.h/2
    cx      = center.x*imgsize.w
    cy      = center.y*imgsize.h
    return int(cx-width2), int(cx+width2), int(cy-height2), int(cy+height2)

#-------------------------------------------------------------------------------
def normalize_box(box):
    img_size = Size(1000, 1000)
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
    xmin = max(xmin, 0)
    xmax = min(xmax, img_size.w-1)
    ymin = max(ymin, 0)
    ymax = min(ymax, img_size.h-1)

    # this happens early in the training when box min and max are outside
    # of the image
    xmin = min(xmin, xmax)
    ymin = min(ymin, ymax)

    center, size = abs2prop(xmin, xmax, ymin, ymax, img_size)
    return Box(box.label, box.labelid, center, size)

#-------------------------------------------------------------------------------
def draw_box(img, box, color):
    img_size = Size(img.shape[1], img.shape[0])
    xmin, xmax, ymin, ymax = prop2abs(box.center, box.size, img_size)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.rectangle(img, (xmin-1, ymin), (xmax+1, ymin-20), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, box.label, (xmin+5, ymin-5), font, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

#-------------------------------------------------------------------------------
class PrecisionSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, labels):
        self.session = session
        self.writer = writer
        self.labels = labels
        self.mAP_placeholder = tf.placeholder(tf.float32)
        self.mAP_summary_op = tf.summary.scalar(sample_name+'_mAP',
                                                self.mAP_placeholder)
        self.placeholders = {}
        self.summary_ops = {}

        for label in labels:
            summary_name = sample_name+'_AP_'+label
            placeholder = tf.placeholder(tf.float32)
            summary_op = tf.summary.scalar(summary_name, placeholder)
            self.placeholders[label] = placeholder
            self.summary_ops[label] = summary_op

    #---------------------------------------------------------------------------
    def push(self, epoch, mAP, APs):
        feed = {self.mAP_placeholder: mAP}
        tensors = [self.mAP_summary_op]
        for label in self.labels:
            feed[self.placeholders[label]] = APs[label]
            tensors.append(self.summary_ops[label])

        summaries = self.session.run(tensors, feed_dict=feed)

        for summary in summaries:
            self.writer.add_summary(summary, epoch)

#-------------------------------------------------------------------------------
class ImageSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, colors):
        self.session = session
        self.writer = writer
        self.colors = colors
        self.img_placeholder = tf.placeholder(tf.float32,
                                              shape=[None, None, None, 3])
        self.img_summary_op = tf.summary.image(sample_name+'_img',
                                               self.img_placeholder)

    #---------------------------------------------------------------------------
    def push(self, epoch, samples):
        imgs = np.zeros((3, 512, 512, 3))
        for i, sample in enumerate(samples):
            img = cv2.imread(sample[0])
            img = cv2.resize(img, (512, 512))
            for _, box in sample[1]:
                draw_box(img, box, self.colors[box.label])
            imgs[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        feed = {self.img_placeholder: imgs}
        summary = self.session.run(self.img_summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)
