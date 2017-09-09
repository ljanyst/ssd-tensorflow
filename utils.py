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
