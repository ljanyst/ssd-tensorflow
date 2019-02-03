#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date: 27.09.2017
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
import sys
import os

import tensorflow as tf

from tensorflow.python.framework import graph_util

if sys.version_info[0] < 3:
    print("This is a Python 3 program. Use Python 3 or higher.")
    sys.exit(1)

#---------------------------------------------------------------------------
# Parse the commandline
#---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Export a tensorflow model')
parser.add_argument('--metagraph-file', default='final.ckpt.meta',
                    help='name of the metagraph file')
parser.add_argument('--checkpoint-file', default='final.ckpt',
                    help='name of the checkpoint file')
parser.add_argument('--output-file', default='model.pb',
                    help='name of the output file')
parser.add_argument('--output-tensors', nargs='+',
                    required=True,
                    help='names of the output tensors')
args = parser.parse_args()

print('[i] Matagraph file:  ', args.metagraph_file)
print('[i] Checkpoint file: ', args.checkpoint_file)
print('[i] Output file:     ', args.output_file)
print('[i] Output tensors:  ', args.output_tensors)

for f in [args.checkpoint_file+'.index', args.metagraph_file]:
    if not os.path.exists(f):
        print('[!] Cannot find file:', f)
        sys.exit(1)

#-------------------------------------------------------------------------------
# Export the graph
#-------------------------------------------------------------------------------
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(args.metagraph_file)
    saver.restore(sess, args.checkpoint_file)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, args.output_tensors)

    with open(args.output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())
