#!/bin/bash

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

mkdir -p trainval
mkdir -p test

(cd trainval && tar xf ../VOCtrainval_06-Nov-2007.tar)
(cd test && tar xf ../VOCtest_06-Nov-2007.tar)
