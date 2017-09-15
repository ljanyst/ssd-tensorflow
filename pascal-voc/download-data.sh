#!/bin/bash

wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

mkdir -p trainval
mkdir -p test

(cd trainval && tar xf ../VOCtrainval_06-Nov-2007.tar)
(cd trainval && tar xf ../VOCtrainval_11-May-2012.tar)
(cd test && tar xf ../VOCtest_06-Nov-2007.tar)
