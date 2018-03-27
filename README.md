
SSD-TensorFlow
==============

Overview
--------

The programs in this repository train and use a Single Shot MultiBox Detector
to take an image and draw bounding boxes around objects of certain classes
contained in this image. The network is based on the VGG-16 model and uses
the approach described in [this paper][1] by Wei Liu et al. The software is
generic and easily extendable to any dataset, although I only tried it with
[Pascal VOC][2] so far. All you need to do to introduce a new dataset is to
create a new `source_xxxxxx.py` file defining it.

Go [here][4] for more info.

Pascal VOC Results
------------------

Images and numbers speak louder than a thousand words, so here they are:

![Example #1][img1]
![Example #2][img2]

| Model  | Training data                    | mAP Train | mAP VOC12 test |
|:------:|:--------------------------------:|:---------:|:--------------:|
| vgg300 | VOC07+12 trainval and VOC07 Test |     79.9% |    [71.8%][3]  |
| vgg512 | VOC07+12 trainval and VOC07 Test |     84.0% |    [75.0%][5]  |

Usage
-----

To train the model on the Pascal VOC data, go to the `pascal-voc` directory
and download the dataset:

    cd pascal-voc
    ./download-data.sh
    cd ..

You then need to preprocess the dataset before you can train the model on it.
It's OK to use the default settings, but if you want something more, it's always
good to try the `--help` parameter.

    ./process_dataset.py

You can then train the whole thing. It will take around 150 to 200 epochs to get
good results. Again, you can try `--help` if you want to do something custom.

    ./train.py

You can annotate images, dump raw predictions, print the AP stats, or export the
results in the Pascal VOC compatible format using the inference script.

    ./infer.py --help

To export the model to an inference optimize graph run (use `result/result`
as the name of the output tensor):

    ./export_model.py

If you want to make detection basing on the inference model, check out:

    ./detect.py


Have Fun!

[1]: https://arxiv.org/pdf/1512.02325.pdf
[2]: http://host.robots.ox.ac.uk/pascal/VOC/
[3]: http://host.robots.ox.ac.uk:8080/anonymous/C3DN60.html
[4]: http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
[5]: http://host.robots.ox.ac.uk:8080/anonymous/CNQPDK.html

[img1]: assets/000232.jpg
[img2]: assets/000032.jpg
