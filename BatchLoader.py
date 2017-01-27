import tensorflow as tf
from scipy import misc
from matplotlib import pyplot as plt
import os
import numpy as np
import functools


def one_use(func):
    """ tf graph construction functions that should only
    be called once will find this decorator useful"""
    attribute = "_cache_" + func.__name__

    @property
    @functools.wraps(func)
    def decorated(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return decorated


class BatchLoader:

    def __init__(self):
        # images will be warped to these dimensions for inference / training
        self.targ_im_h = 100
        self.targ_im_w = 100
        self.targ_mask_h = 50
        self.targ_mask_w = 50
        self.mask_path = "../VOCdevkit/VOC2007/SegmentationClass/"
        self.im_path = "../VOCdevkit/VOC2007/JPEGImages/"
        self.im, self.mask = None, None
        self.init_queues

    @one_use
    def init_queues(self):
        print "initiating file name queues ..."
        names = os.listdir(self.mask_path)
        names = [name[:-4] for name in names if name.endswith(".png")]
        mask_names = [self.mask_path + name + ".png" for name in names]
        im_names = [self.im_path + name + ".jpg" for name in names]

        mask_queue = tf.train.string_input_producer(mask_names, shuffle=False)
        im_queue = tf.train.string_input_producer(im_names, shuffle=False)

        # read single image/mask into tensors :
        # note that it is important to have 2 file readers here (otherwise weird deadlocks occur)!
        reader1 = tf.WholeFileReader()
        reader2 = tf.WholeFileReader()
        mask_key, mask_raw = reader1.read(mask_queue)
        im_key, im_raw = reader2.read(im_queue)
        mask = tf.image.decode_png(mask_raw, channels=3)
        im = tf.image.decode_jpeg(im_raw, channels=3)

        # resize to some target width / height
        self.mask = tf.image.resize_images(mask, [self.targ_mask_h, self.targ_mask_w])
        self.im = tf.image.resize_images(im, [self.targ_im_h, self.targ_im_w])

    def get_batch(self, batch_size=30):
        return tf.train.shuffle_batch([self.im, self.mask], batch_size, 1000, 100)

    """
    print "starting session ..."
    with tf.Session() as sess:

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # visualize results
        #summary_writer = tf.summary.FileWriter('./tmp/logs7', sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)

        #summary = sess.run(summary_op)
        #summary_writer.add_summary(summary)

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)
    """
