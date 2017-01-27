import tensorflow as tf
from scipy import misc
from matplotlib import pyplot as plt
import os
import numpy as np
import functools
from skimage.io import imread, imsave


def one_use(func):
    """ tf graph construction functions that should only
        be called once will find this decorator useful """
    attribute = "_cache_" + func.__name__

    @property
    @functools.wraps(func)
    def decorated(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return decorated


def generate_labels(f_names, out_path):
    """ f_names should be a list of input image names:
        for each png mask in PASCAL_VOC 2007 dataset, generate
        a grayscale png where each pixel has label val from [0, 23) """
    fill_val = 255
    arr = np.full(3000, fill_val, dtype=np.uint16)
    max_ind = np.int16(0)

    for k in range(len(f_names)):
        mask_test = 255.0 * plt.imread(f_names[k]) / 16.0
        transformed = np.zeros([len(mask_test), len(mask_test[0])], dtype=np.uint16)
        for i in range(len(mask_test)):
            for j in range(len(mask_test[0])):
                pixel = mask_test[i][j]
                id = np.int16(pixel[0] * 100 + pixel[1] * 10 + pixel[2])
                if arr[id] == fill_val:
                    arr[id] = max_ind
                    max_ind += 1
                transformed[i][j] = arr[id]

        imsave("../VOCdevkit/VOC2007/SegmentationClassAlt/" + f_names[-10, -4] + ".png", transformed)


class BatchLoader:

    def __init__(self):
        # images will be warped to these dimensions for inference / training
        self.targ_im_h = 100
        self.targ_im_w = 100
        self.targ_mask_h = 50
        self.targ_mask_w = 50
        self.mask_path = "../VOCdevkit/VOC2007/SegmentationClassAlt/"
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
        mask = tf.image.decode_png(mask_raw, channels=1)
        im = tf.image.decode_jpeg(im_raw, channels=3)

        # resize to some target width / height
        self.mask = tf.image.resize_images(mask, [self.targ_mask_h, self.targ_mask_w])
        self.mask = tf.to_int32(self.mask)
        self.mask = tf.squeeze(self.mask, axis=2)
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
