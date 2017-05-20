import tensorflow as tf
import numpy as np
import BatchLoader
import Model
import utils
from matplotlib import pyplot as plt

if __name__ == '__main__':

    loader = BatchLoader.BatchLoader()
    f_name = loader.mask_path + "2011_003145.png"
    im = plt.imread(f_name)
    utils.visualize_label(im)

    """
    with tf.Session() as sess:

        net = Model.Model()

        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for i in range(100):
            print "on batch: " + str(i + 1)
            train_step = net.get_train_step()
            loss = net.get_loss()
            _, loss_val = sess.run([train_step, loss])

            print "current loss: " + str(loss_val)
            print ""

        coord.request_stop()
        coord.join(threads)
    """