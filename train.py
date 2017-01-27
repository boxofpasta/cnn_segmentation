import tensorflow as tf
import numpy as np
import BatchLoader
import Model


if __name__ == '__main__':

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