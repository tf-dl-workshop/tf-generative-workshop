from utils import *

import os
import tensorflow.contrib.layers as contrib_layers
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


class CDGans(object):
    def __init__(self, z_dim, batch_size, image_dim=28 * 28, image_w_h=28):
        self.z_dim = z_dim
        self.image_dim = image_dim
        self.image_w_h = image_w_h
        self.y_dim = 10  # number of classes
        self.learning_rate = 0.00002
        self.batch_size = batch_size

        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.image_dim])
        self.y = tf.placeholder(tf.float32, [None, self.y_dim])
        self.z = tf.placeholder(tf.float32, [None, self.z_dim])

        ## Generator
        self.generator_z = self.generator(self.z, self.y)


        ## Discriminator
        D_logit_real = self.discriminator(self.x, self.y, False)
        D_logit_fake = self.discriminator(self.generator_z, self.y, True)

        ## loss
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_real), logits=D_logit_real))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        self.D_loss = D_loss = D_loss_real + D_loss_fake
        self.G_loss = G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        # Obtain variables relevant to discriminator and geneartor
        D_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        G_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        self.D_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.D_loss,
                                                                                                       var_list=D_var_list)
        self.G_train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.G_loss,
                                                                                                       var_list=G_var_list)

        return

    def generator(self, z, y):
        with tf.variable_scope("generator"):
            hy = tf.layers.dense(y, 512, tf.nn.relu)
            hz = tf.layers.dense(z, 128, tf.nn.relu)


            hyz = tf.concat([hy, hz], axis=1)
            g_prob = tf.layers.dense(hyz, self.image_dim, activation=tf.nn.sigmoid,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())


        return g_prob

    def discriminator(self, x, y, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            # mnist data's shape is (28 , 28 , 1)
            hx = tf.layers.dense(x, 256, tf.nn.relu)
            hy = tf.layers.dense(x, 128, tf.nn.relu)

            # concat
            hxy = tf.concat([hx, hy], 1)

            d_logit = tf.layers.dense(hxy, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            d_prob = tf.nn.sigmoid(d_logit)

        return d_prob


def sample_label(batch_size):
    label_vector = np.zeros((batch_size, 10), dtype=np.float)
    class_label = [0, 1, 5, 8] * 32
    for i in range(0, batch_size):
        label_vector[i, class_label[i]] = 1.0
    return label_vector


def main():
    batch_size = 128
    z_dim = 100

    f_mnist = input_data.read_data_sets('../data/fashion_mnist', one_hot=True)

    cdgan = CDGans(z_dim, batch_size)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0

    for it in range(1000000):
        if it % 1000 == 0:
            samples = sess.run(cdgan.generator_z,
                               feed_dict={cdgan.z: sample_z_uniform(batch_size, z_dim),
                                          cdgan.y: sample_label(batch_size)})

            fig = plot(samples[:16])
            plt.savefig('./out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_mb, Y_mb = f_mnist.train.next_batch(batch_size)

        _, D_loss_curr = sess.run([cdgan.D_train_op, cdgan.D_loss],
                                  feed_dict={cdgan.x: X_mb,
                                             cdgan.y: Y_mb,
                                             cdgan.z: sample_z_uniform(batch_size, z_dim)})
        _, G_loss_curr = sess.run([cdgan.G_train_op, cdgan.G_loss],
                                  feed_dict={cdgan.y: Y_mb,
                                             cdgan.z: sample_z_uniform(batch_size, z_dim)})

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()

        # save_path = saver.save(sess, model_path)


if __name__ == "__main__":
    main()
