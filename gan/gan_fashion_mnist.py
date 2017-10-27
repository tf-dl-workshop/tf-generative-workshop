import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from utils import *


class GANs(object):
    def __init__(self, z_dim, image_dim=28 * 28):
        self.z_dim = z_dim
        self.image_dim = image_dim
        self.learning_rate = 1e-4
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.build_model()

    def build_model(self):
        # Noise input Z
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        # Input Image
        self.x = tf.placeholder(tf.float32, shape=[None, self.image_dim])

        # Generator
        # Take Z as an input and produce fake sample (g_sample)
        def geneartor(z, reuse=False):
            with tf.variable_scope("generator", reuse=reuse):
                g1 = tf.layers.dense(z, 128, activation=tf.nn.relu,
                                     kernel_initializer=self.initializer)
                g_prob = tf.layers.dense(g1, self.image_dim, activation=tf.nn.sigmoid,
                                         kernel_initializer=self.initializer)

                return g_prob

        # Discriminator
        # A classifier returning probability of being a real example
        def discriminator(x, reuse=False):
            with tf.variable_scope("discriminator", reuse=reuse):
                d1 = tf.layers.dense(x, 128, activation=tf.nn.relu,
                                     kernel_initializer=self.initializer)
                d_logit = tf.layers.dense(d1, 1, activation=None, kernel_initializer=self.initializer)
                return d_logit

        # logit output from discriminator for real example
        D_logit_real = discriminator(self.x)

        # generate fake sample from generator
        self.g_sample = geneartor(self.z)
        # logit output from discriminator for fake example
        D_logit_fake = discriminator(self.g_sample, reuse=True)

        # Defining losses
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        # Discriminator Loss
        self.D_loss = D_loss_real + D_loss_fake

        # Generator Loss
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        # Obtain variables relevant to discriminator and geneartor
        D_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        G_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        self.D_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.D_loss, var_list=D_var_list)
        self.G_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.G_loss, var_list=G_var_list)


def main():
    batch_size = 128
    z_dim = 128

    f_mnist = input_data.read_data_sets('../data/fashion_mnist', one_hot=True)

    gan = GANs(z_dim)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0
    for it in range(100000):
        if it % 1000 == 0:
            samples = sess.run(gan.g_sample, feed_dict={gan.z: sample_z_uniform(16, z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        x_image, _ = f_mnist.train.next_batch(batch_size)

        _, D_loss_curr = sess.run([gan.D_train_op, gan.D_loss],
                                  feed_dict={gan.x: x_image, gan.z: sample_z_uniform(batch_size, z_dim)})
        _, G_loss_curr = sess.run([gan.G_train_op, gan.G_loss], feed_dict={gan.z: sample_z_uniform(batch_size, z_dim)})

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()


if __name__ == "__main__":
    main()
