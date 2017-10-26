import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from utils import *


class GANs(object):
    def __init__(self, z_dim, image_dim=28 * 28):
        self.z_dim = z_dim
        self.image_dim = image_dim
        self.generator_dim = 256
        self.discriminator_dim = 128
        self.learning_rate = 1e-5
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.build_model()

    def build_model(self):
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.x = tf.placeholder(tf.float32, shape=[None, self.image_dim])

        ## Generator
        with tf.variable_scope("generator"):
            g1 = tf.layers.dense(self.z, self.generator_dim, activation=tf.nn.relu,
                                 kernel_initializer=self.initializer)
            self.generator_z = tf.layers.dense(g1, self.image_dim, activation=tf.nn.sigmoid,
                                               kernel_initializer=self.initializer)

        ## Discriminator
        def discriminator(x, reuse=False):
            with tf.variable_scope("discriminator", reuse=reuse):
                d1 = tf.layers.dense(x, self.generator_dim, activation=tf.nn.relu,
                                     kernel_initializer=self.initializer)
                d1_dropout = tf.layers.dropout(d1, rate=0.2)
                d_logit = tf.layers.dense(d1_dropout, 1, activation=None, kernel_initializer=self.initializer)
                return d_logit

        D_logit_real = discriminator(self.x)
        D_logit_fake = discriminator(self.generator_z, reuse=True)

        ## loss
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        self.D_loss = D_loss = D_loss_real + D_loss_fake
        self.G_loss = G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        D_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        G_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=D_var_list)
        self.G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=G_var_list)

        return


def main():
    batch_size = 128
    Z_dim = 128

    f_mnist = input_data.read_data_sets('../data/fashion_mnist', one_hot=True)

    model = GANs(Z_dim)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0
    for it in range(1000000):
        if it % 1000 == 0:
            samples = sess.run(model.generator_z, feed_dict={model.z: sample_z(16, Z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_mb, _ = f_mnist.train.next_batch(batch_size)

        _, D_loss_curr = sess.run([model.D_solver, model.D_loss],
                                  feed_dict={model.x: X_mb, model.z: sample_z(batch_size, Z_dim)})
        _, G_loss_curr = sess.run([model.G_solver, model.G_loss], feed_dict={model.z: sample_z(batch_size, Z_dim)})

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()


if __name__ == "__main__":
    main()
