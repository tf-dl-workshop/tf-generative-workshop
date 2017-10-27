import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from utils import *


class GANs(object):
    def __init__(self, z_dim, image_dim=28 * 28):
        self.z_dim = z_dim
        self.image_dim = image_dim
        self.batch_size = 128
        self.generator_dim = 512
        self.discriminator_dim = 256
        self.learning_rate = 0.0002
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.build_model()

    def build_model(self):
        # Noise input Z
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])

        # Input Image
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_dim])
        self.image = tf.reshape(self.x, [self.batch_size, 28, 28, 1])

        def discriminator(x, is_training=True, reuse=False):
            # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
            # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
            with tf.variable_scope("discriminator", reuse=reuse):
                net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
                net = lrelu(conv2d(net, 64, 4, 4, 2, 2, name='d_conv2'))
                net = tf.reshape(net, [self.batch_size, -1])
                net = lrelu(linear(net, 128, scope='d_fc3'))
                out_logit = linear(net, 1, scope='d_fc4')

                return out_logit

        def generator(z, is_training=True, reuse=False):
            # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
            # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
            with tf.variable_scope("generator", reuse=reuse):
                net = tf.nn.relu(linear(z, 256, scope='g_fc1'))
                net = tf.nn.relu(linear(net, 64 * 7 * 7, scope='g_fc2'))
                net = tf.reshape(net, [self.batch_size, 7, 7, 64])
                net = tf.nn.relu(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'))
                out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

                return out

        # logit output from discriminator for real example
        D_logit_real = discriminator(self.image, True)
        # output of D for fake images
        G = generator(self.z, is_training=True, reuse=False)
        D_logit_fake = discriminator(G, is_training=True, reuse=True)

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

        """" Testing """
        # for test
        self.fake_images = generator(self.z, is_training=False, reuse=True)


def main():
    batch_size = 128
    Z_dim = 128

    f_mnist = input_data.read_data_sets('../data/fashion_mnist', one_hot=True)

    gan = GANs(Z_dim)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('outCV/'):
        os.makedirs('outCV/')

    i = 0
    for it in range(100000):
        if it % 1000 == 0:
            samples = sess.run(gan.fake_images, feed_dict={gan.z: sample_z(128, Z_dim)})

            fig = plot(samples[:16])
            plt.savefig('outCV/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        x_image, _ = f_mnist.train.next_batch(batch_size)

        _, D_loss_curr = sess.run([gan.D_train_op, gan.D_loss],
                                  feed_dict={gan.x: x_image, gan.z: sample_z(batch_size, Z_dim)})
        _, G_loss_curr = sess.run([gan.G_train_op, gan.G_loss], feed_dict={gan.z: sample_z(batch_size, Z_dim)})

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()


if __name__ == "__main__":
    main()
