import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from utils import *


class VariantionalAutoencoder(object):
    def __init__(self, latent_dim, image_dim=28 * 28):
        self.hidden_encoder_dim = 256
        self.hidden_decoder_dim = 256
        self.latent_dim = latent_dim
        self.learning_rate = 1e-2

        self.image_dim = image_dim

        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, self.image_dim])

        # Gaussian MLP as encoder
        with tf.variable_scope("gaussian_MLP_encoder"):
            he1 = tf.layers.dense(self.x, self.hidden_encoder_dim, activation=tf.nn.relu)
            he2 = tf.layers.dense(he1, self.hidden_encoder_dim, activation=tf.nn.relu)
            z_mu = tf.layers.dense(he2, self.latent_dim, activation=None)
            z_log_sigma = tf.layers.dense(he2, self.latent_dim, activation=None)

            eps = tf.random_normal(tf.shape(z_mu), 0, 1, dtype=tf.float32)
            self.z = z_mu + tf.exp(z_log_sigma * 0.5) * eps

        with tf.variable_scope("bernoulli_MLP_decoder"):
            hd1 = tf.layers.dense(self.z, self.hidden_decoder_dim, activation=tf.nn.relu)
            hd2 = tf.layers.dense(hd1, self.hidden_decoder_dim, activation=tf.nn.relu)
            self.x_hat = tf.layers.dense(hd2, self.image_dim, activation=tf.nn.sigmoid)

        with tf.variable_scope("loss"):
            # Reconstruction loss
            # Minimize the cross-entropy loss
            # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
            recon_loss = tf.reduce_sum(tf.losses.log_loss(self.x, self.x_hat, reduction="none"), axis=1)
            self.recon_loss = tf.reduce_mean(recon_loss)

            # Latent loss
            # Kullback Leibler divergence: measure the difference between two distributions
            # Here we measure the divergence between the latent distribution and N(0, 1)
            latent_loss = -0.5 * tf.reduce_sum(
                1 + z_log_sigma - tf.square(z_mu) - tf.exp(z_log_sigma), axis=1)

            self.latent_loss = tf.reduce_mean(latent_loss)
            self.total_loss = tf.reduce_mean(recon_loss + latent_loss)
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.total_loss)
        return


def main():
    batch_size = 128
    latent_dim = 100

    f_mnist = input_data.read_data_sets('../data/fashion_mnist', one_hot=True)

    vae = VariantionalAutoencoder(latent_dim)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0
    for it in range(100000):
        if it % 1000 == 0:
            samples = sess.run(vae.x_hat, feed_dict={vae.z: sample_z(16, latent_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        x_image, _ = f_mnist.train.next_batch(batch_size)

        _, loss, recon_loss, latent_loss = sess.run(
            [vae.train_op, vae.total_loss, vae.recon_loss, vae.latent_loss],
            feed_dict={vae.x: x_image}
        )

        if it % 1000 == 0:
            print('[Iter {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
                it, loss, recon_loss, latent_loss))


if __name__ == "__main__":
    main()
