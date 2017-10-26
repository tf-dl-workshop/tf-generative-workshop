import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

class Gan_FMnist:
    def __init__(self, image_dim = 128 ):

        self.g_in_dim = 100
        self.g_out_dim = 784
        self.image_dim = 128

        self.build_model()

    def xavier_init(self,size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def build_model(self):

        self.Z = Z = tf.placeholder(tf.float32, shape=[None, self.g_in_dim])
        self.X = X = tf.placeholder(tf.float32, shape=[None, self.g_out_dim])


        ## Generator

        G_W1 = tf.Variable(self.xavier_init([self.g_in_dim, self.image_dim]))
        G_b1 = tf.Variable(tf.zeros(shape=[self.image_dim]))

        G_W2 = tf.Variable(self.xavier_init([self.image_dim, self.g_out_dim]))
        G_b2 = tf.Variable(tf.zeros(shape=[self.g_out_dim]))

        theta_G = [G_W1, G_W2, G_b1, G_b2]


        def generator(z):
            G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
            G_prob = tf.nn.sigmoid(G_log_prob)

            return G_prob

        ## Discriminator
        D_W1 = tf.Variable(self.xavier_init([self.g_out_dim, self.image_dim]))
        D_b1 = tf.Variable(tf.zeros(shape=[self.image_dim]))

        D_W2 = tf.Variable(self.xavier_init([self.image_dim, 1]))
        D_b2 = tf.Variable(tf.zeros(shape=[1]))

        theta_D = [D_W1, D_W2, D_b1, D_b2]



        def discriminator(x):
            D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
            D_logit = tf.matmul(D_h1, D_W2) + D_b2
            D_prob = tf.nn.sigmoid(D_logit)

            return D_prob, D_logit

        self.G_sample = generator(self.Z)
        D_real, D_logit_real = discriminator(X)
        D_fake, D_logit_fake = discriminator(self.G_sample)

        ## loss
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        self.D_loss = D_loss = D_loss_real + D_loss_fake
        self.G_loss = G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        self.D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
        self.G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

        return self.Z



def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])



def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig
def main():
    mb_size = 128
    Z_dim = 100

    f_mnist = input_data.read_data_sets('../data/fashion_mnist', one_hot=True)

    model = Gan_FMnist()
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out/'):
        os.makedirs('out/')

    i=0
    for it in range(1000000):
        if it % 1000 == 0:
            samples = sess.run(model.G_sample, feed_dict={model.Z: sample_Z(16, Z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        X_mb, _ = f_mnist.train.next_batch(mb_size)

        _, D_loss_curr = sess.run([model.D_solver, model.D_loss], feed_dict={model.X: X_mb, model.Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([model.G_solver, model.G_loss], feed_dict={model.Z: sample_Z(mb_size, Z_dim)})

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'.format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()


if __name__ == "__main__":
    main()