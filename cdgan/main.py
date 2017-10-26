import cv2
import numpy as np
import tensorflow as tf

from ops import batch_normal
from ops import conv2d
from ops import conv_cond_concat
from ops import de_conv
from ops import fully_connect
from ops import lrelu
from utils import *

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.0002
batch_size = 128
EPOCH = 5
display_step = 1
sample_size = 100
y_dim = 10
channel = 1


class CDGans:
    def __init__(self, z_dim, image_dim = 28*28, image_w_h=28):
        self.z_dim = z_dim
        self.image_dim = image_dim
        self.image_w_h = image_w_h
        self.generate_image_w_h = 28
        self.y_dim = 10 # number of classes
        self.learning_rate = 0.0002
        self.batch_size = 128
        self.generator_dim = 100
        self.sample_num = 16



        self.build_model()
    def build_model(self):

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.image_dim])
        self.y = tf.placeholder(tf.float32, [None, self.y_dim])
        self.z = tf.placeholder(tf.float32, [None, sample_size])

        x_images = tf.reshape(self.x, [-1, self.image_w_h, self.image_w_h, channel])

        # weights
        self.weights = {

            'wc1': tf.Variable(tf.random_normal([5, 5, 11, 10], stddev=0.02), name='dis_w1'),
            'wc2': tf.Variable(tf.random_normal([5, 5, 20, 64], stddev=0.02), name='dis_w2'),
            'wc3': tf.Variable(tf.random_normal([64 * 7 * 7 + y_dim, 1024], stddev=0.02), name='dis_w3'),
            'wd': tf.Variable(tf.random_normal([1024 + y_dim, channel], stddev=0.02), name='dis_w4')
        }

        self.biases = {

            'bc1': tf.Variable(tf.zeros([10]), name='dis_b1'),
            'bc2': tf.Variable(tf.zeros([64]), name='dis_b2'),
            'bc3': tf.Variable(tf.zeros([1024]), name='dis_b3'),
            'bd': tf.Variable(tf.zeros([channel]), name='dis_b4')

        }

        self.weights2 = {

            'wd': tf.Variable(tf.random_normal([100 + y_dim, 1024], stddev=0.02), name='genw1'),
            'wc1': tf.Variable(tf.random_normal([1024 + y_dim, 7 * 7 * 2 * 64], stddev=0.02), name='genw2'),
            'wc2': tf.Variable(tf.random_normal([5, 5, 128, 138], stddev=0.02), name='genw3'),
            'wc3': tf.Variable(tf.random_normal([5, 5, channel, 138], stddev=0.02), name='genw4'),

        }

        self.biases2 = {

            'bd': tf.Variable(tf.zeros([1024]), name='genb1'),
            'bc1': tf.Variable(tf.zeros([7 * 7 * 2 * 64]), name='genb2'),
            'bc2': tf.Variable(tf.zeros([128]), name='genb3'),
            'bc3': tf.Variable(tf.zeros([channel]), name='genb4'),

        }

        ## Generator
        self.generator_z = self.generator(self.z, self.y, self.generate_image_w_h)
        self.sample_generator_z = self.sample_generator(self.sample_num, self.z, self.y, self.generate_image_w_h)

        ## Discriminator

        D_pro_real, D_logit_real = self.discriminator(x_images, self.y, self.weights, self.biases, False)

        G_pro_fake, D_logit_fake = self.discriminator(self.generator_z, self.y, self.weights, self.biases, True)


        ## loss
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_real), logits=D_logit_real))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake) ))
        self.D_loss = D_loss = D_loss_real + D_loss_fake
        self.G_loss = G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake) ))

        t_vars = tf.trainable_variables()

        d_var = [var for var in t_vars if 'dis' in var.name]
        g_var = [var for var in t_vars if 'gen' in var.name]

        self.D_solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(D_loss, var_list=d_var)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(G_loss, var_list=g_var)

        return


    def generator(self, z, y, output_size):

        yb = tf.reshape(y, shape=[-1, 1, 1, self.y_dim])

        z = tf.concat([z, y], 1)

        c1, c2 = int(output_size / 4), int(output_size / 2)

        # 10 stand for the num of labels
        d1 = fully_connect(z, self.weights2['wd'], self.biases2['bd'])
        d1 = batch_normal(d1, scope="genbn1")
        d1 = tf.nn.relu(d1)

        d1 = tf.concat([d1, y], 1)

        d2 = fully_connect(d1, self.weights2['wc1'], self.biases2['bc1'])
        d2 = batch_normal(d2, scope="genbn2")

        d2 = tf.nn.relu(d2)
        d2 = tf.reshape(d2, [batch_size, c1, c1, 64 * 2])

        d2 = conv_cond_concat(d2, yb)

        d3 = de_conv(d2, self.weights2['wc2'], self.biases2['bc2'], out_shape=[batch_size, c2, c2, 128])
        d3 = batch_normal(d3, scope="genbn3")
        d3 = tf.nn.relu(d3)

        d3 = conv_cond_concat(d3, yb)

        d4 = de_conv(d3, self.weights2['wc3'], self.biases2['bc3'], out_shape=[batch_size, output_size, output_size, 1])

        return tf.nn.sigmoid(d4)

    def sample_generator(self, sample_size, z, y, output_size):

        yb = tf.reshape(y, shape=[sample_size, 1, 1, y_dim])

        z = tf.concat([z, y], 1)

        c1, c2 = int(output_size / 4), int(output_size / 2)

        # 10 stand for the num of labels
        d1 = fully_connect(z, self.weights2['wd'], self.biases2['bd'])
        d1 = batch_normal(d1, scope="genbn1", reuse=True)
        d1 = tf.nn.relu(d1)

        d1 = tf.concat([d1, y], 1)

        d2 = fully_connect(d1, self.weights2['wc1'], self.biases2['bc1'])
        d2 = batch_normal(d2, scope="genbn2", reuse=True)

        d2 = tf.nn.relu(d2)
        d2 = tf.reshape(d2, [sample_size, c1, c1, 64 * 2])

        d2 = conv_cond_concat(d2, yb)

        d3 = de_conv(d2, self.weights2['wc2'], self.biases2['bc2'], out_shape=[sample_size, c2, c2, 128])
        d3 = batch_normal(d3, scope="genbn3", reuse=True)
        d3 = tf.nn.relu(d3)

        d3 = conv_cond_concat(d3, yb)

        d4 = de_conv(d3, self.weights2['wc3'], self.biases2['bc3'], out_shape=[sample_size, output_size, output_size, 1])

        return tf.nn.sigmoid(d4)

    def discriminator(self, x, y, weights, biases, reuse=False):
        # mnist data's shape is (28 , 28 , 1)

        yb = tf.reshape(y, shape=[batch_size, 1, 1, y_dim])

        # concat
        xy = conv_cond_concat(x, yb)

        conv1 = conv2d(xy, weights['wc1'], biases['bc1'])

        tf.add_to_collection('weight_1', weights['wc1'])
        conv1 = lrelu(conv1)
        conv1 = conv_cond_concat(conv1, yb)

        tf.add_to_collection('ac_1', conv1)

        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = batch_normal(conv2, scope="dis_bn1", reuse=reuse)
        conv2 = lrelu(conv2)

        tf.add_to_collection('weight_2', weights['wc2'])

        tf.add_to_collection('ac_2', conv2)

        conv2 = tf.reshape(conv2, [batch_size, -1])

        conv2 = tf.concat([conv2, y], 1)

        f1 = fully_connect(conv2, weights['wc3'], biases['bc3'])
        f1 = batch_normal(f1, scope="dis_bn2", reuse=reuse)
        f1 = lrelu(f1)
        f1 = tf.concat([f1, y], 1)

        out = fully_connect(f1, weights['wd'], biases['bd'])

        return tf.nn.sigmoid(out), out


def sample_label():
    #num = 64
    num = 16
    label_vector = np.zeros((num , 10), dtype=np.float)
    class_label = [0,1,5,8]
    class_ind = 0
    for i in range(0 , num):
        #-------------------------
        #label_vector[i , int(i/8)] = 1.0

        label_vector[i, int(i / 4)] = 1.0
    return label_vector

def main():
    batch_size = 128
    Z_dim = 100

    f_mnist = input_data.read_data_sets('../data/fashion_mnist', one_hot=True)

    model = CDGans(Z_dim)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0

    for it in range(1000000):
        if it%50 ==0 :
            samples = sess.run(model.sample_generator_z, feed_dict={model.z: sample_z(16, Z_dim), model.y: sample_label()})


            fig = plot(samples)
            plt.savefig('./out/{}.png'.format( str(i).zfill(3)), bbox_inches='tight')
            i += 1
            #plt.close(fig)

        X_mb, Y_mb = f_mnist.train.next_batch(batch_size)




        _, D_loss_curr = sess.run([model.D_solver, model.D_loss],
                                  feed_dict={model.x: X_mb,
                                             model.y: Y_mb,
                                             model.z: sample_z(batch_size, Z_dim)})
        _, G_loss_curr = sess.run([model.G_solver, model.G_loss],
                                  feed_dict={model.y : Y_mb,
                                             model.z: sample_z(batch_size, Z_dim)})

        #if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()

        #save_path = saver.save(sess, model_path)
if __name__ == "__main__":
    main()


