import os

import tensorflow as tf

from model_mnist_2 import dcgan

flags = tf.app.flags
flags.DEFINE_string("dataset" , "mnist" , "the dataset for read images")
flags.DEFINE_string("sample_dir" , "samples_for_test" , "the dir of sample images")
flags.DEFINE_integer("output_size" , 28 , "the size of generate image")
flags.DEFINE_string("log_dir" , "/tmp/tensorflow_mnist" , "the path of tensorflow's log")
flags.DEFINE_string("model_path" , "model/model.ckpt" , "the path of model")
flags.DEFINE_string("visua_path" , "visualization" , "the path of visuzation images")
flags.DEFINE_integer("operation" , 0 , "0 : train ; 1:test ; 2:visualize")

FLAGS = flags.FLAGS
#
if os.path.exists(FLAGS.sample_dir) == False:
    os.makedirs(FLAGS.sample_dir)
if os.path.exists(FLAGS.log_dir) == False:
    os.makedirs(FLAGS.log_dir)
if os.path.exists(FLAGS.model_path) == False:
    os.makedirs(FLAGS.model_path)
if os.path.exists(FLAGS.visua_path) == False:
    os.makedirs(FLAGS.visua_path)

def main(_):
    #dcgan(operation = FLAGS.operation ,data_name=FLAGS.dataset ,  output_size=FLAGS.output_size , sample_path=FLAGS.sample_dir , log_dir=FLAGS.log_dir
    #       , model_path= FLAGS.model_path , visua_path=FLAGS.visua_path)

    dataset = "fashion_mnist"
    sample_dir = "./out"
    output_size = 28
    log_dir = "./log"
    model_path = "./models"
    visual_path = "./out"
    operation  = 0

    if os.path.exists(sample_dir) == False:
        os.makedirs(sample_dir)
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    if os.path.exists(model_path) == False:
        os.makedirs(model_path)
    if os.path.exists(visual_path) == False:
        os.makedirs(visual_path)

    dcgan(operation = operation ,data_name=dataset, output_size=output_size , sample_path=sample_dir , log_dir=log_dir
        , model_path= model_path , visua_path=visual_path)


if __name__ == '__main__':
    tf.app.run()