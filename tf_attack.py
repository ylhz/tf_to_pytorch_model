# encoding:utf-8
"""Implementation of sample attack."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from nets import inception_v3, inception_resnet_v2

slim = tf.contrib.slim

tf.flags.DEFINE_string('checkpoint_path', './nets_weight', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string('input_csv', 'data/val_rs.csv', 'Input directory with images.')
tf.flags.DEFINE_string('input_dir', 'data/val_rs/', 'Input directory with images.')
tf.flags.DEFINE_string('output_dir', 'adv_img_tf/', 'Output directory with adv images.')
tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer('num_classes', 1001, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')
tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer('batch_size', 10, 'How man images process at one time.')
tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')
tf.flags.DEFINE_string('gpu', '3', 'gpu idx.')   

FLAGS = tf.flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu


model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
}


def seed_tensorflow(seed=0):
    """Set a random seed to ensure that the results are reproducible"""  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def mkdir(path):
    """Check if the folder exists, if it does not exist, create it"""
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def load_images(input_dir, csv_file, index, batch_shape):
    """Images for inception classifier are normalized to be in [-1, 1] interval"""
    images = np.zeros(batch_shape)
    filenames = []
    truelabel = []
    idx = 0
    for i in range(index, min(index + batch_shape[0], 1000)):
        img_obj = csv_file.loc[i]
        ImageID = img_obj['filename']
        img_path = os.path.join(input_dir, ImageID)
        images[idx, ...] = np.array(Image.open(img_path)).astype(np.float) / 255.0
        filenames.append(ImageID)
        truelabel.append(img_obj['label'])
        idx += 1

    images = images * 2.0 - 1.0
    return images, filenames, truelabel


def save_images(images, filenames, output_dir):
    """Saves images to the output directory."""
    mkdir(output_dir)
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            image = (images[i, :, :, :] + 1.0) * 0.5
            img = Image.fromarray((image * 255 + 0.5).astype('uint8')).convert('RGB')
            img.save(os.path.join(output_dir, filename), quality=100)


def graph(x, y, i, x_max, x_min, grad):
    """"I-FGSM
    """
    one_hot = tf.one_hot(y, FLAGS.num_classes)
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    # momentum = FLAGS.momentum

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
              x, num_classes=FLAGS.num_classes, is_training=False, reuse=tf.AUTO_REUSE)

    logits = logits_v3
    auxlogits = end_points_v3['AuxLogits']

    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits, label_smoothing=0.0, weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot, auxlogits, label_smoothing=0.0, weights=1.0)

    noise = tf.gradients(cross_entropy, x)[0]

    # MI
    # noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    # noise = momentum * grad + noise

    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise


def stop(x, y, i, x_max, x_min, grad):
    """I-FGSM　Attack stop condition
    """
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def main(_):
    # Because we normalized the input through "input * 2.0 - 1.0" to [-1,1],
    # the corresponding perturbation also needs to be multiplied by 2
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = FLAGS.num_classes
    inc_error_num = 0  # attack success num
    inc_res_v2_error_num = 0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        adv_img = tf.placeholder(tf.float32, shape=batch_shape)
        y = tf.placeholder(tf.int32, shape=batch_shape[0])
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        # white model
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                adv_img, num_classes=FLAGS.num_classes, is_training=False)
        # black model
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
                    adv_img, num_classes=num_classes, is_training=False, scope='InceptionResnetV2')

        pred_inc_res_v2 = tf.argmax(logits_ensadv_res_v2, 1)
        pred_inc_v3 = tf.argmax(logits_v3, 1)

        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad])

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            s2.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            
            dev = pd.read_csv(FLAGS.input_csv)
            for idx in tqdm(range(0, 1000 // FLAGS.batch_size)):
                images, filenames, True_label = load_images(FLAGS.input_dir, dev, idx * FLAGS.batch_size, batch_shape)
                # generate adversarial examples
                my_adv_images = sess.run(x_adv, feed_dict={x_input: images, y: True_label}).astype(np.float32)
                # test attack success rate
                pred_inc_res_v2_ = sess.run(pred_inc_res_v2, feed_dict={adv_img: my_adv_images})
                pred_inc_v3_ = sess.run(pred_inc_v3, feed_dict={adv_img: my_adv_images})
                inc_error_num += (pred_inc_v3_ != True_label).sum()
                inc_res_v2_error_num += (pred_inc_res_v2_ != True_label).sum()
                save_images(my_adv_images, filenames, FLAGS.output_dir)

    print('Inception_V3 success rate = {}'.format(inc_error_num / 1000.0))
    print('Inception_Resnet_V2 success rate = {}'.format(inc_res_v2_error_num / 1000.0))


if __name__ == '__main__':
    seed_tensorflow(0)
    tf.app.run()
