# encoding:utf-8
"""Implementation of sample attack."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from nets import inception_v3, inception_resnet_v2, resnet_v2, inception_v4

slim = tf.contrib.slim

tf.flags.DEFINE_string('checkpoint_path', 'nets_weight', 'Path to checkpoint for inception network.')
tf.flags.DEFINE_string('input_csv', 'data/val_rs.csv', 'Input directory with images.')
# tf.flags.DEFINE_string('input_dir', 'data/val_rs/', 'Input directory with images.')

tf.flags.DEFINE_integer('num_classes', 1001, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')
tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer('batch_size', 10, 'How man images process at one time.')
tf.flags.DEFINE_string('gpu', '3', 'gpu idx.')   

FLAGS = tf.flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

checkpoint_path = FLAGS.checkpoint_path
model_checkpoint_map = {
    'inception_v3': os.path.join(checkpoint_path, 'inception_v3.ckpt'),
    'inception_v4': os.path.join(checkpoint_path, 'inception_v4.ckpt'),
    'resnet_v2_50': os.path.join(checkpoint_path, 'resnet_v2_50.ckpt'),
    'resnet_v2_101': os.path.join(checkpoint_path, 'resnet_v2_101.ckpt'),
    # 'resnet_v2_152': os.path.join(checkpoint_path, 'resnet_v2_152.ckpt'),
    'inception_resnet_v2': os.path.join(checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),

    'adv_inception_v3': os.path.join(checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    }

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

def main(_):
    num_classes = FLAGS.num_classes
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape = batch_shape)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_inc_res_v2, end_points_inc_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_inc_v3, end_points_inc_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_inc_v4, end_points_inc_v4 = inception_v4.inception_v4(
                x_input, num_classes=num_classes, is_training=False)

        # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        #     logits_res_v2_50, end_points_res_v2_50 = resnet_v2.resnet_v2_50(
        #         x_input, num_classes=num_classes, is_training=False, scope='ResnetV2_50')

        # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        #     logits_res_v2_101, end_points_res_v2_101 = resnet_v2.resnet_v2_101(
        #         x_input, num_classes=num_classes, is_training=False, scope='ResnetV2_101')

        # with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        #     logits_res_v2_152, end_points_res_v2_152 = resnet_v2.resnet_v2_152(
        #         x_input, num_classes=num_classes, is_training=False, scope='ResnetV2_152')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_adv_inc_v3, end_points_adv_inc_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False, scope='AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_inc_v3, end_points_ens3_adv_inc_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_inc_v3, end_points_ens4_adv_inc_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ens_adv_inc_res_v2, end_points_ens_adv_inc_res_v2 = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')



        pred_inc_v3 = tf.argmax(logits_inc_v3, 1)
        pred_inc_v4 = tf.argmax(logits_inc_v4, 1)
        # pred_res_v2_50 = tf.argmax(logits_res_v2_50, 1)
        # pred_res_v2_101 = tf.argmax(logits_res_v2_101, 1)
        # pred_res_v2_152 = tf.argmax(logits_res_v2_152, 1)
        pred_inc_res_v2 = tf.argmax(logits_inc_res_v2, 1)


        pred_adv_inc_v3 = tf.argmax(logits_adv_inc_v3, 1)
        pred_ens3_adv_inc_v3 = tf.argmax(logits_ens3_adv_inc_v3, 1)
        pred_ens4_adv_inc_v3 = tf.argmax(logits_ens4_adv_inc_v3, 1)
        pred_ens_adv_inc_res_v2 = tf.argmax(logits_ens_adv_inc_res_v2, 1)

        # Run computation
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        # s3 = tf.train.Saver(slim.get_model_variables(scope='ResnetV2_50'))
        # s4 = tf.train.Saver(slim.get_model_variables(scope='ResnetV2_101'))
        # s5 = tf.train.Saver(slim.get_model_variables(scope='ResnetV2_152'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))

        s7 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s9 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s10 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))


        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])
            s2.restore(sess, model_checkpoint_map['inception_v4'])
            # s3.restore(sess, model_checkpoint_map['resnet_v2_50'])
            # s4.restore(sess, model_checkpoint_map['resnet_v2_101'])
            # s5.restore(sess, model_checkpoint_map['resnet_v2_152'])
            s6.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            s7.restore(sess, model_checkpoint_map['adv_inception_v3'])
            s8.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            s9.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            s10.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])

            dev = pd.read_csv(FLAGS.input_csv)
            inc_v3_num, inc_v4_num, res_v2_50_num, res_v2_101_num, res_v2_152_num, inc_res_v2_num, adv_inc_v3_num, ens3_adv_inc_v3_num, ens4_adv_inc_v3_num, ens_adv_inc_res_v2_num = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            for idx in tqdm(range(0, 1000 // FLAGS.batch_size)):
                images, filenames, True_label = load_images(FLAGS.input_dir, dev, idx * FLAGS.batch_size, batch_shape)
             
                # test attack success rate
                # inc_v3, inc_v4, res_v2_50, res_v2_101, res_v2_152, inc_res_v2, adv_inc_v3, ens3_adv_inc_v3, ens4_adv_inc_v3, ens_adv_inc_res_v2 = sess.run((pred_inc_v3, pred_inc_v4, pred_res_v2_50, pred_res_v2_101, pred_res_v2_152, pred_inc_res_v2, pred_adv_inc_v3, pred_ens3_adv_inc_v3, pred_ens4_adv_inc_v3, pred_ens_adv_inc_res_v2), feed_dict={x_input: images})

                inc_v3, inc_v4, inc_res_v2, adv_inc_v3, ens3_adv_inc_v3, ens4_adv_inc_v3, ens_adv_inc_res_v2 = sess.run((pred_inc_v3, pred_inc_v4, pred_inc_res_v2, pred_adv_inc_v3, pred_ens3_adv_inc_v3, pred_ens4_adv_inc_v3, pred_ens_adv_inc_res_v2), feed_dict={x_input: images})

                inc_v3_num += (inc_v3 == True_label).sum()
                inc_v4_num += (inc_v4 == True_label).sum()
                # res_v2_50_num += (res_v2_50 == True_label).sum()
                # res_v2_101_num += (res_v2_101 == True_label).sum()
                # res_v2_152_num += (res_v2_152 == True_label).sum()
                inc_res_v2_num += (inc_res_v2 == True_label).sum()

                adv_inc_v3_num += (adv_inc_v3 == True_label).sum()
                ens3_adv_inc_v3_num += (ens3_adv_inc_v3 == True_label).sum()
                ens4_adv_inc_v3_num += (ens4_adv_inc_v3 == True_label).sum()
                ens_adv_inc_res_v2_num += (ens_adv_inc_res_v2 == True_label).sum()

    print('Inception_V3 accuracy = {}'.format(inc_v3_num / 1000.0))
    print('Inception_V4 accuracy = {}'.format(inc_v4_num / 1000.0))
    print('res_v2_50 accuracy = {}'.format(res_v2_50_num / 1000.0))
    print('res_v2_101 accuracy = {}'.format(res_v2_101_num / 1000.0))
    # print('res_v2_152 accuracy = {}'.format(res_v2_152_num / 1000.0))
    print('adv_inc_v3 accuracy = {}'.format(adv_inc_v3_num / 1000.0))
    print('ens3_adv_inc_v3 accuracy = {}'.format(ens3_adv_inc_v3_num / 1000.0))
    print('ens4_adv_inc_v3 accuracy = {}'.format(ens4_adv_inc_v3_num / 1000.0))
    print('ens_adv_inc_res_v2 accuracy = {}'.format(ens_adv_inc_res_v2_num / 1000.0))


if __name__ == '__main__':
    tf.app.run()
