# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import argparse
import time
import tensorflow as tf
import errno
import matplotlib.pyplot as plt
import cv2
import sys
import shutil
from tqdm import tqdm

from bts_dataloader import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

checkpoint_path = './models/bts_nyu/model'
# model_name = 'bts_nyu_test'
#
# model_dir = os.path.dirname(checkpoint_path)
# sys.path.append(model_dir)
#
# for key, val in vars(__import__(model_name)).items():
#     if key.startswith('__') and key.endswith('__'):
#         continue
#     vars()[key] = val

from models.bts_nyu.bts_nyu import *


def test():

    params = bts_parameters(
        encoder='densenet161_bts',
        height=480,
        width=640,
        batch_size=None,
        dataset=None,
        max_depth=10,
        num_gpus=None,
        num_threads=None,
        num_epochs=None)

    # image_path = './tests/src/cam_100_img_0000003_1571220946.png'
    image_path = './tests/src/cam_100_img_0000002_1571220931.png'
    image = tf.image.decode_png(tf.read_file(image_path), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    print(image.get_shape().as_list())
    focal = 519

    model = BtsModel(params, 'test', image, None, focal=focal, bn_training=False)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # SAVER
    train_saver = tf.train.Saver()
    num_test_samples = 1

    with tf.device('/GPU:0'):
        restore_path = checkpoint_path

        # RESTORE
        train_saver.restore(sess, restore_path)


        pred_depths = []
        pred_8x8s = []
        pred_4x4s = []
        pred_2x2s = []

        start_time = time.time()
        print('Processing images..')
        for s in tqdm(range(num_test_samples)):
            depth, pred_8x8, pred_4x4, pred_2x2 = sess.run(
                [model.depth_est, model.depth_8x8, model.depth_4x4, model.depth_2x2])
            pred_depths.append(depth[0].squeeze())

            pred_8x8s.append(pred_8x8[0].squeeze())
            pred_4x4s.append(pred_4x4[0].squeeze())
            pred_2x2s.append(pred_2x2[0].squeeze())

        print('Done.')

        save_name = './tests/out/result'

        print('Saving result pngs..')
        if not os.path.exists(os.path.dirname(save_name)):
            try:
                os.makedirs(save_name)
                os.mkdir(save_name + '/raw')
                os.mkdir(save_name + '/cmap')
                os.mkdir(save_name + '/rgb')
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        for s in tqdm(range(num_test_samples)):
            filename_cmap_png = save_name + '/cmap/out_cmap.png'

            pred_depth = pred_depths[s]
            pred_8x8 = pred_8x8s[s]
            pred_4x4 = pred_4x4s[s]
            pred_2x2 = pred_2x2s[s]

            image = sess.run(image)
            _, h, w, _ = image.shape
            print(f'w x h = {w} x {h}')

            pred_depth_cropped = np.zeros((h, w), dtype=np.float32) + 1
            pred_depth_cropped[10:-1 - 10, 10:-1 - 10] = pred_depth[10:-1 - 10, 10:-1 - 10]
            plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='Greys')
            pred_8x8_cropped = np.zeros((h, w), dtype=np.float32) + 1
            pred_8x8_cropped[10:-1 - 10, 10:-1 - 10] = pred_8x8[10:-1 - 10, 10:-1 - 10]
            filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
            plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8_cropped), cmap='Greys')
            pred_4x4_cropped = np.zeros((h, w), dtype=np.float32) + 1
            pred_4x4_cropped[10:-1 - 10, 10:-1 - 10] = pred_4x4[10:-1 - 10, 10:-1 - 10]
            filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
            plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4_cropped), cmap='Greys')
            pred_2x2_cropped = np.zeros((h, w), dtype=np.float32) + 1
            pred_2x2_cropped[10:-1 - 10, 10:-1 - 10] = pred_2x2[10:-1 - 10, 10:-1 - 10]
            filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
            plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2_cropped), cmap='Greys')


def main(_):
    test()


if __name__ == '__main__':
    tf.app.run()



