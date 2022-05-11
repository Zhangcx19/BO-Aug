# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Data utils for TinyImagenet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import copy
import pickle
import os
import os.path
from search import augmentation_transforms
import numpy as np
import search.bo_policies as found_policies
import tensorflow as tf


# pylint:disable=logging-format-interpolation


class TinyIN():
  base_folder = 'tiny-imagenet-200'

  class_number = 200

  def __init__(self, root, train=True,
               transform=None, target_transform=None,
               download=False):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.train = train  # training set or test set
    self.train_data = []
    self.train_labels = []
    self.train_names = []

    self.test_data = []
    self.test_labels = []
    self.test_names = []
    self.Original()

  def Original(self):
    # now load the picked numpy arrays
    if self.train:
      self.train_data = []
      self.train_labels = []
      self.train_names = []
      base_dir = os.path.join(self.root, self.base_folder)
      image_dir = os.path.join(self.root, self.base_folder, 'reduced/')

      with open(base_dir + '/reduced_folder2labels', "rb") as f:
        reduced_folder2labels = pickle.load(f)

      for folder_name in os.listdir(image_dir):
        if os.path.isdir(image_dir + folder_name + '/images/'):
          type_images = os.listdir(image_dir + folder_name + '/images/')
          # Loop through all the images of a type directory
          # batch_index = 0;
          # print ("Loading Class ", type)
          for image in type_images[:-100]:
            image_file = os.path.join(image_dir, folder_name + '/images/', image)

            # reading the images as they are; no normalization, no color editing
            image_data = Image.open(image_file)
            image_data = np.array(image_data)
            if (image_data.shape != (64, 64, 3)):
              # there are images that are not three channels, in this case we have to convert them.
              image_data = Image.open(image_file).convert("RGB")
              image_data = np.array(image_data)

            self.train_data.append(image_data)
            self.train_labels.append(reduced_folder2labels[folder_name])
            self.train_names.append(image)
      self.train_data = np.array(self.train_data)
      self.train_labels = np.array(self.train_labels)
      self.train_names = np.array(self.train_names)
    else:
      self.test_data = []
      self.test_labels = []
      self.test_names = []
      base_dir = os.path.join(self.root, self.base_folder)
      image_dir = os.path.join(self.root, self.base_folder, 'reduced/')

      with open(base_dir + '/reduced_folder2labels', "rb") as f:
        reduced_folder2labels = pickle.load(f)

      for folder_name in os.listdir(image_dir):
        if os.path.isdir(image_dir + folder_name + '/images/'):
          type_images = os.listdir(image_dir + folder_name + '/images/')
          # Loop through all the images of a type directory
          # batch_index = 0;
          # print ("Loading Class ", type)
          for image in type_images[-100:]:
            image_file = os.path.join(image_dir, folder_name + '/images/', image)

            # reading the images as they are; no normalization, no color editing
            image_data = Image.open(image_file)
            image_data = np.array(image_data)
            if (image_data.shape != (64, 64, 3)):
              # there are images that are not three channels, in this case we have to convert them.
              image_data = Image.open(image_file).convert("RGB")
              image_data = np.array(image_data)

            self.test_data.append(image_data)
            self.test_labels.append(reduced_folder2labels[folder_name])
            self.test_names.append(image)
      self.test_data = np.array(self.test_data)
      self.test_labels = np.array(self.test_labels)
      self.test_names = np.array(self.test_names)


class DataSet(object):
  """Dataset object that produces augmented training and eval data."""

  def __init__(self, hparams):
    self.hparams = hparams
    self.epochs = 0
    self.curr_train_index = 0

    self.good_policies = found_policies.good_policies()

    root = './search/tmp/'
    trainset = TinyIN(root=root, train=True)
    testset = TinyIN(root=root, train=False)
    all_data = trainset.train_data
    all_labels = trainset.train_labels.tolist()
    test_data = testset.test_data
    test_labels = testset.test_labels.tolist()

    train_dataset_size = 5000
    num_classes = 10

    all_data = all_data / 255.0
    test_data = test_data / 255.0
    mean = augmentation_transforms.MEANS
    std = augmentation_transforms.STDS
    tf.logging.info('mean:{}    std: {}'.format(mean, std))

    all_data = (all_data - mean) / std
    all_labels = np.eye(num_classes)[np.array(all_labels, dtype=np.int32)]
    test_data = (test_data - mean) / std
    test_labels = np.eye(num_classes)[np.array(test_labels, dtype=np.int32)]
    assert len(all_data) == len(all_labels)
    tf.logging.info(
        'In Imagenet loader, number of images: {}'.format(len(all_data)+len(test_data)))

    # Break off test data
    if hparams.eval_test:
      self.test_images = test_data
      self.test_labels = test_labels

    # Shuffle the rest of the data
    all_data = all_data[:train_dataset_size]
    all_labels = all_labels[:train_dataset_size]
    np.random.seed(0)
    perm = np.arange(len(all_data))
    np.random.shuffle(perm)
    all_data = all_data[perm]
    all_labels = all_labels[perm]

    # Break into train and val
    train_size, val_size = hparams.train_size, hparams.validation_size
    self.train_images = all_data[:train_size]
    self.train_labels = all_labels[:train_size]
    self.val_images = all_data[train_size:train_size + val_size]
    self.val_labels = all_labels[train_size:train_size + val_size]
    self.num_train = self.train_images.shape[0]

  def next_batch(self):
    """Return the next minibatch of augmented data."""
    next_train_index = self.curr_train_index + self.hparams.batch_size
    if next_train_index > self.num_train:
      # Increase epoch number
      epoch = self.epochs + 1
      self.reset()
      self.epochs = epoch
    batched_data = (
        self.train_images[self.curr_train_index:
                          self.curr_train_index + self.hparams.batch_size],
        self.train_labels[self.curr_train_index:
                          self.curr_train_index + self.hparams.batch_size])
    final_imgs = []

    images, labels = batched_data
    for data in images:
      epoch_policy = self.good_policies[np.random.choice(
          len(self.good_policies))]
      final_img = augmentation_transforms.apply_policy(
          epoch_policy, data)
      final_img = augmentation_transforms.random_flip(
          augmentation_transforms.zero_pad_and_crop(final_img, 4))
      # Apply cutout
      final_img = augmentation_transforms.cutout_numpy(final_img)
      final_imgs.append(final_img)
    batched_data = (np.array(final_imgs, np.float32), labels)
    self.curr_train_index += self.hparams.batch_size
    return batched_data

  def reset(self):
    """Reset training data and index into the training data."""
    self.epochs = 0
    # Shuffle the training data
    perm = np.arange(self.num_train)
    np.random.shuffle(perm)
    assert self.num_train == self.train_images.shape[
        0], 'Error incorrect shuffling mask'
    self.train_images = self.train_images[perm]
    self.train_labels = self.train_labels[perm]
    self.curr_train_index = 0


def unpickle(f):
  tf.logging.info('loading file: {}'.format(f))
  fo = tf.gfile.Open(f, 'rb')
  d = pickle.load(fo, encoding='latin1')
  fo.close()
  return d
