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

"""AutoAugment Train/Eval module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import time
import logging

import custom_ops as ops
import data_utils
import helper_utils
import numpy as np
from shake_drop import build_shake_drop_model
from shake_shake import build_shake_shake_model
import tensorflow as tf
from wrn import build_wrn_model
import bo_policies as bo_policies

file_path = "/mnt/home/zhaoxi35/zhangzijian/xz-net/Aug/train_tiny_aug/log/"
logging.basicConfig(level=logging.DEBUG, filename=file_path+"wrn-28-2-aug.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

tf.flags.DEFINE_string('model_name', 'wrn',
                       'wrn, shake_shake_32, shake_shake_96, shake_shake_112, '
                       'pyramid_net')
tf.flags.DEFINE_string('checkpoint_dir', '/mnt/home/zhaoxi35/zhangzijian/xz-net/Aug/train_tiny_aug/tmp/20220425', 'Training Directory.')
tf.flags.DEFINE_string('data_path', '/mnt/home/zhaoxi35/zhangzijian/xz-net/Aug/train_tiny/tmp/',
                       'Directory where dataset is located.')
tf.flags.DEFINE_string('dataset', 'tiny',
                       'Dataset to train with. Either cifar10 or cifar100')
tf.flags.DEFINE_integer('use_cpu', 0, '1 if use CPU, else GPU.')

FLAGS = tf.flags.FLAGS

arg_scope = tf.contrib.framework.arg_scope

MAX = 0

def setup_arg_scopes(is_training):
  """Sets up the argscopes that will be used when building an image model.

  Args:
    is_training: Is the model training or not.

  Returns:
    Arg scopes to be put around the model being constructed.
  """

  batch_norm_decay = 0.9
  batch_norm_epsilon = 1e-5
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      'scale': True,
      # collection containing the moving mean and moving variance.
      'is_training': is_training,
  }

  scopes = []

  scopes.append(arg_scope([ops.batch_norm], **batch_norm_params))
  return scopes


def build_model(inputs, num_classes, is_training, hparams):
  """Constructs the vision model being trained/evaled.

  Args:
    inputs: input features/images being fed to the image model build built.
    num_classes: number of output classes being predicted.
    is_training: is the model training or not.
    hparams: additional hyperparameters associated with the image model.

  Returns:
    The logits of the image model.
  """
  scopes = setup_arg_scopes(is_training)
  with contextlib.ExitStack() as stack:
    tuple(stack.enter_context(cm) for cm in scopes)
    if hparams.model_name == 'pyramid_net':
      logits = build_shake_drop_model(
          inputs, num_classes, is_training)
    elif hparams.model_name == 'wrn':
      logits = build_wrn_model(
          inputs, num_classes, hparams.wrn_size)
    elif hparams.model_name == 'shake_shake':
      logits = build_shake_shake_model(
          inputs, num_classes, hparams, is_training)
  return logits


class CifarModel(object):
  """Builds an image model for Cifar10/Cifar100."""

  def __init__(self, hparams):
    self.hparams = hparams

  def build(self, mode):
    """Construct the cifar model."""
    assert mode in ['train', 'eval']
    self.mode = mode
    self._setup_misc(mode)
    self._setup_images_and_labels()
    self._build_graph(self.images, self.labels, mode)

    self.init = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())

  def _setup_misc(self, mode):
    """Sets up miscellaneous in the cifar model constructor."""
    self.lr_rate_ph = tf.Variable(0.0, name='lrn_rate', trainable=False)
    self.reuse = None if (mode == 'train') else True
    self.batch_size = self.hparams.batch_size
    if mode == 'eval':
      self.batch_size = 25

  def _setup_images_and_labels(self):
    """Sets up image and label placeholders for the cifar model."""
    if FLAGS.dataset == 'cifar10':
      self.num_classes = 10
    else:
      self.num_classes = 200
    self.images = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 3])
    self.labels = tf.placeholder(tf.float32,
                                 [self.batch_size, self.num_classes])

  def assign_epoch(self, session, epoch_value):
    session.run(self._epoch_update, feed_dict={self._new_epoch: epoch_value})

  def _build_graph(self, images, labels, mode):
    """Constructs the TF graph for the cifar model.

    Args:
      images: A 4-D image Tensor
      labels: A 2-D labels Tensor.
      mode: string indicating training mode ( e.g., 'train', 'valid', 'test').
    """
    is_training = 'train' in mode
    if is_training:
      self.global_step = tf.train.get_or_create_global_step()

    logits = build_model(
        images,
        self.num_classes,
        is_training,
        self.hparams)
    self.predictions, self.cost = helper_utils.setup_loss(
        logits, labels)
    self.accuracy, self.eval_op = tf.metrics.accuracy(
        tf.argmax(labels, 1), tf.argmax(self.predictions, 1))
    self._calc_num_trainable_params()

    # Adds L2 weight decay to the cost
    self.cost = helper_utils.decay_weights(self.cost,
                                           self.hparams.weight_decay_rate)

    if is_training:
      self._build_train_op()

    # Setup checkpointing for this child model
    # Keep 2 or more checkpoints around during training.
    with tf.device('/cpu:0'):
      self.saver = tf.train.Saver(max_to_keep=2)

    self.init = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())

  def _calc_num_trainable_params(self):
    self.num_trainable_params = np.sum([
        np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()
    ])
    tf.logging.info('number of trainable params: {}'.format(
        self.num_trainable_params))

  def _build_train_op(self):
    """Builds the train op for the cifar model."""
    hparams = self.hparams
    tvars = tf.trainable_variables()
    grads = tf.gradients(self.cost, tvars)
    if hparams.gradient_clipping_by_global_norm > 0.0:
      grads, norm = tf.clip_by_global_norm(
          grads, hparams.gradient_clipping_by_global_norm)
      tf.summary.scalar('grad_norm', norm)

    # Setup the initial learning rate
    initial_lr = self.lr_rate_ph
    optimizer = tf.train.MomentumOptimizer(
        initial_lr,
        0.9,
        use_nesterov=True)

    self.optimizer = optimizer
    apply_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=self.global_step, name='train_step')
    train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([apply_op]):
      self.train_op = tf.group(*train_ops)


class CifarModelTrainer(object):
  """Trains an instance of the CifarModel class."""

  def __init__(self, hparams):
    self._session = None
    self.hparams = hparams

    self.model_dir = os.path.join(FLAGS.checkpoint_dir, 'model')
    self.log_dir = os.path.join(FLAGS.checkpoint_dir, 'log')
    # Set the random seed to be sure the same validation set
    # is used for each model
    np.random.seed(0)
    self.data_loader = data_utils.DataSet(hparams)
    np.random.seed()  # Put the random seed back to random
    self.data_loader.reset()

  def save_model(self, step=None):
    """Dumps model into the backup_dir.

    Args:
      step: If provided, creates a checkpoint with the given step
        number, instead of overwriting the existing checkpoints.
    """
    model_save_name = os.path.join(self.model_dir, 'model.ckpt')
    if not tf.gfile.IsDirectory(self.model_dir):
      tf.gfile.MakeDirs(self.model_dir)
    self.saver.save(self.session, model_save_name, global_step=step)
    tf.logging.info('Saved child model')

  def extract_model_spec(self):
    """Loads a checkpoint with the architecture structure stored in the name."""
    checkpoint_path = tf.train.latest_checkpoint(self.model_dir)
    if checkpoint_path is not None:
      self.saver.restore(self.session, checkpoint_path)
      tf.logging.info('Loaded child model checkpoint from %s',
                      checkpoint_path)
    else:
      self.save_model(step=0)

  def eval_child_model(self, model, data_loader, mode):
    """Evaluate the child model.

    Args:
      model: image model that will be evaluated.
      data_loader: dataset object to extract eval data from.
      mode: will the model be evalled on train, val or test.

    Returns:
      Accuracy of the model on the specified dataset.
    """
    global MAX
    tf.logging.info('Evaluating child model in mode %s', mode)
    while True:
      try:
        with self._new_session(model):
          accuracy = helper_utils.eval_child_model(
              self.session,
              model,
              data_loader,
              mode)
          tf.logging.info('Eval child model accuracy: {}'.format(accuracy))
          if accuracy > MAX:
                MAX = accuracy
          print('MAX:', MAX)
          logging.info('MAX: {}'.format(MAX))
          # If epoch trained without raising the below errors, break
          # from loop.
          break
      except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
        tf.logging.info('Retryable error caught: %s.  Retrying.', e)

    return accuracy

  @contextlib.contextmanager
  def _new_session(self, m):
    """Creates a new session for model m."""
    # Create a new session for this model, initialize
    # variables, and save / restore from
    # checkpoint.
    self._session = tf.Session(
        '',
        config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
    self.session.run(m.init)

    # Load in a previous checkpoint, or save this one
    self.extract_model_spec()
    try:
      yield
    finally:
      tf.Session.reset('')
      self._session = None

  def _build_models(self):
    """Builds the image models for train and eval."""
    # Determine if we should build the train and eval model. When using
    # distributed training we only want to build one or the other and not both.
    with tf.variable_scope('model', use_resource=False):
      m = CifarModel(self.hparams)
      m.build('train')
      self._num_trainable_params = m.num_trainable_params
      self._saver = m.saver
    with tf.variable_scope('model', reuse=True, use_resource=False):
      meval = CifarModel(self.hparams)
      meval.build('eval')
    return m, meval

  def _calc_starting_epoch(self, m):
    """Calculates the starting epoch for model m based on global step."""
    hparams = self.hparams
    batch_size = hparams.batch_size
    steps_per_epoch = int(hparams.train_size / batch_size)
    with self._new_session(m):
      curr_step = self.session.run(m.global_step)
    total_steps = steps_per_epoch * hparams.num_epochs
    epochs_left = (total_steps - curr_step) // steps_per_epoch
    starting_epoch = hparams.num_epochs - epochs_left
    return starting_epoch

  def _run_training_loop(self, m, curr_epoch):
    """Trains the cifar model `m` for one epoch."""
    start_time = time.time()
    while True:
      try:
        with self._new_session(m):
          train_accuracy = helper_utils.run_epoch_training(
              self.session, m, self.data_loader, curr_epoch)
          tf.logging.info('Saving model after epoch')
          self.save_model(step=curr_epoch)
          break
      except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
        tf.logging.info('Retryable error caught: %s.  Retrying.', e)
    tf.logging.info('Finished epoch: {}'.format(curr_epoch))
    tf.logging.info('Epoch time(min): {}'.format(
        (time.time() - start_time) / 60.0))
    return train_accuracy

  def _compute_final_accuracies(self, meval):
    """Run once training is finished to compute final val/test accuracies."""
    valid_accuracy = self.eval_child_model(meval, self.data_loader, 'val')
    if self.hparams.eval_test:
      test_accuracy = self.eval_child_model(meval, self.data_loader, 'test')
    else:
      test_accuracy = 0
    tf.logging.info('Test Accuracy: {}'.format(test_accuracy))
    return valid_accuracy, test_accuracy

  def run_model(self):
    """Trains and evalutes the image model."""
    hparams = self.hparams

    # Build the child graph
    with tf.Graph().as_default(), tf.device(
        '/cpu:0' if FLAGS.use_cpu else '/gpu:0'):
      m, meval = self._build_models()

      # Figure out what epoch we are on
      starting_epoch = self._calc_starting_epoch(m)

      # Run the validation error right at the beginning
      valid_accuracy = self.eval_child_model(
          meval, self.data_loader, 'val')
      tf.logging.info('Before Training Epoch: {}     Val Acc: {}'.format(
          starting_epoch, valid_accuracy))
      training_accuracy = None

      for curr_epoch in range(starting_epoch, hparams.num_epochs):

        # Run one training epoch
        training_accuracy = self._run_training_loop(m, curr_epoch)

        valid_accuracy = self.eval_child_model(
            meval, self.data_loader, 'test')
        tf.logging.info('Epoch: {}    test Acc: {}'.format(
            curr_epoch, valid_accuracy))

      valid_accuracy, test_accuracy = self._compute_final_accuracies(
          meval)

    tf.logging.info(
        'Train Acc: {}    Valid Acc: {}     Test Acc: {}'.format(
            training_accuracy, valid_accuracy, test_accuracy))
    print('MAX:', MAX)
    logging.info('MAX: {}'.format(MAX))

  @property
  def saver(self):
    return self._saver

  @property
  def session(self):
    return self._session

  @property
  def num_trainable_params(self):
    return self._num_trainable_params


def main(_):
  if FLAGS.dataset not in ['cifar10', 'cifar100', 'tiny']:
    raise ValueError('Invalid dataset: %s' % FLAGS.dataset)
  hparams = tf.contrib.training.HParams(
      train_size=100000,
      validation_size=0,
      eval_test=1,
      dataset=FLAGS.dataset,
      data_path=FLAGS.data_path,
      batch_size=256,
      gradient_clipping_by_global_norm=5.0)
  if FLAGS.model_name == 'wrn':
    hparams.add_hparam('model_name', 'wrn')
    hparams.add_hparam('num_epochs', 120)
    hparams.add_hparam('wrn_size', 32)
    hparams.add_hparam('lr', 0.1)
    hparams.add_hparam('weight_decay_rate', 5e-4)
  elif FLAGS.model_name == 'shake_shake_32':
    hparams.add_hparam('model_name', 'shake_shake')
    hparams.add_hparam('num_epochs', 1800)
    hparams.add_hparam('shake_shake_widen_factor', 2)
    hparams.add_hparam('lr', 0.01)
    hparams.add_hparam('weight_decay_rate', 0.001)
  elif FLAGS.model_name == 'shake_shake_96':
    hparams.add_hparam('model_name', 'shake_shake')
    hparams.add_hparam('num_epochs', 1800)
    hparams.add_hparam('shake_shake_widen_factor', 6)
    hparams.add_hparam('lr', 0.01)
    hparams.add_hparam('weight_decay_rate', 0.001)
  elif FLAGS.model_name == 'shake_shake_112':
    hparams.add_hparam('model_name', 'shake_shake')
    hparams.add_hparam('num_epochs', 1800)
    hparams.add_hparam('shake_shake_widen_factor', 7)
    hparams.add_hparam('lr', 0.01)
    hparams.add_hparam('weight_decay_rate', 0.001)
  elif FLAGS.model_name == 'pyramid_net':
    hparams.add_hparam('model_name', 'pyramid_net')
    hparams.add_hparam('num_epochs', 1800)
    hparams.add_hparam('lr', 0.05)
    hparams.add_hparam('weight_decay_rate', 5e-5)
    hparams.batch_size = 64
  else:
    raise ValueError('Not Valid Model Name: %s' % FLAGS.model_name)

  policies = [74.59492671, 0.36237805, 4.22216973, 0.6892949, 1.11414227,
              184.47725265, 0.95289816, 1.12166779, 0.97034174, 5.90216261,
              166.6672381, 0.97743375, 4.7498256, 0.70251432, 2.70503843,
              170.61345706,   0.37366146,   3.18541466,   0.36306211,   4.61747465,
              105.00565615,   0.47818459,   3.70850263,   0.96498194,   5.7581574,
              87.43685832,   0.87823868,   4.84292846,   0.90731033,   4.50219931,
              147.51363180435672, 0.984372338768757, 5.5285824519706654, 0.6449963388926072, 1.2622244538115026,
              161.48011302433662, 0.9935170037142828, 1.1801603463348727, 0.4658036181615953, 5.3732496764506035,
              105.1724909279528, 0.002275186043534568, 6.481846204028834, 0.6869268226528746, 5.1583644407761415,
              1.52352028e+02, 8.77769322e-01, 2.23426611e+00, 9.65695578e-01, 4.49885070e+00,
              1.61413411e+02, 2.07856827e-01, 3.60585464e+00, 9.31157273e-01, 1.34199690e+00,
              1.71488264e+01, 1.68465268e-01, 9.41388233e-02, 7.04164605e-01, 2.55797013e+00,
              132.410747, 0.0850113539, 4.38492219, 0.662534357, 0.52068686,
              191.437234, 0.45616425, 2.3181548, 0.536892042, 0.745798575,
              170.64691, 0.182257642, 0.51721665, 0.684560009, 6.00586061,
              5.52869348e+01, 1.35400838e-01, 7.75693058e+00, 8.44078671e-01, 2.24554202e+00,
              1.84410906e+02, 8.35614195e-01, 3.53845264e+00, 3.61538332e-01, 3.71963333e+00,
              2.87502624e+01, 8.25932956e-01, 2.30504150e+00, 8.82225343e-02, 2.04503324e+00,
              85.27519470607159, 0.34705151936395495, 6.068544898846506, 0.8997885493449134, 5.056779166575173,
              169.17232208874125, 0.7118675405513857, 6.901967243076479, 0.4892178196162571, 3.8570475099774617,
              37.1324248182375, 0.2053655273130128, 5.61039340388344, 0.3963110846341643, 4.22155715781329,
              39.86747642,   0.37736407,   0.66926481,   0.52113547,   8.93671391,
              173.70769851,   0.3329326,    8.85341563,   0.23311253,   5.0473408,
              47.53902196,   0.95851189,   5.54664908,   0.4071068,    8.6902641]
  bo_policies.construct_good_policies(policies)
  cifar_trainer = CifarModelTrainer(hparams)
  cifar_trainer.run_model()

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
