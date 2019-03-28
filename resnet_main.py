# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""ResNet Train/Eval module.
"""
import time
import six
import sys

import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')

def train(hps):
  """Training loop."""
  class_loss = []
  images1, labels1 = cifar_input.build_input(
          FLAGS.dataset, '/home/fuxianya/data/bin/train_batch', hps.batch_size, FLAGS.mode)
  model = resnet_model.ResNet(hps,  FLAGS.mode)
  model.build_graph(images1, labels1, True)
 
  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
  
  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

  truth = tf.argmax(labels1, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  precision_o = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
  loss_o = model.cost

  vars_ =  {}
  for v in tf.trainable_variables():
      #if v.name.find('bn')==-1:
      #  print(v.name)
      vars_[v.name] = v
  cost, pred, logits= model.forward_prob(images1, labels1, vars_, True)
  inner_grad = tf.gradients(cost, list(vars_.values()))
  inner_grad = [tf.stop_gradient(grad) for grad in inner_grad]
  inner_grad_dict = dict(zip(vars_.keys(), inner_grad))
  new_vars = dict(zip(vars_.keys(), [vars_[key] - model.lrn_rate*inner_grad_dict[key] for key in vars_.keys()]))

  class_preds = []
  costb = []
  for i in range(0,10):
      class_image, class_label = cifar_input.build_input(
              FLAGS.dataset, FLAGS.train_data_path+'_'+str(i) , hps.batch_size, FLAGS.mode)

      cost1, pred, _ = model.forward_prob(class_image,class_label, new_vars, True)
      costb.append(cost1)
      tmp = tf.argmax(class_label, axis=1)
      preds = tf.argmax(pred, axis=1)
      co_pred =  tf.reduce_mean(tf.to_float(tf.equal(preds, tmp)))
      class_preds.append(co_pred)
 
  meta_loss = tf.to_float(0.5)*tf.reduce_sum(costb)/tf.to_float(10)+tf.to_float(0.5)*loss_o
  #meta_loss = tf.reduce_mean(costb, 0, keep_dims=True)
  #meta_optimizer = tf.train.AdamOptimizer(model.lrn_rate).minimize(meta_loss, global_step=global_step)
  trainable_variables = tf.trainable_variables()
  grads = tf.gradients(meta_loss, trainable_variables)
  optimizer = tf.train.MomentumOptimizer(model.lrn_rate, 0.9)
  meta_train_op = optimizer.apply_gradients(
          zip(grads, trainable_variables),
          global_step=model.global_step, name='train_step')
  train_op = [meta_train_op]+model.extra_train_ops
  train_ops = tf.group(*train_op)
  total_accs = tf.reduce_sum(class_preds)/tf.to_float(10)

  '''
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
  config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())
  for i in range(2000):
      sess.run(meta_train_op)
      l, p, tl, tp = sess.run([loss_o, precision_o, meta_loss, total_accs])
      print('epoch:%d, loaa_0:%f, acc_0:%f, loss:%f, acc:%f'%(l,p,tl,tp))
      saver.save(sess, 'ckpt/model.ckpt', global_step=i+1)

  '''
  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.train_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('Precision', total_accs)]))

  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss_o': loss_o,
               'precision_o': precision_o,
               'total precision': total_accs,
               'total losses': meta_loss},
      every_n_iter=100)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.1

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results
      if train_step < 2000:
        self._lrn_rate = 0.1
      elif train_step < 4000:
        self._lrn_rate = 0.01
      elif train_step < 6000:
        self._lrn_rate = 0.001
      else:
        self._lrn_rate = 0.0001
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9) 
  epoch =0
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as mon_sess:
    while not mon_sess.should_stop() and epoch<4000:
        #mon_sess.run(meta_optimizer)
        #mon_sess.run([meta_loss, meta_train_op])
        #mon_sess.run(meta_train_op)
        mon_sess.run(train_ops)
        epoch = epoch+1

def evaluate(hps):
  """Eval loop."""
  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
  model = resnet_model.ResNet(hps,  FLAGS.mode)
  global_step = tf.train.get_or_create_global_step()
  model.build_graph(images, labels, True)
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  best_precision = 0.0
  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)
    
    vars_ =  {}
    for v in tf.trainable_variables():
        vars_[v.name] = v

    total_prediction, correct_prediction, total_preds = 0, 0, 0
    for _ in six.moves.range(FLAGS.eval_batch_count):
      cost, pred, logits= model.forward_prob(images, labels, vars_) 
      tmp = tf.argmax(labels, axis=1)
      preds = tf.argmax(pred, axis=1)
      co_pred =  tf.reduce_mean(tf.to_float(tf.equal(preds, tmp)))
      total_preds += co_pred

      (summaries, loss, predictions, truth, train_step, total_preds) = sess.run(
          [model.summaries, model.cost, model.predictions,
           labels, global_step, total_preds])

      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

    t_preds = total_preds / FLAGS.eval_batch_count
    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f, t preds: %.3f' %
                    (loss, precision, best_precision, t_preds))
    summary_writer.flush()

    if FLAGS.eval_once:
      break

    time.sleep(60)


def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  if FLAGS.mode == 'train':
    batch_size = 128
  elif FLAGS.mode == 'eval':
    batch_size = 100

  if FLAGS.dataset == 'cifar10':
    num_classes = 10
  elif FLAGS.dataset == 'cifar100':
    num_classes = 100

  hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')

  with tf.device(dev):
    if FLAGS.mode == 'train':
      train(hps)
    elif FLAGS.mode == 'eval':
      evaluate(hps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
