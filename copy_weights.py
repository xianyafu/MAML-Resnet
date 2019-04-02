# -*- coding:utf-8 -*-
#!/usr/bin/env python

'''
############################################################
rename tensorflow variable.
############################################################
'''

import tensorflow as tf
import argparse
import os
import re
import numpy as np
def get_parser():
    parser = argparse.ArgumentParser(description='parameters to rename tensorflow variable!')
    parser.add_argument('--ckpt_path', type=str, help='the ckpt file where to load.')
    parser.add_argument('--save_path', type=str, help='the ckpt file where to save.')
    args = parser.parse_args()
    return args

def load_model(model_path, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model_path)
    if (os.path.isfile(model_exp)):
        print('not support: %s' % model_exp)
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

    return saver

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    print(meta_file)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def rename(args):
    '''rename tensorflow variable, just for checkpoint file format.'''


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        #saver = load_model(args.ckpt_path)
        #graph = tf.get_default_graph()
        #test = graph.get_tensor_by_name('init/init_conv/DW:0')
        #print('-----------------')
        #print(test.shape)

        for var_name, _ in tf.contrib.framework.list_variables(args.ckpt_path):
            # Load the variable
            var = tf.contrib.framework.load_variable(args.ckpt_path, var_name)

            # Set the new name
            new_name = var_name

            if var_name.find('bn')==-1 and len(var.shape)==4:
              if var_name == 'init/init_conv/DW' or var_name == 'init/init_conv/DW/Momentum':
               print('===================')
               print(var.dtype)
               tmp = np.zeros([var.shape[0], var.shape[1], var.shape[2], var.shape[3]+3],  dtype=np.float32)
               for i in range(0, var.shape[0]):
                   for j in range(0, var.shape[1]):
                       for k in range(0, var.shape[2]):
                           for l in range(0, var.shape[3]+3):
                               if k<var.shape[2] and l<var.shape[3]:
                                   tmp[i][j][k][l] = var[i][j][k][l]
               # Rename the variable
               print(new_name)
               print(var.shape)
               print(tmp.shape)
               var = tf.Variable(tmp, name=new_name)

              else:
               tmp = np.zeros([var.shape[0], var.shape[1], var.shape[2]+3, var.shape[3]+3],  dtype=np.float32)
               for i in range(0, var.shape[0]):
                   for j in range(0, var.shape[1]):
                       for k in range(0, var.shape[2]+3):
                           for l in range(0, var.shape[3]+3):
                               if k<var.shape[2] and l<var.shape[3]:
                                   tmp[i][j][k][l] = var[i][j][k][l]
               # Rename the variable
               print(new_name)
               print(var.shape)
               print(tmp.shape)
               var = tf.Variable(tmp, name=new_name)
            #else:
            #    var = tf.Variable(var, name=new_name)
            #    print(var_name)
            #    print(var.shape)

        # Save the variables
        saver=tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()
        test = graph.get_tensor_by_name('init/init_conv/DW:0')
        print('-----------------')
        print(test.shape)
        saver.save(sess, args.save_path)

if __name__ == '__main__':
    args = get_parser()
    rename(args)
