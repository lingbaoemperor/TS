# -*- coding: utf-8 -*-
"""
Created on Sat May 12 13:08:18 2018

@author: Jacob
"""

import tensorflow as tf
#import numpy as np

input = tf.Variable(tf.ones([1,3,3,3]))
filter = tf.Variable(tf.ones([1,1,3,1]))

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(op))
    