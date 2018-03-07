# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 20:05:35 2018

@author: Jacob
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#[200,1]
x1 = np.linspace(-0.5,0.5,100)[:,np.newaxis]
noise = np.random.normal(0,0.02,x1.shape)
y1 = np.square(x1)+noise

x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

weight1 = tf.Variable(tf.random_normal([1,10]))
biases1 = tf.Variable(tf.zeros([1,10]))

#中间层
result_L1 = tf.matmul(x,weight1)+biases1
output_L1 = tf.nn.tanh(result_L1)

weight2 = tf.Variable(tf.random_normal([10,1]))
biases2 = tf.Variable(tf.zeros([1,1]))
result_L2 = tf.matmul(output_L1,weight2)+biases2
output_L2 = tf.nn.tanh(result_L2)

loss = tf.reduce_mean(tf.square(y-output_L2))
step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        sess.run(step,feed_dict={x:x1,y:y1})
    
    result = sess.run(output_L2,feed_dict={x:x1})
    plt.figure()
    plt.scatter(x1,y1)
    plt.plot(x1,result,'r-',lw=3)
    plt.show()