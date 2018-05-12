# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 09:37:50 2018

@author: Jacob
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
file_name = os.path.join('data', 'ex2data1.txt')
data = pd.read_table(file_name, sep=',', header=None, quoting=3)

print(data.shape)

rate = 0.001
epoch = 1000
train_x = np.array(data.iloc[:, 0:2])
train_y = np.array(data.iloc[:, 2:3])


scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)

original_x = np.array(data)
xx = train_x[original_x[:,2] == 1]
yy = train_x[original_x[:,2] == 0]

#print(train_x)

x = tf.placeholder("float", [None, 2])
y = tf.placeholder("float", [None, 1])

w = tf.Variable(tf.random_uniform([2, 1], -1., 1.))
b = tf.Variable(tf.zeros([1])) # theta_0

output = tf.matmul(x,w)+b
result = tf.nn.sigmoid(output)

loss = tf.reduce_mean(- y * tf.log(result) - (1 - y) * tf.log(1 - result))

optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #print('w start is ', sess.run(w))  
    #print('b start is ', sess.run(b))
    for index in range(epoch):   
        sess.run(optimizer, {x: train_x, y: train_y}) 
    
        if index % 10 == 0:
            print('w is', sess.run(w), ' b is', sess.run(b), ' loss is', sess.run(loss, {x: train_x, y: train_y}))
    
    print('loss is ', sess.run(loss, {x: train_x, y: train_y})) 
    print('w end is ',sess.run(w))
    print('b end is ',sess.run(b)) 
    print('reult is ', sess.run(result, {x: [[0.25, 0.25]]}))
    Y = sess.run(output,{x:train_x})
    print(Y)
    plt.figure(1)
    plt.scatter(xx[:,0],xx[:,1],c='r')
    plt.scatter(yy[:,0],yy[:,1])
    plt.plot(train_x[:,0],Y,'g-',lw=3)
    plt.show()
    
