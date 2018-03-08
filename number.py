# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:28:37 2018
4-2 dropout使用

@author: ...
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 

data = input_data.read_data_sets("data",one_hot=True)

batch_size = 100
n_batch = data.train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

keep_alive = tf.placeholder(tf.float32)

#输入层
weight1 = tf.Variable(tf.truncated_normal([784,200]))
biases1 = tf.Variable(tf.truncated_normal([200]))
output1 = tf.nn.tanh(tf.matmul(x,weight1)+biases1)
output1_drop = tf.nn.dropout(output1,keep_alive)

#中间层
weight2 = tf.Variable(tf.random_normal([200,200]))
biases2 = tf.Variable(tf.truncated_normal([200]))
output2 = tf.nn.tanh(tf.matmul(output1_drop,weight2)+biases2)
output2_drop = tf.nn.dropout(output2,keep_alive)

weight3 = tf.Variable(tf.truncated_normal([200,100]))
biases3 = tf.Variable(tf.truncated_normal([100]))
output3 = tf.nn.tanh(tf.matmul(output2_drop,weight3)+biases3)
output3_drop = tf.nn.dropout(output3,keep_alive)

#输出层 10
weight4 = tf.Variable(tf.truncated_normal([100,10]))
biases4 = tf.Variable(tf.truncated_normal([10]))
output4 = tf.nn.softmax(tf.matmul(output3_drop,weight4)+biases4)

#交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output4))
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#n个测试数据正确或错误，list bool
result = tf.equal(tf.argmax(y,1),tf.argmax(output4,1))
accuracy = tf.reduce_mean(tf.cast(result,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #20次
    for epoch in range(31):
        #n_batch block
        for batch in range(n_batch):
            batch_x,batch_y = data.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_x,y:batch_y,keep_alive:1.0})
        print("epoch:",epoch,"accuracy_test:",sess.run(accuracy,feed_dict={x:data.test.images,y:data.test.labels,keep_alive:1.0}),
              "accuracy_train:",sess.run(accuracy,feed_dict={x:data.train.images,y:data.train.labels,keep_alive:1.0}))