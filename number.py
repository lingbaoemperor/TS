# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:28:37 2018
5-1 优化器

@author: ...
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 

data = input_data.read_data_sets("data",one_hot=True)

batch_size = 100
n_batch = data.train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#dropout
keep_alive = tf.placeholder(tf.float32)

lr = tf.Variable(0.001,dtype=tf.float32)

weight1 = tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
biases1 = tf.Variable(tf.zeros([500])+0.1)
output1 = tf.nn.tanh(tf.matmul(x,weight1)+biases1)
output1_drop = tf.nn.dropout(output1,keep_alive)

weight2 = tf.Variable(tf.random_normal([500,300],stddev=0.1))
biases2 = tf.Variable(tf.zeros([300])+0.1)
output2 = tf.nn.tanh(tf.matmul(output1_drop,weight2)+biases2)
output2_drop = tf.nn.dropout(output2,keep_alive)

weight3 = tf.Variable(tf.truncated_normal([300,10],stddev=0.1))
biases3 = tf.Variable(tf.zeros([10])+0.1)
output3 = tf.nn.tanh(tf.matmul(output2_drop,weight3)+biases3)
output3_drop = tf.nn.dropout(output3,keep_alive)

#交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output3))
#优化器
train = tf.train.AdamOptimizer(lr).minimize(loss)

#n个测试数据正确或错误，list bool
result = tf.equal(tf.argmax(y,1),tf.argmax(output3,1))
accuracy = tf.reduce_mean(tf.cast(result,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #20次
    for epoch in range(20):
        sess.run(tf.assign(lr,0.001*0.95**epoch))
        #n_batch block
        for batch in range(n_batch):
            batch_x,batch_y = data.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_x,y:batch_y,keep_alive:1.0})
        l_r = sess.run(lr)
        print("epoch:",epoch,"accuracy_test:",sess.run(accuracy,feed_dict={x:data.test.images,y:data.test.labels,keep_alive:1.0}),
              "accuracy_train:",sess.run(accuracy,feed_dict={x:data.train.images,y:data.train.labels,keep_alive:1.0}),
              "lr:",l_r)