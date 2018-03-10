# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:28:37 2018
6-2 卷积

@author: ...
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 

data = input_data.read_data_sets("data",one_hot=True)

batch_size = 100
n_batch = data.train.num_examples // batch_size

#28*28=784
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

x_img = tf.reshape(x,[-1,28,28,1])

#卷积层1,权值，偏置
weight1_conv = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
biases1_conv = tf.Variable(tf.constant(0.1,shape=[32]))
#input'S shape : [batch,height,width,channels]
#filter's shape : [height,width,in_channels,out_channels]
#strides[0] = stride[3] = 1 , stride[1]:x stride , stride[2]:y stride
#padding : SAME or VALID
res1_conv = tf.nn.relu(tf.nn.conv2d(x_img,weight1_conv,[1,1,1,1],'SAME')+biases1_conv)  #28*28*32
#参数同上卷积类似
#池化,窗口2*2,步长2
pool1 = tf.nn.max_pool(res1_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')   #14*14*32

#卷积层2
weight2_conv = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
biases2_conv = tf.Variable(tf.constant(0.1,shape=[64]))
res2_conv = tf.nn.relu(tf.nn.conv2d(pool1,weight2_conv,[1,1,1,1],'SAME')+biases2_conv)   #14*14*64
#池化
pool2 = tf.nn.max_pool(res2_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  #7*7*64个神经元

#全连接层1
weight1 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1,shape=[1024]))
#pool2降维
pool2_flat = tf.reshape(pool2,[-1,7*7*64])
output1 = tf.nn.relu(tf.matmul(pool2_flat,weight1)+biases1)

#dropout
keep_alive = tf.placeholder(tf.float32)
output1_drop = tf.nn.dropout(output1,keep_alive)

#全连接层2
weight2 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1,shape=[10]))
output2 = tf.nn.softmax(tf.matmul(output1_drop,weight2)+biases2)

#lr = tf.Variable(0.001,dtype=tf.float32)

#交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output2))
#优化器
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

#n个测试数据正确或错误，list bool
result = tf.equal(tf.argmax(y,1),tf.argmax(output2,1))
accuracy = tf.reduce_mean(tf.cast(result,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #20次
    for epoch in range(10):
#        sess.run(tf.assign(lr,0.001*0.95**epoch))
        #n_batch block
        for batch in range(n_batch):
            batch_x,batch_y = data.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_x,y:batch_y,keep_alive:0.7})
#        l_r = sess.run(lr)
        print("epoch:",epoch,"accuracy_test:",sess.run(accuracy,feed_dict={x:data.test.images,y:data.test.labels,keep_alive:1.0}))