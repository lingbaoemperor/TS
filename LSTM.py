# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:18:50 2018

LSTM 7-2
@author: Jacob
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

data = input_data.read_data_sets('data',one_hot=True)

n_input = 28 #每次一行
max_time = 28   #28个序列
lstm_size = 100 #隐层单元 100个block
n_class = 10
batch_size = 50 #每次训练50个样本
n_batch = data.train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#输入权值
weight = tf.Variable(tf.truncated_normal([lstm_size,10],stddev=0.1))
biases = tf.Variable(tf.constant(0.1,shape=[10]))


#lstm_size个block
lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)

#inputs : [batch_size,max_time,n_input]
inputs = tf.reshape(x,[-1,max_time,n_input])
#final_state最后的输出（第28次）
#final_state[0] cell state (内部输出)
#final_state[1] hidden_state (block输出)
#final_state[state,batch_size,final_state]
#outputs:全部过程的输出
#if time_major == False
#outputs : [batch_size,max_time,cell.output_size] 50 28 28
#if time_major == True
#outputs : [max_time,batch_size,cell.output_size] 50 28 28

outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
results = tf.nn.softmax(tf.matmul(final_state[1],weight)+biases)

#代价函数,交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=results,labels=y))

train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


#预测结果列表
correct_list = tf.equal(tf.arg_max(y,1),tf.arg_max(results,1))
#统计求正确率
accuracy = tf.reduce_mean(tf.cast(correct_list,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(7):
        for batch in range(n_batch):
            batch_x,batch_y = data.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_x,y:batch_y})
        print("epoch:",epoch,"accuracy:",str(sess.run(accuracy,feed_dict={x:data.test.images,y:data.test.labels})))