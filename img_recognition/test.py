# -*- coding: utf-8 -*-
"""
Created on Thu May 10 20:15:46 2018

@author: Jacob
"""
import os
import tensorflow as tf
import numpy as np

path = './img/'
batch_count = 0
batch_size = 20
epoch_num = 10
resize_w = 128
resize_h = 128
label_size = 2

def get_filelist(path):
    #文件名与类型对应,即一个文件夹下的全图图片种类是该文件夹名字
    global batch_size
    global batch_count
    global label_size
    file_list = list()
    label_list = list()
    count = 0
    for label in os.listdir(path):
        single = [0 for i in range(label_size)]
        single[count] = 1
        for file in os.listdir(path+label):
            file_list.append(path+label+'/'+file)
            label_list.append(single)
        count += 1
    file_list = np.array(file_list)[:,np.newaxis]
    label_list = np.array(label_list)
    res = np.hstack([file_list,label_list])
    np.random.shuffle(res)
    file_list = res[:,0]
    label_list = res[:,1:]
    label_list = label_list.astype(np.int32)
    batch_count = len(label_list)//batch_size+1
    return file_list,label_list

#创建文件名队列
def load_data(files,labels):
    global batch_size
    global resize_w
    global resize_h
    file_list = tf.cast(files,tf.string)
    label_list = tf.cast(labels,tf.int32)
    #创建输入队列，文件名-标签
    queue = tf.train.slice_input_producer([file_list,label_list],num_epochs=None,shuffle=False)
    #数据和标签
    files = queue[0]
    labels = queue[1]
    #加载数据，这里不是一次全部加载
    image = tf.read_file(files)
    #解码
    data = tf.image.decode_jpeg(image,channels=3)
    #大小
    data = tf.image.resize_image_with_crop_or_pad(data,resize_h,resize_w)
    data = tf.image.per_image_standardization(data)
    
    #每次batch_size张
    image_batch,label_batch = tf.train.batch([data,labels],batch_size=batch_size,num_threads=3)
    #要转成float才能训练
#    image_batch = tf.cast(image_batch,tf.float32)
    return image_batch,label_batch

#模型参数
def parameters():
    #128*128 64*64 32*32 16*16
    w = [tf.Variable(tf.truncated_normal([5,5,3,16],stddev=0.1)),
         tf.Variable(tf.truncated_normal([5,5,16,32],stddev=0.1)),
         tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1)),
         tf.Variable(tf.truncated_normal([16*16*64,128],stddev=0.1)),
         tf.Variable(tf.truncated_normal([128,2],stddev=0.1))]
    
    b = [tf.Variable(tf.constant(0.1,shape=[16])),
         tf.Variable(tf.constant(0.1,shape=[32])),
         tf.Variable(tf.constant(0.1,shape=[64])),
         tf.Variable(tf.constant(0.1,shape=[128])),
         tf.Variable(tf.constant(0.1,shape=[2]))]
    return w,b

#模型
def model(train_data,labels,w,b):
    res1_conv = tf.nn.relu(tf.nn.conv2d(train_data,w[0],[1,1,1,1],padding='SAME')+b[0])
    pool1 = tf.nn.max_pool(res1_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  #64*64*16
    norm1 = tf.nn.lrn(pool1,depth_radius = 4,bias = 1,alpha = 0.001/9.0,beta = 0.75)
    
    res2_conv = tf.nn.relu(tf.nn.conv2d(norm1,w[1],[1,1,1,1],padding='SAME')+b[1])
    pool2 = tf.nn.max_pool(res2_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  #32*32*32
    norm2 = tf.nn.lrn(pool2,depth_radius = 4,bias = 1,alpha = 0.001/9.0,beta = 0.75)
    
    res3_conv = tf.nn.relu(tf.nn.conv2d(norm2,w[2],[1,1,1,1],padding='SAME')+b[2])  #32*32*64
    pool3 = tf.nn.max_pool(res3_conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  #16*16*64
    norm3 = tf.nn.lrn(pool3,depth_radius = 4,bias = 1,alpha = 0.001/9.0,beta = 0.75)
    
    pool3 = tf.reshape(norm3,[-1,16*16*64])
    res1 = tf.nn.relu(tf.matmul(pool3,w[3])+b[3])
    
#    keep_alive = tf.placeholder(tf.float32)
#    output1_drop = tf.nn.dropout(pool3,keep_alive)
    
    output = tf.nn.softmax(tf.matmul(res1,w[4])+b[4])
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=output))
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
#    #准确率
    result = tf.equal(tf.argmax(labels,1),tf.argmax(output,1))
    accuracy = tf.reduce_mean(tf.cast(result,tf.float32))
    
    return loss,train,accuracy

    
files,labels = get_filelist(path)
image_batch,label_batch = load_data(files,labels)
w,b = parameters()
loss,train,accuracy = model(image_batch,label_batch,w,b)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #填充文件队列线程
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess,coord)
    try:
        for i in range(epoch_num):
            print('epoch:',i)
            for i in range(batch_count):
                sess.run(train)
            lo,acc = sess.run([loss,accuracy])
            print("loss:",lo,"acc:",acc)
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    coord.request_stop()
    coord.join(thread)