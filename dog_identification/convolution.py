# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:33:39 2018

@author: Jacob
"""

import tensorflow as tf
import numpy as np
import pandas as pd

img_path = './data/labels.csv'
#批次
batch_count = 0
batch_size = 20
#遍历数据集次数
epoch_num = 10
#图像大小
resize_w = 128
resize_h = 128
label_size = 0


#一开始是文件名字对应字符串种类，一共120种
#把字符串转换成数字，保存一次（file_name_to_number），后面可能有用
#读取文件名和种类对应到两个列表
def get_file_list():
    global batch_size
    global batch_count
    global label_size
    file_list = list()
    label_list = list()
    data = pd.read_csv('./data/labels.csv')
    #去重复，取种类标签·
    labels = data.drop_duplicates(['breed'],keep='first')
    labels = labels.iloc[:,1].values
    #dataframe用编号替换字符串种类
    data['breed'].replace(labels,np.arange(120),inplace=True)
    data.to_csv('./data/file_name_to_number.csv',index=False)
    file_list = data.iloc[:,0].values
    label_list = data.iloc[:,1].values
    print("文件列表长和标签长：",len(file_list),len(label_list))
    #批次数量为总数量除以一个批次大小
    batch_count = len(label_list)//batch_size+1
    return file_list,label_list

#创建文件名队列
def crete_queue(file_list,label_list):
    global batch_size
    global resize_w
    global resize_h
    #转换成tf.string
    file_list = tf.cast(file_list,tf.string)
    label_list = tf.one_hot(label_list,120)
#   创建输入队列，文件名-标签
    queue = tf.train.slice_input_producer([file_list,label_list],num_epochs=None,shuffle=False)
    return queue
#    数据和标签
    file_list = queue[0]
    label_list = queue[1]
#   加载数据，这里不是一次全部加载，一次加载batch_size个
    images = tf.read_file(file_list)
#   解码
    data = tf.image.decode_jpeg(images,channels=3)
#   调整大小
    data = tf.image.resize_image_with_crop_or_pad(data,resize_h,resize_w)
#   归一化
    data = tf.image.per_image_standardization(data)
#   每次batch_size张
    image_batch,label_batch = tf.train.batch([data,label_list],batch_size=batch_size,num_threads=2)
#   要转成float才能训练
    image_batch = tf.cast(image_batch,tf.float32)
#   返回一个批次的图片数据和对应标签
    return image_batch,label_batch

#模型参数定义
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

if __name__ == '__main__':
    [file_list,label_list] = get_file_list()
    queue = crete_queue(file_list,label_list)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #填充文件队列线程
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess,coord)
        lb = sess.run(queue)
        print(lb)
        coord.request_stop()
        coord.join(thread)