import tensorflow as tf
import numpy as np
    
file_list = ['img1','img2','img3','img4','img5']
label_list = [0,1,2,3,4]
queue = tf.train.slice_input_producer([file_list,label_list],num_epochs=None,shuffle=False)
data = queue[0]
labels = queue[1]
image_batch,label_batch = tf.train.batch([data,labels],batch_size=5,num_threads=3)
#a = tf.Variable(10,dtype=tf.int32)
#b = a*10
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'./net/my-test.ckpt')
    coor = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess,coord=coor)
    try:
        for i in range(1):
            im,la = sess.run([image_batch,label_batch])
            print(im,la)
    except tf.errors.OutOfRangeError:
        print("Error!!!")
    coor.request_stop()
    coor.join(thread)