import tensorflow as tf
import numpy as np

def generate_data():
#    num = 8
    label = [1,2,3,4,5,6,7,8]
    
    images = [11,22,33,44,55,66,77,88]
    return label, images

def get_batch_data():
    label, images = generate_data()
    images = tf.cast(images, tf.float32)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([images, label], shuffle=True,num_epochs=None)
#    queue = input_queue[1]
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=2)
#    image_batch = tf.cast(image_batch,tf.float32)  
    return image_batch, label_batch,input_queue

#image_batch, label_batch,queue = get_batch_data()
#print(type(image_batch),type(label_batch))
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(sess, coord)
#    try:
#        print(sess.run(queue))
#        for i in range(1):  # 每一轮迭代 
#            print('*********')
#            for j in range(5):
#                image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
#                print(image_batch_v, label_batch_v,type(image_batch_v))
#    except tf.errors.OutOfRangeError:
#        print("done")
#    finally:
#        coord.request_stop()
#    coord.join(threads)
    
x = tf.Variable([1,1],dtype=tf.float32)
y = tf.nn.softmax(x)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))