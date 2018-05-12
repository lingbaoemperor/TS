import tensorflow as tf
import os
import pandas as pd
import numpy as np

def load_train_data(file_name, normalize=True):
    x_train, y_train = [], []
    with open(file_name) as my_file:
        header = my_file.readline()
        for line in my_file.readlines():
            line = line.strip().split(',')
            x_train.append(line[1:])
            y_train.append(int(line[0]))

    x_train = np.array(x_train).astype('float32')
    y_train = np.array(y_train)

    if normalize == True:
        x_train /= 255

    return x_train, y_train

def load_test_data(file_name, normalize=True):
    x_test = []
    with open(file_name) as my_file:
        header = my_file.readline()
        for line in my_file.readlines():
            line = line.strip().split(',')
            x_test.append(line)

    x_test = np.array(x_test).astype('float32')
    if normalize == True:
        x_test /= 255

    return x_test

def save_result(y_pred, file_name):
    result_df = pd.DataFrame({'ImageId': range(1, len(y_pred) + 1), 'Label': y_pred})
    result_df.to_csv(file_name, index=False)


train_file = os.path.join('digit', 'train.csv')
test_file = os.path.join('digit', 'test.csv')
x_train, y_train = load_train_data(train_file)
x_test = load_test_data(test_file)
print(y_train)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

h1_W = tf.Variable(tf.random_normal([784, 256]))
h1_b = tf.Variable(tf.random_normal([256]))

h2_W = tf.Variable(tf.random_normal([256, 100]))
h2_b = tf.Variable(tf.random_normal([100]))

out_W = tf.Variable(tf.random_normal([100, 10]))
out_b = tf.Variable(tf.random_normal([10]))

hidden_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, h1_W), h1_b))
hidden_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_1, h2_W), h2_b))
y_ = tf.matmul(hidden_layer_2, out_W) + out_b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

nb_epoch = 10
batch_size = 1024
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(nb_epoch):
        avg_cost = 0.
        total_batch = int(len(x_train) / batch_size)

        for i in range(total_batch):
            batch_xs = x_train[i*batch_size:(i+1)*batch_size]
            batch_ys = y_train[i*batch_size:(i+1)*batch_size]

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})

        print('epoch: %d, cost: %.9f' % (epoch+1, avg_cost))

    y_pred = sess.run(y_, {x: x_test})

    y_pred = np.argmax(y_pred, axis=-1)
    save_file = os.path.join('digit', 'mlp_tensordlow.csv')
    save_result(y_pred, save_file)