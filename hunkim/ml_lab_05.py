# 참고자료
# 모두를 위한 머신러닝/딥러닝 강의
# 홍콩과기대 김성훈
# http://hunkim.github.io/ml

import tensorflow as tf
import numpy as np

# Data
xy = np.loadtxt('ml_lab_05.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# variable
W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))

# Hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1.0, 1.0 + tf.exp(-h))

# Cost
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# Gradient descent algorithm
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

# predict
print(sess.run(hypothesis, feed_dict={X: [[1], [2], [2]]}) > 0.5)
print(sess.run(hypothesis, feed_dict={X: [[1], [5], [5]]}) > 0.5)
print(sess.run(hypothesis, feed_dict={X: [[1, 1], [4, 3], [3, 5]]}) > 0.5)
