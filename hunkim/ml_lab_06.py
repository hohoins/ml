# 참고자료
# 모두를 위한 머신러닝/딥러닝 강의
# 홍콩과기대 김성훈
# http://hunkim.github.io/ml

# Import
import tensorflow as tf
import numpy as np

# Data
xy = np.loadtxt('ml_lab_06.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

# Variables
X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.zeros([3, 3]))
learning_rate = 0.001

# Hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W))

# Cost
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

# Gradient descent algorithm
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

# Initilization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print("\n", step, "cost: ", sess.run(cost, feed_dict={X: x_data, Y: y_data}), "\n", sess.run(W))

# Predict
a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
print("\n", a, sess.run(tf.arg_max(a, 1)))

a = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
print("\n", a, sess.run(tf.arg_max(a, 1)))

a = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
print("\n", a, sess.run(tf.arg_max(a, 1)))

a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
print("\n", a, sess.run(tf.arg_max(a, 1)))
