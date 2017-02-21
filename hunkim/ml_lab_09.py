# 참고자료
# 모두를 위한 머신러닝/딥러닝 강의
# 홍콩과기대 김성훈
# http://hunkim.github.io/ml

# Import
import tensorflow as tf
import numpy as np

# Data
xy = np.loadtxt('ml_lab_09.txt', unpack=True)
x_data = xy[0:-1]
y_data = xy[-1]

# Variables
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# Hypothesis - sigmoid
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))

# Cost - cross entropy
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# Gradient descent algorithm
a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Initilization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train
for step in range(1000):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

# Test
correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X: x_data, Y: y_data}))
print("accuracy:", sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))

# Predict