# 참고자료
# 모두를 위한 머신러닝/딥러닝 강의
# 홍콩과기대 김성훈
# http://hunkim.github.io/ml

# Import
import tensorflow as tf
import numpy as np

# Data
xy = np.loadtxt('ml_lab_09.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.transpose(xy[2:3])

# Variables
X = tf.placeholder(tf.float32, [None, 2], name='X-input')
Y = tf.placeholder(tf.float32, [None, 1], name='Y-input')

W1 = tf.Variable(tf.random_uniform([2, 5], -0.5, 0.5), name='Weight1')
W2 = tf.Variable(tf.random_uniform([5, 4], -0.5, 0.5), name='Weight2')
W3 = tf.Variable(tf.random_uniform([4, 1], -0.5, 0.5), name='Weight3')
w1_hist = tf.summary.histogram('Weight1', W1)
w2_hist = tf.summary.histogram('Weight2', W2)
w3_hist = tf.summary.histogram('Weight3', W3)

b1 = tf.Variable(tf.zeros([5]), name='Bias1')
b2 = tf.Variable(tf.zeros([4]), name='Bias2')
b3 = tf.Variable(tf.zeros([1]), name='Bias3')
b1_hist = tf.summary.histogram('Bias1', b1)
b2_hist = tf.summary.histogram('Bias2', b2)
b3_hist = tf.summary.histogram('Bias3', b3)

a = tf.Variable(0.2, name="rate")

# Hypothesis
with tf.name_scope('layer2') as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
with tf.name_scope('layer3') as scope:
    L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
with tf.name_scope('layer4') as scope:
    hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)

# Cost - cross entropy
with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1.0-hypothesis))
    tf.summary.scalar('cost', cost)

# Gradient descent algorithm
with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

# Initilization
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/xor_logs', sess.graph)


    # Train
    for step in range(50001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 5000 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
            summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)

    # Test
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X: x_data, Y: y_data}))
    print("\n accuracy:", sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))