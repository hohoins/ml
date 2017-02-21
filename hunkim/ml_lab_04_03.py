# 참고자료
# 모두를 위한 머신러닝/딥러닝 강의
# 홍콩과기대 김성훈
# http://hunkim.github.io/ml

import tensorflow as tf
import numpy as np

# data
xy = np.loadtxt('ml_lab_04.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

print('xy', xy)
print('x', x_data)
print('y', y_data)

# variable
W = tf.Variable(tf.random_uniform([1, 3], -1.0, 1.0))
hypothesis = tf.matmul( W, x_data)
cost = tf.reduce_mean(tf.square(hypothesis-y_data))

# Minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))