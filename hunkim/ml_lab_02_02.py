# 참고자료
# 모두를 위한 머신러닝/딥러닝 강의
# 홍콩과기대 김성훈
# http://hunkim.github.io/ml

import tensorflow as tf

# train data
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# weight and bias
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# H(x) = Wx + b
hypothesis = W * X + b

# cost = 1/m sum(0~m) (H(x) - y)^2
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
a = tf.Variable(0.1) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train
for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if (step % 20 == 0):
        print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b))

# predic
print(sess.run(W), sess.run(b))
print(sess.run(hypothesis, feed_dict={X: 5}))
print(sess.run(hypothesis, feed_dict={X: 2.5}))
