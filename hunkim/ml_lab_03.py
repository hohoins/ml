# 참고자료
# 모두를 위한 머신러닝/딥러닝 강의
# 홍콩과기대 김성훈
# http://hunkim.github.io/ml

import tensorflow as tf
import matplotlib.pyplot as plt

X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_sample = len(X)

W = tf.placeholder(tf.float32)

hypothesis = tf.multiply(X, W)

cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2))/(m)

init = tf.global_variables_initializer()

W_val = []
coat_val = []

sess = tf.Session()
sess.run(init)

for i in range(-30, 50):
    print(i*0.1, sess.run(cost, feed_dict={W: i*0.1}))
    W_val.append(i*0.1)
    coat_val.append(sess.run(cost, feed_dict={W: i*0.1}))

# print
plt.plot(W_val, coat_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()