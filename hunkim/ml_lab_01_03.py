# 참고자료
# 모두를 위한 머신러닝/딥러닝 강의
# 홍콩과기대 김성훈
# http://hunkim.github.io/ml

import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
# mul = tf.mul(a, b) # tensotflow 1.0 미만
mul = tf.multiply(a, b) # tensotflow 1.0 이상

with tf.Session() as sess:
    print("+ with var: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("* with var: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))
