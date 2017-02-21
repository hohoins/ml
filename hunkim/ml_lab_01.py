# 참고자료
# 모두를 위한 머신러닝/딥러닝 강의
# 홍콩과기대 김성훈
# http://hunkim.github.io/ml

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

print(hello)

sess = tf.Session()

print(sess.run(hello))

print(sess.run(hello).decode(encoding='utf-8'))
