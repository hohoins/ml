# 참고자료
# 모두를 위한 머신러닝/딥러닝 강의
# 홍콩과기대 김성훈
# http://hunkim.github.io/ml

import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# decent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
decent = W - tf.multiply(0.1, tf.reduce_mean( ((W * X) - Y) * X) )
update = W.assign(decent)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

arrStep = []
arrCost = []
arrWeight = []

for step in range(20):
    sess.run(update, feed_dict={X: x_data, Y: y_data})

    arrStep.append(step)
    arrCost.append(sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    arrWeight.append(sess.run(W))

    print(step, arrCost[step], arrWeight[step])

print('10', sess.run(hypothesis, feed_dict={X: 10}))
print('100', sess.run(hypothesis, feed_dict={X: 100}))

# print
plt.plot(arrStep, arrCost)
plt.xlabel('Step')
plt.ylabel('Cost')
plt.show()
