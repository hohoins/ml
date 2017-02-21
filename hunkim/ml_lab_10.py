# 참고자료
# 모두를 위한 머신러닝/딥러닝 강의
# 홍콩과기대 김성훈
# http://hunkim.github.io/ml

from __future__ import print_function
import tensorflow as tf
import random as ran
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

def xavier_init(n_inputs, n_outputs, uniform=True):
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)

# Parameters
learning_rate = 0.0006
training_epochs = 60
batch_size = 512
display_step = 1

# tf Graph Input
X = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
Y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
dropout_rate = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[784, 512], initializer=xavier_init(784, 512))
W2 = tf.get_variable("W2", shape=[512, 512], initializer=xavier_init(512, 512))
W3 = tf.get_variable("W3", shape=[512, 512], initializer=xavier_init(512, 512))
W4 = tf.get_variable("W4", shape=[512, 256], initializer=xavier_init(512, 256))
W5 = tf.get_variable("W5", shape=[256, 256], initializer=xavier_init(256, 256))
W6 = tf.get_variable("W6", shape=[256, 256], initializer=xavier_init(256, 256))
W8 = tf.get_variable("W8", shape=[256, 10], initializer=xavier_init(256, 10))

b1 = tf.Variable(tf.random_normal([512]))
b2 = tf.Variable(tf.random_normal([512]))
b3 = tf.Variable(tf.random_normal([512]))
b4 = tf.Variable(tf.random_normal([256]))
b5 = tf.Variable(tf.random_normal([256]))
b6 = tf.Variable(tf.random_normal([256]))
b8 = tf.Variable(tf.random_normal([10]))

# Construct model
_L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(_L1, dropout_rate)

_L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(_L2, dropout_rate)

_L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(_L3, dropout_rate)

_L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(_L4, dropout_rate)

_L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
L5 = tf.nn.dropout(_L5, dropout_rate)

_L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
L6 = tf.nn.dropout(_L6, dropout_rate)

hypothesis = tf.matmul(L6, W8) + b8

# Minimize error using cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 0.7})
            # Compute average loss
            c = sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 0.7})
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_rate: 1}))


    r = ran.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Predication: ", sess.run(tf.argmax(hypothesis, 1), {X: mnist.test.images[r:r+1], dropout_rate: 1}))
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()


