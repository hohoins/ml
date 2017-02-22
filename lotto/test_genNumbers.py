import tensorflow as tf
import math
import numpy as np
import ast

str = open('test_input.data', 'r').read()
data = np.array(ast.literal_eval(str))
x_data = data[:, 0:1]
y_data = data[:, 1:]

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
learning_rate = 0.1
training_epochs = 201
display_step = 1

# tf Graph Input
X = tf.placeholder(tf.float32, [None, 1]) # mnist data image of shape 28*28=784
Y = tf.placeholder(tf.float32, [None, 45]) # 0-9 digits recognition => 10 classes

W1 = tf.get_variable("W1", shape=[1, 128], initializer=xavier_init(1, 128))
W2 = tf.get_variable("W2", shape=[128, 128], initializer=xavier_init(128, 128))
W3 = tf.get_variable("W3", shape=[128, 128], initializer=xavier_init(128, 128))
W4 = tf.get_variable("W4", shape=[128, 45], initializer=xavier_init(128, 45))
W5 = tf.get_variable("W5", shape=[45, 45], initializer=xavier_init(45, 45))
W6 = tf.get_variable("W6", shape=[45, 45], initializer=xavier_init(45, 45))
Wo = tf.get_variable("Wo", shape=[45, 45], initializer=xavier_init(45, 45))

b1 = tf.Variable(tf.random_normal([128]))
b2 = tf.Variable(tf.random_normal([128]))
b3 = tf.Variable(tf.random_normal([128]))
b4 = tf.Variable(tf.random_normal([45]))
b5 = tf.Variable(tf.random_normal([45]))
b6 = tf.Variable(tf.random_normal([45]))
bo = tf.Variable(tf.random_normal([45]))

# Construct model
L1 = tf.nn.relu(tf.matmul( X, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
hypothesis = tf.matmul(L6, Wo) + bo

# Minimize error using cross entropy
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(Y), logits=hypothesis))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=hypothesis))
# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

#
def getIdx(array, val):
  for i in range( len( array ) ):
      if array[i] == val:
          return i+1
  return -1

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})

        if (epoch) % display_step == 0:
            c = sess.run( cost, feed_dict={X: x_data, Y: y_data})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")
    print()

    steps = int(len(x_data) / 20)
    for i in range(1, len(x_data), steps):
        pred = sess.run(hypothesis, feed_dict={X: [[i]]})[0]
        sorted = np.sort(pred)
        print(i, ": ", getIdx(pred, sorted[44]))
        # print(i, ": ", getIdx(pred, sorted[44]), getIdx(pred, sorted[43]), getIdx(pred, sorted[42]), getIdx(pred, sorted[41]), getIdx(pred, sorted[39]), getIdx(pred, sorted[38]))
