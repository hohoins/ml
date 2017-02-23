import tensorflow as tf
import math
import numpy as np


# http://archive.ics.uci.edu/ml/datasets/Iris
# load train data set
input = np.loadtxt('iris_train.data', unpack=True, dtype='float')
data = np.transpose(input)
x_data = data[:, 0:4]
y_data = data[:, 4:]

# load test data set
input = np.loadtxt('iris_test.data', unpack=True, dtype='float')
data = np.transpose(input)
x_test = data[:, 0:4]
y_test = data[:, 4:]

def xavier_init(n_inputs, n_outputs, uniform=True):
  if uniform:
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)

# parameters
features = 4
labels = 3
wide = 26
deep = 3
learning_rate = 0.01
training_epochs = 50001
display_step = 2000

# dropout
dropout_rate = tf.placeholder(tf.float32)
DROPOUT_TRAIN = 0.9
DROPOUT_TEST = 1.0

# DNN
def makeDNN(hidden_layer):
    # input from X
    prevLayer = X

    # make layers
    for i in range(hidden_layer):
        if i==0:
            newWeight = tf.get_variable("W0%d" % i, shape=[features, wide], initializer=tf.contrib.layers.xavier_initializer())
        else:
            newWeight = tf.get_variable("W0%d" % i, shape=[wide, wide], initializer=tf.contrib.layers.xavier_initializer())
        newBias = tf.Variable(tf.random_normal([wide]))
        newLayer = tf.nn.relu(tf.matmul(prevLayer, newWeight) + newBias)
        newDropLayer = tf.nn.dropout(newLayer, dropout_rate)
        prevLayer = newDropLayer

    # make output layers
    Wo = tf.get_variable("Wo", shape=[wide, labels], initializer=tf.contrib.layers.xavier_initializer())
    bo = tf.Variable(tf.random_normal([labels]))
    return tf.matmul(prevLayer, Wo) + bo

# tf Graph Input
X = tf.placeholder(tf.float32, [None, features])
Y = tf.placeholder(tf.float32, [None, labels])

# hypothesis
hypothesis = makeDNN(deep)

# minimize error
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.nn.softmax(Y), logits=hypothesis))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=hypothesis))
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=hypothesis))
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data, dropout_rate: DROPOUT_TRAIN})

        if (epoch) % display_step == 0:
            c = sess.run( cost, feed_dict={X: x_data, Y: y_data, dropout_rate: DROPOUT_TRAIN})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

    print("*************************")
    print("* Optimization Finished *")
    print("*************************")

    # Test
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: x_test, Y: y_test, dropout_rate: DROPOUT_TEST}))
