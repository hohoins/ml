import tensorflow as tf
import numpy as np

char_rdic = ['h', 'e', 'l', 'o']
char_dic = {w:i for i, w in enumerate(char_rdic)}
x_data = np.array([[1.0,0,0,0], # h
                  [0,1,0,0], # e
                  [0,0,1,0], # l
                  [0,0,1,0]], # l
                  dtype='f')

sample = [char_dic[c] for c in "hello"] # to index


# configuration
char_vocab_size = len(char_dic)
rnn_size = char_vocab_size # 1 hot coding (one of 4)
time_step_size = 4 # 'hell' -> predict 'ello'
batch_size = 1 # one sample


# RNN model
rnn_cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, rnn_cell.state_size])

# tf.split(split_dim, num_splits, value) –>
# tf.split(value, num_or_size_splits, axis)
X_split = tf.split(x_data, time_step_size, axis=0)

# outputs, state = tf.nn.rnn(rnn_cell, X_split, state)
outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, X_split, state)
#outputs, states = tf.contrib.rnn.static_rnn(lstm_cell_1, [tf.constant([[1.0],[1.0]], dtype=tf.float32)], dtype=tf.float32)


# logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols].
# targets: list of 1D batch-sized int32 Tensors of the same length as logits.
# weights: list of 1D batch-sized float-Tensors of the same length as logits.

"""
logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
https://www.tensorflow.org/install/migration
tf.concat: keyword argument concat_dim should be renamed to axis
arguments have been reordered to tf.concat(values, axis, name='concat').
tf 1.0에서 concat의 인자 순서가 바뀜 조심해야함
"""
logits = tf.reshape(tf.concat(outputs, 1), [-1, rnn_size])

targets = tf.reshape(sample[1:],[-1])
weights = tf.ones([time_step_size * batch_size])

# loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
# loss = tf.contrib.rnn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)


# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.arg_max(logits, 1))
        print (result, [char_rdic[t] for t in result])
