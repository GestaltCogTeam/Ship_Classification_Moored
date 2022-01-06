# -*- coding: UTF-8 -*-


import tensorflow as tf


def sct_cal(x, weights, biases, keep_prob, n_hidden, seq_length, attention):

    rnn_variant = "GRU"

    x = tf.transpose(x, [1, 0, 2])  # transform position for different dimension

    GRUCell = tf.nn.rnn_cell.GRUCell(n_hidden)

    # GRUCell_1 = tf.nn.rnn_cell.GRUCell(n_hidden)
    # GRUCell_2 = tf.nn.rnn_cell.GRUCell(n_hidden)

    # GRUCell = tf.nn.rnn_cell.MultiRNNCell([GRUCell_1, GRUCell_2], state_is_tuple=True)

    # define dynamic rnn model
    GRUCcell = tf.nn.rnn_cell.DropoutWrapper(GRUCell, output_keep_prob=keep_prob)

    (outputs, states) = tf.nn.dynamic_rnn(GRUCell, x, time_major=True, dtype=tf.float32)

    new_outputs, attention_name = attention(outputs, attention_size=300, time_major=True, return_alphas=False)

    val = tf.matmul(new_outputs, weights['out'])+biases['out']

    train_details = rnn_variant + ' ' + attention_name

    return val, train_details
