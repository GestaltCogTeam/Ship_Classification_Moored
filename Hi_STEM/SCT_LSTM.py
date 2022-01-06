# -*- coding: UTF-8 -*-

import tensorflow as tf


def sct_cal(x, weights, biases, keep_prob, n_hidden, seq_length, attention):

    rnn_variant = "LSTM"

    x = tf.transpose(x, [1, 0, 2])  # transform position for different dimension

    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

    # define dynamic rnn model
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    (outputs, states) = tf.nn.dynamic_rnn(lstm_cell, x, time_major=True, dtype=tf.float32)

    new_outputs, attention_name = attention(outputs, attention_size=300, time_major=True, return_alphas=False)

    val = tf.matmul(new_outputs, weights['out'])+biases['out']

    train_details = rnn_variant + ' ' + attention_name

    return val, train_details

