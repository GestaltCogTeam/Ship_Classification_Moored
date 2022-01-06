# -*- coding: UTF-8 -*-


import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans,SpectralClustering
import datetime
from collections import defaultdict

import matplotlib as plt

from sklearn.manifold import TSNE

from Hi_STEM.data_read import DataReader
import matplotlib.pyplot as plt
from Hi_STEM.MacroF1 import MacroF1
from Hi_STEM.attention import attention

def RNN(x, weights, biases, keep_prob, n_hidden, seq_length):

    # print("原始x维度")
    # print(x.shape)
    x = tf.transpose(x, [1, 0, 2])  # transform position for different dimension

    # print("time major 后x的维度")
    # print(x.shape)
    # forward, state_is_tuple=True, return c_state and m_state
    fw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # # use dropout
    fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell, output_keep_prob=keep_prob)  # use dropout
    bw_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell, output_keep_prob=keep_prob)
    #
    # # bidirectional LSTM
    cell_fw = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell], state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell], state_is_tuple=True)
    # # dynamic rnn
    (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, dtype=tf.float32, time_major=True, sequence_length=seq_length)  # ,dtype=tf.float32,time_major=True  ,initial_state_fw=istate_fw,initial_state_bw=istate_bw
    new_outputs = tf.concat(outputs, 2)[-1]
    # new_outputs = attention(outputs, attention_size=300, time_major=True, return_alphas=False)

# LSTM
#     lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
# #
# #     # define dynamic rnn model
#     lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
#     (outputs, states) = tf.nn.dynamic_rnn(lstm_cell, x, time_major=True, dtype=tf.float32)
#     new_outputs =outputs[-1]




# GRU
#     GRUCell = tf.nn.rnn_cell.GRUCell(n_hidden)
#
#     # GRUCell_1 = tf.nn.rnn_cell.GRUCell(n_hidden)
#     # GRUCell_2 = tf.nn.rnn_cell.GRUCell(n_hidden)
#
#     # GRUCell = tf.nn.rnn_cell.MultiRNNCell([GRUCell_1, GRUCell_2], state_is_tuple=True)
#
#     # define dynamic rnn model
#     GRUCell = tf.nn.rnn_cell.DropoutWrapper(GRUCell, output_keep_prob=keep_prob)
#
#     (outputs, states) = tf.nn.dynamic_rnn(GRUCell, x, time_major=True, dtype=tf.float32)
#     new_outputs =outputs[-1]
    # print("双向lstm outputs的维度")
    # # print(size(outputs))
    # ???
    # new_outputs = tf.concat(outputs, 2)[0]

    # new_outputs = attention(outputs, attention_size=300, time_major=True, return_alphas=False)

    # print("拼接output后的维度")
    # print(new_outputs.shape)
    # # matrix multiplication
    #
    # print('三个参数维度分别为')
    # print(new_outputs[-1].shape)
    # print(weights['out'].shape)
    # print(biases['out'].shape)

    val = tf.matmul(new_outputs, weights['out'])+biases['out']
    #val=tf.add(tf.add(tf.matmul(fw_output[-1],weights['out']),tf.matmul(fb_output[-1],weights['out'])), biases['out'])

    # print("rnn计算值维度")
    # print(val.shape)
    return val


class Hi_STEM:
    def __init__(self, learning_rate, batch_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.keep_prob = tf.placeholder(tf.float32)  # rate = 1 - keep_prob

        self.n_input = 250  # embedding size
        self.n_hidden = 300  # hidden size

        self.n_classes = 4  # class number

        self.x = tf.placeholder("float", [self.batch_size, None, self.n_input])  #

        self.y_out = tf.placeholder("float", [self.batch_size, self.n_classes])

        self.weights = {
            'out': tf.Variable(tf.random_normal([2 * self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
        self.seq_length = tf.placeholder(tf.int32, [None])
        self.pred = RNN(self.x, self.weights, self.biases, self.keep_prob, self.n_hidden, self.seq_length)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y_out))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)  #
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y_out, 1))  # 1
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.sess = tf.Session()

        self.reader = DataReader()

    def save(self, model_name):
        saver = tf.train.Saver()
        saver.save(self.sess, model_name)

    def restore(self, model_name):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_name)

    def init_model(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def cluster(self):
        X_input = []
        while True:
            # x_in, y_in, seq_len = self.reader.read_train_data_mixup(batch_size)
            # x_in, y_in, seq_len = self.reader.read_train_data(batch_size)
            x_in, y_in, seq_len = self.reader.read_test_data(batch_size)

            if x_in is None:
                break

            nowVec = self.sess.run(self.pred, feed_dict={self.x: x_in, self.y_out: y_in, self.keep_prob: 1.0,
                                                         self.seq_length: seq_len})  #

            # print(nowVec[0])
            # input()
            X_input.append(nowVec[0])

        kmeans = KMeans(n_clusters=4).fit(X_input)


        tsne = TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(X_input)

        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))

        point_size = 30
        for i in range(0, 100):
            plt.scatter(X_norm[i, 0], X_norm[i, 1], c='r', s=point_size, label="cargo")

        for i in range(100, 200):
            plt.scatter(X_norm[i, 0], X_norm[i, 1], c='b', s=point_size, label="fish")

        for i in range(200, 300):
            plt.scatter(X_norm[i, 0], X_norm[i, 1], c='g', s=point_size, label="oil")

        for i in range(300, X_norm.shape[0]):
            plt.scatter(X_norm[i, 0], X_norm[i, 1], c='k', s=point_size, label="passenger")


        plt.show()

        label = kmeans.labels_
        # print(label[0:600])
        # print(label[600:1200])
        # print(label[1200:1800])
        # print(label[1800:])


    def train_(self, batch_size=1):
        while True:
            # x_in, y_in, seq_len = self.reader.read_train_data_mixup(batch_size)
            x_in, y_in, seq_len = self.reader.read_train_data(batch_size)

            if x_in is None:
                break

            self.sess.run(self.optimizer,
                          feed_dict={self.x: x_in, self.y_out: y_in, self.keep_prob: 0.5, self.seq_length: seq_len})

            cost = self.sess.run(self.cost,
                          feed_dict={self.x: x_in, self.y_out: y_in, self.keep_prob: 0.5, self.seq_length: seq_len})

        print("cost %f" %cost)

    def get_results_details(self, batch_size=1):

        macrof1_dict = dict()
        test_count = 0

        user_dict = defaultdict(int)
        while True:

            x_in, y_in, seq_len = self.reader.read_test_data(batch_size)

            if x_in is None:
                break

            for batch_num in range(0, batch_size):
                for index in range(0, 1):
                    true_uesr = y_in[batch_num].index(1)
                    macrof1_dict[true_uesr] = MacroF1(true_uesr)

        while True:

            x_in, y_in, seq_len = self.reader.read_test_data(batch_size)

            if x_in is None:
                break

            nowVec = self.sess.run(self.pred, feed_dict={self.x: x_in, self.y_out: y_in, self.keep_prob: 1.0,
                                                         self.seq_length: seq_len})  #

            # predictList = np.argpartition(a=-nowVec, kth=5)
            predictList_top1 = np.argpartition(a=-nowVec, kth=1)

            # for batch_num in range(0, batch_size):
            #     for index in range(0, 5):
            #         if predictList[batch_num][index] == y_in[batch_num].index(1):
            #             acc_top5 += 1
            #             break

            for batch_num in range(0, batch_size):
                for index in range(0, 1):
                    test_count += 1
                    predict_user = predictList_top1[batch_num][index]
                    true_uesr = y_in[batch_num].index(1)

                    if true_uesr == predict_user:
                        user_dict[true_uesr] += 1
                    print(true_uesr, true_uesr == predict_user)

        for key in user_dict.keys():
            print(key, user_dict[key])

    def test_(self, batch_size=1):

        acc_top1 = 0
        acc_top5 = 0

        macrof1_dict = dict()
        test_count = 0

        while True:

            x_in, y_in, seq_len = self.reader.read_test_data(batch_size)

            if x_in is None:
                break

            for batch_num in range(0, batch_size):
                for index in range(0, 1):
                    true_uesr = y_in[batch_num].index(1)
                    macrof1_dict[true_uesr] = MacroF1(true_uesr)

        while True:

            x_in, y_in, seq_len = self.reader.read_test_data(batch_size)

            if x_in is None:
                break

            nowVec = self.sess.run(self.pred, feed_dict={self.x: x_in, self.y_out: y_in, self.keep_prob: 1.0, self.seq_length: seq_len})  #

            # predictList = np.argpartition(a=-nowVec, kth=5)
            predictList_top1 = np.argpartition(a=-nowVec, kth=1)

            # for batch_num in range(0, batch_size):
            #     for index in range(0, 5):
            #         if predictList[batch_num][index] == y_in[batch_num].index(1):
            #             acc_top5 += 1
            #             break

            for batch_num in range(0, batch_size):
                for index in range(0, 1):
                    test_count += 1
                    predict_user = predictList_top1[batch_num][index]
                    true_uesr = y_in[batch_num].index(1)

                    if predict_user == true_uesr:
                        macrof1_dict[true_uesr].TP += 1
                    else:
                        macrof1_dict[true_uesr].FN += 1
                        if predict_user in macrof1_dict.keys():
                            macrof1_dict[predict_user].FP += 1


                    if predictList_top1[batch_num][index] == y_in[batch_num].index(1):
                        acc_top1 += 1
                        break

        print(acc_top1 / test_count)
        print(acc_top5 / test_count)

        TP = sum([macrof1_dict[key].TP for key in macrof1_dict.keys()])
        FP = sum([macrof1_dict[key].FP for key in macrof1_dict.keys()])
        FN = sum([macrof1_dict[key].FN for key in macrof1_dict.keys()])

        print(TP, FP, FN)

        microP = TP / (TP + FP)
        microR = TP / (TP + FN)

        microf1 = 2 * (microP * microR) / (microP + microR)

        macrof1 = np.mean([macrof1_dict[key].get_marcof1() for key in macrof1_dict.keys()])

        macroP = np.mean([macrof1_dict[key].get_P() for key in macrof1_dict.keys()])
        macroR = np.mean([macrof1_dict[key].get_R() for key in macrof1_dict.keys()])

        print(microP, microR, microf1, macrof1)

        print(macroP, macroR, 2 * (macroP * macroR) / (macroP + macroR))

        # for key in macrof1_dict.keys():
        #     print(macrof1_dict[key].toString())
        #
        # input("please input any keys")


if __name__ == "__main__":

    learning_rate = 0.001
    batch_size = 1
    iterations = 30
    model_name = "./hi_stem.ckpt"

    model = Hi_STEM(learning_rate, batch_size)

    # bi_lstm.restore(model_name)
    model.init_model()

    # bi_lstm.cluster()
    # bi_lstm.test_(batch_size=batch_size)
    # bi_lstm.get_results_details(batch_size=batch_size)
    for i in range(0, iterations):
        print(datetime.datetime.now())
        print("epoch number: ", i)
        model.train_(batch_size=batch_size)
        model.test_(batch_size=batch_size)
        model.save(model_name)


