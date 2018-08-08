import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from utlis.net_cfg_parser import parser_cfg_file

class CRNN(object):

    def __init__(self, net_params, inputs, seq_len, batch_size, trainable=False, pretrain=False):

        self._input_height = int(net_params['input_height'])
        self._input_width = int(net_params['input_width'])
        self._class_num = int(net_params['classes_num'])

        self._inputs = inputs
        self._batch_size = batch_size
        self._seq_len = seq_len

    def construct_graph(self):
        """
        构建网络
        :return:
        """
        # 进入cnn网络层 shape [batch, length, 32 ,1]
        cnn_out = self._cnn(self._inputs)

        # 送入rnn前将cnn进行reshape
        reshaped_cnn_output = tf.reshape(cnn_out, [self._batch_size, -1, 512])
        max_char_count = reshaped_cnn_output.get_shape().as_list()[1]
        print(max_char_count)

        crnn_model = self._rnn(reshaped_cnn_output, self._seq_len)
        logits = tf.reshape(crnn_model, [-1, 512])

        W = tf.Variable(tf.truncated_normal([512, self._class_num], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[self._class_num]), name="b")

        logits = tf.matmul(logits, W) + b

        logits = tf.reshape(logits, [self._batch_size, -1, self._class_num])

        # 网络层输出
        net_output = tf.transpose(logits, (1, 0, 2))

        # 解析网络输出
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_output, self._seq_len)

        return net_output, decoded, max_char_count

    def _cnn(self, inputs):
        """
        cnn网络结构
        :param inputs:
        :return:
        """
        # 64 / 3 x 3 / 1 / 1
        conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        # 2 x 2 / 1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # 128 / 3 x 3 / 1 / 1
        conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        # 2 x 2 / 1
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # 256 / 3 x 3 / 1 / 1
        conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        # Batch normalization layer
        bnorm1 = tf.layers.batch_normalization(conv3)

        # 256 / 3 x 3 / 1 / 1
        conv4 = tf.layers.conv2d(inputs=bnorm1, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        # 1 x 2 / 1
        pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[1, 2], padding="same")

        # 512 / 3 x 3 / 1 / 1
        conv5 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        # Batch normalization layer
        bnorm2 = tf.layers.batch_normalization(conv5)

        # 512 / 3 x 3 / 1 / 1
        conv6 = tf.layers.conv2d(inputs=bnorm2, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        # 1 x 2 / 2
        pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[1, 2], padding="same")

        # 512 / 2 x 2 / 1 / 0
        conv7 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[2, 2], padding="valid", activation=tf.nn.relu)

        return conv7

    def _rnn(self, inputs, seq_len):
        """
        双向rnn
        :return:
        """
        with tf.variable_scope(None, default_name="bidirectional-rnn-1"):
            # 前向rnn
            lstm_fw_cell_1 = rnn.BasicLSTMCell(256)
            # 反向rnn
            lstm_bw_cell_1 = rnn.BasicLSTMCell(256)

            inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, seq_len,
                                                              dtype=tf.float32)

            inter_output = tf.concat(inter_output, 2)

        with tf.variable_scope(None, default_name="bidirectional-rnn-2"):
            # 前向rnn
            lstm_fw_cell_2 = rnn.BasicLSTMCell(256)
            # 反向rnn
            lstm_bw_cell_2 = rnn.BasicLSTMCell(256)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, inter_output, seq_len,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, 2)

        return outputs
