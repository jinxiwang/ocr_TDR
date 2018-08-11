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
        self._trainable = trainable

    def construct_graph(self):
        """
        构建网络
        :return:
        """
        # 进入cnn网络层 shape [batch, length, 32 ,1]
        cnn_out = self._cnn(self._inputs)

        # 送入rnn前将cnn进行reshape
        #reshaped_cnn_output = tf.reshape(cnn_out, [self._batch_size, -1, 512])
        max_char_count = cnn_out.get_shape().as_list()[1]
        print(max_char_count)

        crnn_model = self._rnn(cnn_out, self._seq_len)
        logits = tf.reshape(crnn_model, [-1, 512])

        W = tf.Variable(tf.truncated_normal([512, self._class_num], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[self._class_num]), name="b")

        logits = tf.matmul(logits, W) + b

        logits = tf.reshape(logits, [self._batch_size, -1, self._class_num])

        # 网络层输出
        net_output = tf.transpose(logits, (1, 0, 2))

        # 解析网络输出
        decoded, log_prob = tf.nn.ctc_greedy_decoder(net_output, self._seq_len)

        return net_output, decoded, max_char_count

    def _conv2d(self,inputs,filters,padding,batch_norm,name):

        if batch_norm:
            activation = None
        else:
            activation = tf.nn.relu

        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)

        top = tf.layers.conv2d(inputs,
                               filters=filters,
                               kernel_size=3,
                               padding=padding,
                               activation=activation,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               name=name)
        if batch_norm:
            top = tf.layers.batch_normalization(top, axis=3, training=self._trainable, name=name)
            top = tf.nn.relu(top, name=name + '_relu')

        return top

    def _cnn(self, inputs):
        """
        cnn网络结构
        :param inputs:
        :return:
        """
        conv1 = self._conv2d(inputs=inputs, filters=64, padding="valid",batch_norm=False,name='conv1')
        conv2 = self._conv2d(inputs=conv1, filters=64, padding="same",batch_norm=True,name='conv2')
        pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=[2,2], padding='valid')

        conv3 = self._conv2d(inputs=pool1, filters=128, padding="same",batch_norm=True,name='conv3')
        conv4 = self._conv2d(inputs=conv3, filters=128, padding="same",batch_norm=True, name='conv4')
        pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=2, strides=[2, 1], padding="valid")

        conv5 = self._conv2d(inputs=pool2, filters=256, padding="same", batch_norm=True, name='conv5')
        conv6 = self._conv2d(inputs=conv5, filters=256, padding="same", batch_norm=True, name='conv6')
        pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=2, strides=[2, 1], padding="valid")

        conv7 = self._conv2d(inputs=pool3, filters=512, padding="same", batch_norm=True, name='conv7')
        conv8 = self._conv2d(inputs=conv7, filters=512, padding="same", batch_norm=True, name='conv8')
        pool4 = tf.layers.max_pooling2d(inputs=conv8, pool_size=[3,1], strides=[3,1], padding="valid")

        # 去掉维度为1的维度
        features = tf.squeeze(pool4, axis=1, name='features')

        return features

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
