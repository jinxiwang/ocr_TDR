import tensorflow as tf
from crnn import CRNN
from utlis.net_cfg_parser import parser_cfg_file

class Train_CRNN(object):

    def __init__(self):
        net_params, train_params = parser_cfg_file('./net.cfg')

        self.input_height = int(net_params['input_height'])
        self.input_width = int(net_params['input_width'])
        self.batch_size = int(train_params['batch_size'])

        self._inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_width, 32, 1])

        # label
        self._label = tf.sparse_placeholder(tf.int32, name='label')

        # The length of the sequence [32] * 64
        self._seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        crnn_net = CRNN(net_params, self._inputs, self._seq_len, self.batch_size)
        self._net_output, self._dense_decoded, self._max_char_count = crnn_net.construct_graph()

    def _compute_accuracy(self, label, pridicted):

        accuracy = tf.reduce_mean(tf.edit_distance(tf.cast(pridicted, tf.int32), label))
        return accuracy

    def _compute_loss(self):

        loss = tf.nn.ctc_loss(self._net_output, self._label, self._seq_len)
        loss = tf.reduce_mean(loss)
        return loss

    def train(self):

        pass


