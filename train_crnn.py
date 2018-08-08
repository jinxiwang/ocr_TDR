import time
import logging
import tensorflow as tf
from crnn import CRNN
from dataload import Dataload
from utlis.net_cfg_parser import parser_cfg_file

class Train_CRNN(object):

    def __init__(self, pre_train=False, start_step=0):
        net_params, train_params = parser_cfg_file('./net.cfg')

        self.input_height = int(net_params['input_height'])
        self.input_width = int(net_params['input_width'])
        self.batch_size = int(train_params['batch_size'])
        self._learning_rate = float(train_params['learning_rate'])
        self._max_iterators = int(train_params['max_iterators'])
        self._train_logger_init()
        self._pre_train = pre_train
        self._model_save_path = str(train_params['model_save_path'])
        self._start_step = start_step

        self._inputs = tf.placeholder(tf.float32, [self.batch_size, self.input_width, 32, 1])

        # label
        self._label = tf.sparse_placeholder(tf.int32, name='label')

        # The length of the sequence [32] * 64
        self._seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        crnn_net = CRNN(net_params, self._inputs, self._seq_len, self.batch_size)
        self._net_output, self._decoded, self._max_char_count = crnn_net.construct_graph()
        self.dense_decoded = tf.sparse_tensor_to_dense(self._decoded[0], default_value=-1)

    def _compute_accuracy(self, label, pridicted):

        accuracy = tf.reduce_mean(tf.edit_distance(tf.cast(pridicted, tf.int32), label))
        return accuracy

    def train(self):
        loss = tf.nn.ctc_loss(self._label, self._net_output, self._seq_len)
        loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

        accuracy = self._compute_accuracy(self._label, self._decoded[0])

        data = Dataload(self.batch_size, './data/dataset_label.txt')

        # 保存模型
        saver = tf.train.Saver()

        with tf.Session() as sess:
            if self._pre_train:
                saver.restore(sess, self._model_save_path)
            else:
                sess.run(tf.global_variables_initializer())

            for step in range(1, self._max_iterators):
                batch_data, batch_label = data.get_train_batch()

                feed_dict = {self._inputs: batch_data,
                             self._label: batch_label,
                             self._seq_len: [self._max_char_count]*self.batch_size}
                sess.run(train_op, feed_dict=feed_dict)

                if step%10 == 0:
                    train_loss = sess.run(loss, feed_dict=feed_dict)
                    # accuracy = sess.run(accuracy, feed_dict=feed_dict)
                    self.train_logger.info('step:%d, train accuracy: %6f, total loss: %6f' % (step, 0, train_loss))

                if step % 50 == 0:
                    self.train_logger.info('saving model...')
                    save_path = saver.save(sess, self._model_save_path, global_step=(self._start_step + step))
                    self.train_logger.info('model saved at %s' % save_path)


    def _train_logger_init(self):
        """
        初始化log日志
        :return:
        """
        self.train_logger = logging.getLogger('train')
        self.train_logger.setLevel(logging.DEBUG)

        # 添加文件输出
        log_file = './train_logs/' + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.logs'
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        self.train_logger.addHandler(file_handler)

        # 添加控制台输出
        consol_handler = logging.StreamHandler()
        consol_handler.setLevel(logging.DEBUG)
        consol_formatter = logging.Formatter('%(message)s')
        consol_handler.setFormatter(consol_formatter)
        self.train_logger.addHandler(consol_handler)


if __name__ == "__main__":
    train = Train_CRNN()
    train.train()