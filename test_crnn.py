import numpy as np
import tensorflow as tf
import scipy.misc as sm
from crnn import CRNN
from utlis.net_cfg_parser import parser_cfg_file

class Test_CRNN(object):
    def __init__(self):
        self.net_params, train_params = parser_cfg_file('./net.cfg')
        self._model_save_path = str(self.net_params['model_save_path'])
        self.input_img_height = int(self.net_params['input_height'])
        self.input_img_width = int(self.net_params['input_width'])

        f = open('./data/word_onehot.txt', 'r')
        data = f.read()
        words_onehot_dict = eval(data)
        self.words_list = list(words_onehot_dict.keys())
        self.words_onehot_list = [words_onehot_dict[self.words_list[i]] for i in range(len(self.words_list))]

    def _get_crnn_net(self, input_tensor, seq_len, batch_size):

        crnn_net = CRNN(self.net_params, input_tensor, seq_len, batch_size)
        net_output, decoded, max_char_count = crnn_net.construct_graph()

        return net_output, decoded, max_char_count

    def test_one_img(self, img_path):
        img = sm.imread(img_path, mode='L')
        resized_img = self._resize_img(img).reshape([1, self.input_img_width, self.input_img_height, 1])

        inputs = tf.placeholder(tf.float32, [1, self.input_img_width, self.input_img_height, 1])
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        net_output, decoded, max_char_count = self._get_crnn_net(inputs, seq_len, 1)

        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

        with tf.Session() as sess:
            feed_dict = {inputs: resized_img, seq_len: [max_char_count]}

            decoded = sess.run(dense_decoded, feed_dict=feed_dict)

            words = self._predict_to_words(decoded[0])
        return img, words

    def _predict_to_words(self, decoded):
        words = ''

        for onehot in decoded:
            if onehot == -1:
                continue
            words += self.words_list[self.words_onehot_list.index(onehot)]

        return words




    def _resize_img(self, img):
        """
        将图像先转为灰度图，并将图像进行resize
        :param img:
        :return:
        """
        height, width = np.shape(img)

        if width > self.input_img_width:
            width = self.input_img_width
            ratio = float(self.input_img_width) / width
            outout_img = sm.imresize(img, [self.input_img_height, self.input_img_width])
        else:
            outout_img = np.zeros([self.input_img_height, self.input_img_width])
            ratio = self.input_img_height / height
            img_resized = sm.imresize(img, (self.input_img_height, int(width * ratio)))
            outout_img[:, 0:np.shape(img_resized)[1]] = img_resized

        return outout_img
