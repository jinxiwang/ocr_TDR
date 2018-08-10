import numpy as np
import tensorflow as tf
import scipy.misc as sm
import cv2
from crnn import CRNN
from dataload import Dataload
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

    def test_one_img(self, img_path):

        load = Dataload(2, './data/dataset_label.txt', 32, 900)
        data, label = load.get_train_batch()

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = self._resize_img(img)
        reshape_img = resized_img.reshape([1, self.input_img_height, self.input_img_width, 1])
        img_norm = reshape_img / 255 * 2 - 1

        inputs = tf.placeholder(tf.float32, [2, self.input_img_height, self.input_img_width, 1])
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        crnn_net = CRNN(self.net_params, inputs, seq_len, 2, True)
        net_output, decoded, max_char_count = crnn_net.construct_graph()
        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "/home/tony/ocr/model/93/ckpt")

            feed_dict = {inputs: data, seq_len: [max_char_count]*2}

            predict = sess.run(dense_decoded, feed_dict=feed_dict)
            words = self._predict_to_words(predict[0])
        print(words)
        cv2.imshow('dw', img)
        cv2.waitKey()

        return img, words

    # def _get_crnn_net(self, input_tensor, seq_len, batch_size):
    #
    #     crnn_net = CRNN(self.net_params, input_tensor, seq_len, batch_size, False)
    #     net_output, decoded, max_char_count = crnn_net.construct_graph()
    #
    #     return net_output, decoded, max_char_count

    # def test_one_img(self, img_path):
    #     img = cv2.imread(img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     resized_img = self._resize_img(img)
    #     reshape_img = resized_img.reshape([1, self.input_img_height, self.input_img_width, 1])
    #     img_norm = reshape_img / 255 * 2 - 1
    #
    #     inputs = tf.placeholder(tf.float32, [1, self.input_img_height, self.input_img_width, 1])
    #     seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
    #
    #     crnn_net = CRNN(self.net_params, inputs, seq_len, 1, False)
    #     net_output, decoded, max_char_count = crnn_net.construct_graph()
    #     dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
    #
    #     with tf.Session() as sess:
    #         saver = tf.train.Saver()
    #         # saver.restore(self._get_session(), "./models/test_model/model.ckpt")
    #         saver.restore(sess, "/home/tony/ocr/model/93/ckpt")
    #
    #         feed_dict = {inputs: img_norm, seq_len: [max_char_count]}
    #
    #         predict = sess.run(dense_decoded, feed_dict=feed_dict)
    #         print(predict)
    #         words = self._predict_to_words(predict[0])
    #
    #     print(words)
    #     cv2.imshow('dw', resized_img)
    #     cv2.waitKey()
    #
    #     return img, words

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
            outout_img = cv2.resize(img, (self.input_img_width,self.input_img_height))
        else:
            outout_img = np.zeros([self.input_img_height, self.input_img_width])
            ratio = self.input_img_height / height
            img_resized = cv2.resize(img, (int(width * ratio), self.input_img_height))
            outout_img[:, 0:np.shape(img_resized)[1]] = img_resized

        return outout_img


if __name__ == "__main__":
    a = Test_CRNN()
    a.test_one_img('/home/tony/ocr/dataset/18_2.jpg')