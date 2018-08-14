import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from crnn import CRNN
from utlis.net_cfg_parser import parser_cfg_file

class Test_CRNN(object):
    def __init__(self, batch_size=None):
        net_params, train_params = parser_cfg_file('./net.cfg')
        self._model_save_path = str(net_params['model_save_path'])
        self.input_img_height = int(net_params['input_height'])
        self.input_img_width = int(net_params['input_width'])
        if batch_size is None:
            self.test_batch_size = int(net_params['test_batch_size'])
        else:
            self.test_batch_size = batch_size

        # 加载label onehot
        f = open('./data/word_onehot.txt', 'r')
        data = f.read()
        words_onehot_dict = eval(data)
        self.words_list = list(words_onehot_dict.keys())
        self.words_onehot_list = [words_onehot_dict[self.words_list[i]] for i in range(len(self.words_list))]

        # 构建网络
        self.inputs_tensor = tf.placeholder(tf.float32, [self.test_batch_size, self.input_img_height, self.input_img_width, 1])
        self.seq_len_tensor = tf.placeholder(tf.int32, [None], name='seq_len')

        crnn_net = CRNN(net_params, self.inputs_tensor, self.seq_len_tensor, self.test_batch_size, True)
        net_output, decoded, self.max_char_count = crnn_net.construct_graph()
        self.dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, "/home/tony/ocr/model/93/ckpt")

    def _get_input_img(self, img_path_list):

        batch_size = len(img_path_list)

        batch_data = np.zeros([batch_size,
                               self.input_img_height,
                               self.input_img_width,
                               1])
        img_list = []

        for i in range(batch_size):
            img = cv2.imread(img_path_list[i], 0)
            img_list.append(img)
            # print(np.shape(img))
            # print(img_path_list[i])
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = self._resize_img(img)
            reshape_img = resized_img.reshape([1, self.input_img_height, self.input_img_width, 1])
            img_norm = reshape_img / 255 * 2 - 1
            batch_data[i] = img_norm

        return batch_data, batch_size, img_list


    def test_img(self, img_path_list, is_show_res=False):

        batch_data, batch_size, img_list= self._get_input_img(img_path_list)
        if batch_size != self.test_batch_size:
            error = '网络构建batch size:'+str(self.test_batch_size)+'和实际输入batch size:'+str(batch_size)+'不一样'
            assert 0, error

        feed_dict = {self.inputs_tensor: batch_data, self.seq_len_tensor: [self.max_char_count]*batch_size}
        predict = self.sess.run(self.dense_decoded, feed_dict=feed_dict)
        predict_seq = self._predict_to_words(predict)

        if is_show_res:
            for i in range(batch_size):
                print(img_path_list[i], ':', predict_seq[i])
                cv2.imshow(img_path_list[i], img_list[i])
            cv2.waitKey()

        return predict_seq

    def _predict_to_words(self, decoded):
        words = []

        for seq in decoded:
            seq_words = ''
            for onehot in seq:
                if onehot == -1:
                    break
                seq_words += self.words_list[self.words_onehot_list.index(onehot)]
            words.append(seq_words)
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

    test_img_list = ['/home/tony/ocr/test_data/00023.jpg']
    a = Test_CRNN()
    a.test_img(test_img_list)

    # test_list = []
    # res_list = []
    # name_list = []
    #
    # for i in range(0, 32):
    #     test_list.append('/home/tony/ocr/test_data/%05d.jpg' % (i+945))
    #     name = [('%05d' % (i+945))]
    #     name_list.append(('%05d' % (i+945)))
    #
    #     if (i+1) % 32 == 0:
    #         print('test....', (i+1) / 32)
    #         res = a.test_img(test_list)
    #         res = [i for i in res]
    #         res_list.extend(res)
    #         test_list.clear()
    #
    #
    # save = []
    #
    # for i in range(len(name_list)):
    #     res_dict = {}
    #     res_dict[name_list[i]] = res_list[i]
    #     save.append(res_dict)
    # print(save)
    #
    # f = open('1.json', 'w')
    # f.write(str(save))


    # for i in range(0, 977):
    #     test_list.append('/home/tony/ocr/test_data/%05d.jpg'%(i))
    #     name_list.append('%05d' % (i))
    #
    # a.test_img(['/home/tony/ocr/test_data/00021.jpg'])