import numpy as np
import tensorflow as tf
import scipy.misc as sm


class Dataload(object):

    def __init__(self, batch_size, label_path,img_height=32, img_width=1050):
        self.batch_size = batch_size
        self.input_img_height = img_height
        self.input_img_width = img_width
        self.label_path = label_path

        f = open(self.label_path, 'r')
        data = f.read()
        self.data_dict = eval(data)
        self.img_path_list = list(self.data_dict.keys())

        self.current_index = 0
        self.epoch = 0

    def get_train_batch(self):
        """
        获取训练batch
        :return:
        """
        if self.current_index + self.batch_size > len(self.img_path_list):
            self.current_index = len(self.img_path_list) - self.batch_size
            self.epoch += 1

        batch_data = np.zeros([self.batch_size,
                               self.input_img_height,
                               self.input_img_width])
        batch_label = []

        for i in range(self.batch_size):
            img = sm.imread(self.img_path_list[self.current_index], mode='L')
            img_resized = self._resize_img(img)
            batch_data[i] = img_resized
            batch_label.append(self.data_dict[self.img_path_list[self.current_index]])
            self.current_index += 1

        batch_label = self._sparse_tuple_from(batch_label)

        if self.current_index + self.batch_size == len(self.img_path_list):
            self.current_index = 0
        else:
            self.current_index += self.batch_size

        batch_data = np.reshape(batch_data, [self.batch_size,
                                             self.input_img_width,
                                             self.input_img_height,
                                             1])

        return batch_data, batch_label

    def _resize_img(self, img):
        """
        将图像先转为灰度图，并将图像进行resize
        :param img:
        :return:
        """
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = np.shape(img)

        if width > self.input_img_width:
            width = self.input_img_width
            ratio = float(self.input_img_width) / width
            outout_img = sm.imresize(img, [int(self.input_img_height * ratio), self.input_img_width])
        else:
            outout_img = np.zeros([self.input_img_height, self.input_img_width])
            ratio = self.input_img_height / height
            img_resized = sm.imresize(img, (self.input_img_height, int(width * ratio)))
            outout_img[:, 0:np.shape(img_resized)[1]] = img_resized

        return outout_img

    def _sparse_tuple_from(self, sequences, dtype=np.int32):
        """
        将矩阵转为稀疏矩阵存储方式
        :param sequences:
        :param dtype:
        :return:
        """

        indices = []
        values = []
        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), [i for i in range(len(seq))]))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape

if __name__ == "__main__":
    a = Dataload(2, './data/dataset_label.txt')

    b, c = a.get_train_batch()

    #print(c)