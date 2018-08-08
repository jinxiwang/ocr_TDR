import os
import cv2
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

class Analysis_Data(object):
    def __init__(self, words_dict):

        self.words_dict = words_dict
        self.csv_data = self._dict2csv()

    def analysis_data_distribution(self):
        """
        分析数据分布
        :return:
        """
        if self.words_dict == None:
            assert 0, print("字典为空，无法分析")

        titanic = sb.load_dataset("titanic")
        #print(titanic)
        #sb.distplot(self.csv_data['word_num'], kde = True, rug = True)
        sb.catplot(x="word", kind="count", palette="ch:.25",
                   data=self.csv_data[0:20])

        plt.show()
        print(self.csv_data.sort_values(by="word_num", ascending=False))

    def analysis_words_num(self):
        """
        统计出现的单词数量
        :return:
        """
        return self.csv_data.word.size

    def analysis_which_greater(self, num):
        """
        获取当前出现次数大于num的汉字和数量
        :param num:
        :return:
        """
        greater = self.csv_data.loc[(self.csv_data["word_num"] > num)]
        all = self.csv_data.word_num.size
        num = greater.word_num.size
        print(num,all-num)
        return greater, num

    def analysis_longgest_label(self):
        """
        获取最长的长度,和key
        :return:
        """
        f = open("../data/dataset_label.txt", 'r')
        data = f.read()
        dataset_label_dict = eval(data)

        length = 0
        keys = []

        for i in dataset_label_dict.keys():
            if length < len(dataset_label_dict[i]):
                keys.clear()
                keys.append(i)
                length = len(dataset_label_dict[i])
            elif length == len(dataset_label_dict[i]):
                keys.append(i)

        return length, keys

    def analysis_img_mess(self):
        """
        分析图像的信息
        :return:
        """
        path = '/home/tony/ocr/dataset/'

        width_list = []
        height_list = []
        img_num = len(os.listdir(path))

        for i in os.listdir(path):
            img = cv2.imread(path+i)
            shape = np.shape(img)

            height_list.append(shape[0])
            width_list.append(shape[1])

        width_list.sort()
        max_width = width_list[-1]
        min_width = width_list[0]

        x = [i for i in range(max_width)]
        plt.subplot(121)
        plt.xlabel('width')
        plt.ylabel('img number')
        plt.hist(width_list, 30)

        plt.subplot(122)
        plt.xlabel('height')
        plt.ylabel('img number')
        plt.hist(height_list, 30)
        # num_bins = 10
        #
        # fig, ax = plt.subplots()
        #
        # # the histogram of the data
        # n, bins, patches = ax.hist(x, num_bins)
        #
        # ax.plot(10, width_list)
        # ax.set_xlabel('width')
        # ax.set_ylabel('img number')
        # ax.set_title('Histogram of img message')
        plt.show()


    def _dict2csv(self):
        """
        将字典结构转为csv类型，方便处理
        :return:
        """
        words_list = list(self.words_dict.keys())
        words_num_list = [self.words_dict[key] for key in words_list]

        csv_data = pd.DataFrame({'word':words_list, 'word_num':words_num_list})
        return csv_data

if __name__ == '__main__':
    read = open("../data/word_num_dict.txt", 'r')
    data = read.read()
    data_dict = eval(data)

    a = Analysis_Data(data_dict)
    a.analysis_img_mess()

    # 查看出现次数大于num的汉字个数
    # print(a.analysis_words_num())
    # greater, num = a.analysis_which_greater(1)
    # a.analysis_data_distribution()

    # 查看label长度
    # length, keys = a.analysis_longgest_label()
    # print(keys)
    # print(length)
    # for k in keys:
    #     print(k, data_dict[k])

