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
        return self.csv_data.word.sort

    def analysis_which_greater(self, num):
        """
        获取当前出现次数大于num的汉字和数量
        :param num:
        :return:
        """
        greater = self.csv_data.loc[(self.csv_data["word_num"] < num)]
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

    def analysis_width_label(self):
        path = '/home/tony/ocr/dataset/'

        f = open('../data/dataset_label.txt', 'r')
        label_dict = f.read()
        label_dict = eval(label_dict)

        img_path_list = list(label_dict.keys())
        width_label_dict = {}
        img_num = len(img_path_list)

        for i in img_path_list:
            img = cv2.imread(i)
            shape = np.shape(img)
            width = shape[1]
            label_length = len(label_dict[i])
            if width in width_label_dict.keys():
                width_label_dict[width] = [width_label_dict[width][0]+label_length,
                                           width_label_dict[width][1]+1]
            else:
                width_label_dict[width] = [label_length, 1]

        width_list = list(width_label_dict.keys())
        width_list.sort()
        label_list = [width_label_dict[width_list[i]][0]/width_label_dict[width_list[i]][1] for i in range(len(width_list))]

        fig = plt.figure(1)
        # 子表1绘制加速度传感器数据
        plt.subplot()
        plt.xlabel('width')
        plt.ylabel('avg label')
        plt.plot(width_list, label_list)
        plt.show()


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
        print(max_width)
        print(min_width)

        x = [i for i in range(max_width)]
        plt.subplot(121)
        plt.xlabel('width')
        plt.ylabel('img number')
        plt.hist(width_list, 10)

        plt.subplot(122)
        plt.xlabel('height')
        plt.ylabel('img number')
        plt.hist(height_list, 10)
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

        csv_data = pd.DataFrame({'word': words_list, 'word_num': words_num_list})
        return csv_data

    def pie_chart(self):
        labels = 'number > 30', 'number < 30'
        sizes = [883, 1791]
        explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.show()

if __name__ == '__main__':
    read = open("../data/word_num_dict.txt", 'r')
    data = read.read()
    data_dict = eval(data)

    a = Analysis_Data(data_dict)
    #a.analysis_img_mess()
    #a.analysis_width_label()
    #a.pie_chart()

    # 查看出现次数大于num的汉字个数
    #print(a.analysis_words_num())
    greater, num = a.analysis_which_greater(2)
    print([i for i in greater.word])
    #a.analysis_data_distribution()

    # 查看label长度
    # length, keys = a.analysis_longgest_label()
    # print(keys)
    # print(length)
    # for k in keys:
    #     print(k, data_dict[k])

