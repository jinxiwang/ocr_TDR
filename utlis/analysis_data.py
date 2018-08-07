import os
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
        greater = self.csv_data.loc[(self.csv_data["word_num"] > num)]
        all = self.csv_data.word_num.size
        num = greater.word_num.size
        print(num,all-num)
        return greater, num

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
    read = open("../word_num_dict.txt", 'r')
    data = read.read()
    data_dict = eval(data)

    a = Analysis_Data(data_dict)
    print(a.analysis_words_num())
    greater, num = a.analysis_which_greater(1)

    a.analysis_data_distribution()

