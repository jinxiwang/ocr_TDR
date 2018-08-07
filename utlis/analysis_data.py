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

        words_list = list(self.words_dict.keys())
        words_num_list = [self.words_dict[key] for key in words_list]

        sb.distplot(words_num_list, kde=False)

        plt.show()

    def analysis_words_num(self):
        return self.csv_data.word.size


    def _dict2csv(self):
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
