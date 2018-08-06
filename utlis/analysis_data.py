import os
import numpy as np
import pandas as pd

class Analysis_Data(object):

    words_dict = {}

    def __init__(self, words_dir):

        self.words_dir = words_dir

    def word2dict(self):
        """
        提取文本数据中的汉子，并统计每个汉子出现的次数
        :return:
        """
        files_list = os.listdir(self.words_dir)
        files_list.sort()
        for file in files_list:
            if 'txt' in file:
                self._find_words(file)

        return self.words_dict

    def _find_words(self, file):
        """
        从file文件中读取数据，并将数据存放在words_dict中
        :param file:
        :return:
        """
        lines = open(self.words_dir + file)

        for line in lines:
            comma_num = 0
            for i in range(len(line)):
                # 开始解析label字符
                if comma_num >= 4:
                    # if line[i].isupper():
                    #     print('出现大写%s,文件%s,行内容：%s'%(line[i],file,line))
                    # if line[i].islower():
                    #     print('出现小写%s,文件%s,行内容：%s'%(line[i],file,line))

                    if line[i] == 'e' and line[i+1]=='m' and line[i+2]=='p' and line[i+3]=='t' \
                            and line[i+4]=='y':

                        if 'empty' in self.words_dict.keys():
                            self.words_dict['empty'] += 1
                        else:
                            self.words_dict['empty'] = 1
                        break
                    if line[i] == '\"':
                        if line[i] in self.words_dict.keys():
                            self.words_dict[line[i]] += 1
                        else:
                            self.words_dict[line[i]] = 1

                        print('出现问题%s,文件%s,行内容：%s' % (line[i], file, line))
                        break

                    if line[i] == '」':
                        print('出现问题%s,文件%s,行内容：%s' % (line[i], file, line))
                    if line[i] == 13:
                        print("1111111111111111111111111111111")
                    if line[i] == 10:
                        print('2222222222222222222222222222222')

                    if line[i] in self.words_dict.keys():
                        self.words_dict[line[i]] += 1
                    else:
                        self.words_dict[line[i]] = 1
                if line[i] == ",":
                    comma_num += 1
            if comma_num < 4:
                print("文件%s,问题行:%s"%(file, line))

    def datal2csv(self, save_file):

        word_num = len(self.words_dict.keys())
        word_labels = [i for i in range(word_num)]
        word_keys = [i for i in self.words_dict.keys()]

        words_label = pd.DataFrame({'label':word_labels, 'word':word_keys})

        words_label.to_csv(save_file, index=False)


    def data2label(self):
        self.label = {}
        if self.words_dict is not None:
            num = 0
            for key in self.words_dict.keys():
                label[key] = num
                num = num + 1
        return self.label


if __name__ == "__main__":
    a = Analysis_Data('../../train/')
    a.word2dict()
    a.datal2csv('../word_label.csv')


