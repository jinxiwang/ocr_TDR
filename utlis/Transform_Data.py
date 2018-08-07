import os
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

class Transform_Data(object):

    words_dict = {}

    def __init__(self, words_dir=None):

        self.words_dir = words_dir

    def word2dict(self):
        """
        提取文本数据中的汉子，并统计每个汉子出现的次数
        :return:
        """
        if self.words_dir is None:
            assert 0, print('数度读取地址为空')

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
            line = line.replace('\n', '')
            for i in range(len(line)):
                # 开始解析label字符
                if comma_num >= 4:

                    # 用于检测是否包含大小写字符
                    if line[i].isupper():
                        print('出现大写%s,文件%s,行内容：%s'%(line[i], file, line))
                    if line[i].islower():
                        print('出现小写%s,文件%s,行内容：%s'%(line[i], file, line))

                    # if line[i] == 'e' and line[i+1]=='m' and line[i+2]=='p' and line[i+3]=='t' \
                    #         and line[i+4]=='y':
                    #
                    #     if 'empty' in self.words_dict.keys():
                    #         self.words_dict['empty'] += 1
                    #     else:
                    #         self.words_dict['empty'] = 1
                    #     break
                    #
                    if line[i] == ' ':
                        continue
                    if line[i] == '　':
                        continue

                    # if line[i]=='”':
                    #     print('出现%s,文件%s,行内容：%s' % (line[i], file, line))

                    if self._find_repetition(line[i], '，', ','): continue
                    if self._find_repetition(line[i], '？', '?'): continue
                    if self._find_repetition(line[i], '。', '.'): continue
                    if self._find_repetition(line[i], '！', '!'): continue
                    if self._find_repetition(line[i], '（', '('): continue
                    if self._find_repetition(line[i], '）', ')'): continue
                    if self._find_repetition(line[i], '；', ';'): continue

                    if line[i] in self.words_dict.keys():
                        self.words_dict[line[i]] += 1
                    else:
                        self.words_dict[line[i]] = 1
                if line[i] == ",":
                    comma_num += 1
            if comma_num < 4:
                print("文件%s,问题行:%s"%(file, line))

    def datal2onehot(self, save_file):
        one_hot = {}
        if self.words_dict is not None:
            num = 0
            for key in self.words_dict.keys():
                one_hot[key] = num
                num = num + 1
        else:
            assert 0,print('无数据字典')

        f = open(save_file, 'w')
        f.write(str(one_hot))
        f.close()

    def _find_repetition(self, line, chinese, engish):
        if line == chinese or line == engish:
            if chinese in self.words_dict.keys():
                self.words_dict[chinese] += 1
            else:
                self.words_dict[chinese] = 1
            # if engish in self.words_dict.keys():
            #     self.words_dict[engish] += 1
            # else:
            #     self.words_dict[engish] = 1

            return True
        return False


if __name__ == "__main__":
    a = Transform_Data('../../train/')
    dict = a.word2dict()

    writer = open("../word_num_dict.txt", 'w')
    writer.write(str(dict))
    writer.close()
    # a.datal2onehot('../word_onehot.txt')

    # read = open("../word_num_dict.txt", 'r')
    # data = read.read()
    # data_dict = eval(data)
