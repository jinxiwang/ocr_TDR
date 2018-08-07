import os
import cv2
import numpy as np

RAW_DATASET = '../../train/'
SAVE_DATASET = '../../dataset/'

dataset_dict = {}

def cut_img_and_save_label(img_file, bbox_list, words_list):
    """
    将img图像根据bbox大小进行剪裁
    :param bbox: [left_x,left_y,right_x,right_y]
    :param img: 读取的img
    :return:
    """
    img_name = img_file.replace('.jpg', '')
    img = cv2.imread(RAW_DATASET + img_file)

    if len(bbox_list) != len(words_list):
        assert 0, print('bbox_list长度和words_list长度不相等')

    # 将图像名和其label进行存储
    label_list = words_list2label_list(words_list)
    img_name_list = []

    index = 0
    for i in range(len(bbox_list)):
        # 将切割的图片进行保存并命名为img_name_i.jpg
        cut_img = img[bbox_list[i][1]:bbox_list[i][3], bbox_list[i][0]:bbox_list[i][2]]
        # img_gray = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
        save_name = SAVE_DATASET + img_name + '_' + str(index) + '.jpg'

        cv2.imwrite(save_name, cut_img)
        img_name_list.append(save_name)
        dataset_dict[img_name_list[i]] = label_list[i]
        index += 1

    return img_name_list, label_list

def words_list2label_list(words_list):
    """
    将图像中单个文字label拼成label_list
    :param words:
    :return:
    """
    label_list = []

    read = open('../word_label.txt', 'r')
    all_label_dict = read.read()
    all_label_dict = eval(all_label_dict)

    for words in words_list:
        list = []
        for i in words:
            i = __english_symbol(i)
            if i in all_label_dict.keys():
                list.append(all_label_dict[i])
            else:
                print(i)
        label_list.append(list)

    read.close()

    return label_list

def __english_symbol(symbol):
    """
    将英文符号转为中文符号
    :param symbol:
    :return:
    """
    if symbol is '.':
        return '。'
    elif symbol is '?':
        return '？'
    elif symbol is '!':
        return '！'
    elif symbol is '(':
        return '（'
    elif symbol is ')':
        return '）'
    elif symbol is ';':
        return '；'
    else:
        return symbol


def extract_bbox_words(txt_file):
    """
    从txt_file中获取bbox,和words
    :param txt_file:
    :return:
    """
    bbox_list = []
    words_list = []

    read = open(RAW_DATASET+txt_file,'r')
    lines = read.readlines()
    for line in lines:
        comma_index = []
        i = 0
        while len(comma_index) <4:
            if line[i] == ',':
                comma_index.append(i)
            i += 1
        # try:
        #     a = int(line[0:comma_index[0]])
        # except ValueError:
        #     print(txt_file)

        # bbox中包含float，需要做转换string->float->int
        bbox = [int(float(line[0:comma_index[0]])),
                int(float(line[comma_index[0]+1:comma_index[1]])),
                int(float(line[comma_index[1] + 1:comma_index[2]])),
                int(float(line[comma_index[2] + 1:comma_index[3]]))]
        line = line.replace('\n', '')
        bbox_list.append(bbox)
        words_list.append(line[comma_index[3]+1:])

    read.close()

    return bbox_list, words_list

def make_dataset():
    listdir = os.listdir(RAW_DATASET)
    listdir.sort()
    for file in listdir:
        if 'jpg' in file:
            txt_file = file.replace('jpg', 'txt')
            bbox_list, words_list = extract_bbox_words(txt_file)
            cut_img_and_save_label(file, bbox_list, words_list)

make_dataset()
f = open('../dataset_dict.txt', 'w')
f.write(str(dataset_dict))
f.close()

# bbox_list, words_list = extract_bbox_words('0.txt')
# img_name_list, label_list = cut_img_and_save_label('0.jpg', bbox_list, words_list)
# print(img_name_list)
# print(label_list)