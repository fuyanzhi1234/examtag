import re
import json
import numpy as np
import pandas as pd

def is_nan(nan):
    return nan != nan

def clean_text(text):
    if is_nan(text):
        return ''
    # 替换所有的数字、字母、换行符和空格
    text = re.sub(r"[\n\s]", "", text)
    text.replace('A.', '')
    text.replace('B.', '')
    text.replace('C.', '')
    text.replace('D.', '')
    return text

# 定义读取数据集函数
def read_excel_data(labelnum, is_test=False):
    if is_test:
        filepath = 'raw_data/testdata.csv'
    else:
        filepath = 'raw_data/traindata.csv'

    data = pd.read_csv(filepath)
    # 取出知识点列
    knowledge = data.loc[:, 'knowledge']
    # 取出题目列
    exams = data.loc[:, 'exam']
    
    # 遍历exams
    total = len(exams)
    for index in range(total):
        examClean = clean_text(exams[index])
        if len(examClean) <= 10:
            continue
        # 将labels转换为One-hot表示
        labels = [float(1) if str(i) in knowledge[index].split(';') else float(0) for i in range(labelnum)]

        print({"text": clean_text(examClean), "labels": labels})
        yield {"text": clean_text(examClean), "labels": labels}

def readVocab(jsonName):
    with open(jsonName, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(data)
        return data

# 定义读取数据集函数
def read_custom_data(is_test=False, is_one_hot=True):

    file_num = 6 if is_test else 48
    filepath = 'raw_data/test/' if is_test else 'raw_data/train/'

    for i in range(file_num):
        f = open('{}labeled_{}.txt'.format(filepath, i))
        while True:
            line = f.readline()
            if not line:
                break
            data = line.strip().split('\t')
            # 标签用One-hot表示
            if is_one_hot:
                labels = [float(1) if str(i) in data[1].split(',') else float(0) for i in range(20)]
            else:
                labels = [int(d) for d in data[1].split(',')]
            yield {"text": clean_text(data[0]), "labels": labels}
        f.close()