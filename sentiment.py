from collections import defaultdict
import os
import re
import jieba
import codecs
import pandas as pd
from tqdm import tqdm
import time

jieba.load_userdict('E:/情绪/dict1.txt')

stopwords = set()
with codecs.open('E:/情绪/停用词.txt', 'r', encoding='utf-8') as fr:
    for word in fr:
        stopwords.add(word.strip())

not_word_file = codecs.open('E:/情绪/否定词.txt', 'r+', encoding='utf-8')
not_word_list = not_word_file.readlines()
not_word_list = [w.strip() for w in not_word_list]

degree_file = codecs.open('E:/情绪/程度副词.txt', 'r+', encoding='utf-8')
degree_list = degree_file.readlines()
degree_dict = defaultdict()
for i in degree_list:
    degree_dict[i.split(',')[0]] = float(i.split(',')[1])  # 将得分转换为float类型

with codecs.open('E:/情绪/stopwords.txt', 'w', encoding='utf-8') as f:
    for word in stopwords:
        if(word not in not_word_list) and (word not in degree_dict.keys()):
            f.write(word+'\n')

def seg_word(sentence):
    seg_list = jieba.cut(sentence)
    seg_result = []
    for i in seg_list:
        seg_result.append(i)

    stopwords = set()
    with codecs.open('E:/情绪/stopwords.txt', 'r', encoding='utf-8') as fr:
        for i in fr:
            stopwords.add(i.strip())
    return list(filter(lambda x: x not in stopwords, seg_result))

def classify_words(word_list):
    sen_file = codecs.open('E:/情绪/冯总补充后.txt', 'r+', encoding='utf-8')
    sen_list = sen_file.readlines()
    sen_dict = defaultdict()
    for i in sen_list:
        if len(i.split(' ')) == 2:
            sen_dict[i.split(' ')[0]] = i.split(' ')[1]

    not_word_file = codecs.open('E:/情绪/否定词.txt', 'r+', encoding='utf-8')
    not_word_list = not_word_file.readlines()
    sen_word = dict()
    not_word = dict()
    degree_word = dict()

    for i in range(len(word_list)):
        word = word_list[i]
        if word in sen_dict.keys() and word not in not_word_list and word not in degree_dict.keys():
            sen_word[i] = sen_dict[word]
        elif word in not_word_list and word not in degree_dict.keys():
            not_word[i] = -1
        elif word in degree_dict.keys():
            degree_word[i] = degree_dict[word]

    sen_file.close()
    not_word_file.close()
    return sen_word, not_word, degree_word

def score_sentiment(sen_word, not_word, degree_word, seg_result):
    W = 1
    pos_score = neg_score = 0
    sentiment_index = -1
    sentiment_index_list = list(sen_word.keys())

    not_word_count = defaultdict(int)
    degree_effect = 0
    for i in range(len(seg_result)):
        if i in degree_word:
            degree_effect += float(degree_word[i])
        if i in sen_word:
            score = W * float(sen_word[i])
            for j in range(i-1, -1, -1):
                if j in not_word:
                    not_word_count[i] += 1
            
            for j in range(i-1, -1, -1):
                if j in not_word:
                    score *= -1      
            if degree_effect > 0:
                score *= degree_effect  
            if score > 0:
                pos_score += score
            elif score < 0:
                neg_score += abs(score)

    total_score = (abs(pos_score) - abs(neg_score)) / (abs(pos_score) + abs(neg_score)) if (abs(pos_score) + abs(neg_score)) != 0 else 0
    return total_score

data = pd.read_excel('E:/预测返修/上海内容.xlsx').astype(str)
sentence = data.content

def sentiment_score(sentence):
    seg_list = seg_word(sentence)
    sen_word, not_word, degree_word = classify_words(seg_list)
    score = score_sentiment(sen_word, not_word, degree_word, seg_list)
    return score

tqdm.pandas(desc="Processing", position=0)
data['cut_content'] = data.progress_apply(lambda row: seg_word(row['content']), axis=1)
data['sentiment_score'] = data.progress_apply(lambda row: sentiment_score(row['content']), axis=1)
data.to_excel("E:/返修结果存放/上海情绪.xlsx", index=False)
