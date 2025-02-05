#!/usr/bin/env python
# coding: utf-8

# # sklearn-LDA

# 代码示例：https://mp.weixin.qq.com/s/hMcJtB3Lss1NBalXRTGZlQ （玉树芝兰） <br>
# 可视化：https://blog.csdn.net/qq_39496504/article/details/107125284  <br>
# sklearn lda参数解读:https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
# <br>中文版参数解读：https://blog.csdn.net/TiffanyRabbit/article/details/76445909
# <br>LDA原理-视频版：https://www.bilibili.com/video/BV1t54y127U8
# <br>LDA原理-文字版：https://www.jianshu.com/p/5c510694c07e
# <br>score的计算方法：https://github.com/scikit-learn/scikit-learn/blob/844b4be24d20fc42cc13b957374c718956a0db39/sklearn/decomposition/_lda.py#L729
# <br>主题困惑度1：https://blog.csdn.net/weixin_43343486/article/details/109255165
# <br>主题困惑度2：https://blog.csdn.net/weixin_39676021/article/details/112187210

# ## 1.预处理

# In[3]:


import os
import pandas as pd
import re
import jieba
import jieba.posseg as psg
jieba.load_userdict('E:/lda/stop_dic/dict.txt')

# In[4]:


output_path = 'E:/lda/result'
file_path = 'E:/lda'
os.chdir(file_path)
data=pd.read_excel("E:/总表.xlsx")#content type
os.chdir(output_path)
dic_file = "E:/lda/stop_dic/dict1.txt"
stop_file = "E:/lda/stop_dic/stopwords.txt"


# In[35]:

jieba.load_userdict(dic_file)
jieba.load_userdict(stop_file)
jieba.initialize()
try:
    stopword_list = open(stop_file,encoding ='utf-8')
except:
    stopword_list = []
    print("error in stop_file")
stop_list = []
flag_list = ['n','nz','vn']
for line in stopword_list:
    line = re.sub(u'\n|\\r', '', line)
    stop_list.append(line)

def chinese_word_cut(mytext):  
    word_list = []  
    seg_list = psg.cut(mytext)  
    for seg_word in seg_list:  
        word = re.sub(u'[^\u4e00-\u9fa5]','',seg_word.word)  
        find = 0  
        for stop_word in stop_list:  
            if stop_word == word or len(word)<2:     #this word is stopword  
                find = 1  
                break  
        if find == 0 and seg_word.flag in flag_list:  
            word_list.append(word)        
    return (" ").join(word_list)  
  
 
if not stop_list:  
    print("Error in stop file.")  

data["content"]=data.content.astype(str)
data["content_cutted"] = data.content.apply(chinese_word_cut)


# ## 2.LDA分析

# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[38]:


def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword


# In[39]:


n_features = 1000 
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words="english",
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(data.content_cutted)


# In[40]:


n_topics = 3
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='batch',
                                learning_offset=50,
#                                 doc_topic_prior=0.1,
#                                 topic_word_prior=0.01,
                               random_state=0)
lda.fit(tf)


# In[11]:
n_top_words = 10

# 使用 get_feature_names_out() 获取特征名
tf_feature_names = tf_vectorizer.get_feature_names_out()

# 假设 print_top_words 函数是自定义函数，你需要检查函数的定义是否接受正确的参数
topic_word = print_top_words(lda, tf_feature_names, n_top_words)



# ### 2.2输出每篇文章对应主题 

# In[12]:
import numpy as np

# In[13]:

topics=lda.transform(tf)


# In[28]:


topic = []
for t in topics:
    topic.append("Topic #"+str(list(t).index(np.max(t))))
data['概率最大的主题序号']=topic
data['每个主题对应概率']=list(topics)
data.to_excel("E:/data_topic.xlsx",index=False)



