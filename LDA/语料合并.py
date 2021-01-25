# -*- coding: UTF-8 -*-
__author__ = 'zy'
__time__ = '2020/12/6 17:06'
file_list=['1.txt','2.txt','3.txt','4.txt','5.txt','6.txt']
result=""
for i in file_list:
    with open(i,'r',encoding='utf-8') as f:
        tmp=f.read()
        result=result+tmp
result=result.replace(' ',"").replace('\n',' ')
res=result.split('。')
print(len(res))
print(res)
res_list=[]
import jieba
for i in res:
    if i.strip()!="":
        res_list.append(' '.join(jieba.cut(i)))
    else:
        pass
print(res_list)

import jieba

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = res_list
vector = TfidfVectorizer()
tfidf = vector.fit_transform(corpus)
# print (tfidf)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#这个迭代次数至少上千次

#多进行计算
lda = LatentDirichletAllocation(n_topics=10, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
docres = lda.fit_transform(tfidf)

print(len(lda.components_[1])) #4306个词语
print(docres)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# % matplotlib
# inline


####
# Per topic: (token, pseudocount)
# pseudocount represents the number of times word j was assigned to topic i
#
# We can to convert these to a normalized form -- well you don't have to
# but it's easier to understand the output in this form.  Also, this is
# consistent with how Gensim performs.  After some consideration, we will plot these out.
####
import pandas as pd

def display_topics(model, feature_names, no_words=10, plot=False, plot_dim=(5, 2)):
    topics_tokens = []

    for topic_idx, topic in enumerate(model.components_):

        topic = zip(feature_names, topic)
        topic = sorted(topic, key=lambda pair: pair[1])

        topic_words = [(token, counts)
                       for token, counts in topic[:-no_words - 1:-1]]

        topics_tokens.append(topic_words)

        if not plot:
            print("Topic %d:" % (topic_idx))
            print(topic_words)

    if plot:

        plot_matrix = np.arange(10).reshape(5, 2)

        fig, ax = plt.subplots(figsize=(10, 10), nrows=5, ncols=2)

        topics = [
            {key: value for key, value in topic}
            for topic in topics_tokens
        ]

        row = 0

        for topic_id, topic in enumerate(topics):
            column = (0 if topic_id % 2 == 0 else 1)

            chart = pd.DataFrame([topic]).iloc[0].sort_values(axis=0)
            chart.plot(kind="barh", title="Topic %d" % topic_id, ax=ax[row, column])

            row += 1 if column == 1 else 0

        plt.tight_layout()


display_topics(lda , vector.get_feature_names(), no_words=10, plot=True)

plt.show()

