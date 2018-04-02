
# coding: utf-8

# 1) 【导入常用的处理包】
# ### 如何安装jieba：https://pypi.python.org/pypi/jieba/ ###

import pandas as pd
import numpy as np
import os
import jieba #需要先安装jieba库

os.chdir('/Users/a1/Desktop/算法实战/贝叶斯_新闻分类/贝叶斯-新闻分类/data')

# 2) 【导入数据】
#这里使用的是pd.read_table: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_table.html
df_news = pd.read_table('val.txt', names=['category', 'theme', 'URL', 'content'], encoding = 'utf-8')
df_news = df_news.dropna()
df_news.head()

#df_newss = pd.read_table('val.txt')
#df_newss 

df_news.shape #看一下表格大概结构

# 3）【使用jieba对每一条content内容分词操作，存入list】
content = df_news.content.values.tolist() #在pandas中拿某一列的值，并把它转化为list格式
print(content[1000])

content_S = []
for line in content: #对每一个content中内容循环，分词操作
    current_segment = jieba.lcut(line) #使用jieba.lcut()命令进行分词
    if len(current_segment) > 1 and current_segment != '\r\r': #挑出content中长度大于1，且不是分隔符的字符
        content_S.append(current_segment) 

content_S[1000] #查看分词结果

df_content = pd.DataFrame({'content_S':content_S})
df_content.head()


# 4) 【开始清洗：去除停用词】

#os.chdir('/Users/a1/Desktop/算法实战/贝叶斯_新闻分类/贝叶斯-新闻分类/')

# 取出停用词表
stopwords=pd.read_csv("stopwords.txt",index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8')
stopwords

type(stopwords)
type(df_content)

#定义一个函数，输入：list格式的文档与停用词表， 输出：去除停用词的文档，与去除的文档词
def drop_stopwords(contents,stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean,all_words #注意缩进
    #print (contents_clean)
        

contents = df_content.content_S.values.tolist()    
stopwords = stopwords.stopword.values.tolist()
contents_clean,all_words = drop_stopwords(contents,stopwords)

#df_content.content_S.isin(stopwords.stopword)
#df_content=df_content[~df_content.content_S.isin(stopwords.stopword)]
#df_content.head()

# 清洗完成第一步，去除了contents中在停词表出现过的词汇

df_content = pd.DataFrame({'contents_clean' : contents_clean}) #把处理完成的list，改成字典格式，传入DataFrame格式
df_content.head()

df_all_words = pd.DataFrame({'all_words':all_words}) #把处理完成的所有的all_words打印出来
df_all_words


# 5）【统计词频，groupby & matplotlib】

#计算词频groupby: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
import numpy
words_count = df_all_words.groupby(by=['all_words'])['all_words'].agg({"count":numpy.size})
words_count = words_count.reset_index().sort_values(by=["count"], ascending = False)
words_count.head()


# 6) 【使用词云wordcloud展示】

#使用词云 ：https://github.com/amueller/word_cloud
from wordcloud import WordCloud #导入安装的词云库
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib


#使用画图：https://matplotlib.org/users/customizing.html
matplotlib.rcParams['figure.figsize'] = [10.0, 8.0]

wordcloud = WordCloud(font_path='simhei.ttf', background_color = "white", max_font_size = 80)
word_frequence = {x[0]:x[1] for x in words_count.head(100).values}  #前100个词，画出来
wordcloud = wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)

