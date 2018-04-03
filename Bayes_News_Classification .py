
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



### 8) 使用TF-IDF 提取关键字
import jieba.analyse
#打印未清洗数据
index = 2400
print(df_news['content'][index])

#打印清洗过后的数据
#使用jieba，连接起来，提取关键词
print(df_content['contents_clean'][index])
print("--------------------------------------------------")
content_S_str = "".join(contents_clean[index])
print(" ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))

### 9) LDA 主题模型
#    格式要求：list of list 形式，分词好的整个语料
#    找出文章中主题
#类似于无监督模型
#导入gensim库，自然语言处理，参考官网：https://radimrehurek.com/gensim/
from gensim import corpora, models, similarities
import gensim
#http://radimrehurek.com/gensim/

#首先做映射表（字典），相当于词袋，
dictionary = corpora.Dictionary(contents_clean)
corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean]

#使用gensim中的lda模型，可以参考：https://blog.csdn.net/angela2016/article/details/78208754
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20) #类似Kmeans自己指定K值

#一号分类结果（一共分了20个主题）
#在一号中，找出前5个
print (lda.print_topic(1, topn=5))

# 20个主题中，排名前5个主题词
for topic in lda.print_topics(num_topics=20, num_words=5):
    print (topic[1])

### 10）贝叶斯分类器对新闻数据进行分类
#课外学习“if __name__ == '__main__' ”，参考：http://blog.konghy.cn/2017/04/24/python-entry-program/
# 1）拿到数据，导入为DataFrame格式，带标签的
df_train=pd.DataFrame({'contents_clean':contents_clean,'label':df_news['category']})
df_train

#2）找出多少种label
df_train.label.unique()

#3）因为sklearn不认识这些label，需要把这些label转换为数字，用字典来做映射
label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育":5, "教育": 6,"文化": 7,"军事": 8,"娱乐": 9,"时尚": 0}
#把label进行替换 map()
df_train['label'] = df_train['label'].map(label_mapping)
df_train.head()

#4）sklearn, 先进行数据切分，分为训练集与测试集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, random_state=1)

#5）把训练集中的list转换为string，注意输入格式，后续需要把每条content转换为对应的向量，通过sklearn中的向量构造器
words = []
for line_index in range(len(x_train)):
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        #需要把list转换为string格式，可以用.join形式组合，并且用‘ ’空格取分开
        #join()用法参考：https://blog.csdn.net/weixin_40475396/article/details/78227747
        words.append(' '.join(x_train[line_index])) #words里面是训练数据
    except:
        print (line_index,word_index)
words[0]
print (len(words))

# 6）构造向量，有了上面的words的标准格式的内容，开始构建特征向量
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(analyzer='word', max_features=4000,  lowercase = False)
#fit一下
vec.fit(words)

# 7） 导入在sklearn中，把beyes拿出来，有输入特征，还有label值
from sklearn.naive_bayes import MultinomialNB
#实例化分类器对象
classifier = MultinomialNB()
#把words向量传入classifier
classifier.fit(vec.transform(words), y_train)

# 8）测试集同样的3步处理操作：内容转换为string，空格分开 --- 把内容转换为特征向量 ---- 导入贝叶斯模型传入（输入特征向量与label值）
test_words = []
for line_index in range(len(x_test)):
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        test_words.append(' '.join(x_test[line_index]))
    except:
         print (line_index,word_index)
test_words[0]

#9) 基于词频向量的，进行贝叶斯，结果
classifier.score(vec.transform(test_words), y_test)

# 10) 另一种构造向量的方式，不采用词频，采用TF-IDF模式构造向量，发现最后结果稍好一些
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word', max_features=4000,  lowercase = False)
vectorizer.fit(words)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)

#11）执行当前的结果，贝叶斯分类器的精度
classifier.score(vectorizer.transform(test_words), y_test)

#12）补充知识点：构造特征向量的格式问题：
#先举例
from sklearn.feature_extraction.text import CountVectorizer
#我有4篇文章
#【格式要求】：这里输入为list，每篇文章用逗号分开，每篇文章里面用空格分开，不能写成字符串
texts=["dog cat fish","dog cat cat","fish bird", 'bird']
#向量构造器，实例化一个对象
cv = CountVectorizer()
#进行向量转换
cv_fit=cv.fit_transform(texts)
#现在语料库中不重复的词有几个，一共是4中
print(cv.get_feature_names())
#打印转换好的向量，解释第一行：‘bird’单词在第一篇文章中出现0次，所以第一个值为0，‘cat’出现一次，所以为1，以此类推
print(cv_fit.toarray())
print(cv_fit.toarray().sum(axis=0))


from sklearn.feature_extraction.text import CountVectorizer
texts=["dog cat fish","dog cat cat","fish bird", 'bird']
#注意这里有 ngram_range ，可以让词组合，让种类更多更复杂，向量从原来的4维转为了9维，当然也别太多
#通常为ngram 为 2就行
cv = CountVectorizer(ngram_range=(1,4))
cv_fit=cv.fit_transform(texts)
print(cv.get_feature_names())
print(cv_fit.toarray())
print(cv_fit.toarray().sum(axis=0))
