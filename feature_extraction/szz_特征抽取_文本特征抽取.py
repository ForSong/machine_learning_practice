from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import jieba


def contvec():
    """
    对文本进行特征抽取

    1. 统计所有的此，重复的只看做一次
    2. 对每一个列表中的元素（一句话）在词的列表中进行统计每个此出现的次数
    3. 单个字母是不统计，因为不会影响文本的分析
    :return: None
    """
    # 实例化
    cv0 = CountVectorizer()
    cv1 = CountVectorizer()

    data0 = cv0.fit_transform(["life is short, i like python", "life is too long, i like python"])
    data1 = cv1.fit_transform(["人生 苦短，我 喜欢 python", "人生 漫长，我 喜欢 python"])
    print(cv0.get_feature_names())  # ['is', 'life', 'like', 'long', 'python', 'short', 'too']
    # data 是一个基于numpy的sparse矩阵
    print(data0.toarray())  # [[1 1 1 0 1 1 0] [1 1 1 1 1 0 1]]
    print(type(data0.toarray()))

    print(cv1.get_feature_names())
    print(data1.toarray())


def cutword():
    '''
    分词方法
    :return:
    '''

    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好")
    con2 = jieba.cut("时间是一条长河，裹挟着时光，带走了青春")
    con3 = jieba.cut("如果历史是一面镜子，它总能照出人类的丑陋，而人类却视而不见")

    # 将生成器转换为列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 把列表转换为字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)
    return c1, c2, c3


def hanzivec():
    """
    中文特征值话
     对于中文来说，需要进行中文分词，可以使用jieba包进行分词
    :return:
    """
    c1, c2, c3 = cutword()
    print(c1, c2, c3)
    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names())
    print(data.toarray())


if __name__ == '__main__':
    # contvec()
    hanzivec()
