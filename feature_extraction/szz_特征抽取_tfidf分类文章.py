from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba
import numpy as np


def cutword(line1, line2, line3):
    list = [line1, line2, line3]
    list_con = []
    for line in list:
        con = jieba.cut(line)
        content = ' '.join(con)
        list_con.append(content)
    return list_con


def hanzivec():
    """
    使用tfidf类来比较词语的重要性
    :return:
    """
    list_lines = cutword("今天很残酷，明天更残酷，后天很美好", "时间是一条长河，裹挟着时光，带走了青春", "如果历史是一面镜子，它总能照出人类的丑陋，而人类却视而不见")

    tf = TfidfVectorizer()
    data = tf.fit_transform([list_lines[0], list_lines[1], list_lines[2]])
    print(tf.get_feature_names())
    # print(data.toarray())
    toarray = data.toarray()
    print(np.around(toarray,decimals=2))


if __name__ == '__main__':
    hanzivec()
