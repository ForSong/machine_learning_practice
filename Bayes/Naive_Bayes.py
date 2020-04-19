# 导入新闻的数据库集
from sklearn.datasets import fetch_20newsgroups
# 导入分割训练集和测试集的包
from sklearn.model_selection import train_test_split
# 导入特征抽取的包
from sklearn.feature_extraction.text import TfidfVectorizer
# 到如朴素贝叶斯算法的包
from sklearn.naive_bayes import MultinomialNB
# 导入分类的精确率和召回率的包
from sklearn.metrics import classification_report


def naive_bayes():
    """
    朴素贝叶斯进行文本分类
    :return: None
    """
    # 首先获取数据
    news = fetch_20newsgroups()
    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()
    # 以训练集单重的词的列表，进行每篇文章重要性统计
    x_train = tf.fit_transform(x_train)
    print(tf.get_feature_names())
    # 将x_test转化
    x_test = tf.transform(x_test)
    # 进行朴素贝叶斯算法的预测
    mlt = MultinomialNB(alpha=1.0)
    # 打印训练集的内容
    print(x_train.toarray())
    # 传入训练集的特征值和训练集的结果
    mlt.fit(x_train, y_train)
    # 预测测试集中的结果
    y_predict = mlt.predict(x_test)
    print("预测的文章类别为：", y_predict)
    # 得出准确率, 这个准确率很难提高，因为没有超参数可以调整
    # 所以朴素贝叶斯算法的好坏只取决于训练集的好坏
    print("预测的准确率为：", mlt.score(x_test, y_test))

    # 打印每个类别的精确率和召回率
    print("每个类别的精确率和召回率：", classification_report(y_test, y_predict, target_names=news.target_names))


if __name__ == '__main__':
    naive_bayes()
