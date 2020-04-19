from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split

# li = load_iris()

# print("获取特征值")
# print(li.data)
# print("目标值")
# print(li.target)
#
# print(li.DESCR)

# 注意返回值，训练集 train  x_train表示训练集中的特征值, y_trian表示训练集中的目标值 和测试集 test
# x_test 测试集中的特征值， y_test 表示测试集中的目标值
# 训练集占75%
# x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
#
# print("训练集特征值和目标值：", x_train, y_train)
# print("测试集特征值和目标值：", x_test, y_test)

# 新闻数据集合
# news = fetch_20newsgroups(subset='all')
# print(news.data)
# print(news.target)

lb = load_boston()

print("获取特征值")
print(lb.data)
print("目标值")
print(lb.target)
print(lb.DESCR)
