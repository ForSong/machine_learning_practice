from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
li = load_iris()

data = li.data
print(data)

target = li.target
print(target)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

# 特征工程
std = StandardScaler()
x_train = std.fit_transform(x_train)

x_test = std.transform(x_test)

# 使用KNN算法
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
# 得出预测结果
predict = knn.predict(x_test)

print("预测结果", predict)

# 得出准确率
print("预测的准确率", knn.score(x_test,y_test))

