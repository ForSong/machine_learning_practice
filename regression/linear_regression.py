from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def my_linear():
    """
    线性回归直接预测房子价格
    :return: None
    """
    # 获取数据
    lb = load_boston()
    # 分割数据及到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    print(y_train, y_test)
    # 进行标准化处理
    # 目标值和特征值都需要进行标准化处理，因为是计算公式计算处理的，如果值进行特征值的标准化，那么计算
    # 的结果会和目标值的值相差非常大，因此，两者都需要进行标准化，最终结果再转换回来即可，有对应的API
    # 这里需要实例化两个标准化API
    std_x = StandardScaler()

    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()
    # 转换器，estimator 要求数据必须是二维形状
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # estimator预测
    lr = LinearRegression()

    lr.fit(x_train, y_train)

    # 打印参数，也就是很多w的值
    print(lr.coef_)

    # 在线性回归中不再用准确率评估
    # 预测测试集的房子价格
    y_predict = std_y.inverse_transform(lr.predict(x_test))

    print("测试集中额每个样本的测试价格：", y_predict)


if __name__ == '__main__':
    my_linear()
