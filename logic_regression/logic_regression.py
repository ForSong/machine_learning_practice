from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.externals import joblib
import pandas as pd
import numpy as np


def logic_regression():
    """
    逻辑回归做二分类进行癌症预测(根据细胞的属性特征)
    :return: None
    """
    # 读取数据
    # 如果没有列名，需要先指定类别名字
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        names=column_name)

    # print(data)

    # 缺失值进行处理
    data = data.replace(to_replace='?', value=np.nan)

    data = data.dropna()

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(data[column_name[1:10]], data[column_name[10]], test_size=0.25)
    # 进行标准化处理
    std = StandardScaler()

    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)
    # 打印对应特征的系数
    print(lg.coef_)

    y_predict = lg.predict(x_test)
    print("准确率：", lg.score(x_test, y_test))
    print("召回率", classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"]))


if __name__ == '__main__':
    logic_regression()
