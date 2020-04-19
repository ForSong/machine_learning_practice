import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
# 导入随机森林包
from sklearn.ensemble import RandomForestClassifier


def decision():
    """
    决策树对泰坦尼克号进行预测生死
    :return: None
    """
    # 获取数据
    titan = pd.read_csv()

    # 处理数据，找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']
    print(x)
    # 缺失值处理
    x['age'].fillna(x['age'].mean, inplace=True)
    # 分割数据集到训练集和测试机
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    print(x_train)
    # 进行处理(特征工程) 特征->类别->one hot编码
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.todict(orient="records"))
    print(dict.get_feature_names())
    x_test = dict.fit_transform(x_test.todict(orient="records"))
    # print(x_train)
    # 用决策数进行预测
    # dec = DecisionTreeClassifier(max_depth=5)
    # dec.fit(x_train, y_train)
    # # 预测准确率
    # print("预测的准确率：", dec.score(x_test, y_test))
    # # 导出树的结构
    # export_graphviz(dec, out_file="./tree.dot",
    #                 feature_names=["年龄", "pclass=1st", "pclass=2st", "pclass=3st", "女性", "男性"])
    # 随机森林进行预测(超参数调优)
    rf = RandomForestClassifier()

    # 构造参数字典
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    # 网格搜索和交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)

    print("准确率：", gc.score(x_test, y_test))

    print("查看选择的参数模型：", gc.best_estimator_)


if __name__ == '__main__':
    decision()
