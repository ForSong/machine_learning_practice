from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# 导入网格搜索包
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


def knncls():
    """
    K-邻近预测用户签到位置
    :return: None
    """
    # 读取数据
    data = pd.read_csv("./data/train.csv")  # 返回DataFrame类型的数据

    # print(data.head(2))
    # 处理数据
    # 1. 减小数据量
    # 按照x,y的位置缩小
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")
    # 2. 处理时间的数据,注意这个方法在pd下
    time_value = pd.to_datetime(data['time'], unit='s')

    # print(time_value)

    # 把日期格式转换为字典格式
    time_value = pd.DatetimeIndex(time_value)

    # 构造一些特征
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    # 把时间戳特征删除, 注意在sklearn中列是0 但是在pandas中的列是1 ！！！
    data = data.drop(['time'], axis=1)
    # 将签到人数少于n个的目标位置删除
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()
    # 对原来的data进行处理，值保留签到次数大于3的，做法就是和tf中的数据进行比较
    data = data[data['place_id'].isin(tf.place_id)]

    # 取出数据中的特征值和目标值
    y = data['place_id']

    x = data.drop(['place_id', 'row_id'], axis=1)

    # 进行数据的分割，分割成训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # print(data.head(2))
    # 特征工程（标准化）
    std = StandardScaler()

    # 对测试机和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)

    x_test = std.transform(x_test)

    # 进行算法流程
    knn = KNeighborsClassifier()

    # # fit方法 predict,score
    # knn.fit(x_train, y_train)
    #
    # # 得出预测结果knn.predict
    # y_predict = knn.predict(x_test)
    #
    # print("预测的目标签到位置为：", y_predict)
    #
    # # 得出准确率
    # print("预测的准确率：", knn.score(x_test, y_test))

    # 构造一些参数的值进行搜索
    param = {"n_neighbors": [3, 5, 10]}

    # 进行网格搜索
    gc = GridSearchCV(knn, param_grid=param, cv=10)
    gc.fit(x_train, y_train)
    # 预测准确率
    print("在测试集上的准确率：", gc.score(x_test, y_test))

    print("在交叉验证中最好的验证结果", gc.best_score_)

    print("选择最好的模型是", gc.best_estimator_)

    print("每个超参数每次交叉验证的结果", gc.cv_results_)


if __name__ == '__main__':
    knncls()
