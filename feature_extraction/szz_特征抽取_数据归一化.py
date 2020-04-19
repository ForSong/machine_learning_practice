from sklearn.preprocessing import MinMaxScaler


def mm():
    """
    对数据进行归一化
    目的是使得一个特征对最终的结果不会造成更大的影响，例如某一列的值很大，
    """
    mm = MinMaxScaler(feature_range=(2, 3))
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)


if __name__ == '__main__':
    mm()
