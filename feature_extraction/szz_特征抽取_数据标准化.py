from sklearn.preprocessing import StandardScaler


def stand():
    """
    标准化缩放
    在数据量较大的情况下可以减少极端数据的影响
    """
    std = StandardScaler()
    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])

    print(data)


if __name__ == '__main__':
    stand()
