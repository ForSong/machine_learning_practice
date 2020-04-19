# 特征抽取

# 导包
from sklearn.feature_extraction import DictVectorizer


def dictvec():
    """
    字典数据抽取
    如果是类别的类型：比如城市，就会单独作为一列抽出
    例如：输入
    [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '郑州', 'temperature': 100}]

    输出（sparse=False）：
    ['city=上海', 'city=北京', 'city=郑州', 'temperature']
    [[  0.   1.   0. 100.]
    [  1.   0.   0.  60.]
    [  0.   0.   1. 100.]]

    使用One-Hot编码处理类别型的特征

    再到（sparse=False）：
    (0, 1)	1.0
    (0, 3)	100.0
    (1, 0)	1.0
    (1, 3)	60.0
    (2, 2)	1.0
    (2, 3)	100.0
    第一列是上面二维数组中的定为，第二列是对应的值，值为0的位置忽略
    如果是数组的形式，则先转换为字典
    :return: None
    """
    list1 = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '郑州', 'temperature': 100}]
    # 实例化
    dict = DictVectorizer()

    # 调用fit_transform
    # 将会返回scipy.sparse类型的数据
    # 1. 节约内存
    data = dict.fit_transform(list1)
    print(dict.get_feature_names())
    print(data)
    print(type(data))
    return None


if __name__ == "__main__":
    dictvec()
