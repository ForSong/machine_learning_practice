from sklearn.impute import SimpleImputer
import numpy as np


def im():
    """
    缺失值处理
    :return:None
    """
    im = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = im.fit_transform([[1., 2.], [np.nan, 3.], [7., 6.]])

    print(data)


if __name__ == '__main__':
    im()
