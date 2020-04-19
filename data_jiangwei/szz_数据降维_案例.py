import pandas as pd
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)  # 显示完整的列
pd.set_option('display.max_rows', None)  # 显示完整的行
# 读取数据
prior = pd.read_csv("./instacart-market-basket-analysis/order_products__prior.csv")
orders = pd.read_csv("./instacart-market-basket-analysis/orders.csv")
aisles = pd.read_csv("./instacart-market-basket-analysis/aisles.csv")
products = pd.read_csv("./instacart-market-basket-analysis/products.csv")

# 合并四张表格：最终目的是得到  用户-物品类别
merge = pd.merge(prior, orders, on=['order_id', 'order_id'])
merge = pd.merge(merge, products, on=['product_id', 'product_id'])
mt = pd.merge(merge, aisles, on=['aisle_id', 'aisle_id'])

# 交叉表（特殊的分组工具）
crosstab = pd.crosstab(mt['user_id'], mt['aisle'])

# print(crosstab.head(2))

# 进行主成分分析
pca = PCA(n_components=0.9)
data = pca.fit_transform(crosstab)
print(data)
