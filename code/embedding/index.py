import openai
import os
from openai.embeddings_utils import get_embedding, cosine_similarity
# 导入 pandas 包。Pandas 是一个用于数据处理和分析的 Python 库
# 提供了 DataFrame 数据结构，方便进行数据的读取、处理、分析等操作。
import pandas as pd
import datetime 
import ast
# 前置处理
os.environ["http_proxy"]="127.0.0.1:50918"
os.environ["https_proxy"]="127.0.0.1:50918"
with open('../api_key.txt', 'r') as f:
    OPENAI_API_KEY = f.readline().strip()
openai.api_key = OPENAI_API_KEY

# 任务目标
# 1. 读取data.csv的数据
# 2. 打印出对应的数据
# 3. 根据 "好吃" 找到对应的向量
# 4. 根据 "好吃" 和 "难吃" 找到对应的相似度

input_datapath = "data.csv"
df = pd.read_csv(input_datapath, index_col=0)
df = df[["comment","address","Score"]]
# 去掉空值
df = df.dropna()
# df.head(3)
# # 打印查看 这个时候会输出表格
# print(df.head(3))
# print('-------------------')
# # 打印表格第一行的comment
# print(df.iloc[0, 0])
# print('-------------------')
# # 打印comment列的所有数据
# print(df["address"])

# 模型类型
# 建议使用官方推荐的第二代嵌入模型：text-embedding-ada-002
# 转过一次了就可以不进行转化了
# # 将comment列的所有数据转换成向量
embedding_model = "text-embedding-ada-002"
output_datapath = "embeddings.csv"
# # output_datapath = f"embeddings_1k_{date_time_str}.csv"
# df["embedding"] = df.comment.apply(lambda x: get_embedding(x, engine=embedding_model))

# now = datetime.datetime.now()
# date_format = "%Y-%m-%d-%H-%M-%S"
# date_time_str = now.strftime(date_format)
# df.to_csv(output_datapath)

# 根据对应的词找到最接近的词
# 先读取对应的向量
df_embedded = pd.read_csv(output_datapath, index_col=0)
# print(df_embedded["embedding"])

# print(len(df_embedded["embedding"][0]))
# print(type(df_embedded["embedding"][0]))
# 读取到的是字符串，此时需要将字符串转成向量
df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)
# print(len(df_embedded["embedding_vec"][0]))

# 找到对应的向量

def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine=embedding_model
    )
    df["similarity"] = df.embedding_vec.apply(lambda x: cosine_similarity(x, product_embedding))
    print(f'与 {product_description} 最相近的是: \n')
    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .comment
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results
res = search_reviews(df_embedded, '难吃', n=3)

# 使用 t-SNE 可视化 数据

# 导入 NumPy 包，NumPy 是 Python 的一个开源数值计算扩展。这种工具可用来存储和处理大型矩阵，
# 比 Python 自身的嵌套列表（nested list structure)结构要高效的多。
import numpy as np
# 从 matplotlib 包中导入 pyplot 子库，并将其别名设置为 plt。
# matplotlib 是一个 Python 的 2D 绘图库，pyplot 是其子库，提供了一种类似 MATLAB 的绘图框架。
import matplotlib.pyplot as plt
import matplotlib

# 从 sklearn.manifold 模块中导入 TSNE 类。
# TSNE (t-Distributed Stochastic Neighbor Embedding) 是一种用于数据可视化的降维方法，尤其擅长处理高维数据的可视化。
# 它可以将高维度的数据映射到 2D 或 3D 的空间中，以便我们可以直观地观察和理解数据的结构。
from sklearn.manifold import TSNE
# 首先，确保你的嵌入向量都是等长的
assert df_embedded['embedding_vec'].apply(len).nunique() == 1
# 将嵌入向量列表转换为二维 numpy 数组
matrix = np.vstack(df_embedded['embedding_vec'].values)
colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
# 创建一个基于预定义颜色的颜色映射对象
colormap = matplotlib.colors.ListedColormap(colors)
def showByScore():
    # print(matrix)
    # 创建一个 t-SNE 模型，t-SNE 是一种非线性降维方法，常用于高维数据的可视化。
    # n_components 表示降维后的维度（在这里是2D）
    # perplexity 可以被理解为近邻的数量
    # random_state 是随机数生成器的种子
    # init 设置初始化方式
    # learning_rate 是学习率。
    tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='random', learning_rate=200)
    # 使用 t-SNE 对数据进行降维，得到每个数据点在新的2D空间中的坐标
    vis_dims = tsne.fit_transform(matrix)
    # 定义了五种不同的颜色，用于在可视化中表示不同的等级
    # 从降维后的坐标中分别获取所有数据点的横坐标和纵坐标
    x = [x for x,y in vis_dims]
    y = [y for x,y in vis_dims]

    # print(vis_dims)
    # 根据数据点的评分（减1是因为评分是从1开始的，而颜色索引是从0开始的）获取对应的颜色索引
    color_indices = df_embedded.Score.values - 1

    # 确保你的数据点和颜色索引的数量匹配
    assert len(vis_dims) == len(df_embedded.Score.values)
    # 使用 matplotlib 创建散点图，其中颜色由颜色映射对象和颜色索引共同决定，alpha 是点的透明度
    plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)

    # 为图形添加标题
    plt.title("Amazon ratings visualized in language using t-SNE")
    plt.show()

# showByScore()

# 聚类分析 使用 K-Means 聚类，然后使用 t-SNE 可视化
# 从 scikit-learn中导入 KMeans 类。KMeans 是一个实现 K-Means 聚类算法的类。
from sklearn.cluster import KMeans
# 提前设定好有5类了
n_clusters = 5

# 创建一个 KMeans 对象，用于进行 K-Means 聚类。
# n_clusters 参数指定了要创建的聚类的数量；
# init 参数指定了初始化方法（在这种情况下是 'k-means++'）；
# random_state 参数为随机数生成器设定了种子值，用于生成初始聚类中心。
# n_init=10 消除警告 'FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4'
kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42, n_init=10)

# 使用 matrix（我们之前创建的矩阵）来训练 KMeans 模型。这将执行 K-Means 聚类算法。
kmeans.fit(matrix)

# kmeans.labels_ 属性包含每个输入数据点所属的聚类的索引。
# 这里，我们创建一个新的 'Cluster' 列，在这个列中，每个数据点都被赋予其所属的聚类的标签。
df_embedded['Cluster'] = kmeans.labels_

print("----------------------------\n",df_embedded[['comment','Cluster']])

# 然后，你可以使用 t-SNE 来降维数据。这里，我们只考虑 'embedding_vec' 列。
tsne_model = TSNE(n_components=2, perplexity=5,random_state=42)
vis_data = tsne_model.fit_transform(matrix)

# 现在，你可以从降维后的数据中获取 x 和 y 坐标。
x = vis_data[:, 0]
y = vis_data[:, 1]

# 'Cluster' 列中的值将被用作颜色索引。
color_indices = df_embedded['Cluster'].values

# 使用 matplotlib 创建散点图，其中颜色由颜色映射对象和颜色索引共同决定
plt.scatter(x, y, c=color_indices, cmap=colormap)

# 为图形添加标题
plt.title("Clustering visualized in 2D using t-SNE")

# 显示图形
plt.show()