import numpy as np
import pandas as pd
# import kmeans form sklearn
from sklearn.cluster import KMeans

"""添加新特征, 并保存文件train.npy, test.npy"""

def kmeans(X, k):
    # kmeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    # get labels
    labels = kmeans.labels_
    # get centroids
    centroids = kmeans.cluster_centers_
    return labels, centroids
    
def difference_velocity(df):
    """添加速度的差值特征

    Args:
        df (_type_): _description_
    """
    a = abs(df["spd-x1"] - df["spd-x2"])
    b = abs(df["spd-y1"] - df["spd-y2"])
    df['spd_diff'] = np.sqrt(a**2 + b**2)

def difference_distance(df):
    """添加距离的差值特征

    Args:
        df (_type_): _description_
    """
    a = abs(df["pos-x1"] - df["pos-x2"])
    b = abs(df["pos-y1"] - df["pos-y2"])
    df['pos_diff'] = np.sqrt(a**2 + b**2)
    

def difference_time(df):
    """发送时间和传输时间的差值

    Args:
        df (_type_): _description_
    """
    df['transfer_time'] = df['sendtime_2'] - df['sendtime_1']


df = pd.read_csv('../Dataset30/at.csv')
df.columns
label = df['AttackerType']
df.drop(['AttackerType'], axis=1, inplace=True)
difference_velocity(df)
difference_distance(df)
difference_time(df)
df['AttackerType'] = label
df.to_csv('../Dataset30/at_add.csv', index=False)