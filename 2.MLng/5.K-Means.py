# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('datasets/sales_data_sample.csv',encoding='unicode_escape')
df.head()

# %%
df.info()

# %%
df.isnull().sum()

# %%
df.columns

# %% [markdown]
# ## Reducing dimensionality by dropping redundant columns

# %%
# Dropping redundant columns
drop_columns = ['CUSTOMERNAME', 'PHONE','ADDRESSLINE1', 'ADDRESSLINE2', 'CITY', 'STATE', 'POSTALCODE', 'TERRITORY', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME']
df.drop(columns=drop_columns, axis=1, inplace=True)

# %%
df.shape

# %%
df.head()

# %%
df.isnull().sum()

# %% [markdown]
# ## Converting categorical to quantitative values

# %%
# Converting categorical data to quantitative data

# Here are some of the benefits of converting categorical data to quantitative data 
# before performing K-means clustering:

# 1. It makes the clustering process more efficient and accurate.
# 2. It makes the results of the clustering analysis easier to interpret.
# 3. It allows you to use a wider range of distance metrics, such as Euclidean distance and Manhattan distance.
# Overall, it is generally recommended to convert categorical data to quantitative data before performing K-means clustering.


# %%
# Using OneHotEncoding for COUNTRY and PRODUCTLINE
df = pd.get_dummies(df, columns=['COUNTRY','PRODUCTLINE'], dtype=int)
df

# %%
df['DEALSIZE'].unique()

# %%
# Using Label Encoding for DEAL SIZE

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['DEALSIZE'] = encoder.fit_transform(df['DEALSIZE'])

df['DEALSIZE']

# %%
df['STATUS'].unique()

# %%
# Using LabelEncoding for STATUS 
# Assigning 1 for success 2 for issues and 0 for in progress

status_map = {'Shipped':1, 'Disputed':2 , 'Cancelled': 2, 'On Hold': 2,'In Process': 0,'Resolved': 0}
df['STATUS'] = df['STATUS'].map(status_map)
df['STATUS']

# %%
df.info()

# We need to convert object data types to integer for K Means Algorithm

# %%
len(df['PRODUCTCODE'].unique())

# %%
# Since number of unique product codes is more we will use LabelEncoding for PRODUCTCODE instead of OneHotEncoding
# Using OneHotEncoding will increase dimensions too much

df['PRODUCTCODE'] = encoder.fit_transform(df['PRODUCTCODE'])
df['PRODUCTCODE']

# %% [markdown]
# ## Converting remaining object data type to integer

# %%
# Converting object date data type numeric dates (object -> datetime -> int) 
# Integer datatype is required for scaling and K means

df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
df['ORDERDATE'] = pd.to_numeric(df['ORDERDATE']).astype(int) / 10**9

# %%
df.head()

# %%
# Now we will remove further redundant columns 

# If you have two highly correlated columns in your dataset, and you include both of them in your K-means clustering
# model, the algorithm will be essentially using the same information twice. This can lead to overfitting and 
# can reduce the accuracy of the clustering results.

# %%
# Taking first 12 columns
plt.figure(figsize=(10,12))
sns.heatmap(df.iloc[:,:12].corr(), cmap='coolwarm', annot=True)

# %%
# Dropping highly correlated columns ORDERDATE [corr(YEAR_ID,ORDERDATE) = 0.9] and QTR_ID [corr(QTR_ID,MONTH_ID) = 0.98]

df.drop(columns=['ORDERDATE','QTR_ID'], axis=1, inplace=True)
df.head()

# %%
# Creating datasets for K Means and Hierarchical clustering
dfk = df.copy()
dfh = df.copy()

# %% [markdown]
# # K-Means Clustering

# %%
# Cleaning outliers as K Means is sensitive to outliers
# Probable outliers can be present in ['QUANTITYORDERED','PRICEEACH','SALES']

def clean_outlier(data):
    ninety = np.percentile(data, 90)
    ten = np.percentile(data, 10)
    
    data = np.where(data >= ninety, ninety, data)
    data = np.where(data <= ten, ten, data) 
    

outlier_colums = ['QUANTITYORDERED','PRICEEACH','SALES']
for col in outlier_colums:
    clean_outlier(dfk[col])

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
dfk_scaled = scaler.fit_transform(dfk)
dfk_scaled

# %% [markdown]
# ## Elbow Method

# %%
from sklearn.cluster import KMeans

# %%
wcss = []
for i in range(1,16):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_predict(dfk_scaled)
    wcss.append(kmeans.inertia_)

wcss

# %%
plt.plot(range(1,16),wcss, marker='o')
plt.title('Elbow Curve')
plt.ylabel('Inertia')
plt.xlabel('K')

# %% [markdown]
# ## Silhoutte Coefficient

# %%
from sklearn.metrics import silhouette_score

silhoutte_avg = []

for i in range(2,16): # Start from 2 
    kmeans = KMeans(n_clusters=i)
    labels= kmeans.fit_predict(dfk_scaled)
    silhoutte_avg.append(silhouette_score(dfk_scaled,labels))
    
silhoutte_avg

# %%
plt.figure(figsize=(10,20))
plt.plot(range(2,16),silhoutte_avg, marker='o')
plt.title('Silhoutte Curve')
plt.ylabel('Silhoutte Coeff')
plt.xlabel('Clusters')

# %%
kmeans = KMeans(n_clusters=12, init='k-means++')
kmeans.fit_predict(dfk_scaled)
labels = kmeans.labels_
labels

# %%
cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=dfk.columns)
cluster_centers

# %% [markdown]
# ## Hierarchical Clustering

# %%
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10,7))
dend = shc.dendrogram(shc.linkage(dfh, method='ward'))

# %%
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
labels = cluster.fit_predict(dfh)

# %%
labels

# %%



