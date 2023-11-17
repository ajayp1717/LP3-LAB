# %%
import warnings
warnings.filterwarnings('ignore')

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('./datasets/uber.csv')
df.head()

# %%
df.shape

# %% [markdown]
# **Preprocessing the dataset**

# %%
df.info()

# %%
len(df['key'].unique())

# %%
# Dropping redundant columns
df.drop(columns=['Unnamed: 0','key'], axis=1, inplace=True)

# %%
df.head()

# %%
df.isnull().sum()

# %%
df.columns

# %%
df.dropna(inplace=True)
df.isnull().sum()

# %%
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime']).astype(int) / 10**9
df.head(10)

# %%
df.info()

# %% [markdown]
# **Handling Outliers**

# %%
for col in df.columns:
    plt.figure(figsize=(5,5))
    plt.boxplot(x=df[col])
    plt.title(col)

# %%
# Handling outliers in longitudes and latitudes

df = df[
    (df['pickup_latitude'] <= 90) & (df['dropoff_latitude'] <= 90) & 
    (df['pickup_latitude'] >= -90) & (df['dropoff_latitude'] >= -90) &
    (df['pickup_longitude'] <= 180) & (df['dropoff_longitude'] <= 180) &
    (df['pickup_longitude'] >= -180) & (df['dropoff_longitude'] >= -180)
]

df.shape

# %% [markdown]
# **Correlation**

# %%
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# %% [markdown]
# **Calculating distance parameter**

# %%
def calc_dist(lat_1, lat_2, lon_1, lon_2):
    lat_1,lat_2,lon_1,lon_2 = map(np.radians, [lat_1,lat_2,lon_1,lon_2])
    diff_lat = lat_2 - lat_1
    diff_lon = lon_2 - lon_1
    
    dist = 2 * 6371 * np.arcsin(np.sqrt(np.sin(diff_lat/2)**2 + np.cos(lat_1)*np.cos(lat_2)*np.sin(diff_lon/2)**2))

    return dist

# %%
df['Distance'] = [calc_dist(
                    df['dropoff_latitude'][i],
                    df['pickup_latitude'][i],
                    df['dropoff_longitude'][i],
                    df['pickup_longitude'][i])
                    for i in df.index
                 ]
df.head(10)

# %%
sns.boxplot(x=df['Distance'])

# %%
# Removing distance outliers

q1 = np.percentile(df['Distance'],25)
q3 = np.percentile(df['Distance'],75)

iqr = q3-q1 
upper_limit = q3 + 1.5*iqr
lower_limit = q1 - 1.5*iqr

df = df[(df['Distance'] < upper_limit) & (df['Distance'] > lower_limit)]

df.shape

# %% [markdown]
# **Regression Models**

# %%
X = df[['pickup_datetime','passenger_count','Distance']]
y = df.iloc[:,0]

# %%
X

# %%
y

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=0.2)

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train

# %%
display(X_train.shape)
display(X_test.shape)
display(y_train.shape)
display(y_test.shape)

# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

# %%
# X_train = X_train.values.reshape(-1,1)
y_train = y_train.values.reshape(-1,1)
# X_test = X_test.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)

# %%
lr.fit(X_train, y_train)

# %%
y_pred = lr.predict(X_test)
y_pred

# %%
y_test

# %%
from sklearn import metrics

print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: ", metrics.mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2 Score: ", metrics.r2_score(y_test, y_pred))

# %% [markdown]
# **Random Forest Regression**

# %%
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100, random_state=2)

# %%
rfr.fit(X_train, y_train)

# %%
y_pred_rfr = rfr.predict(X_test)
y_pred_rfr

# %%
from sklearn import metrics

print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, y_pred_rfr))
print("Mean Squared Error: ", metrics.mean_squared_error(y_test, y_pred_rfr))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred_rfr)))
print("R2 Score: ", metrics.r2_score(y_test, y_pred_rfr))

# %%
# Hyparameter Tuning

r2Scores = []
for i in range(1,31):
    rfr = RandomForestRegressor(n_estimators=i, random_state=2)
    rfr.fit(X_train, y_train)
    y_pred_rfr = rfr.predict(X_test)
    r2Scores.append(metrics.r2_score(y_test, y_pred_rfr))

# %%
plt.figure(figsize=(10,8))
plt.plot(range(1,31), r2Scores)
plt.xlabel("Decision Trees")
plt.ylabel("Accuracy")
plt.title("Elbow Plot - Random Forest Regression")

# %%
X_train

# %%
plt.scatter(X_train, y_train)
plt.plot(X_train.iloc[:,-1], rfr.predict(X_train), color='red')
plt.xlabel('Distance')
plt.ylabel('Fare Amount')

# %% [markdown]
# **Testing without using distance**

# %%
df.head()

# %%
X2 = df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','normalized_distance','normalized_dates']]
y2 = df['fare_amount']
X2

# %%
y2

# %%
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=2)

# %%
lr2 = LinearRegression()
lr2.fit(X2_train, y2_train)

# %%
y2_pred = lr2.predict(X2_test)
y2_pred

# %%
from sklearn import metrics

print("Mean Absolute Error: ", metrics.mean_absolute_error(y2_test, y2_pred))
print("Mean Squared Error: ", metrics.mean_squared_error(y2_test, y2_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y2_test, y2_pred)))
print("R2 Score: ", metrics.r2_score(y2_test, y2_pred))

# %%
rfr2 = RandomForestRegressor(n_estimators=100,random_state=2)
rfr2.fit(X2_train, y2_train)

# %%
y2_pred_rfr = rfr2.predict(X2_test)

# %%
from sklearn import metrics

print("Mean Absolute Error: ", metrics.mean_absolute_error(y2_test, y2_pred))
print("Mean Squared Error: ", metrics.mean_squared_error(y2_test, y2_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y2_test, y2_pred)))
print("R2 Score: ", metrics.r2_score(y2_test, y2_pred))


