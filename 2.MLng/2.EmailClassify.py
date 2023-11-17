# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('datasets/emails.csv')
df.head()

# %%
# Checking if the dataset is not too large
df.shape

# %%
# Checking for any empty values
df.isnull().sum()   

# %%
df.info()

# %%
df.describe()

# %%
# Checking if the data is biased
df['Prediction'].value_counts()[0]    

# %%
df['Prediction'].value_counts()[1]

# %%
# Dropping unnecessary columns
df.drop(columns=['Email No.'],axis=1,inplace=True)    

# %%
df.corr()

# %%
X = df.iloc[:,:-1]
X

# %%
y = df.iloc[:,-1].values
y

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# %%
X_train

# %%
y_train

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
X_train

# %% [markdown]
# ## KNN Classifier

# %%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

# %%
knn.fit(X_train, y_train)

# %%
from sklearn.metrics import accuracy_score

y_pred = knn.predict(X_test)

accuracy_score(y_test,y_pred)

# %%
scores = []

for i in range(1,16):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))

# %%
plt.plot(range(1,16),scores)
# Best classification at k = 2

# %% [markdown]
# ## SVM [Support Vector Machines]

# %%
# Since our dataset has more than 2 features for classification we have to use RBF Kernel

# %%
# Trying using linear kernel SVM
from sklearn.svm import SVC

model_Linear = SVC(kernel='linear', C=1)
model_Linear.fit(X_train,y_train)

# %%
y_pred_Linear_SVC = model_Linear.predict(X_test)

# %%
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred_Linear_SVC)

# %%
# Trying using RBF kernel SVM

model_RBF = SVC(kernel='rbf',gamma='auto')
model_RBF.fit(X_train,y_train)

# %%
y_pred_RBF_SVC = model_RBF.predict(X_test)

# %%
accuracy_score(y_test,y_pred_RBF_SVC)

# %%



