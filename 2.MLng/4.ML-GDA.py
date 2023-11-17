# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

# %%
def objective(x):
    return (x+3)**2

# %%
def derivate(x):
    return 2*(x+3)

# %%
def gradientDescent(start, lr, epoch): # lr -> learning rate, epoch -> iterations
    x_old = start
    x_list = []
    for i in range(epoch):
        slope = derivate(x_old)
        x_new = x_old - (slope * lr)
        x_list.append(x_new)
        x_old = x_new
    return x_list

# %%
lr = 0.1
start = 2
epoch = 100

# %%
x_list = gradientDescent(start, lr, epoch)
x_list

# %%
# Calculating derivatives at start and end points

from sympy import Derivative
from sympy.abc import x

df = Derivative((x+3)**2, x, evaluate=True)
df

# %%
print(df.doit().subs(x,x_list[0]))
print(df.doit().subs(x,x_list[-1]))

# %%
X = np.linspace(-30,30,100)
plt.plot(X, objective(X), '-')
plt.plot(start, objective(start), 'ro')

# %%
X = np.linspace(-4,4,100)
plt.plot(X, objective(X), '-')

x_list_arr = np.array(x_list)
plt.plot(x_list, objective(x_list_arr), 'r.-')

# %%



