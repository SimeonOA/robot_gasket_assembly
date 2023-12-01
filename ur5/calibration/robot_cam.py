import numpy as np
import pandas as pd
import IPython
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import load, dump


data = pd.read_csv('sorted_cam_data.csv')
im_x = data.im_x
im_y = data.im_y
real_x = data.real_x
real_y = data.real_y

num_data_pts = im_x.shape[0]
X = np.array([[im_x[_], im_y[_]] for _ in range(num_data_pts)])
y = np.array([[real_x[_], real_y[_]] for _ in range(num_data_pts)])

scaler = StandardScaler()
scaler.fit(X)
mean_val = scaler.mean_
var_val = scaler.var_

X_tr = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_tr, y, train_size=0.8, test_size=0.2, random_state=1)

reg = LinearRegression().fit(X_train, y_train)
print (reg.score(X_test, y_test))
print (reg.coef_)
print (reg.intercept_)

model_path = './cam_model/'
model_name = 'cam_robot_regr.pkl'
scaler_name = 'cam_robot_scaler.pkl'

dump(reg, open(model_path+model_name, 'wb'))
dump(scaler, open(model_path+scaler_name, 'wb'))
