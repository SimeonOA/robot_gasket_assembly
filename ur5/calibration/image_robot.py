
import numpy as np
import pandas as pd
import IPython
import csv

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import load, dump
import pdb

class ImageRobot():

    def __init__(self, model_path = './'):
        model_path = './'
        model_name = 'cam_robot_regr.pkl'
        scaler_name = 'cam_robot_scaler.pkl'
        self.load_model(model_path, model_path + model_name, scaler_name)

    def load_model(self, model_path, model_name, scaler_name):
        self.model = load(open(model_path+model_name, 'rb'))
        self.scaler = load(open(model_path+scaler_name, 'rb'))

    def image_pt_to_rw_pt(self, image_pt):
        #print('Transform...')
        image_pt_tr = self.scaler.transform([image_pt])
        #print('Predict...')
        return self.model.predict(image_pt_tr)

    def train_model(self, calibration_path='./cam_cal_09_11_23.csv'):

        print('Calibrating...')
        data = pd.read_csv(calibration_path)
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
            X_tr, y, train_size=0.9, test_size=0.1, random_state=1)

        reg = LinearRegression().fit(X_train, y_train)
        # reg = MLPRegressor(hidden_layer_sizes=(100,100,100), random_state=1,
        #                     max_iter=50000).fit(X_train, y_train)
        print (reg.score(X_test, y_test))
        print (reg.coef_)
        print (reg.intercept_)

        model_path = './'
        model_name = 'cam_robot_regr.pkl'
        scaler_name = 'cam_robot_scaler.pkl'

        dump(reg, open(model_path+model_name, 'wb'))
        dump(scaler, open(model_path+scaler_name, 'wb'))

def main():
    ir = ImageRobot()
    ir.train_model()
    image_pt = [36,72]
    rw_pt  = ir.image_pt_to_rw_pt(image_pt)
    print ('Real world point is', rw_pt)
    print ('Real world point should be: 109, -618')

if __name__ == '__main__':
    main()
