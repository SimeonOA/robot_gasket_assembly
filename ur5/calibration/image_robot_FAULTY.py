
import numpy as np
import pandas as pd
import IPython
import csv

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import load, dump
import pdb

class ImageRobot():

    def __init__(self, model_path = ''):
        self.load_model()

    def load_model(self):
        self.model_x = load(open('/home/gasket/robot_cable_insertion/ur5/calibration/im_to_real_x.pkl', 'rb'))
        self.model_y = load(open('/home/gasket/robot_cable_insertion/ur5/calibration/im_to_real_y.pkl', 'rb'))

    def image_pt_to_rw_pt(self, image_pt):
        #print('Predict...')
        if type(image_pt) is list or type(image_pt) is tuple or image_pt.shape != (1,2):
            image_pt = np.array(image_pt).reshape((1,2))
        # this version is for the lienar regression
        return np.array([self.model_x.predict(image_pt)[0][0], self.model_y.predict(image_pt)[0][0]])
        # return np.array([self.model_x.predict(image_pt)[0], self.model_y.predict(image_pt)[0]])

    def train_model(self, calibration_path='/home/gasket/robot_cable_insertion/ur5/calibration/cam_cal_2_21_24_final.csv'):

        print('Calibrating...')
        df = pd.read_csv(calibration_path)
        im_x = df['im_x'].values.reshape(-1, 1)
        real_x = df['real_x'].values.reshape(-1, 1)
        im_y = df['im_y'].values.reshape(-1, 1)
        real_y = df['real_y'].values.reshape(-1, 1)

        im_coords = np.column_stack((im_x, im_y))
        model_x = LinearRegression()
        model_y = LinearRegression()
        model_x.fit(im_coords, real_x)
        model_y.fit(im_coords, real_y)

        dump(model_x, open('/home/gasket/robot_cable_insertion/ur5/calibration/im_to_real_x.pkl', 'wb'))
        dump(model_y, open('/home/gasket/robot_cable_insertion/ur5/calibration/im_to_real_y.pkl', 'wb'))

def main():
    ir = ImageRobot()
    ir.train_model()
    image_pt = [553.6, 814.4]
    rw_pt  = ir.image_pt_to_rw_pt(image_pt)
    print ('Real world point is', rw_pt)
    print ('Real world point should be: [-447.1	-312]')

if __name__ == '__main__':
    main()
