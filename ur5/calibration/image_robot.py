
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pickle import load, dump

class ImageRobot():
    def __init__(self, model_path = ''):
        self.load_model()

    def load_model(self):
        self.model_x = load(open('ur5/calibration/im_to_real_x.pkl', 'rb'))
        self.model_y = load(open('ur5/calibration/im_to_real_y.pkl', 'rb'))

    def image_pt_to_rw_pt(self, image_pt):
        if type(image_pt) is list or type(image_pt) is tuple or image_pt.shape != (1,2):
            image_pt = np.array(image_pt).reshape((1,2))
        return np.array([self.model_x.predict(image_pt)[0][0], self.model_y.predict(image_pt)[0][0]])

    # TODO: make sure to update cam_cal.csv with values for your camera and robot!
    def train_model(self, calibration_path='ur5/calibration/cam_cal.csv'):
        print('Calibrating...')
        df = pd.read_csv(calibration_path)
        # im_x are the x-values of the black points in ur5_workspace_calibration_example.png
        im_x = df['im_x'].values.reshape(-1, 1)
        # im_x are the x-values of the black points in ur5_workspace_calibration_example.png
        real_x = df['real_x'].values.reshape(-1, 1)
        # im_x are the x-values of the black points in ur5_workspace_calibration_example.png
        im_y = df['im_y'].values.reshape(-1, 1)
        # im_x are the x-values of the black points in ur5_workspace_calibration_example.png
        real_y = df['real_y'].values.reshape(-1, 1)

        im_coords = np.column_stack((im_x, im_y))
        model_x = LinearRegression()
        model_y = LinearRegression()
        model_x.fit(im_coords, real_x)
        model_y.fit(im_coords, real_y)

        dump(model_x, open('ur5/calibration/im_to_real_x.pkl', 'wb'))
        dump(model_y, open('ur5/calibration/im_to_real_y.pkl', 'wb'))

def main():
    ir = ImageRobot()
    ir.train_model()
    # TODO: fill these in with your values!
    image_pt = ...
    rw_pt  = ir.image_pt_to_rw_pt(image_pt)
    print ('Real world point is', rw_pt)
    # TODO: Use a point you can verify with the robot in the real world
    print ('Real world point should be: ...')

if __name__ == '__main__':
    main()