from ur5py import UR5Robot
import numpy as np

if __name__ == "__main__":
    ur = UR5Robot()
    input("enter to enter freedrive")
    ur.start_teach()
    input("enter tro end freedrive")
    ur.stop_teach()
