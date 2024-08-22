This is the code repo for the paper **Automating Deformable Gasket Assembly** accepted for [CASE 2024](https://2024.ieeecase.org/)

The website for the paper: **Automating Deformable Gasket Assembly** including videos, results and CAD files, can be found [here](https://berkeleyautomation.github.io/robot-gasket/).

## Installation
```shell
git clone git@github.com:SimeonOA/robot_gasket_assembly.git
cd robot_gasket_assembly/
conda env create -f environment.yml
```
Note: You will need to separately install ur5py, ur_rtde (make sure it's version 1.4.2!), and pyzed (if you want to use a ZED camera!)

## Calibration
This implementation requires an overhead camera calibrated to the robot such that a pixel from the overhead's camera can be translated to the appropriate x,y coordinate of that point in the robot's frame. Reference calibration/image_robot.py if necessary.

Optional: Use if you do not have another calibration method you prefer and make sure to update cam_cal.csv with your values!
```shell
cd ur5/calibration/
python image_robot.py
```
## Quickstart
Across the codebase are TODOs that you will need to update for your given workspace for the code to work! 
After that's done you can run the following:
```shell
cd ur5/
python main.py <include args here!>
```
