This is the code repo for the paper **Automating Deformable Gasket Assembly** accepted for [CASE 2024](https://2024.ieeecase.org/)

The website for the paper: **Automating Deformable Gasket Assembly** including videos, results and CAD files, can be found [here](https://berkeleyautomation.github.io/robot-gasket/).

## Installation
```shell
git clone <insert repo name here!>
<include command to install correct dependencies in a conda env>
```

## Calibration
This implementation requires an overhead camera calibrated to the robot such that a pixel from the overhead's camera can be provided and the associated x,y coordinate of that point in the

```shell
cd <repo name here>/ur5/calibration
python main.py --
```
## Quickstart

```shell
cd <repo name here>/ur5
python main.py --
```
