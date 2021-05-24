# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## Rubric points

1. FP.1 - The bounding boxes of prevFrame and currFrame are matched by building a nxm voting matrix and then by pairing the boxes with the highest votes
2. FP.2 - The Lidar based TTC is calculated by finding the closes LidarPoint on the preceding vehicle. The outliers are removed by performing clustering using a distance tolerance
3. FP.3 - Bounding box is updated with the key-points which fall within the ROI of the particular bounding box. The addition of key-points is made robust by computing median of key-points distance values and ensuring that the key-point being added is not far away from it's corresponding match
4. FP.4 - Distance ratios are calculated and the camera based TTC is calculated based on the median distance ratio
5. FP.5 - 

Here, in the table we have the Lidar TTC estimates for 18 of the 19 images.

| Image number   | 0 |   1   |   2   |   3   |   4  |   5   |   6   |   7   |   8  |   9   |   10  |  11  |   12  |  13  |   14  |  15  |  16  |  17  |  18  |
|----------------|:-:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:----:|:-----:|:----:|:-----:|:----:|:----:|:----:|:----:|
| Lidar TTC(in s)|   | 12.97 | 12.26 | 13.92 | 7.12 | 16.25 | 12.42 | 34.34 | 9.34 | 18.13 | 18.03 | 3.83 | 10.85 | 9.22 | 10.97 | 8.09 | 3.18 | 9.99 | 8.31 |

Average Lidar TTC = 12.18 s

We observe that the Lidar TTC estimate for the **7th image** and **11th image** are extremly high and extremly less compared to the average TTC estimate. This is due to the fact that Lidar TTC is computed using a constant velocity model and for these two particular frames *though* the closest lidar point on the preceding vehicle is not incorrect, the TTC estimates are wrong. These outliers and a form of oscillation in the TTC estimate can be noticed in the following graph as the vehicles accelerates and decelerates i.e. estimates of TTC are incorrect when we see abrupt change in acceleration and in turn the velocity of the vehicle.
![](./lidar_ttc.png)

6. FP.6 - 