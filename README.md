# point_cloud_segmentation

master: [![Build Status](https://travis-ci.org/swri-robotics/point_cloud_segmentation.svg?branch=master)](https://travis-ci.org/swri-robotics/point_cloud_segmentation)

devel: [![Build Status](https://travis-ci.org/swri-robotics/point_cloud_segmentation.svg?branch=devel)](https://travis-ci.org/swri-robotics/point_cloud_segmentation)

## Description

This package contains tools for annotating point clouds based on associated images. The idea is that 2D feature detectors are a mature technology, but detecting features in 3D point clouds is much harder. It is possible to detect features in the 2D images that are often associated with pointclouds (e.g. from depth cameras) and annotate the point clouds based on the 2D detectors. This data can then be aggregated over the course of a 3D scan to result in a semantically labelled 3D reconstruction.

One important feature of this meta-package is that the majority of the subpackages are ROS-independents. They are pure cmake 3.5 packages that expose cmake targets that can be used in a variety of settings. While pcs_ros is a ROS 1 wrapper around many of the packages's functions, a ROS 2 wrapper pull request would also be welcome.

## Package Overview
* pcs_detection - Contains functions for doing 2D detection
* pcs_msgs - Contains ROS msgs
* pcs_ros - Exposes the functions in the other packages as ros nodes as well as provides a variety of utility nodes that are useful for piecing together a complete system.
* pcs_scan_integration - Contains functions used to aggregate multiple annotated point clouds over the course of a scan.



