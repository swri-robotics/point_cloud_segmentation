.. SphinxTest documentation master file, created by
   sphinx-quickstart on Tue Oct  3 11:09:13 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================
Welcome to the Point Cloud Segmentation wiki
=============================


Core Packages
-----------------------

* **pcs_detection** – Contains functions for doing 2D detection
* **pcs_scan_integration** - Contains functions used during scanning

ROS Packages
----------------------

* **pcs_msgs** – This package contains examples using tesseract and tesseract_ros for motion planning and collision checking.
* **pcs_ros** – This contains plugins for collision and kinematics which are automatically loaded by the monitors.

.. Warning:: These packages are under heavy development and are subject to change.


Packages
------------

.. toctree::
   :maxdepth: 1

   pcs_detection <_source/pcs_detection.rst>
   pcs_msgs <_source/pcs_msgs.rst>
   pcs_ros <_source/pcs_ros.rst>
   pcs_scan_integration <_source/psc_scan_integration.rst>

FAQ
---
.. toctree::
   :maxdepth: 2

   Questions?<_source/FAQ.rst>
