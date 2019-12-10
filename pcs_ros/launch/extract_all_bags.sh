#!/bin/bash
# This bash script loops through all files in the directory and calls the image extractor launch file.
# The results for each bag file are placed in a different directory

for file in .* *; do 
  echo "${file}"
  mkdir "${file}_images"
  roslaunch pcs_ros extract_images_from_bag.launch filepath:="${PWD}/${file}" results_dir:="${PWD}/${file}_images"
done



