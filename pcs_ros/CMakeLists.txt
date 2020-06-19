cmake_minimum_required(VERSION 3.5.0)
project(pcs_ros)

add_compile_options(-std=c++14)

find_package(pcs_scan_integration REQUIRED)
find_package(pcs_detection REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_conversions
  pcl_ros
  pcs_msgs
  roscpp
  sensor_msgs
  )

#find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)

#catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES
  CATKIN_DEPENDS
    cv_bridge
    pcl_conversions
    pcl_ros
    pcs_msgs
    roscpp
    sensor_msgs
#  DEPENDS
)

###########
## Build ##
###########

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_hsv_thresholding_node src/hsv_thresholding_node.cpp)
add_dependencies(${PROJECT_NAME}_hsv_thresholding_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_hsv_thresholding_node ${catkin_LIBRARIES} pcs::pcs_detection_hsv_thresholding)

add_executable(${PROJECT_NAME}_image_channel_splitter_node src/image_channel_splitter_node.cpp)
add_dependencies(${PROJECT_NAME}_image_channel_splitter_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_image_channel_splitter_node ${catkin_LIBRARIES})

add_executable(${PROJECT_NAME}_image_extractor_node src/image_extractor_node.cpp)
add_dependencies(${PROJECT_NAME}_image_extractor_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_image_extractor_node ${catkin_LIBRARIES} pcs::pcs_detection_hsv_thresholding)

add_executable(${PROJECT_NAME}_octomap_mesh_mask_node src/octomap_mesh_mask_node.cpp)
add_dependencies(${PROJECT_NAME}_octomap_mesh_mask_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_octomap_mesh_mask_node ${catkin_LIBRARIES} pcs::pcs_scan_integration_octomap_mesh_masking )

add_executable(${PROJECT_NAME}_point_cloud_annotator_node src/point_cloud_annotator_node.cpp)
add_dependencies(${PROJECT_NAME}_point_cloud_annotator_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_point_cloud_annotator_node ${catkin_LIBRARIES} pcs::pcs_detection_point_cloud_annotator)

add_executable(${PROJECT_NAME}_point_cloud_xyzi_to_xyzrgb_node src/point_cloud_xyzi_to_xyzrgb_node.cpp)
add_dependencies(${PROJECT_NAME}_point_cloud_xyzi_to_xyzrgb_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_point_cloud_xyzi_to_xyzrgb_node ${catkin_LIBRARIES})

#############
## Install ##
#############
 install(PROGRAMS
   src/histogram_backprojection_node
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

# Mark executables and/or libraries for installation
 install(TARGETS
     ${PROJECT_NAME}_hsv_thresholding_node
     ${PROJECT_NAME}_image_channel_splitter_node
     ${PROJECT_NAME}_image_extractor_node
     ${PROJECT_NAME}_octomap_mesh_mask_node
     ${PROJECT_NAME}_point_cloud_annotator_node
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

## Mark cpp header files for installation
# install(
   #DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
 install(FILES
   launch/extract_images_from_bag.launch
   launch/demo.launch
   config/demo.rviz
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
 )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/point_cloud_segmentation.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
