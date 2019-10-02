cmake_minimum_required(VERSION 3.5.0)
project(pcs_ros)

add_compile_options(-std=c++11)

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
find_package(OpenCV 3 REQUIRED)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES pc_masker
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

# Declare a C++ library
 add_library(pc_masker SHARED
   src/${PROJECT_NAME}/pc_masker.cpp
 )
add_dependencies(pc_masker ${pc_masker_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pc_masker PUBLIC
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${OpenCV_LIBRARIES}
  pcs::pcs_detection_hsv_thresholding)

# Declare a C++ executable
add_executable(pc_masker_node src/pc_masker_node.cpp)
add_dependencies(pc_masker_node ${pc_masker_node_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pc_masker_node pc_masker ${catkin_LIBRARIES})

add_executable(${PROJECT_NAME}_hsv_thresholding_node src/hsv_thresholding_node.cpp)
add_dependencies(${PROJECT_NAME}_hsv_thresholding_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_hsv_thresholding_node ${catkin_LIBRARIES} pcs::pcs_detection_hsv_thresholding)

add_executable(${PROJECT_NAME}_image_channel_splitter_node src/image_channel_splitter_node.cpp)
add_dependencies(${PROJECT_NAME}_image_channel_splitter_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_image_channel_splitter_node ${catkin_LIBRARIES})

add_executable(${PROJECT_NAME}_image_extractor_node src/image_extractor_node.cpp)
add_dependencies(${PROJECT_NAME}_image_extractor_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_image_extractor_node ${catkin_LIBRARIES})

add_executable(${PROJECT_NAME}_octomap_mesh_mask_node src/octomap_mesh_mask_node.cpp)
add_dependencies(${PROJECT_NAME}_octomap_mesh_mask_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_octomap_mesh_mask_node ${catkin_LIBRARIES} pcs::pcs_scan_integration_octomap_mesh_masking )



#############
## Install ##
#############
 install(PROGRAMS
   src/histogram_backprojection_node
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

# Mark executables and/or libraries for installation
 install(TARGETS pc_masker_node ${PROJECT_NAME}_hsv_thresholding_node ${PROJECT_NAME}_image_extractor_node
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

## Mark cpp header files for installation
 install(DIRECTORY include/${PROJECT_NAME}/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.h"
   PATTERN ".svn" EXCLUDE
 )

## Mark other files for installation (e.g. launch and bag files, etc.)
 install(FILES
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
