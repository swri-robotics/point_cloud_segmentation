cmake_minimum_required(VERSION 3.5)
project(pcs_ros2)

if ($ENV{ROS_VERSION} VERSION_EQUAL "2")
message(STATUS "ROS 2 found. Building pcs_ros2")

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcs_scan_integration REQUIRED)
find_package(pcs_detection REQUIRED)
find_package(pcs_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(OpenCV 3 REQUIRED)

###########
## Build ##
###########

#add_executable(${PROJECT_NAME}_hsv_thresholding_node src/hsv_thresholding_node.cpp)
#add_dependencies(${PROJECT_NAME}_hsv_thresholding_node ${catkin_EXPORTED_TARGETS})
#target_link_libraries(${PROJECT_NAME}_hsv_thresholding_node ${catkin_LIBRARIES} pcs::pcs_detection_hsv_thresholding)

#add_executable(${PROJECT_NAME}_image_channel_splitter_node src/image_channel_splitter_node.cpp)
#add_dependencies(${PROJECT_NAME}_image_channel_splitter_node ${catkin_EXPORTED_TARGETS})
#target_link_libraries(${PROJECT_NAME}_image_channel_splitter_node ${catkin_LIBRARIES})

#add_executable(${PROJECT_NAME}_image_extractor_node src/image_extractor_node.cpp)
#add_dependencies(${PROJECT_NAME}_image_extractor_node ${catkin_EXPORTED_TARGETS})
#target_link_libraries(${PROJECT_NAME}_image_extractor_node ${catkin_LIBRARIES} pcs::pcs_detection_hsv_thresholding)

add_executable(${PROJECT_NAME}_octomap_mesh_mask_node src/octomap_mesh_mask_node.cpp)
ament_target_dependencies(${PROJECT_NAME}_octomap_mesh_mask_node rclcpp rclcpp_action cv_bridge pcs_msgs tf2_ros)
target_link_libraries(${PROJECT_NAME}_octomap_mesh_mask_node pcs::pcs_scan_integration_octomap_mesh_masking)
target_include_directories(${PROJECT_NAME}_octomap_mesh_mask_node PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")

add_executable(${PROJECT_NAME}_point_cloud_annotator_node src/point_cloud_annotator_node.cpp)
ament_target_dependencies(${PROJECT_NAME}_point_cloud_annotator_node cv_bridge rclcpp rclcpp_action cv_bridge pcs_msgs tf2_ros std_srvs)
target_link_libraries(${PROJECT_NAME}_point_cloud_annotator_node pcs::pcs_detection_point_cloud_annotator)
target_include_directories(${PROJECT_NAME}_point_cloud_annotator_node PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME}_point_cloud_annotator_node SYSTEM PUBLIC
    ${cv_bridge_INCLUDE_DIRS})

#add_executable(${PROJECT_NAME}_point_cloud_xyzi_to_xyzrgb_node src/point_cloud_xyzi_to_xyzrgb_node.cpp)
#add_dependencies(${PROJECT_NAME}_point_cloud_xyzi_to_xyzrgb_node ${catkin_EXPORTED_TARGETS})
#target_link_libraries(${PROJECT_NAME}_point_cloud_xyzi_to_xyzrgb_node ${catkin_LIBRARIES})


#############
## Install ##
#############
 install(PROGRAMS
#   src/histogram_backprojection_node
   src/fcn8_node
   DESTINATION lib/${PROJECT_NAME}
 )

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
  PATTERN ".svn" EXCLUDE
)

install(
  TARGETS
#    ${PROJECT_NAME}_hsv_thresholding_node
#    ${PROJECT_NAME}_image_channel_splitter_node
#    ${PROJECT_NAME}_image_extractor_node
    ${PROJECT_NAME}_octomap_mesh_mask_node
    ${PROJECT_NAME}_point_cloud_annotator_node
  EXPORT ${PACKAGE_NAME}-targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)
ament_export_interfaces(${PACKAGE_NAME}-targets)


#############
## Testing ##
#############
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

#############
## Ament ##
#############
ament_export_dependencies(
    ament_cmake
    cv_bridge
    pcs_scan_integration
    pcl_conversions
    pcl_ros
    rclcpp
    sensor_msgs
    OpenCV)

ament_package()

endif()
