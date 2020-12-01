cmake_minimum_required(VERSION 3.5.0)
project(pcs_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

if ($ENV{ROS_VERSION} VERSION_EQUAL "1")
  message(STATUS "Building crs_msgs for ROS 1")

  find_package(catkin REQUIRED COMPONENTS
    actionlib_msgs
    message_generation
    message_runtime
    sensor_msgs
    std_msgs
  )

  ## Generate messages in the 'msg' folder
  # add_message_files(
  #   FILES
  #   Message1.msg
  #   Message2.msg
  # )

  # Generate services in the 'srv' folder
  add_service_files(
    FILES
    ImageProcessing.srv
  )

  # Generate actions in the 'action' folder
   add_action_files(
     FILES
     ApplyOctomapMeshMask.action
   )

  ## Generate added messages and services with any dependencies listed here
   generate_messages(
     DEPENDENCIES
     actionlib_msgs
     sensor_msgs
     std_msgs
   )

  catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES pcs_msgs
    CATKIN_DEPENDS
      actionlib_msgs
      message_generation
      sensor_msgs
      std_msgs
      message_runtime
  #  DEPENDS system_lib
  )

  include_directories(
  # include
    ${catkin_INCLUDE_DIRS}
  )

else()
  message(STATUS "Building crs_msgs for ROS 2")

    find_package(ament_cmake REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(rosidl_default_generators REQUIRED)

  rosidl_generate_interfaces(${PROJECT_NAME}
    srv/ImageProcessing.srv
    action/ApplyOctomapMeshMask.action
    DEPENDENCIES
      sensor_msgs
      std_msgs
    )

  ament_export_dependencies(rosidl_default_runtime)
  ament_package()
endif()
