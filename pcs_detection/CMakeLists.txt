cmake_minimum_required(VERSION 3.5.0)

# Extract package name and version
find_package(cmake_common_scripts REQUIRED)
extract_package_metadata(pkg)
project(${pkg_extracted_name} VERSION ${pkg_extracted_version} LANGUAGES CXX)

find_package(console_bridge REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

###########
## Build ##
###########

add_library(${PROJECT_NAME}_hsv_thresholding INTERFACE)
target_link_libraries(${PROJECT_NAME}_hsv_thresholding INTERFACE console_bridge ${OpenCV_LIBS})
target_compile_options(${PROJECT_NAME}_hsv_thresholding INTERFACE -Wsuggest-override -Wconversion -Wsign-conversion)
target_cxx_version(${PROJECT_NAME}_hsv_thresholding INTERFACE VERSION 11)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(${PROJECT_NAME}_hsv_thresholding INTERFACE -mno-avx)
else()
  message(WARNING "Non-GNU compiler detected. If using AVX instructions, Eigen alignment issues may result.")
endif()
target_include_directories(${PROJECT_NAME}_hsv_thresholding INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME}_hsv_thresholding SYSTEM INTERFACE
    ${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME}_point_cloud_annotator SHARED src/${PROJECT_NAME}/point_cloud_annotator.cpp)
target_link_libraries(${PROJECT_NAME}_point_cloud_annotator PUBLIC  console_bridge ${PCL_LIBRARIES} ${OpenCV_LIBS})
target_compile_options(${PROJECT_NAME}_point_cloud_annotator PRIVATE -Wsuggest-override -Wconversion -Wsign-conversion)
target_cxx_version(${PROJECT_NAME}_point_cloud_annotator INTERFACE VERSION 11)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME}_point_cloud_annotator PUBLIC -mno-avx)
  else()
    message(WARNING "Non-GNU compiler detected. If using AVX instructions, Eigen alignment issues may result.")
  endif()
target_include_directories(${PROJECT_NAME}_point_cloud_annotator PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME}_point_cloud_annotator SYSTEM PUBLIC
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

#############
## Python ##
#############
add_subdirectory(src_python)

#############
## Install ##
#############
configure_package(NAMESPACE pcs TARGETS ${PROJECT_NAME}_hsv_thresholding ${PROJECT_NAME}_point_cloud_annotator)

# Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE
 )

#############
## Test ##
#############
if (ENABLE_TESTS)
  enable_testing()
  add_run_tests_target(ENABLE ${PCS_ENABLE_RUN_TESTS})
  add_subdirectory(test)
endif()

