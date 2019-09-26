cmake_minimum_required(VERSION 3.5.0)
project(pcs_scan_integration VERSION 0.1.0 LANGUAGES CXX)

find_package(console_bridge REQUIRED)
find_package(PCL REQUIRED)
find_package(tesseract_collision REQUIRED)

list(FIND CMAKE_CXX_COMPILE_FEATURES cxx_std_11 CXX_FEATURE_FOUND)

###########
## Build ##
###########

add_library(${PROJECT_NAME}_octomap_mesh_masking src/${PROJECT_NAME}/octomap_mesh_masking.cpp)
target_link_libraries(${PROJECT_NAME}_octomap_mesh_masking tesseract::tesseract_collision_bullet console_bridge ${PCL_LIBRARIES})
target_compile_options(${PROJECT_NAME}_octomap_mesh_masking PUBLIC -Wsuggest-override -Wconversion -Wsign-conversion)
if(CXX_FEATURE_FOUND EQUAL "-1")
    target_compile_options(${PROJECT_NAME}_octomap_mesh_masking PUBLIC -std=c++11)
else()
    target_compile_features(${PROJECT_NAME}_octomap_mesh_masking PRIVATE cxx_std_11)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME}_octomap_mesh_masking PUBLIC -mno-avx)
  else()
    message(WARNING "Non-GNU compiler detected. If using AVX instructions, Eigen alignment issues may result.")
  endif()
target_include_directories(${PROJECT_NAME}_octomap_mesh_masking PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME}_octomap_mesh_masking SYSTEM PUBLIC
    ${PCL_INCLUDE_DIRS})


#############
## Install ##
#############
install(
  TARGETS
    ${PROJECT_NAME}_octomap_mesh_masking
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
 )

install(FILES package.xml DESTINATION share/${PROJECT_NAME})

install(
  DIRECTORY test
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING
    PATTERN "*.bt"
    PATTERN "*.stl"
    PATTERN "*.ply"
)

if (ENABLE_TESTS)
  enable_testing()
  add_custom_target(run_tests ALL
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIGURATION> -V)

  add_subdirectory(test)
endif()

