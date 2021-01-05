cmake_minimum_required(VERSION 3.5.0)

# Extract package name and version
find_package(cmake_common_scripts REQUIRED)
extract_package_metadata(pkg)
project(${pkg_extracted_name} VERSION ${pkg_extracted_version} LANGUAGES CXX)

find_package(console_bridge REQUIRED)
find_package(PCL REQUIRED)
find_package(tesseract_collision REQUIRED)

###########
## Build ##
###########

add_library(${PROJECT_NAME}_octomap_mesh_masking src/${PROJECT_NAME}/octomap_mesh_masking.cpp)
target_link_libraries(${PROJECT_NAME}_octomap_mesh_masking tesseract::tesseract_collision_bullet console_bridge ${PCL_LIBRARIES})
target_compile_options(${PROJECT_NAME}_octomap_mesh_masking PUBLIC -Wsuggest-override -Wconversion -Wsign-conversion)
target_cxx_version(${PROJECT_NAME}_octomap_mesh_masking INTERFACE VERSION 11)
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
configure_package(NAMESPACE pcs TARGETS ${PROJECT_NAME}_octomap_mesh_masking)

# Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE
 )

if (ENABLE_TESTS)
  enable_testing()
  add_run_tests_target(ENABLE ${PCS_ENABLE_RUN_TESTS})
  add_subdirectory(test)
endif()

