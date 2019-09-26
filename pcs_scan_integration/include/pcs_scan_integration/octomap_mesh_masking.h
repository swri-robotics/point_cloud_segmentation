/**
 * @file octomap_mesh_masking.h
 * @brief Masks a mesh based on an octomap
 *
 * @author Matthew Powelson
 * @date Sept 23, 2019
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2013, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef PCS_SCAN_INTEGRATION_OCTOMAP_MESH_MASKING_H
#define PCS_SCAN_INTEGRATION_OCTOMAP_MESH_MASKING_H

#include <tesseract_common/macros.h>
TESSERACT_COMMON_IGNORE_WARNINGS_PUSH
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
TESSERACT_COMMON_IGNORE_WARNINGS_POP

#include <tesseract_geometry/geometries.h>

namespace pcs_scan_integration
{
/**
 * @brief colorPassthrough Filter for PCL Pointcloud that filters based on color. Similar to Passthrough filter
 *
 * This currently applies one filter to all 3 channels and all 3 channels must satisfy it to be included. ie if r and g
 * are inside the threshold but b is outside, then the point is considered outside
 * @param input_cloud ConstPtr to input cloud
 * @param lower_limit All RGB values must be > limit to be included in output
 * @param upper_limit All RGB values must be < limit to be included in output
 * @param limit_negative If true return the data outside of the threshold rather than inside it
 * @return Ptr to filtered cloud where
 */
pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorPassthrough(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr input_cloud,
                                                        const int& lower_limit,
                                                        const int& upper_limit,
                                                        const bool& limit_negative = false);

class OctomapMeshMask
{
public:
  /**
   * @brief Used to determine the method used for masking
   *
   * RETURN_INSIDE - Returns only the mesh inside the masking
   * RETURN_OUTSIDE - Returns only the mesh outside the masking
   * RETURN_COLORIZED - Returns the entire mesh but colorizes it based on what is inside/outside the masking
   */
  enum class MaskType
  {
    RETURN_INSIDE,
    RETURN_OUTSIDE,
    RETURN_COLORIZED
  };

  OctomapMeshMask(){};

  /**
   * @brief Sets the input mesh from an absolute filepath using assimp
   * @param filepath Absolute filepath to a mesh.
   */
  void setInputMesh(std::string& filepath);
  /**
   * @brief Sets the mesh that will be masked. Mesh is stored as a ConstPtr
   * @param input_mesh Mesh to be masked
   */
  inline void setInputMesh(tesseract_geometry::Mesh::ConstPtr input_mesh) { input_mesh_ = input_mesh; }

  /**
   * @brief Returns the original input mesh before masking
   * @return ConstPointer to the input mesh.
   */
  inline tesseract_geometry::Mesh::ConstPtr getInputMesh() { return input_mesh_; }

  /**
   * @brief Sets the Tesseract_Geometry octree that will be used to mask the mesh from points in a point cloud that meet
   * the specified color threshold
   *
   * This simply takes the input point cloud and applies 3 sequential passthrough filters based on r,g, and b before
   * calling the octree constructor with the resulting point cloud
   * @param point_cloud Colorized point cloud
   * @param resolution Smallest resolution of the octree
   * @param lower_limit Default: 0. Lower limit of allowed RGB color values
   * @param upper_limit Default: 255. Upper limit of allowed RGB color values
   * @param limit_negative Default: false.	If true use the data outside of the threshold rather than inside it
   */
  void setOctree(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud,
                 const double resolution,
                 const int& lower_limit = 0,
                 const int& upper_limit = 255,
                 const bool& limit_negative = false);

  /**
   * @brief Sets the Tesseract_Geometry octree that will be used to mask the mesh directly
   * @param octree Octree to be used to mask the mesh. No filtering is applied internally.
   */
  void setOctree(const tesseract_geometry::Octree::Ptr& octree) { octree_ = std::move(octree); }

  /**
   * @brief Returns the octree that will be used to mask the mesh
   * @return Returns a ConstPtr to the octree used to mask the mesh
   */
  inline tesseract_geometry::Octree::ConstPtr getOctree() { return octree_; }

  /**
   * @brief Apply the octomap mask to the mesh based on the type of mask selected
   * @param mask_type Determines the way the output is processed
   *
   * RETURN_INSIDE - Save the mesh that falls inside the octomap
   * RETURN_OUTSIDE - Save the mesh that alls outside the octomap
   * RETURN_COLORIZED - Colorize the mesh based on regions that are within the octomap
   * @return True if successful
   */
  bool maskMesh(const MaskType& mask_type);

  /**
   * @brief Returns the last successfully masked mesh
   * @return The last successfully masked mesh
   */
  inline tesseract_geometry::Mesh::Ptr getMaskedMesh() { return masked_mesh_; }

  /**
   * @brief Saves the last successfully masked mesh to a file
   * @param filepath Absolute filepath to which the mesh is saved
   * @return True if successful
   */
  bool saveMaskedMesh(std::string& filepath);

protected:
  tesseract_geometry::Mesh::ConstPtr input_mesh_;
  tesseract_geometry::Mesh::Ptr masked_mesh_;
  std::vector<Eigen::Vector3i> mesh_vertices_color_;
  tesseract_geometry::Octree::Ptr octree_;
};

}  // namespace pcs_scan_integration

#endif
