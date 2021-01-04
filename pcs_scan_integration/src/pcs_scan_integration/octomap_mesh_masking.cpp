#include <tesseract_common/macros.h>
TESSERACT_COMMON_IGNORE_WARNINGS_PUSH
#include <console_bridge/console.h>
#include <pcl/filters/passthrough.h>
#include <tesseract_collision/bullet/bullet_discrete_simple_manager.h>
#include <tesseract_collision/core/common.h>
#include <tesseract_geometry/mesh_parser.h>
TESSERACT_COMMON_IGNORE_WARNINGS_POP

#include "pcs_scan_integration/octomap_mesh_masking.h"

using namespace pcs_scan_integration;

// This was commented 10/23/2019. It seems that r,g, and b values are not independent fields, so you can't use this
// approach. However we should look into that more later. If no one has fixed this in say a year, just delete it.
// pcl::PointCloud<pcl::PointXYZRGB>::Ptr
// pcs_scan_integration::colorPassthrough(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr input_cloud,
//                                       const int& lower_limit,
//                                       const int& upper_limit,
//                                       const bool& limit_negative)
//{
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_r(new pcl::PointCloud<pcl::PointXYZRGB>());
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_g(new pcl::PointCloud<pcl::PointXYZRGB>());
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered_b(new pcl::PointCloud<pcl::PointXYZRGB>());
//  {
//    pcl::PassThrough<pcl::PointXYZRGB> pass;
//    pass.setInputCloud(input_cloud);
//    pass.setFilterFieldName("r");
//    pass.setFilterLimits(static_cast<float>(lower_limit), static_cast<float>(upper_limit));
//    pass.setFilterLimitsNegative(limit_negative);
//    pass.filter(*cloud_filtered_r);
//  }
//  {
//    pcl::PassThrough<pcl::PointXYZRGB> pass;
//    pass.setInputCloud(cloud_filtered_r);
//    pass.setFilterFieldName("g");
//    pass.setFilterLimits(static_cast<float>(lower_limit), static_cast<float>(upper_limit));
//    pass.setFilterLimitsNegative(limit_negative);
//    pass.filter(*cloud_filtered_g);
//  }
//  {
//    pcl::PassThrough<pcl::PointXYZRGB> pass;
//    pass.setInputCloud(cloud_filtered_g);
//    pass.setFilterFieldName("b");
//    pass.setFilterLimits(static_cast<float>(lower_limit), static_cast<float>(upper_limit));
//    pass.setFilterLimitsNegative(limit_negative);
//    pass.filter(*cloud_filtered_b);
//  }
//  return cloud_filtered_b;
//}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
pcs_scan_integration::colorPassthrough(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr input_cloud,
                                       const int& lower_limit,
                                       const int& upper_limit,
                                       const bool& limit_negative)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
  cloud_filtered->header = input_cloud->header;
  cloud_filtered->width = input_cloud->width;
  cloud_filtered->height = input_cloud->height;
  cloud_filtered->is_dense = input_cloud->is_dense;
  cloud_filtered->points.resize(input_cloud->points.size());

  if (lower_limit > upper_limit)
  {
    CONSOLE_BRIDGE_logError("colorPassthrough: Lower limit greater than upper limit");
    assert(false);
  }
  int i = 0;
  for (const pcl::PointXYZRGB& point : input_cloud->points)
  {
    if (!limit_negative)
    {
      if (((point.r >= lower_limit) && (point.r <= upper_limit)) &&
          ((point.g >= lower_limit) && (point.g <= upper_limit)) &&
          ((point.b >= lower_limit) && (point.b <= upper_limit)))
      {
        cloud_filtered->points[i] = point;
        i++;
      }
    }
    else
    {
      if (((point.r <= lower_limit) || (point.r >= upper_limit)) &&
          ((point.g <= lower_limit) || (point.g >= upper_limit)) &&
          ((point.b <= lower_limit) || (point.b >= upper_limit)))
      {
        cloud_filtered->points[i] = point;
        i++;
      }
    }
  }
  cloud_filtered->resize(i);
  return cloud_filtered;
}

void OctomapMeshMask::setInputMesh(std::string& filepath)
{
  tesseract_geometry::Mesh::Ptr mesh =
      tesseract_geometry::createMeshFromPath<tesseract_geometry::Mesh>(filepath, Eigen::Vector3d(1, 1, 1), true)[0];
  setInputMesh(mesh);
}

void OctomapMeshMask::setOctree(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr point_cloud,
                                const double resolution,
                                const int& lower_limit,
                                const int& upper_limit,
                                const bool& limit_negative)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered =
      colorPassthrough(point_cloud, lower_limit, upper_limit, limit_negative);

  octree_ = std::make_shared<tesseract_geometry::Octree>(
      *cloud_filtered, resolution, tesseract_geometry::Octree::SubType::BOX, true);
}

bool OctomapMeshMask::maskMesh(const MaskType& mask_type)
{
  // ----------- First perform collision check to find triangles inside octomap ------------
  tesseract_collision::tesseract_collision_bullet::BulletDiscreteSimpleManager checker;

  // Create Octomap collision object
  tesseract_collision::CollisionShapePtr dense_octomap(octree_);
  Eigen::Isometry3d octomap_pose = Eigen::Isometry3d::Identity();

  tesseract_collision::CollisionShapesConst obj1_shapes;
  tesseract_common::VectorIsometry3d obj1_poses;
  obj1_shapes.push_back(dense_octomap);
  obj1_poses.push_back(octomap_pose);
  checker.addCollisionObject("octomap_link", 0, obj1_shapes, obj1_poses);

  // Create mesh collision object
  tesseract_collision::CollisionShapesConst obj2_shapes;
  tesseract_common::VectorIsometry3d obj2_poses;
  obj2_shapes.push_back(input_mesh_);
  Eigen::Isometry3d mesh_pose = Eigen::Isometry3d::Identity();
  obj2_poses.push_back(mesh_pose);
  checker.addCollisionObject("mesh_link", 0, obj2_shapes, obj2_poses);

  // Set the active collision objects and transforms
  checker.setActiveCollisionObjects({ "octomap_link", "mesh_link" });
  checker.setDefaultCollisionMarginData(0.0);

  tesseract_common::TransformMap location;
  location["octomap_link"] = Eigen::Isometry3d::Identity();
  location["mesh_link"] = Eigen::Isometry3d::Identity();
  location["mesh_link"].translation() = Eigen::Vector3d(0, 0, 0);
  checker.setCollisionObjectsTransform(location);

  // Perform collision check and get results
  tesseract_collision::ContactResultMap result;
  checker.contactTest(result, tesseract_collision::ContactTestType::ALL);
  tesseract_collision::ContactResultVector result_vector;
  flattenResults(std::move(result), result_vector);

  const tesseract_collision::CollisionShapesConst& geom = checker.getCollisionObjectGeometries("mesh_link");
  const auto& mesh2 = std::static_pointer_cast<const tesseract_geometry::Mesh>(geom.at(0));
  const auto& mesh_vertices = mesh2->getVertices();
  const auto& mesh_triangles = mesh2->getTriangles();

  // ----------- Return correct triangles ------------
  if (mask_type == MaskType::RETURN_COLORIZED)
  {
    // default color is green
    std::vector<Eigen::Vector3i> mesh_vertices_color(mesh_vertices->size(), Eigen::Vector3i(0, 128, 0));

    // Loop over all
    for (tesseract_collision::ContactResult& result : result_vector)
    {
      int idx = 0;
      if (result.link_names[0] != "mesh_link")
        idx = 1;

      mesh_vertices_color[static_cast<std::size_t>((*mesh_triangles)[4 * result.subshape_id[idx] + 1])] =
          Eigen::Vector3i(255, 0, 0);
      mesh_vertices_color[static_cast<std::size_t>((*mesh_triangles)[4 * result.subshape_id[idx] + 2])] =
          Eigen::Vector3i(255, 0, 0);
      mesh_vertices_color[static_cast<std::size_t>((*mesh_triangles)[4 * result.subshape_id[idx] + 3])] =
          Eigen::Vector3i(255, 0, 0);
    }
    mesh_vertices_color_ = mesh_vertices_color;
    masked_mesh_ = std::make_shared<tesseract_geometry::Mesh>(input_mesh_->getVertices(), input_mesh_->getTriangles());
    return true;
  }
  else if (mask_type == MaskType::RETURN_INSIDE)
  {
    // Allocate memory for resulting vertices/triangles
    tesseract_common::VectorVector3d vertices;
    // I don't know of good way to get this size ahead of time, so make it plenty big
    vertices.reserve(result_vector.size() * 3);
    Eigen::VectorXi triangles = Eigen::VectorXi::Zero(static_cast<Eigen::Index>(result_vector.size()) * 4);

    // Loop over all
    int result_idx = 0;
    for (tesseract_collision::ContactResult& result : result_vector)
    {
      // Subshape will have 2 objects. Get the correct idx depending on if the mesh is subshape 0 or subshape 1
      int idx = 0;
      if (result.link_names[0] != "mesh_link")
        idx = 1;

      vertices.push_back((*mesh_vertices)[(*mesh_triangles)[4 * result.subshape_id[idx] + 1]]);
      vertices.push_back((*mesh_vertices)[(*mesh_triangles)[4 * result.subshape_id[idx] + 2]]);
      vertices.push_back((*mesh_vertices)[(*mesh_triangles)[4 * result.subshape_id[idx] + 3]]);

      // tesseract_geometry::Mesh accepts all polygons. The format is [ num_vertices vert_1 vert_2 vert_3 num_vertices
      // vert_1 ... ]. Thus there are 4 entries for each triangle that are the last 3 entries in the vertex list (since
      // we are just blindly pushing them back). Note that this method results in duplicate vertices - potentially a lot
      // of them.
      triangles[result_idx + 0] = 3;
      triangles[result_idx + 1] = vertices.size() - 3;
      triangles[result_idx + 2] = vertices.size() - 2;
      triangles[result_idx + 3] = vertices.size() - 1;
      result_idx += 4;
    }
    // Construct the mesh from the vertices and triangles
    auto vertices_ptr = std::make_shared<tesseract_common::VectorVector3d>(vertices);
    auto triangles_ptr = std::make_shared<Eigen::VectorXi>(triangles);
    masked_mesh_ = std::make_shared<tesseract_geometry::Mesh>(vertices_ptr, triangles_ptr);

    // TODO: Rethink the handling of color here. It should probably go in the Mesh object itself and pass through the
    // colors from the input (if given)
    mesh_vertices_color_.assign(masked_mesh_->getVertices()->size(), Eigen::Vector3i(0, 128, 0));
    return true;
  }
  else if (mask_type == MaskType::RETURN_OUTSIDE)
  {
    // This will keep track of what subshapes are in collision
    std::vector<bool> subshape_in_collision(mesh_triangles->size() / 4, false);

    // Annotate which subshapes are in collision
    for (tesseract_collision::ContactResult& result : result_vector)
    {
      // Subshape will have 2 objects. Get the correct idx depending on if the mesh is subshape 0 or subshape 1
      int idx = 0;
      if (result.link_names[0] != "mesh_link")
        idx = 1;

      subshape_in_collision[result.subshape_id[idx]] = true;
    }

    int num_collision_free_triangles =
        std::count_if(subshape_in_collision.begin(), subshape_in_collision.end(), [](bool i) { return i == false; });

    // Allocate memory for resulting vertices/triangles
    tesseract_common::VectorVector3d vertices;
    vertices.reserve(num_collision_free_triangles * 3);
    Eigen::VectorXi triangles = Eigen::VectorXi::Zero(static_cast<Eigen::Index>(num_collision_free_triangles) * 4);

    // Add the ones that are not in collision to the masked_mesh_
    int result_idx = 0;
    for (int subshape_id = 0; subshape_id < subshape_in_collision.size(); subshape_id++)
    {
      if (!subshape_in_collision[subshape_id])
      {
        vertices.push_back((*mesh_vertices)[(*mesh_triangles)[4 * subshape_id + 1]]);
        vertices.push_back((*mesh_vertices)[(*mesh_triangles)[4 * subshape_id + 2]]);
        vertices.push_back((*mesh_vertices)[(*mesh_triangles)[4 * subshape_id + 3]]);

        // tesseract_geometry::Mesh accepts all polygons. The format is [ num_vertices vert_1 vert_2 vert_3 num_vertices
        // vert_1 ... ]. Thus there are 4 entries for each triangle that are the last 3 entries in the vertex list
        // (since we are just blindly pushing them back). Note that this method results in duplicate vertices -
        // potentially a lot of them.
        triangles[result_idx + 0] = 3;
        triangles[result_idx + 1] = vertices.size() - 3;
        triangles[result_idx + 2] = vertices.size() - 2;
        triangles[result_idx + 3] = vertices.size() - 1;
        result_idx += 4;
      }
    }
    // Construct the mesh from the vertices and triangles
    auto vertices_ptr = std::make_shared<tesseract_common::VectorVector3d>(vertices);
    auto triangles_ptr = std::make_shared<Eigen::VectorXi>(triangles);
    masked_mesh_ = std::make_shared<tesseract_geometry::Mesh>(vertices_ptr, triangles_ptr);
    // TODO: Rethink the handling of color here. It should probably go in the Mesh object itself and pass through the
    // colors from the input (if given)
    mesh_vertices_color_.assign(masked_mesh_->getVertices()->size(), Eigen::Vector3i(0, 128, 0));
    return true;
  }

  return false;
}

bool OctomapMeshMask::saveMaskedMesh(std::string& filepath)
{
  return tesseract_collision::writeSimplePlyFile(filepath,
                                                 *(masked_mesh_->getVertices()),
                                                 mesh_vertices_color_,
                                                 *(masked_mesh_->getTriangles()),
                                                 masked_mesh_->getTriangleCount());
}
