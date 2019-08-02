#include <iostream>
#include <thread>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>

int main (int argc, char** argv)
{
  const char* fn_target_cloud =
    "../data/stationary0_section3_part1_s0_filtered020.aligned.pcd";
  const char* fn_input_cloud =
    "../data/stationary0_section4_part1_s0_filtered020.aligned.pcd";
  const char* fn_output_cloud =
    "../data/stationary0_section4_part1_s0_filtered020.refined.pcd";

  const bool filter_on_target_cloud = true;
  const bool filter_on_input_cloud = true;
  const float filter_leaf_size = 0.2;

  // load target point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(
    new pcl::PointCloud<pcl::PointXYZ>
  );
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(fn_target_cloud, *target_cloud) == -1)
  {
    PCL_ERROR("Couldn't read the target point cloud file \n");
    return (-1);
  }
  std::cout << "Loaded " << target_cloud->size ()
            << " data points from the target point cloud file" << std::endl;

  // load input point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
    new pcl::PointCloud<pcl::PointXYZ>
  );
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(fn_input_cloud, *input_cloud) == -1)
  {
    PCL_ERROR("Couldn't read the input point cloud file \n");
    return (-1);
  }
  std::cout << "Loaded " << input_cloud->size ()
            << " data points from the input point cloud file" << std::endl;

  // filter point clouds
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
  approximate_voxel_filter.setLeafSize(
    filter_leaf_size, filter_leaf_size, filter_leaf_size
  );

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target_cloud(
    new pcl::PointCloud<pcl::PointXYZ>
  );
  if (filter_on_target_cloud)
  {
    approximate_voxel_filter.setInputCloud(target_cloud);
    approximate_voxel_filter.filter(*filtered_target_cloud);
    std::cout << "Filtered cloud contains " << filtered_target_cloud->size()
              << " data points from the target point cloud" << std::endl;
  }
  else
  {
    filtered_target_cloud = target_cloud;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_input_cloud(
    new pcl::PointCloud<pcl::PointXYZ>
  );
  if (filter_on_input_cloud)
  {
    approximate_voxel_filter.setInputCloud(input_cloud);
    approximate_voxel_filter.filter(*filtered_input_cloud);
    std::cout << "Filtered cloud contains " << filtered_input_cloud->size()
              << " data points from the input point cloud" << std::endl;
  }
  else
  {
    filtered_input_cloud = input_cloud;
  }

  // initialize a Normal Distributions Transform (NDT)
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

  // set scale dependent NDT parameters
  ndt.setTransformationEpsilon(0.1); // error tolerance
  ndt.setStepSize(0.1);              // maximum line search step size
  ndt.setResolution(1.0);            // NDT grid resolution
  ndt.setMaximumIterations(30);      // maximum registration iterations

  // specify point clouds
  ndt.setInputSource(filtered_input_cloud);
  ndt.setInputTarget(filtered_target_cloud);

  // provide initial alignment estimate
  Eigen::AngleAxisf init_rotation(0.0, Eigen::Vector3f::UnitZ());
  Eigen::Translation3f init_translation(0.0, 0.0, 0.0);
  Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();

  // align filtered point clouds from initial guess
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_output_cloud(
    new pcl::PointCloud<pcl::PointXYZ>
  );
  ndt.align(*filtered_output_cloud, init_guess);

  std::cout << "Aligned with Normal Distributions Transform ("
            << "converged: " << ndt.hasConverged () << ", "
            << "score: "     << ndt.getFitnessScore () << ")"
            << std::endl;

  // align original input cloud using the transform found
  pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(
    new pcl::PointCloud<pcl::PointXYZ>
  );
  pcl::transformPointCloud(
    *input_cloud, *output_cloud, ndt.getFinalTransformation()
  );

  // save output point cloud
  pcl::io::savePCDFileASCII(fn_output_cloud, *output_cloud);

  // initialize a point cloud visualizer
  pcl::visualization::PCLVisualizer::Ptr visualizer(
    new pcl::visualization::PCLVisualizer("Point Cloud Visualizer")
  );
  visualizer->setBackgroundColor(0, 0, 0);

  target_cloud = filtered_target_cloud;
  input_cloud = filtered_input_cloud;
  output_cloud = filtered_output_cloud;

  // visualize the target point cloud in green color
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(
    target_cloud, 0, 255, 0
  );
  visualizer->addPointCloud<pcl::PointXYZ>(
    target_cloud, target_color, "target"
  );
  visualizer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target"
  );

  // visualize the input point cloud in blue color
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> input_color(
    input_cloud, 0, 0, 255
  );
  visualizer->addPointCloud<pcl::PointXYZ>(
    input_cloud, input_color, "input"
  );
  visualizer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "input"
  );

  // visualize the output point cloud in red color
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> output_color(
    output_cloud, 255, 0, 0
  );
  visualizer->addPointCloud<pcl::PointXYZ>(
    output_cloud, output_color, "output"
  );
  visualizer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "output"
  );

  // launch visualizer
  visualizer->addCoordinateSystem(1.0, "global");
  visualizer->initCameraParameters();

  // wait until visualizer window is closed
  while (!visualizer->wasStopped())
  {
    visualizer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  return (0);
}
