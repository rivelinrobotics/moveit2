/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018 Pilz GmbH & Co. KG
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Pilz GmbH & Co. KG nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include <pilz_industrial_motion_planner/trajectory_logger.hpp>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include <moveit/utils/logger.hpp>

namespace pilz_industrial_motion_planner
{
namespace
{
rclcpp::Logger getLogger()
{
  return moveit::getLogger("moveit.planners.pilz.trajectory_logger");
}

/// Set once by configure() during planner manager initialization, read-only afterwards.
bool g_enabled = false;
std::string g_directory;
std::string g_pipeline_name;

using JointTrajectoryPoint = trajectory_msgs::msg::JointTrajectoryPoint;
using Series = std::vector<double> JointTrajectoryPoint::*;

std::string timestamp()
{
  const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm local_time;
  localtime_r(&now, &local_time);
  std::ostringstream oss;
  oss << std::put_time(&local_time, "%Y%m%d_%H%M%S");
  return oss.str();
}

/**
 * @brief Positions of the group's active joints within trajectory.joint_names.
 *
 * The generators name their joints in std::map order, which is alphabetical rather than the
 * group's active joint order. Reordering here keeps the CSV columns comparable across planners.
 */
bool activeJointColumns(const moveit::core::RobotModelConstPtr& robot_model, const std::string& group,
                        const trajectory_msgs::msg::JointTrajectory& trajectory, std::vector<size_t>& columns)
{
  const moveit::core::JointModelGroup* jmg = robot_model->getJointModelGroup(group);
  if (jmg == nullptr)
  {
    RCLCPP_ERROR_STREAM(getLogger(), "Cannot log trajectory, unknown planning group: " << group);
    return false;
  }

  for (const std::string& joint_name : jmg->getActiveJointModelNames())
  {
    const auto it = std::find(trajectory.joint_names.begin(), trajectory.joint_names.end(), joint_name);
    if (it == trajectory.joint_names.end())
    {
      RCLCPP_ERROR_STREAM(getLogger(), "Cannot log trajectory, joint \"" << joint_name
                                                                         << "\" is missing from the trajectory");
      return false;
    }
    columns.push_back(std::distance(trajectory.joint_names.begin(), it));
  }
  return true;
}

std::string csvHeader(const std::string& group, size_t n_joints)
{
  std::string header = "time";
  for (size_t i = 0; i < n_joints; ++i)
  {
    header += "," + group + "_" + std::to_string(i + 1);
  }
  return header;
}

void writeSeries(const std::filesystem::path& filepath, const std::string& csv_header,
                 const std::vector<size_t>& columns, const trajectory_msgs::msg::JointTrajectory& trajectory,
                 Series series)
{
  std::ofstream file(filepath);
  if (!file)
  {
    RCLCPP_ERROR_STREAM(getLogger(), "Failed to open " << filepath.string() << " for writing");
    return;
  }

  file << csv_header << "\n";
  for (const JointTrajectoryPoint& point : trajectory.points)
  {
    const std::vector<double>& values = point.*series;
    if (values.size() != trajectory.joint_names.size())
    {
      RCLCPP_ERROR_STREAM(getLogger(), "Trajectory point does not carry a value per joint, "
                                           << filepath.filename().string() << " is incomplete");
      return;
    }
    file << rclcpp::Duration(point.time_from_start).seconds();
    for (size_t column : columns)
    {
      file << "," << values.at(column);
    }
    file << "\n";
  }
}
}  // namespace

void TrajectoryLogger::configure(const rclcpp::Node::SharedPtr& node, const std::string& parameter_namespace)
{
  // The parameter namespace a pipeline is configured under is its pipeline name.
  g_pipeline_name = parameter_namespace;

  const std::string prefix = parameter_namespace.empty() ? "" : parameter_namespace + ".";
  const std::string enabled_param = prefix + "trajectory_logging.enabled";
  const std::string directory_param = prefix + "trajectory_logging.directory";

  g_enabled = node->has_parameter(enabled_param) ? node->get_parameter(enabled_param).as_bool() :
                                                   node->declare_parameter(enabled_param, false);
  g_directory = node->has_parameter(directory_param) ? node->get_parameter(directory_param).as_string() :
                                                       node->declare_parameter(directory_param, std::string());

  if (!g_enabled)
  {
    return;
  }

  if (g_directory.empty())
  {
    RCLCPP_ERROR_STREAM(getLogger(), "Trajectory logging requested but '" << directory_param
                                                                          << "' is empty. Logging stays disabled.");
    g_enabled = false;
    return;
  }

  RCLCPP_WARN_STREAM(getLogger(), "Trajectory logging enabled, writing CSVs under "
                                      << (std::filesystem::path(g_directory) / "trajectories").string());
}

bool TrajectoryLogger::isEnabled()
{
  return g_enabled;
}

void TrajectoryLogger::log(const moveit::core::RobotModelConstPtr& robot_model, const std::string& group,
                           const std::string& pipeline_id, const std::string& planner_id,
                           const trajectory_msgs::msg::JointTrajectory& trajectory)
{
  if (trajectory.points.empty())
  {
    return;
  }

  std::vector<size_t> columns;
  if (!activeJointColumns(robot_model, group, trajectory, columns))
  {
    return;
  }

  // A request only carries a pipeline_id when the caller names a pipeline explicitly; callers
  // relying on the default pipeline leave it empty, so fall back to the configured name.
  std::string subdirectory = group;
  const std::string& pipeline = pipeline_id.empty() ? g_pipeline_name : pipeline_id;
  if (!pipeline.empty())
  {
    subdirectory += "_" + pipeline;
  }
  if (!planner_id.empty())
  {
    subdirectory += "_" + planner_id;
  }
  subdirectory += "_" + timestamp();

  const std::filesystem::path directory = std::filesystem::path(g_directory) / "trajectories" / subdirectory;

  try
  {
    std::filesystem::create_directories(directory);
    const std::string csv_header = csvHeader(group, columns.size());
    writeSeries(directory / "position.csv", csv_header, columns, trajectory, &JointTrajectoryPoint::positions);
    writeSeries(directory / "velocity.csv", csv_header, columns, trajectory, &JointTrajectoryPoint::velocities);
    writeSeries(directory / "acceleration.csv", csv_header, columns, trajectory,
                &JointTrajectoryPoint::accelerations);
  }
  catch (const std::exception& ex)
  {
    // A debug setting must never fail a plan.
    RCLCPP_ERROR_STREAM(getLogger(), "Failed to log trajectory to " << directory.string() << ": " << ex.what());
    return;
  }

  RCLCPP_INFO_STREAM(getLogger(), "Logged " << trajectory.points.size() << " trajectory points to "
                                            << directory.string());
}

}  // namespace pilz_industrial_motion_planner
