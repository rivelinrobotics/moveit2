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

#pragma once

#include <string>

#include <moveit/robot_model/robot_model.hpp>
#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace pilz_industrial_motion_planner
{
/**
 * @brief Optional CSV dump of generated trajectories, for offline comparison against other planners.
 *
 * When enabled, every trajectory produced by a generator is written to
 * <directory>/trajectories/<group>_<pipeline_id>_<planner_id>_<YYYYmmdd_HHMMSS>/ as three files,
 * position.csv, velocity.csv and acceleration.csv. Each file holds a "time" column
 * followed by one column per active joint of the group, named <group>_1 ... <group>_N
 * in the group's active joint order.
 *
 * Logging is off unless the parameters below are set on the move_group node, where
 * <ns> is the planning pipeline's parameter namespace (e.g. "pilz_industrial_motion_planner"):
 *   <ns>.trajectory_logging.enabled   (bool, default false)
 *   <ns>.trajectory_logging.directory (string, no default)
 */
class TrajectoryLogger
{
public:
  /**
   * @brief Declare and read the logging parameters. Called once while the planner manager
   * initializes, before any generator runs.
   */
  static void configure(const rclcpp::Node::SharedPtr& node, const std::string& parameter_namespace);

  static bool isEnabled();

  /**
   * @brief Write @p trajectory as CSVs. Never throws; failures are reported to the log so that
   * a broken debug setting cannot fail a plan.
   */
  static void log(const moveit::core::RobotModelConstPtr& robot_model, const std::string& group,
                  const std::string& pipeline_id, const std::string& planner_id,
                  const trajectory_msgs::msg::JointTrajectory& trajectory);
};

}  // namespace pilz_industrial_motion_planner
