/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2026, Rivelin Robotics Ltd.
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
 *   * Neither the name of the copyright holder nor the names of its
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

#include <moveit/collision_detection_fcl/collision_common.hpp>
#include <moveit/collision_detection_fcl/collision_env_fcl.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/utils/robot_model_test_utils.hpp>

#include <geometric_shapes/shape_operations.h>
#include <gtest/gtest.h>

#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

class SharedGeometryTest : public testing::Test
{
protected:
  SharedGeometryTest()
    : robot_model_(moveit::core::loadTestingRobotModel("panda"))
    , c_env_(std::make_shared<collision_detection::CollisionEnvFCL>(robot_model_))
    , robot_state_(robot_model_)
    , acm_(robot_model_->getLinkModelNames(), false)
  {
    for (const auto& [first, second] :
         std::vector<std::pair<std::string, std::string>>{ { "panda_link0", "panda_link1" },
                                                           { "panda_link1", "panda_link2" },
                                                           { "panda_link2", "panda_link3" },
                                                           { "panda_link3", "panda_link4" },
                                                           { "panda_link4", "panda_link5" },
                                                           { "panda_link5", "panda_link6" },
                                                           { "panda_link6", "panda_link7" },
                                                           { "panda_link7", "panda_hand" },
                                                           { "panda_hand", "panda_rightfinger" },
                                                           { "panda_hand", "panda_leftfinger" },
                                                           { "panda_rightfinger", "panda_leftfinger" },
                                                           { "panda_link5", "panda_link7" },
                                                           { "panda_link6", "panda_hand" } })
    {
      acm_.setEntry(first, second, true);
    }
    robot_state_.setToDefaultValues();
    robot_state_.setVariablePositions({ { "panda_joint2", -0.785 },
                                        { "panda_joint4", -2.356 },
                                        { "panda_joint6", 1.571 },
                                        { "panda_joint7", 0.785 } });
    robot_state_.update();
  }

  moveit::core::RobotModelPtr robot_model_;
  collision_detection::CollisionEnvPtr c_env_;
  moveit::core::RobotState robot_state_;
  collision_detection::AllowedCollisionMatrix acm_;
};

TEST_F(SharedGeometryTest, AttachedMeshGeometrySharedAcrossThreads)
{
  shapes::ShapeConstPtr mesh(shapes::createMeshFromShape(shapes::Box(0.1, 0.1, 0.1)));
  moveit::core::AttachedBody body(robot_model_->getLinkModel("panda_hand"), "part", Eigen::Isometry3d::Identity(),
                                  { mesh }, { Eigen::Isometry3d::Identity() }, {},
                                  trajectory_msgs::msg::JointTrajectory());
  moveit::core::AttachedBody body_copy(body);

  collision_detection::FCLGeometryConstPtr geometry = collision_detection::createCollisionGeometry(mesh, &body, 0);
  collision_detection::FCLGeometryConstPtr other_thread_geometry;
  std::thread([&] { other_thread_geometry = collision_detection::createCollisionGeometry(mesh, &body_copy, 0); }).join();

  EXPECT_EQ(geometry->collision_object_->collisionGeometry(),
            other_thread_geometry->collision_object_->collisionGeometry());
}

TEST_F(SharedGeometryTest, CollisionObjectCopiesAreIndependent)
{
  shapes::ShapeConstPtr mesh(shapes::createMeshFromShape(shapes::Box(0.1, 0.1, 0.1)));
  collision_detection::World::Object owner("part");
  collision_detection::FCLGeometryConstPtr geometry = collision_detection::createCollisionGeometry(mesh, &owner);
  fcl::Transform3d a_pose = fcl::Transform3d::Identity();
  a_pose.translation() = fcl::Vector3d(1.0, 0.0, 0.0);
  fcl::Transform3d b_pose = fcl::Transform3d::Identity();
  b_pose.translation() = fcl::Vector3d(0.0, 2.0, 0.0);
  fcl::Transform3d moved_pose = fcl::Transform3d::Identity();
  moved_pose.translation() = fcl::Vector3d(0.0, 0.0, 3.0);

  collision_detection::FCLCollisionObjectPtr a = collision_detection::createCollisionObject(geometry, a_pose);
  collision_detection::FCLCollisionObjectPtr b = collision_detection::createCollisionObject(geometry, b_pose);
  EXPECT_LT((a->getTranslation() - a_pose.translation()).norm(), 1e-9);
  EXPECT_LT((a->getAABB().center() - a_pose.translation()).norm(), 1e-9);
  a->setTransform(moved_pose);
  a->computeAABB();

  EXPECT_EQ(a->collisionGeometry(), geometry->collision_object_->collisionGeometry());
  EXPECT_EQ(b->collisionGeometry(), geometry->collision_object_->collisionGeometry());
  EXPECT_EQ(a->getUserData(), geometry->collision_geometry_data_.get());
  EXPECT_EQ(b->getUserData(), geometry->collision_geometry_data_.get());
  EXPECT_LT((a->getAABB().center() - moved_pose.translation()).norm(), 1e-9);
  EXPECT_LT((b->getTranslation() - b_pose.translation()).norm(), 1e-9);
  EXPECT_LT((b->getAABB().center() - b_pose.translation()).norm(), 1e-9);
  EXPECT_LT(geometry->collision_object_->getTranslation().norm(), 1e-9);
  EXPECT_LT(geometry->collision_object_->getAABB().center().norm(), 1e-9);
}

TEST_F(SharedGeometryTest, MovedWorldObjectCollidesAtNewPose)
{
  shapes::ShapeConstPtr box(new shapes::Box(0.1, 0.1, 0.1));
  Eigen::Isometry3d far_pose = Eigen::Isometry3d::Identity();
  far_pose.translation().x() = 2.0;
  Eigen::Isometry3d robot_pose = Eigen::Isometry3d::Identity();
  robot_pose.translation().z() = 0.3;
  c_env_->getWorld()->addToObject("box", far_pose, box, Eigen::Isometry3d::Identity());
  std::vector<bool> collisions;

  for (const Eigen::Isometry3d& pose : { far_pose, robot_pose, far_pose })
  {
    c_env_->getWorld()->setObjectPose("box", pose);
    collision_detection::CollisionRequest req;
    collision_detection::CollisionResult res;
    c_env_->checkRobotCollision(req, res, robot_state_, acm_);
    collisions.push_back(res.collision);
  }

  EXPECT_EQ(collisions, (std::vector<bool>{ false, true, false }));
}

TEST_F(SharedGeometryTest, ReattachedShapeKeepsCorrectOwnerType)
{
  shapes::ShapeConstPtr mesh(shapes::createMeshFromShape(shapes::Box(0.1, 0.1, 0.1)));
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation().z() = 0.3;
  const Eigen::Isometry3d pose_in_hand = robot_state_.getGlobalLinkTransform("panda_hand").inverse() * pose;
  collision_detection::CollisionRequest req;
  req.contacts = true;
  req.max_contacts = 100;
  auto part_body_types = [](const collision_detection::CollisionResult& res) {
    std::set<collision_detection::BodyType> types;
    for (const auto& [names, contacts] : res.contacts)
    {
      for (const collision_detection::Contact& contact : contacts)
      {
        if (contact.body_name_1 == "part")
          types.insert(contact.body_type_1);
        if (contact.body_name_2 == "part")
          types.insert(contact.body_type_2);
      }
    }
    return types;
  };
  std::vector<std::set<collision_detection::BodyType>> body_types;

  c_env_->getWorld()->addToObject("part", mesh, pose);
  collision_detection::CollisionResult world_res;
  c_env_->checkRobotCollision(req, world_res, robot_state_, acm_);
  body_types.push_back(part_body_types(world_res));

  c_env_->getWorld()->removeObject("part");
  robot_state_.attachBody("part", pose_in_hand, { mesh }, { Eigen::Isometry3d::Identity() }, std::set<std::string>(),
                          "panda_hand");
  robot_state_.update();
  collision_detection::CollisionResult attached_res;
  c_env_->checkSelfCollision(req, attached_res, robot_state_, acm_);
  body_types.push_back(part_body_types(attached_res));

  robot_state_.clearAttachedBody("part");
  c_env_->getWorld()->addToObject("part", mesh, pose);
  collision_detection::CollisionResult detached_res;
  c_env_->checkRobotCollision(req, detached_res, robot_state_, acm_);
  body_types.push_back(part_body_types(detached_res));

  EXPECT_EQ(body_types,
            (std::vector<std::set<collision_detection::BodyType>>{ { collision_detection::BodyTypes::WORLD_OBJECT },
                                                                   { collision_detection::BodyTypes::ROBOT_ATTACHED },
                                                                   { collision_detection::BodyTypes::WORLD_OBJECT } }));
}

class UnknownShape : public shapes::Shape
{
public:
  shapes::Shape* clone() const override
  {
    return new UnknownShape(*this);
  }

  void scaleAndPadd(double /*scale*/, double /*padd*/) override
  {
  }
};

TEST_F(SharedGeometryTest, UnsupportedShapeIsSkipped)
{
  shapes::ShapeConstPtr shape = std::make_shared<const UnknownShape>();
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation().z() = 0.3;
  collision_detection::World::Object owner("unknown");
  c_env_->getWorld()->addToObject("unknown", shape, pose);
  collision_detection::CollisionRequest req;
  collision_detection::CollisionResult res;

  c_env_->checkRobotCollision(req, res, robot_state_, acm_);

  EXPECT_EQ(collision_detection::createCollisionGeometry(shape, &owner), nullptr);
  EXPECT_FALSE(res.collision);
}

TEST_F(SharedGeometryTest, ConcurrentChecksOnSharedGeometry)
{
  constexpr std::size_t THREAD_COUNT = 8;
  constexpr std::size_t ITERATIONS = 50;
  shapes::ShapeConstPtr mesh(shapes::createMeshFromShape(shapes::Box(0.1, 0.1, 0.1)));
  const std::set<std::string> touch_links{ "panda_hand", "panda_leftfinger", "panda_rightfinger" };
  Eigen::Isometry3d in_front = Eigen::Isometry3d::Identity();
  in_front.translation().z() = 0.15;
  Eigen::Isometry3d beside = in_front;
  beside.translation().y() = 0.2;
  robot_state_.attachBody("part_a", in_front, { mesh }, { Eigen::Isometry3d::Identity() }, touch_links, "panda_hand");
  robot_state_.attachBody("part_b", beside, { mesh }, { Eigen::Isometry3d::Identity() }, touch_links, "panda_hand");
  robot_state_.update();
  c_env_->getWorld()->addToObject("obstacle", mesh,
                                  robot_state_.getGlobalLinkTransform("panda_hand") * in_front *
                                      Eigen::Translation3d(0.05, 0.0, 0.0));
  auto contact_counts = [&] {
    collision_detection::CollisionRequest req;
    req.contacts = true;
    req.max_contacts = 100;
    req.max_contacts_per_pair = 10;
    collision_detection::CollisionResult res;
    c_env_->checkRobotCollision(req, res, robot_state_, acm_);
    std::map<std::pair<std::string, std::string>, std::size_t> counts;
    for (const auto& [names, contacts] : res.contacts)
      counts[names] = contacts.size();
    return counts;
  };
  auto minimum_distance = [&] {
    collision_detection::DistanceRequest req;
    req.acm = &acm_;
    req.enable_nearest_points = true;
    req.enable_signed_distance = true;
    collision_detection::DistanceResult res;
    c_env_->distanceRobot(req, res, robot_state_);
    return std::make_tuple(res.minimum_distance.distance, res.minimum_distance.link_names[0],
                           res.minimum_distance.link_names[1]);
  };
  const auto reference_counts = contact_counts();
  const auto reference_distance = minimum_distance();
  ASSERT_FALSE(reference_counts.empty());

  std::vector<std::size_t> mismatches(THREAD_COUNT, 0);
  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < THREAD_COUNT; ++i)
  {
    threads.emplace_back([&, i] {
      for (std::size_t iteration = 0; iteration < ITERATIONS; ++iteration)
      {
        if (contact_counts() != reference_counts || minimum_distance() != reference_distance)
          ++mismatches[i];
      }
    });
  }
  for (std::thread& thread : threads)
    thread.join();

  EXPECT_EQ(mismatches, std::vector<std::size_t>(THREAD_COUNT, 0));
}

using BodyOwner = std::pair<std::string, collision_detection::BodyType>;

enum class ShapeKind
{
  BOX,
  MESH
};

struct OwnerCase
{
  ShapeKind first;
  ShapeKind second;
  bool same_shape;
};

shapes::ShapeConstPtr makeShape(ShapeKind kind)
{
  if (kind == ShapeKind::MESH)
    return shapes::ShapeConstPtr(shapes::createMeshFromShape(shapes::Box(0.1, 0.1, 0.1)));
  return std::make_shared<const shapes::Box>(0.1, 0.1, 0.1);
}

class ContactOwnersTest : public SharedGeometryTest, public testing::WithParamInterface<OwnerCase>
{
protected:
  ContactOwnersTest()
    : first_shape_(makeShape(GetParam().first))
    , second_shape_(GetParam().same_shape ? first_shape_ : makeShape(GetParam().second))
    , second_(robot_model_->getLinkModel("panda_hand"), "second", Eigen::Isometry3d::Identity(), { second_shape_ },
              { Eigen::Isometry3d::Identity() }, std::set<std::string>(), trajectory_msgs::msg::JointTrajectory())
    , first_geometry_(collision_detection::createCollisionGeometry(first_shape_, &first_))
    , second_geometry_(collision_detection::createCollisionGeometry(second_shape_, &second_, 0))
  {
  }

  shapes::ShapeConstPtr first_shape_;
  shapes::ShapeConstPtr second_shape_;
  collision_detection::World::Object first_{ "first" };
  moveit::core::AttachedBody second_;
  collision_detection::FCLGeometryConstPtr first_geometry_;
  collision_detection::FCLGeometryConstPtr second_geometry_;
};

TEST_P(ContactOwnersTest, ContactsNameEachBody)
{
  fcl::Transform3d second_pose = fcl::Transform3d::Identity();
  second_pose.translation() = fcl::Vector3d(0.05, 0.03, 0.02);
  collision_detection::FCLCollisionObjectPtr first_object = collision_detection::createCollisionObject(first_geometry_);
  collision_detection::FCLCollisionObjectPtr second_object =
      collision_detection::createCollisionObject(second_geometry_, second_pose);
  collision_detection::CollisionRequest req;
  req.contacts = true;
  req.max_contacts = 10;
  std::set<std::set<BodyOwner>> contact_owners;

  for (const auto& [o1, o2] : { std::make_pair(first_object.get(), second_object.get()),
                                std::make_pair(second_object.get(), first_object.get()) })
  {
    collision_detection::CollisionResult res;
    collision_detection::CollisionData data(&req, &res, nullptr);
    collision_detection::collisionCallback(o1, o2, &data);
    for (const auto& [names, contacts] : res.contacts)
    {
      for (const collision_detection::Contact& contact : contacts)
        contact_owners.insert(
            { { contact.body_name_1, contact.body_type_1 }, { contact.body_name_2, contact.body_type_2 } });
    }
  }

  if (GetParam().same_shape)
  {
    EXPECT_EQ(first_geometry_->collision_object_->collisionGeometry(),
              second_geometry_->collision_object_->collisionGeometry());
  }
  EXPECT_EQ(contact_owners,
            (std::set<std::set<BodyOwner>>{ { { "first", collision_detection::BodyTypes::WORLD_OBJECT },
                                              { "second", collision_detection::BodyTypes::ROBOT_ATTACHED } } }));
}

TEST_P(ContactOwnersTest, DistanceNamesEachBodyWithItsNearestPoint)
{
  using NearestPointOwner = std::tuple<std::string, collision_detection::BodyType, bool>;
  fcl::Transform3d second_pose = fcl::Transform3d::Identity();
  second_pose.translation().x() = 0.5;
  collision_detection::FCLCollisionObjectPtr first_object = collision_detection::createCollisionObject(first_geometry_);
  collision_detection::FCLCollisionObjectPtr second_object =
      collision_detection::createCollisionObject(second_geometry_, second_pose);
  collision_detection::DistanceRequest req;
  req.enable_nearest_points = true;
  std::set<std::set<NearestPointOwner>> nearest_point_owners;

  for (const auto& [o1, o2] : { std::make_pair(first_object.get(), second_object.get()),
                                std::make_pair(second_object.get(), first_object.get()) })
  {
    collision_detection::DistanceResult res;
    collision_detection::DistanceData data(&req, &res);
    double min_dist = std::numeric_limits<double>::max();
    collision_detection::distanceCallback(o1, o2, &data, min_dist);
    const collision_detection::DistanceResultsData& result = res.minimum_distance;
    nearest_point_owners.insert({ { result.link_names[0], result.body_types[0], result.nearest_points[0].x() > 0.25 },
                                  { result.link_names[1], result.body_types[1], result.nearest_points[1].x() > 0.25 } });
  }

  EXPECT_EQ(nearest_point_owners, (std::set<std::set<NearestPointOwner>>{
                                      { { "first", collision_detection::BodyTypes::WORLD_OBJECT, false },
                                        { "second", collision_detection::BodyTypes::ROBOT_ATTACHED, true } } }));
}

INSTANTIATE_TEST_SUITE_P(OwnerPairs, ContactOwnersTest,
                         testing::Values(OwnerCase{ ShapeKind::BOX, ShapeKind::MESH, false },
                                         OwnerCase{ ShapeKind::MESH, ShapeKind::BOX, false },
                                         OwnerCase{ ShapeKind::BOX, ShapeKind::BOX, false },
                                         OwnerCase{ ShapeKind::MESH, ShapeKind::MESH, false },
                                         OwnerCase{ ShapeKind::MESH, ShapeKind::MESH, true }));

TEST_F(SharedGeometryTest, PaddedLinkGeometrySurvivesCacheClean)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(0.43, 0.0, 0.55);
  c_env_->getWorld()->addToObject("box", shapes::ShapeConstPtr(new shapes::Box(0.1, 0.1, 0.1)), pose);
  std::vector<bool> collisions;

  for (double padding : { 0.08, 0.0 })
  {
    c_env_->setLinkPadding("panda_hand", padding);
    collision_detection::cleanCollisionGeometryCache();
    collision_detection::CollisionRequest req;
    collision_detection::CollisionResult res;
    c_env_->checkRobotCollision(req, res, robot_state_, acm_);
    collisions.push_back(res.collision);
  }

  EXPECT_EQ(collisions, (std::vector<bool>{ true, false }));
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
