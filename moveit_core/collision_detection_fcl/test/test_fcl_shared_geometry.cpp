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
#include <ostream>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

// One FCL geometry is now built per shape and shared by every thread and every owner (link, attached body, world
// object) that uses the shape. Who a collision object belongs to therefore lives on the fcl::CollisionObject, not on
// the geometry. These tests check the sharing itself, and that collision results still name the right owner.

using namespace collision_detection;

namespace
{
shapes::ShapeConstPtr makeBox()
{
  return std::make_shared<const shapes::Box>(0.1, 0.1, 0.1);
}

// The same 0.1 m cube as makeBox(), as a triangle mesh.
shapes::ShapeConstPtr makeMesh()
{
  return shapes::ShapeConstPtr(shapes::createMeshFromShape(shapes::Box(0.1, 0.1, 0.1)));
}

fcl::Transform3d translation(double x, double y, double z)
{
  fcl::Transform3d transform = fcl::Transform3d::Identity();
  transform.translation() = fcl::Vector3d(x, y, z);
  return transform;
}

bool isNear(const fcl::Vector3d& actual, const fcl::Vector3d& expected)
{
  return (actual - expected).norm() < 1e-9;
}

// The body types that the contacts in res report for the body called name.
std::set<BodyType> bodyTypesOf(const std::string& name, const CollisionResult& res)
{
  std::set<BodyType> types;
  for (const auto& [names, contacts] : res.contacts)
  {
    for (const Contact& contact : contacts)
    {
      if (contact.body_name_1 == name)
        types.insert(contact.body_type_1);
      if (contact.body_name_2 == name)
        types.insert(contact.body_type_2);
    }
  }
  return types;
}

// A shape type the FCL plugin has no geometry for.
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

// A Panda in its "ready" pose, not in collision with itself: neighbouring links are allowed to touch.
class SharedGeometryTest : public testing::Test
{
protected:
  SharedGeometryTest()
    : robot_model_(moveit::core::loadTestingRobotModel("panda"))
    , c_env_(std::make_shared<CollisionEnvFCL>(robot_model_))
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

  // An attached body on the hand made of one shape. It is not attached to robot_state_.
  moveit::core::AttachedBody makeHandBody(const std::string& name, const shapes::ShapeConstPtr& shape) const
  {
    return moveit::core::AttachedBody(robot_model_->getLinkModel("panda_hand"), name, Eigen::Isometry3d::Identity(),
                                      { shape }, { Eigen::Isometry3d::Identity() }, std::set<std::string>(),
                                      trajectory_msgs::msg::JointTrajectory());
  }

  bool robotCollidesWithWorld() const
  {
    CollisionRequest req;
    CollisionResult res;
    c_env_->checkRobotCollision(req, res, robot_state_, acm_);
    return res.collision;
  }

  // A pose that puts a 0.1 m cube inside the robot near its base.
  static Eigen::Isometry3d poseInsideRobot()
  {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation().z() = 0.3;
    return pose;
  }

  moveit::core::RobotModelPtr robot_model_;
  CollisionEnvPtr c_env_;
  moveit::core::RobotState robot_state_;
  AllowedCollisionMatrix acm_;
};

// The two shapes used by ContactOwnersTest: one for a world object, one for an attached body.
struct ShapePair
{
  std::string name;
  shapes::ShapeConstPtr first;
  shapes::ShapeConstPtr second;
};

ShapePair boxAndMesh()
{
  return { "BoxMesh", makeBox(), makeMesh() };
}

ShapePair sameMeshTwice()
{
  shapes::ShapeConstPtr mesh = makeMesh();
  return { "SharedMesh", mesh, mesh };
}

void PrintTo(const ShapePair& shapes, std::ostream* os)
{
  *os << shapes.name;
}

// Calls the collision and distance callbacks directly on two collision objects: world object "first" and attached
// body "second".
//
// FCL reports each contact or nearest point against a geometry, and we have to work out which of the two objects
// that geometry belongs to. The two shape pairs are the two ways that can go wrong:
//  - BoxMesh: FCL only implements mesh-vs-box, so when given box-vs-mesh it swaps the pair internally and its result
//    comes back in the opposite order to the call.
//  - SharedMesh: both objects use the very same geometry, so the geometry pointer cannot tell them apart.
// Every test passes the objects in both orders, so each pair is exercised swapped and unswapped.
class ContactOwnersTest : public SharedGeometryTest, public testing::WithParamInterface<ShapePair>
{
protected:
  using ObjectPair = std::pair<fcl::CollisionObjectd*, fcl::CollisionObjectd*>;

  // "first" at the origin and "second" at second_pose, in both call orders.
  std::vector<ObjectPair> bothOrders(const fcl::Transform3d& second_pose)
  {
    first_object_ = createCollisionObject(first_geometry_);
    second_object_ = createCollisionObject(second_geometry_, second_pose);
    return { { first_object_.get(), second_object_.get() }, { second_object_.get(), first_object_.get() } };
  }

  World::Object first_{ "first" };
  moveit::core::AttachedBody second_ = makeHandBody("second", GetParam().second);
  FCLGeometryConstPtr first_geometry_ = createCollisionGeometry(GetParam().first, &first_);
  FCLGeometryConstPtr second_geometry_ = createCollisionGeometry(GetParam().second, &second_, 0);
  FCLCollisionObjectPtr first_object_;
  FCLCollisionObjectPtr second_object_;
};
}  // namespace

// The headline behaviour: one mesh used by an attached body, by a copy of that body on another thread, and by a
// world object is backed by a single FCL geometry.
TEST_F(SharedGeometryTest, MeshGeometrySharedAcrossThreadsAndOwners)
{
  shapes::ShapeConstPtr mesh = makeMesh();
  moveit::core::AttachedBody body = makeHandBody("part", mesh);
  moveit::core::AttachedBody body_copy(body);
  World::Object world_object("part");

  FCLGeometryConstPtr geometry = createCollisionGeometry(mesh, &body, 0);
  FCLGeometryConstPtr other_thread_geometry;
  std::thread([&] { other_thread_geometry = createCollisionGeometry(mesh, &body_copy, 0); }).join();
  FCLGeometryConstPtr world_geometry = createCollisionGeometry(mesh, &world_object);

  EXPECT_EQ(other_thread_geometry->collision_object_->collisionGeometry(),
            geometry->collision_object_->collisionGeometry());
  EXPECT_EQ(world_geometry->collision_object_->collisionGeometry(), geometry->collision_object_->collisionGeometry());
}

// createCollisionObject() copies the cached prototype object rather than building on the geometry again. Each copy
// points at the shared geometry and at its owner's data, but has its own pose and bounding box, and the prototype
// itself is left at the origin.
TEST_F(SharedGeometryTest, CollisionObjectsShareGeometryButNotPose)
{
  shapes::ShapeConstPtr mesh = makeMesh();
  World::Object owner("part");
  FCLGeometryConstPtr geometry = createCollisionGeometry(mesh, &owner);
  const fcl::CollisionObjectd& prototype = *geometry->collision_object_;

  FCLCollisionObjectPtr a = createCollisionObject(geometry, translation(1.0, 0.0, 0.0));
  FCLCollisionObjectPtr b = createCollisionObject(geometry, translation(0.0, 2.0, 0.0));

  EXPECT_EQ(a->collisionGeometry(), prototype.collisionGeometry());
  EXPECT_EQ(b->collisionGeometry(), prototype.collisionGeometry());
  EXPECT_EQ(a->getUserData(), geometry->collision_geometry_data_.get());
  EXPECT_EQ(b->getUserData(), geometry->collision_geometry_data_.get());
  EXPECT_TRUE(isNear(a->getAABB().center(), fcl::Vector3d(1.0, 0.0, 0.0)));
  EXPECT_TRUE(isNear(b->getAABB().center(), fcl::Vector3d(0.0, 2.0, 0.0)));
  EXPECT_TRUE(isNear(prototype.getAABB().center(), fcl::Vector3d::Zero()));
}

// Moving a world object no longer recomputes the shared geometry's local bounding box, only the object's own. The
// box starts away from the robot, is moved into it and then away again, and must collide only while it is inside.
TEST_F(SharedGeometryTest, MovedWorldObjectCollidesAtNewPose)
{
  Eigen::Isometry3d far_pose = Eigen::Isometry3d::Identity();
  far_pose.translation().x() = 2.0;
  c_env_->getWorld()->addToObject("box", far_pose, makeBox(), Eigen::Isometry3d::Identity());
  std::vector<bool> collisions;

  for (const Eigen::Isometry3d& pose : { far_pose, poseInsideRobot(), far_pose })
  {
    c_env_->getWorld()->setObjectPose("box", pose);
    collisions.push_back(robotCollidesWithWorld());
  }

  EXPECT_EQ(collisions, (std::vector<bool>{ false, true, false }));
}

// The same mesh goes from world object, to attached body, back to world object, as when a part is picked and placed.
// The geometry is reused each time, so the owner type in the contacts has to come from the current owner and not
// from whoever built the geometry first.
TEST_F(SharedGeometryTest, ReattachedShapeKeepsCorrectOwnerType)
{
  shapes::ShapeConstPtr mesh = makeMesh();
  const Eigen::Isometry3d pose = poseInsideRobot();
  const Eigen::Isometry3d pose_in_hand = robot_state_.getGlobalLinkTransform("panda_hand").inverse() * pose;
  CollisionRequest req;
  req.contacts = true;
  req.max_contacts = 100;
  std::vector<std::set<BodyType>> body_types;

  c_env_->getWorld()->addToObject("part", mesh, pose);
  CollisionResult world_res;
  c_env_->checkRobotCollision(req, world_res, robot_state_, acm_);
  body_types.push_back(bodyTypesOf("part", world_res));

  c_env_->getWorld()->removeObject("part");
  robot_state_.attachBody("part", pose_in_hand, { mesh }, { Eigen::Isometry3d::Identity() }, std::set<std::string>(),
                          "panda_hand");
  robot_state_.update();
  CollisionResult attached_res;
  c_env_->checkSelfCollision(req, attached_res, robot_state_, acm_);
  body_types.push_back(bodyTypesOf("part", attached_res));

  robot_state_.clearAttachedBody("part");
  c_env_->getWorld()->addToObject("part", mesh, pose);
  CollisionResult detached_res;
  c_env_->checkRobotCollision(req, detached_res, robot_state_, acm_);
  body_types.push_back(bodyTypesOf("part", detached_res));

  EXPECT_EQ(body_types, (std::vector<std::set<BodyType>>{ { BodyTypes::WORLD_OBJECT },
                                                          { BodyTypes::ROBOT_ATTACHED },
                                                          { BodyTypes::WORLD_OBJECT } }));
}

// A shape FCL cannot represent gets no geometry. A world object made of one, placed inside the robot, is skipped by
// the collision check instead of crashing it.
TEST_F(SharedGeometryTest, UnsupportedShapeIsSkipped)
{
  shapes::ShapeConstPtr shape = std::make_shared<const UnknownShape>();
  World::Object owner("unknown");
  c_env_->getWorld()->addToObject("unknown", shape, poseInsideRobot());

  EXPECT_EQ(createCollisionGeometry(shape, &owner), nullptr);
  EXPECT_FALSE(robotCollidesWithWorld());
}

// Eight threads run collision and distance checks at once against one mesh that is used three times: by two attached
// bodies and by a world obstacle overlapping one of them. Every result has to match the one computed beforehand on a
// single thread.
//
// A data race on the shared geometry rarely changes a result, so on a normal build this mostly shows the checks run
// and agree. Its real value is under ThreadSanitizer, which reports any unsynchronised write to the geometry.
TEST_F(SharedGeometryTest, ConcurrentChecksOnSharedGeometry)
{
  constexpr std::size_t THREAD_COUNT = 8;
  constexpr std::size_t ITERATIONS = 50;
  shapes::ShapeConstPtr mesh = makeMesh();
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
  // Number of contacts between each pair of bodies.
  auto contact_counts = [&] {
    CollisionRequest req;
    req.contacts = true;
    req.max_contacts = 100;
    req.max_contacts_per_pair = 10;
    CollisionResult res;
    c_env_->checkRobotCollision(req, res, robot_state_, acm_);
    std::map<std::pair<std::string, std::string>, std::size_t> counts;
    for (const auto& [names, contacts] : res.contacts)
      counts[names] = contacts.size();
    return counts;
  };
  // Smallest robot-to-world distance and the two bodies it is between.
  auto minimum_distance = [&] {
    DistanceRequest req;
    req.acm = &acm_;
    req.enable_nearest_points = true;
    req.enable_signed_distance = true;
    DistanceResult res;
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

// "second" overlaps "first", offset a little along every axis. Whichever order the objects are passed in, every
// contact must name "first" as a world object and "second" as an attached body, with the normal pointing from
// body 1 to body 2. A name on the wrong side would show up as a normal pointing the wrong way.
TEST_P(ContactOwnersTest, ContactsNameEachBody)
{
  using BodyOwner = std::pair<std::string, BodyType>;
  const fcl::Vector3d first_to_second(0.05, 0.03, 0.02);
  CollisionRequest req;
  req.contacts = true;
  req.max_contacts = 10;
  std::set<std::set<BodyOwner>> contact_owners;
  std::size_t reversed_normals = 0;

  for (const auto& [o1, o2] : bothOrders(translation(first_to_second.x(), first_to_second.y(), first_to_second.z())))
  {
    CollisionResult res;
    CollisionData data(&req, &res, nullptr);
    collisionCallback(o1, o2, &data);
    for (const auto& [names, contacts] : res.contacts)
    {
      for (const Contact& contact : contacts)
      {
        contact_owners.insert(
            { { contact.body_name_1, contact.body_type_1 }, { contact.body_name_2, contact.body_type_2 } });
        const fcl::Vector3d body_1_to_body_2 =
            contact.body_name_1 == "first" ? first_to_second : fcl::Vector3d(-first_to_second);
        if (contact.normal.dot(body_1_to_body_2) <= 0)
          ++reversed_normals;
      }
    }
  }

  EXPECT_EQ(contact_owners, (std::set<std::set<BodyOwner>>{ { { "first", BodyTypes::WORLD_OBJECT },
                                                              { "second", BodyTypes::ROBOT_ATTACHED } } }));
  EXPECT_EQ(reversed_normals, 0u);
}

// "second" sits 0.5 m along x from "first". Whichever order the objects are passed in, each name and type must be
// reported alongside that body's own nearest point: "first" with the point near x = 0, "second" with the one near
// x = 0.5.
TEST_P(ContactOwnersTest, DistanceNamesEachBodyWithItsNearestPoint)
{
  // Name, type, and whether the nearest point reported for it is on the far ("second") side.
  using NearestPointOwner = std::tuple<std::string, BodyType, bool>;
  DistanceRequest req;
  req.enable_nearest_points = true;
  std::set<std::set<NearestPointOwner>> nearest_point_owners;

  for (const auto& [o1, o2] : bothOrders(translation(0.5, 0.0, 0.0)))
  {
    DistanceResult res;
    DistanceData data(&req, &res);
    double min_dist = std::numeric_limits<double>::max();
    distanceCallback(o1, o2, &data, min_dist);
    const DistanceResultsData& result = res.minimum_distance;
    nearest_point_owners.insert({ { result.link_names[0], result.body_types[0], result.nearest_points[0].x() > 0.25 },
                                  { result.link_names[1], result.body_types[1], result.nearest_points[1].x() > 0.25 } });
  }

  EXPECT_EQ(nearest_point_owners,
            (std::set<std::set<NearestPointOwner>>{ { { "first", BodyTypes::WORLD_OBJECT, false },
                                                      { "second", BodyTypes::ROBOT_ATTACHED, true } } }));
}

INSTANTIATE_TEST_SUITE_P(OwnerPairs, ContactOwnersTest, testing::Values(boxAndMesh(), sameMeshTwice()),
                         [](const testing::TestParamInfo<ShapePair>& info) { return info.param.name; });

// Changing a link's padding rebuilds that link's collision object from a scaled copy of its shape. The copy is
// temporary, so the cache drops its entry at the next clean, and the link has to keep working from its own reference
// to the geometry. A box sits just in front of the hand: only the padded hand reaches it.
TEST_F(SharedGeometryTest, PaddedLinkGeometrySurvivesCacheClean)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(0.43, 0.0, 0.55);
  c_env_->getWorld()->addToObject("box", makeBox(), pose);
  std::vector<bool> collisions;

  for (double padding : { 0.08, 0.0 })
  {
    c_env_->setLinkPadding("panda_hand", padding);
    cleanCollisionGeometryCache();
    collisions.push_back(robotCollidesWithWorld());
  }

  EXPECT_EQ(collisions, (std::vector<bool>{ true, false }));
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
