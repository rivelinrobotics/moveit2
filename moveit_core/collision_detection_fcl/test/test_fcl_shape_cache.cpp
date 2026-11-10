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

#include <moveit/collision_detection_fcl/fcl_shape_cache.hpp>

#if (MOVEIT_FCL_VERSION >= FCL_VERSION_CHECK(0, 6, 0))
#include <fcl/geometry/shape/sphere.h>
#else
#include <fcl/shape/geometric_shapes.h>
#endif

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

// These tests drive FCLShapeCache directly with stand-in build functions, so they are only about the caching:
// what gets built, how often, and when it is released. The real FCL geometry is covered in
// test_fcl_shared_geometry.cpp.

namespace
{
using collision_detection::FCLShapeCache;
using ObjectPtr = FCLShapeCache::CollisionObjectConstPtr;
using ObjectWeakPtr = std::weak_ptr<const fcl::CollisionObjectd>;

shapes::ShapeConstPtr makeShape()
{
  return std::make_shared<const shapes::Box>(0.1, 0.1, 0.1);
}

// What a build function returns. The cache never looks inside it, so any FCL object will do.
ObjectPtr makeObject()
{
  return std::make_shared<const fcl::CollisionObjectd>(std::make_shared<fcl::Sphered>(0.1));
}

// A build function that counts its calls in build_count.
FCLShapeCache::BuildFunction countingBuild(int& build_count)
{
  return [&build_count](const shapes::Shape& /*shape*/) {
    ++build_count;
    return makeObject();
  };
}

// A build function whose first call blocks until release() is called, so a test can keep one build "in progress"
// while other threads use the cache. Later calls return straight away.
class BlockingFirstBuild
{
public:
  FCLShapeCache::BuildFunction function()
  {
    return [this](const shapes::Shape& /*shape*/) {
      if (++count_ == 1)
      {
        started_.set_value();
        released_.wait();
      }
      return makeObject();
    };
  }

  // False if the first build has not started within 5 s.
  bool waitUntilStarted()
  {
    return started_.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  }

  void release()
  {
    release_.set_value();
  }

  int count() const
  {
    return count_;
  }

private:
  std::atomic<int> count_{ 0 };
  std::promise<void> started_;
  std::promise<void> release_;
  std::shared_future<void> released_{ release_.get_future().share() };
};
}  // namespace

// The point of the cache: asking for the same shape again returns the object built the first time.
TEST(FCLShapeCache, SameShapeBuildsOnce)
{
  int build_count = 0;
  FCLShapeCache cache(countingBuild(build_count));
  shapes::ShapeConstPtr shape = makeShape();

  ObjectPtr first = cache.get(shape);
  ObjectPtr second = cache.get(shape);

  EXPECT_EQ(build_count, 1);
  EXPECT_EQ(first, second);
}

// Two shapes are never confused with each other, even when they are identical boxes: the key is the shape pointer.
TEST(FCLShapeCache, DifferentShapesBuildSeparately)
{
  int build_count = 0;
  FCLShapeCache cache(countingBuild(build_count));
  shapes::ShapeConstPtr first_shape = makeShape();
  shapes::ShapeConstPtr second_shape = makeShape();

  ObjectPtr first = cache.get(first_shape);
  ObjectPtr second = cache.get(second_shape);

  EXPECT_EQ(build_count, 2);
  EXPECT_NE(first, second);
}

// Eight threads ask for the same new shape at once. One of them builds it and the other seven wait for that build
// instead of starting their own.
//
// Thread 0 goes first and its build is held open. The other seven are then started and given 100 ms to reach the
// cache before the build is released. If a slow machine starts one of them later than that, it simply gets an
// ordinary cache hit, so the test can lose strength but cannot fail by timing alone.
TEST(FCLShapeCache, ConcurrentFirstUseBuildsOnce)
{
  constexpr std::size_t THREAD_COUNT = 8;
  BlockingFirstBuild build;
  FCLShapeCache cache(build.function());
  shapes::ShapeConstPtr shape = makeShape();
  std::vector<ObjectPtr> objects(THREAD_COUNT);
  std::vector<std::thread> threads;

  threads.emplace_back([&] { objects[0] = cache.get(shape); });
  const bool build_started = build.waitUntilStarted();
  for (std::size_t i = 1; i < THREAD_COUNT; ++i)
    threads.emplace_back([&, i] { objects[i] = cache.get(shape); });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  build.release();
  for (std::thread& thread : threads)
    thread.join();

  EXPECT_TRUE(build_started);
  EXPECT_EQ(build.count(), 1);
  EXPECT_EQ(std::set<ObjectPtr>(objects.begin(), objects.end()).size(), 1u);
}

// A slow build (a big mesh, say) must not hold up other shapes: the cache lock is not held while building.
//
// One thread's build of slow_shape is held open. While it is stuck, another thread asks for fast_shape and has to get
// its answer back. If the lock were held during a build, that second thread would hang until the 5 s timeout.
TEST(FCLShapeCache, BuildDoesNotBlockOtherShapes)
{
  BlockingFirstBuild build;
  FCLShapeCache cache(build.function());
  shapes::ShapeConstPtr slow_shape = makeShape();
  shapes::ShapeConstPtr fast_shape = makeShape();

  std::thread slow_thread([&] { cache.get(slow_shape); });
  const bool slow_build_started = build.waitUntilStarted();
  std::future<ObjectPtr> fast = std::async(std::launch::async, [&] { return cache.get(fast_shape); });
  const std::future_status fast_status = fast.wait_for(std::chrono::seconds(5));
  build.release();
  slow_thread.join();

  EXPECT_TRUE(slow_build_started);
  EXPECT_EQ(fast_status, std::future_status::ready);
}

// A build that throws reaches the caller, and the failure is not remembered: the next call builds again.
TEST(FCLShapeCache, FailedBuildIsRethrownAndNotCached)
{
  int build_count = 0;
  FCLShapeCache cache([&](const shapes::Shape& /*shape*/) {
    if (++build_count == 1)
      throw std::runtime_error("build failed");
    return makeObject();
  });
  shapes::ShapeConstPtr shape = makeShape();

  EXPECT_THROW(cache.get(shape), std::runtime_error);
  EXPECT_NE(cache.get(shape), nullptr);
  EXPECT_EQ(build_count, 2);
}

// The cache keeps each object alive until its shape is gone and the cache is cleaned. Only weak pointers to the
// objects are held here, so "expired" means the cache has let go of it.
TEST(FCLShapeCache, CleanExpiredReleasesOnlyExpiredShapes)
{
  int build_count = 0;
  FCLShapeCache cache(countingBuild(build_count));
  shapes::ShapeConstPtr live_shape = makeShape();
  shapes::ShapeConstPtr expired_shape = makeShape();
  ObjectWeakPtr live_object = cache.get(live_shape);
  ObjectWeakPtr expired_object = cache.get(expired_shape);
  expired_shape.reset();
  ASSERT_FALSE(expired_object.expired());

  cache.cleanExpired();

  EXPECT_TRUE(expired_object.expired());
  EXPECT_FALSE(live_object.expired());
}

// Adding a new shape also cleans up, so the cache cannot grow without bound when nobody calls cleanExpired().
TEST(FCLShapeCache, InsertReleasesObjectOfExpiredShape)
{
  int build_count = 0;
  FCLShapeCache cache(countingBuild(build_count));
  shapes::ShapeConstPtr expired_shape = makeShape();
  ObjectWeakPtr expired_object = cache.get(expired_shape);
  expired_shape.reset();
  shapes::ShapeConstPtr new_shape = makeShape();

  cache.get(new_shape);

  EXPECT_TRUE(expired_object.expired());
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
