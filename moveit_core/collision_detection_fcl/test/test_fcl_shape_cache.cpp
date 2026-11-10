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

namespace
{
using collision_detection::FCLShapeCache;

shapes::ShapeConstPtr makeShape()
{
  return shapes::ShapeConstPtr(new shapes::Box(0.1, 0.1, 0.1));
}

FCLShapeCache::CollisionObjectConstPtr makeCollisionObject()
{
  return std::make_shared<const fcl::CollisionObjectd>(std::make_shared<fcl::Sphered>(0.1));
}

class CountingBuild
{
public:
  FCLShapeCache::BuildFunction function()
  {
    return [this](const shapes::Shape& /*shape*/) {
      ++count_;
      return makeCollisionObject();
    };
  }

  int count() const
  {
    return count_;
  }

private:
  std::atomic<int> count_{ 0 };
};
}  // namespace

/** \brief Gets the same shape twice.
 *  Checks the build function runs once and both calls return the same object. */
TEST(FCLShapeCache, SameShapeBuildsOnce)
{
  CountingBuild build;
  FCLShapeCache cache(build.function());
  shapes::ShapeConstPtr shape = makeShape();

  FCLShapeCache::CollisionObjectConstPtr first = cache.get(shape);
  FCLShapeCache::CollisionObjectConstPtr second = cache.get(shape);

  EXPECT_EQ(build.count(), 1);
  EXPECT_EQ(first, second);
}

/** \brief Gets two different shapes.
 *  Checks each shape is built separately and gets its own object. */
TEST(FCLShapeCache, DifferentShapesBuildSeparately)
{
  CountingBuild build;
  FCLShapeCache cache(build.function());
  shapes::ShapeConstPtr first_shape = makeShape();
  shapes::ShapeConstPtr second_shape = makeShape();

  FCLShapeCache::CollisionObjectConstPtr first = cache.get(first_shape);
  FCLShapeCache::CollisionObjectConstPtr second = cache.get(second_shape);

  EXPECT_EQ(build.count(), 2);
  EXPECT_NE(first, second);
}

/** \brief Gets a shape with a build function that records what it was given.
 *  Checks the build function is called with exactly the requested shape. */
TEST(FCLShapeCache, BuildReceivesRequestedShape)
{
  std::vector<const shapes::Shape*> built_shapes;
  FCLShapeCache cache([&](const shapes::Shape& shape) {
    built_shapes.push_back(&shape);
    return makeCollisionObject();
  });
  shapes::ShapeConstPtr shape = makeShape();

  cache.get(shape);

  EXPECT_EQ(built_shapes, std::vector<const shapes::Shape*>{ shape.get() });
}

/** \brief Starts one thread getting an uncached shape and holds its build open, starts 7 more threads getting the
 *  same shape, waits 100 ms so they reach the cache while the build is still open, then lets the build finish.
 *  Checks the build started, only one build ran and all 8 threads received the same object. The overlap relies on the
 *  100 ms wait: a thread that starts later only exercises a cache hit. */
TEST(FCLShapeCache, ConcurrentFirstUseBuildsOnce)
{
  constexpr std::size_t THREAD_COUNT = 8;
  std::atomic<int> build_count{ 0 };
  std::promise<void> build_started;
  std::promise<void> release_build;
  std::shared_future<void> release = release_build.get_future().share();
  FCLShapeCache cache([&](const shapes::Shape& /*shape*/) {
    if (++build_count == 1)
      build_started.set_value();
    release.wait();
    return makeCollisionObject();
  });
  shapes::ShapeConstPtr shape = makeShape();

  std::vector<FCLShapeCache::CollisionObjectConstPtr> objects(THREAD_COUNT);
  std::vector<std::thread> threads;
  threads.emplace_back([&] { objects[0] = cache.get(shape); });
  const std::future_status first_build_status = build_started.get_future().wait_for(std::chrono::seconds(5));
  for (std::size_t i = 1; i < THREAD_COUNT; ++i)
    threads.emplace_back([&, i] { objects[i] = cache.get(shape); });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  release_build.set_value();
  for (std::thread& thread : threads)
    thread.join();

  EXPECT_EQ(first_build_status, std::future_status::ready);
  EXPECT_EQ(build_count, 1);
  EXPECT_EQ(std::set<FCLShapeCache::CollisionObjectConstPtr>(objects.begin(), objects.end()).size(), 1u);
}

/** \brief Holds the build of one shape open, then gets a second shape from another thread.
 *  Checks the first build started and the second shape is returned within 5 s, i.e. a build in progress does not hold
 *  the cache lock. */
TEST(FCLShapeCache, BuildDoesNotBlockOtherShapes)
{
  shapes::ShapeConstPtr slow_shape = makeShape();
  shapes::ShapeConstPtr fast_shape = makeShape();
  std::promise<void> slow_build_started;
  std::promise<void> release_slow_build;
  std::shared_future<void> release = release_slow_build.get_future().share();
  FCLShapeCache cache([&](const shapes::Shape& shape) {
    if (&shape == slow_shape.get())
    {
      slow_build_started.set_value();
      release.wait();
    }
    return makeCollisionObject();
  });

  std::thread slow_thread([&] { cache.get(slow_shape); });
  const std::future_status slow_build_status = slow_build_started.get_future().wait_for(std::chrono::seconds(5));
  std::future<FCLShapeCache::CollisionObjectConstPtr> fast =
      std::async(std::launch::async, [&] { return cache.get(fast_shape); });
  std::future_status fast_status = fast.wait_for(std::chrono::seconds(5));
  release_slow_build.set_value();
  slow_thread.join();

  EXPECT_EQ(slow_build_status, std::future_status::ready);
  EXPECT_EQ(fast_status, std::future_status::ready);
}

/** \brief Gets a shape whose first build throws, then gets it again.
 *  Checks the first call rethrows the build error and the second call rebuilds and returns an object. */
TEST(FCLShapeCache, FailedBuildIsRethrownAndNotCached)
{
  int build_count = 0;
  FCLShapeCache cache([&](const shapes::Shape& /*shape*/) {
    if (++build_count == 1)
      throw std::runtime_error("build failed");
    return makeCollisionObject();
  });
  shapes::ShapeConstPtr shape = makeShape();

  EXPECT_THROW(cache.get(shape), std::runtime_error);
  EXPECT_NE(cache.get(shape), nullptr);
  EXPECT_EQ(build_count, 2);
}

/** \brief Gets a shape twice with a build function that returns null, as for an unsupported shape type.
 *  Checks both calls return null and the build runs only once. */
TEST(FCLShapeCache, NullBuildResultIsCached)
{
  int build_count = 0;
  FCLShapeCache cache([&](const shapes::Shape& /*shape*/) {
    ++build_count;
    return FCLShapeCache::CollisionObjectConstPtr();
  });
  shapes::ShapeConstPtr shape = makeShape();

  EXPECT_EQ(cache.get(shape), nullptr);
  EXPECT_EQ(cache.get(shape), nullptr);
  EXPECT_EQ(build_count, 1);
}

/** \brief Gets a shape, destroys the shape, then cleans the cache.
 *  Checks the cache still holds the object until the clean, and releases it after. */
TEST(FCLShapeCache, CleanExpiredReleasesObjectOfExpiredShape)
{
  CountingBuild build;
  FCLShapeCache cache(build.function());
  shapes::ShapeConstPtr shape = makeShape();
  std::weak_ptr<const fcl::CollisionObjectd> object = cache.get(shape);
  shape.reset();
  ASSERT_FALSE(object.expired());

  cache.cleanExpired();

  EXPECT_TRUE(object.expired());
}

/** \brief Gets a shape, keeps the shape alive, then cleans the cache.
 *  Checks the clean does not release the object of a shape that still exists. */
TEST(FCLShapeCache, CleanExpiredKeepsObjectOfLiveShape)
{
  CountingBuild build;
  FCLShapeCache cache(build.function());
  shapes::ShapeConstPtr shape = makeShape();
  std::weak_ptr<const fcl::CollisionObjectd> object = cache.get(shape);

  cache.cleanExpired();

  EXPECT_FALSE(object.expired());
}

/** \brief Gets a shape, destroys it, then gets a new shape.
 *  Checks adding the new shape also releases the object of the destroyed one. */
TEST(FCLShapeCache, InsertReleasesObjectOfExpiredShape)
{
  CountingBuild build;
  FCLShapeCache cache(build.function());
  shapes::ShapeConstPtr expired_shape = makeShape();
  std::weak_ptr<const fcl::CollisionObjectd> object = cache.get(expired_shape);
  expired_shape.reset();
  shapes::ShapeConstPtr new_shape = makeShape();

  cache.get(new_shape);

  EXPECT_TRUE(object.expired());
}

/** \brief Gets a shape, destroys it, then creates a new shape at the same address and gets it. Skipped when the
 *  allocator does not reuse the address, e.g. under AddressSanitizer.
 *  Checks the new shape is built again and gets a different object, i.e. a new shape is never mistaken for an expired
 *  one at the same address. */
TEST(FCLShapeCache, NewShapeAtExpiredShapeAddressBuildsAgain)
{
  CountingBuild build;
  FCLShapeCache cache(build.function());
  shapes::ShapeConstPtr shape = makeShape();
  const shapes::Shape* expired_address = shape.get();
  FCLShapeCache::CollisionObjectConstPtr first = cache.get(shape);
  shape.reset();
  shape = makeShape();
  if (shape.get() != expired_address)
    GTEST_SKIP() << "The allocator did not reuse the expired shape's address";

  FCLShapeCache::CollisionObjectConstPtr second = cache.get(shape);

  EXPECT_EQ(build.count(), 2);
  EXPECT_NE(first, second);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
