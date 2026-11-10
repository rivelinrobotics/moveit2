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

namespace collision_detection
{
FCLShapeCache::FCLShapeCache(BuildFunction build) : build_(std::move(build))
{
}

FCLShapeCache::CollisionObjectConstPtr FCLShapeCache::get(const shapes::ShapeConstPtr& shape)
{
  const ShapeKey key(shape);
  std::promise<CollisionObjectConstPtr> promise;
  std::shared_future<CollisionObjectConstPtr> future;
  bool build = false;
  {
    std::scoped_lock lock(mutex_);
    auto [it, inserted] = map_.try_emplace(key);
    if (inserted)
    {
      it->second = promise.get_future().share();
      build = true;
      cleanExpiredLocked();
    }
    future = it->second;
  }
  if (build)
  {
    try
    {
      promise.set_value(build_(*shape));
    }
    catch (...)
    {
      promise.set_exception(std::current_exception());
      std::scoped_lock lock(mutex_);
      map_.erase(key);
    }
  }
  return future.get();
}

void FCLShapeCache::cleanExpired()
{
  std::scoped_lock lock(mutex_);
  cleanExpiredLocked();
}

void FCLShapeCache::cleanExpiredLocked()
{
  // clean-up for cache (we don't want to keep infinitely large number of weak ptrs stored)
  for (auto it = map_.begin(); it != map_.end();)
  {
    if (it->first.expired())
      it = map_.erase(it);
    else
      ++it;
  }
}
}  // namespace collision_detection
