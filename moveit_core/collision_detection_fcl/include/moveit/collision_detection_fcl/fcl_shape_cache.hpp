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

#pragma once

#include <moveit/collision_detection_fcl/fcl_compat.hpp>
#include <geometric_shapes/shapes.h>

#if (MOVEIT_FCL_VERSION >= FCL_VERSION_CHECK(0, 6, 0))
#include <fcl/narrowphase/collision_object.h>
#else
#include <fcl/collision_object.h>
#endif

#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>

namespace collision_detection
{
class FCLShapeCache
{
public:
  using CollisionObjectConstPtr = std::shared_ptr<const fcl::CollisionObjectd>;
  using BuildFunction = std::function<CollisionObjectConstPtr(const shapes::Shape&)>;

  explicit FCLShapeCache(BuildFunction build);

  CollisionObjectConstPtr get(const shapes::ShapeConstPtr& shape);

  void cleanExpired();

private:
  using ShapeKey = std::weak_ptr<const shapes::Shape>;

  void cleanExpiredLocked();

  BuildFunction build_;
  std::mutex mutex_;
  std::map<ShapeKey, std::shared_future<CollisionObjectConstPtr>, std::owner_less<ShapeKey>> map_;
};
}  // namespace collision_detection
