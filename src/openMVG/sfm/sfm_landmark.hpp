// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_SFM_SFM_LANDMARK_HPP
#define OPENMVG_SFM_SFM_LANDMARK_HPP

#include "openMVG/types.hpp"
#include <cassert>
#include <cstddef>
#include <new>
#include <utility>
#include <type_traits>

template <typename T>
struct RawStorage {
  alignas(T) unsigned char data[sizeof(T)];
};

template <typename Key, typename Value, size_t InlineCapacity>
class SmallMap {
public:
  typedef std::pair<Key, Value> value_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef value_type* iterator;
  typedef const value_type* const_iterator;
  typedef size_t size_type;

  SmallMap()
    : size_(0),
    capacity_(InlineCapacity),
    data_(InlineData()) {
  }

  SmallMap(const SmallMap& other)
    : size_(0),
    capacity_(InlineCapacity),
    data_(InlineData()) {
    Reserve(other.size_);
    for (size_t i = 0; i < other.size_; ++i) {
      new (data_ + i) value_type(other.data_[i]);
    }
    size_ = other.size_;
  }

  SmallMap(SmallMap&& other) noexcept
    : size_(0),
    capacity_(InlineCapacity),
    data_(InlineData()) {
    MoveFrom(std::move(other));
  }

  SmallMap& operator=(const SmallMap& other) {
    if (this == &other) {
      return *this;
    }

    Clear();
    Reserve(other.size_);
    for (size_t i = 0; i < other.size_; ++i) {
      new (data_ + i) value_type(other.data_[i]);
    }
    size_ = other.size_;
    return *this;
  }

  SmallMap& operator=(SmallMap&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    DestroyAndFree();
    size_ = 0;
    capacity_ = InlineCapacity;
    data_ = InlineData();
    MoveFrom(std::move(other));
    return *this;
  }

  ~SmallMap() {
    DestroyAndFree();
  }

  size_t size() const {
    return size_;
  }

  bool empty() const {
    return size_ == 0;
  }

  iterator begin() {
    return data_;
  }

  iterator end() {
    return data_ + size_;
  }

  const_iterator begin() const {
    return data_;
  }

  const_iterator end() const {
    return data_ + size_;
  }

  const_iterator cbegin() const {
    return data_;
  }

  const_iterator cend() const {
    return data_ + size_;
  }

  void clear() {
    Clear();
  }

  void Clear() {
    for (size_t i = 0; i < size_; ++i) {
      data_[i].~value_type();
    }
    size_ = 0;
  }

  void reserve(size_t newCapacity) {
    Reserve(newCapacity);
  }

  iterator find(const Key& key) {
    for (size_t i = 0; i < size_; ++i) {
      if (data_[i].first == key) {
        return data_ + i;
      }
    }
    return end();
  }

  const_iterator find(const Key& key) const {
    for (size_t i = 0; i < size_; ++i) {
      if (data_[i].first == key) {
        return data_ + i;
      }
    }
    return end();
  }

  size_t count(const Key& key) const {
    return find(key) != end() ? 1 : 0;
  }

  Value& at(const Key& key) {
    iterator it = find(key);
    assert(it != end());
    return it->second;
  }

  const Value& at(const Key& key) const {
    const_iterator it = find(key);
    assert(it != end());
    return it->second;
  }

  Value& operator[](const Key& key) {
    iterator it = find(key);
    if (it != end()) {
      return it->second;
    }

    if (size_ == capacity_) {
      GrowFor(size_ + 1);
    }

    new (data_ + size_) value_type(key, Value());
    ++size_;
    return data_[size_ - 1].second;
  }

  std::pair<iterator, bool> insert(const value_type& value) {
    iterator it = find(value.first);
    if (it != end()) {
      return std::make_pair(it, false);
    }

    if (size_ == capacity_) {
      GrowFor(size_ + 1);
    }

    new (data_ + size_) value_type(value);
    ++size_;
    return std::make_pair(data_ + (size_ - 1), true);
  }

  std::pair<iterator, bool> insert(value_type&& value) {
    iterator it = find(value.first);
    if (it != end()) {
      return std::make_pair(it, false);
    }

    if (size_ == capacity_) {
      GrowFor(size_ + 1);
    }

    new (data_ + size_) value_type(std::move(value));
    ++size_;
    return std::make_pair(data_ + (size_ - 1), true);
  }

  template <typename PairLike>
  std::pair<iterator, bool> insert(PairLike&& value) {
    value_type tmp(std::forward<PairLike>(value));
    return insert(std::move(tmp));
  }

  /// Append without duplicate check. Caller guarantees key uniqueness.
  /// This is O(1) amortized vs O(n) for insert().
  void push_back_unchecked(const value_type& value) {
    if (size_ == capacity_) {
      GrowFor(size_ + 1);
    }
    new (data_ + size_) value_type(value);
    ++size_;
  }

  void push_back_unchecked(value_type&& value) {
    if (size_ == capacity_) {
      GrowFor(size_ + 1);
    }
    new (data_ + size_) value_type(std::move(value));
    ++size_;
  }

  iterator erase(iterator it) {
    assert(it >= begin() && it < end());
    size_t idx = static_cast<size_t>(it - begin());

    data_[idx].~value_type();
    for (size_t i = idx; i + 1 < size_; ++i) {
      new (data_ + i) value_type(std::move(data_[i + 1]));
      data_[i + 1].~value_type();
    }

    --size_;
    return data_ + idx;
  }

  size_t erase(const Key& key) {
    iterator it = find(key);
    if (it == end()) {
      return 0;
    }
    erase(it);
    return 1;
  }

  void swap(SmallMap& other) noexcept {
    if (this == &other) {
      return;
    }

    SmallMap tmp(std::move(other));
    other = std::move(*this);
    *this = std::move(tmp);
  }

private:
  size_t size_;
  size_t capacity_;
  value_type* data_;
  RawStorage<value_type> inlineStorage_[InlineCapacity > 0 ? InlineCapacity : 1];

  value_type* InlineData() {
    return reinterpret_cast<value_type*>(inlineStorage_);
  }

  const value_type* InlineData() const {
    return reinterpret_cast<const value_type*>(inlineStorage_);
  }

  bool UsingInline() const {
    return data_ == InlineData();
  }

  static value_type* Allocate(size_t count) {
    return static_cast<value_type*>(::operator new(sizeof(value_type) * count));
  }

  void Reserve(size_t newCapacity) {
    if (newCapacity <= capacity_) {
      return;
    }

    value_type* newData = Allocate(newCapacity);
    size_t i = 0;
    try {
      for (; i < size_; ++i) {
        new (newData + i) value_type(std::move(data_[i]));
      }
    }
    catch (...) {
      for (size_t j = 0; j < i; ++j) {
        newData[j].~value_type();
      }
      ::operator delete(newData);
      throw;
    }

    for (size_t j = 0; j < size_; ++j) {
      data_[j].~value_type();
    }

    if (!UsingInline()) {
      ::operator delete(data_);
    }

    data_ = newData;
    capacity_ = newCapacity;
  }

  void GrowFor(size_t minCapacity) {
    size_t newCapacity = capacity_ ? (capacity_ * 2) : InlineCapacity;
    if (newCapacity < minCapacity) {
      newCapacity = minCapacity;
    }
    if (newCapacity < InlineCapacity) {
      newCapacity = InlineCapacity;
    }
    Reserve(newCapacity);
  }

  void MoveFrom(SmallMap&& other) noexcept {
    if (!other.UsingInline()) {
      data_ = other.data_;
      size_ = other.size_;
      capacity_ = other.capacity_;

      other.data_ = other.InlineData();
      other.size_ = 0;
      other.capacity_ = InlineCapacity;
      return;
    }

    if (other.size_ > capacity_) {
      Reserve(other.size_);
    }

    for (size_t i = 0; i < other.size_; ++i) {
      new (data_ + i) value_type(std::move(other.data_[i]));
    }
    size_ = other.size_;
    other.Clear();
  }

  void DestroyAndFree() {
    for (size_t i = 0; i < size_; ++i) {
      data_[i].~value_type();
    }

    if (!UsingInline()) {
      ::operator delete(data_);
    }
  }
};

namespace openMVG {
namespace sfm {

/// Define 3D-2D tracking data: 3D landmark with its 2D observations
struct Observation
{
  Observation() : id_feat(UndefinedIndexT) {}
  Observation(const Vec2& p, IndexT idFeat) : x(p), id_feat(idFeat) {}

  Vec2 x;
  IndexT id_feat;

  template <class Archive>
  void save(Archive& ar) const;

  template <class Archive>
  void load(Archive& ar);
};

/// Observations are indexed by their View_id
using Observations = SmallMap<IndexT, Observation, 8>;

inline Observations::const_iterator FindObservation(
  const Observations& obs,
  IndexT viewId)
{
  for (Observations::const_iterator it = obs.begin(); it != obs.end(); ++it)
  {
    if (it->first == viewId)
    {
      return it;
    }
  }
  return obs.end();
}

/// Define a landmark (a 3D point, with its 2d observations)
struct Landmark
{
  Vec3 X;
  Observations obs;

  // Serialization
  template <class Archive>
  void save( Archive & ar) const;

  template <class Archive>
  void load( Archive & ar);
};

/// Define a collection of landmarks are indexed by their TrackId
using Landmarks = Hash_Map<IndexT, Landmark>;

} // namespace sfm
} // namespace openMVG

#endif // OPENMVG_SFM_SFM_LANDMARK_HPP
