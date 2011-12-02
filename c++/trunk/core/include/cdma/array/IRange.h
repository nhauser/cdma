// ******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
// ******************************************************************************
#ifndef __CDMA_IRANGE_H__
#define __CDMA_IRANGE_H__

#include <yat/memory/SharedPtr.h>
#include <yat/threading/Mutex.h>

#include <cdma/exception/Exception.h>
#include <cdma/IObject.h>

namespace cdma
{

//==============================================================================
/// Interface IRange
/// Represents a set of integers. It can be used as an index for Arrays applied
/// to one particular dimension.
//==============================================================================
class IRange : public IObject
{
public:
  /// d-tor
  virtual ~IRange()
  {
  }

  /// Create a new Range by composing a Range that is relative to this Range.
  ///
  /// @param r range relative to base
  /// @return combined Range, may be EMPTY
  ///
  virtual IRangePtr compose(const IRange& r) throw ( Exception ) = 0;

  /// Create a new Range by compacting this Range by removing the stride. first
  /// = first/stride, last=last/stride, stride=1.
  ///
  /// @return compacted Range
  ///
  virtual IRangePtr compact() throw ( Exception ) = 0;

  /// Create a new Range shifting this range by a constant factor.
  ///
  /// @param origin subtract this from first, last
  /// @return shift range
  ///
  virtual IRangePtr shiftOrigin(int origin) throw ( Exception ) = 0;

  /// Create a new Range by intersecting with a Range using same interval as
  /// this Range. NOTE: intersections when both Ranges have strides are not
  /// supported.
  ///
  /// @param r range to intersect
  /// @return intersected Range, may be EMPTY
  ///
  virtual IRangePtr intersect(const IRange& r) throw ( Exception ) = 0;

  /// Determine if a given Range intersects this one. NOTE: we dont yet support
  /// intersection when both Ranges have strides
  ///
  /// @param r range to intersect
  /// @return true if they intersect
  ///
  virtual bool intersects(const IRange& r) = 0;

  /// Create a new Range by making the union with a Range using same interval
  /// as this Range. NOTE: no strides.
  ///
  /// @param r range to add
  /// @return intersected Range, may be EMPTY
  ///
  virtual IRangePtr unionRanges(const IRange& r) throw ( Exception ) = 0;

  /// Get the number of elements in the range.
  ///
  /// @return the number of elements in the range.
  ///
  virtual int length() const = 0;

  /// Get i-th element.
  ///
  /// @param i index of the element
  /// @return the i-th element of a range.
  ///
  virtual int element(int i) throw ( Exception ) = 0;

  /// Get the index for this element: inverse of element.
  ///
  /// @param elem the element of the range
  /// @return index
  ///
  virtual int index(int elem) throw ( Exception ) = 0;

  /// Is the ith element contained in this Range?
  ///
  /// @param i index in the original Range
  /// @return true if the ith element would be returned by the Range iterator
  ///
  virtual bool contains(int i) const = 0 ;

  /// @return first element's index in range
  ///
  virtual int first() const = 0;

  /// @return last element's index in range, inclusive
  ///
  virtual int last() const = 0;

  /// @return stride, must be >= 1
  ///
  virtual int stride() const = 0;

  /// Get name.
  ///
  /// @return name, or empty string
  ///
  virtual std::string getName() const = 0;

  /// Find the smallest element k in the Range, such that
  /// <ul>
  /// <li>k >= first
  /// <li>k >= start
  /// <li>k <= last
  /// <li>k = first + i * stride for some integer i.
  /// </ul>
  ///
  /// @param start starting index
  /// @return first in interval, else -1 if there is no such element.
  ///
  virtual int getFirstInInterval(int start) = 0;
};
} //namespace CDMACore
#endif //__CDMA_IRANGE_H__

