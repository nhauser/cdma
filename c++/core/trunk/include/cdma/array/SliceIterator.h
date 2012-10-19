//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
//
// This file is part of cdma-core library.
//
// The cdma-core library is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free 
// Software Foundation; either version 2 of the License, or (at your option) 
// any later version.
//
// The CDMA library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
//
// Contributors :
// See AUTHORS file 
//******************************************************************************

#ifndef __CDMA_SLICEITERATOR_H__
#define __CDMA_SLICEITERATOR_H__

#include <vector>
#include <cdma/exception/Exception.h>
#include <cdma/array/IArray.h>
#include <cdma/array/ArrayIterator.h>

/// @cond clientAPI

namespace cdma
{

// Forward declaration
DECLARE_CLASS_SHARED_PTR(SliceIterator);

//==============================================================================
/// @brief Iterator for slicing an Array
///
/// This is a way to iterate over slices of arrays.
/// Each iteration returns an array of dimension i-th, representing the last i-th
/// dimensions of the input array.
//==============================================================================
class CDMA_DECL SliceIterator
{
public:
  // Consrtuctor
  SliceIterator(const SliceIterator& iterator);
  SliceIterator(const IArrayPtr& array, int dim);
  
  // D-structor
  ~SliceIterator();

  /// Prefix operator: increments the ArrayIterator before
  /// returning the reference of the next indexed element
  ///
  SliceIterator& operator++(void);
  
  /// Suffix operator: increments the ArrayIterator after having 
  /// returned the reference of the current indexed element
  ///
  SliceIterator operator++(int);
  
  /// Access operator: returns a reference on the currently targeted
  /// slice.
  ///
  /// @return IArrayPtr reference of the current array slice
  ///
  IArrayPtr& operator*(void) const;
  
  /// Comparison operator: egality
  ///
  /// @param it SliceIterator to compare with this instance
  /// @return true if both iterator refers to the same position
  ///
  bool operator==(const SliceIterator& it);
  
  /// Comparison operator: difference
  ///
  /// @param it SliceIterator to compare with this instance
  /// @return true if both iterator refers to different position
  ///
  bool operator!=(const SliceIterator& it);

  /// Jump to the next slice.
  ///
  void next();

  /// Get the shape of any slice that is returned. This could be used when a
  /// temporary array of the right shape needs to be created.
  ///
  /// @return dimensions of a single slice from the iterator
  ///
  /// @throw cdma::Exception, yat::Exception
  ///
  std::vector<int> getSliceShape() throw ( Exception );

  /// Get the slice position in the whole array from which this slice iterator
  /// was created.
  /// @return <code>int</code> array of the current position of the slice
  /// @note rank of the returned position is the same as the IArray shape we are slicing
  ///
  std::vector<int> getPosition() const;
  
private:
  void get() const;

private:
  IArrayPtr         m_array;      // Whole array to slice
  ArrayIteratorPtr  m_iterator;   // Current position of the slice
  int               m_dimension;  // Dimension of the slice
  mutable IArrayPtr m_slice;      // Reference of the last read slice

};

}

/// @endcond

#endif
