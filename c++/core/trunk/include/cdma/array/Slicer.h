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

#ifndef __CDMA_SLICER_H__
#define __CDMA_SLICER_H__

//==============================================================================
/// Slicer of an Array.
/// This item gives a way to have slices of an array. It provides iterators that
/// permit to iterate over slices of an array that are sub-parts of this array.
/// Slices are sub-part of the array considering the deepest dimension (moving
/// fastest dimension).
//==============================================================================

#include <vector>
#include <cdma/exception/Exception.h>
#include <cdma/array/IArray.h>
#include <cdma/array/ArrayIterator.h>
#include <cdma/array/SliceIterator.h>

/// @cond clientAPI

namespace cdma
{

//==============================================================================
/// @brief Return begin and end SliceIterator(s) on an array following a 
///        given dimension
///
/// @todo Write a long description here
//==============================================================================
class CDMA_DECL Slicer
{
public:
  /// c-tor
  ///
  /// @param array: the array we want to slice
  /// @param dim: rank of the slice
  ///
  Slicer(const IArrayPtr& array, int dim);

  /// d-tor
  ~Slicer();

  /// Returns the a SliceIterator positionned at the beginning of the array
  ///
  SliceIterator begin();
  
  /// Returns the a SliceIterator positionned at the end of the array
  ///
  SliceIterator end();
  
  /// Returns the whole array
  ///
  const IArrayPtr& array();

  /// Get the shape of any slice that is returned. This could be used when a
  /// temporary array of the right shape needs to be created.
  ///
  /// @return dimensions of a single
  ///
  std::vector<int> getSliceShape();

private:
  IArrayPtr m_array; // Array from which slices are desired
  int       m_rank;  // Desired rank of the slices
};

/// Declaration of shared pointer SlicerPtr
DECLARE_SHARED_PTR(Slicer);

}

/// @endcond

#endif // __CDMA_SLICER_H__
