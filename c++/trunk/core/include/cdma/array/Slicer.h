// *****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 10/01/2012
// Author   : Rodriguez Clément
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// *****************************************************************************

#ifndef __CDMA_SLICER_H__
#define __CDMA_SLICER_H__

//==============================================================================
/// Slicer of an Array.
/// This is a way to have slices of an array.
//==============================================================================

#include <vector>
#include <cdma/exception/Exception.h>
#include <cdma/array/Array.h>
#include <cdma/array/ArrayIterator.h>

namespace cdma
{

class Slicer
{
private:
  ArrayPtr m_array; // Array from which slices are desired
  int      m_rank;  // Desired rank of the slices

public:
  Slicer(const ArrayPtr& array, int dim);
  ~Slicer();

  SliceIterator begin();
  SliceIterator end();
  const ArrayPtr& array();

  /// Get the shape of any slice that is returned. This could be used when a
  /// temporary array of the right shape needs to be created.
  ///
  /// @return dimensions of a single
  ///
  std::vector<int> getSliceShape();

 };
}
#endif // __CDMA_SLICER_H__
