//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IINDEX_H__
#define __CDMA_IINDEX_H__

#include <string>
#include <vector>

#include <cdma/exception/Exception.h>
#include <cdma/IObject.h>

namespace cdma
{

//==============================================================================
/// Indexes for Multidimensional Arrays. An Index refers to a particular
/// element of an Array.
//==============================================================================
class IIndex : public IObject
{
public:
  /// d-tor
  virtual ~IIndex()
  {
  }

  /// Get the number of dimensions in the array.
  ///
  /// @return integer value
  ///
  virtual int getRank() = 0;

  /// Get the shape: length of array in each dimension.
  ///
  /// @return array of integer
  ///
  virtual std::vector<int> getShape() = 0;

  /// Get the origin: first index of array in each dimension.
  ///
  /// @return array of integer
  ///
  virtual std::vector<int> getOrigin() = 0;

  /// Get the total number of elements in the array.
  ///
  /// @return long value
  ///
  virtual long getSize() = 0;

  /// Get the stride: for each dimension number elements to jump in the array between two
  /// consecutive element of the same dimension
  ///
  /// @return array of integer representing the stride
  ///
  virtual std::vector<int> getStride() = 0;

  /// Get the current element's index into the 1D backing array.
  ///
  /// @return integer value
  ///
  virtual int currentElement() = 0;

  /// Get the last element's index into the 1D backing array.
  ///
  /// @return integer value
  ///
  virtual int lastElement() = 0;

  /// Set the current element's index. General-rank case.
  ///
  /// @param index array of integer
  /// @return this, so you can use A.get(i.set(i))
  ///
  virtual void set(std::vector<int> index) = 0;

  /// set current element at dimension dim to v.
  ///
  /// @param dim   integer value
  /// @param value integer value
  ///
  virtual void setDim(int dim, int value) = 0;

  /// set the origin on each dimension for this index
  ///
  /// @param origin array of integers
  ///
  virtual void setOrigin(std::vector<int> origin) = 0;

  /// set the given shape for this index
  ///
  /// @param value array of integers
  ///
  virtual void setShape(std::vector<int> shape) = 0;

  /// set the stride for this index. The stride is the number of
  /// cells between two consecutive cells in the same dimension.
  ///
  /// @param stride array of integers
  ///
  virtual void setStride(std::vector<int> stride) = 0;

  /// Return the current location of the index.
  ///
  /// @return java array of integer Created on 18/06/2008
  ///
  virtual std::vector<int> getCurrentCounter() = 0;

  /// Set the name of one of the indices.
  ///
  /// @param dim       which index
  /// @param indexName name of index
  ///
  virtual void setIndexName(int dim, const std::string& indexName) = 0;

  /// Get the name of one of the indices.
  ///
  /// @param dim which index
  /// @return name of index, or null if none.
  ///
  virtual std::string getIndexName(int dim) = 0;

  /// Remove all index with length one.
  /// @return the new IIndex
  ///
  virtual IIndexPtr reduce() = 0;

  /// Eliminate the specified dimension.
  ///
  /// @param dim dimension to eliminate: must be of length one, else
  ///            IllegalArgumentException
  /// @return the new index
  ///
  virtual IIndexPtr reduce(int dim) throw ( Exception ) = 0;
};

} //namespace CDMACore
#endif //__CDMA_IINDEX_H__

