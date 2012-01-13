// ******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
// ******************************************************************************

#ifndef __CDMA_SLICEITERATOR_H__
#define __CDMA_SLICEITERATOR_H__

//==============================================================================
/// Iterator for slicing an Array.
/// This is a way to iterate over slices of arrays, and should eventually be
/// incorporated into the Gumtree Data Model rather than lying around here. Each
/// iteration returns an array of dimension dim, representing the last dim
/// dimensions of the input array. So for 3D data consisting of a set of 2D
/// arrays, each of the 2D arrays will be returned.
//==============================================================================

#include <vector>
#include <cdma/exception/Exception.h>
#include <cdma/array/Array.h>
#include <cdma/array/ArrayIterator.h>

namespace cdma
{

//==============================================================================
/// SliceIterator default implementation
//==============================================================================
class SliceIterator
{
private:
  ArrayPtr m_array;
  ArrayIteratorPtr m_iterator;
  int m_dimension;
  ArrayPtr m_slice;

public:
  SliceIterator(const SliceIterator& iterator);
  SliceIterator(const ArrayPtr& array, int dim);
  ~SliceIterator();

  SliceIterator& operator++(void); // prefix operator
  SliceIterator& operator++(int);  // suffix operator
  ArrayPtr& operator*(void);
  bool operator==(const SliceIteratorPtr& it);
  bool operator!=(const SliceIteratorPtr& it);

  /// Jump to the next slice.
  ///
  void next();
/*
  /// Get the next slice of Array.
  ///
  /// @return IArray
  ///
  /// @throw cdma::Exception, yat::Exception
  ///
  ArrayPtr getArrayNext() throw ( Exception );

  /// Get the current slice of Array.
  ///
  /// @return IArray
  ///
  /// @throw cdma::Exception, yat::Exception
  ///
  ArrayPtr getArrayCurrent() throw ( Exception );
*/
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
  std::vector<int> getPosition();

  //@{ IObject interface
  CDMAType::ModelType getModelType() const { return CDMAType::Other; };
  std::string getFactoryName() const { return m_array->getFactoryName(); };
  //@}
  
private:
  void get();
  
 };
}
#endif
