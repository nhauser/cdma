//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_ISLICEITERATOR_H__
#define __CDMA_ISLICEITERATOR_H__

#include <vector>

#include <yat/memory/SharedPtr.h>
#include <yat/threading/Mutex.h>

#include <cdma/IObject.h>
#include <cdma/exception/Exception.h>

namespace cdma
{

//==============================================================================
/// Interface ISliceIterator
/// Iterator for slicing an Array.
/// This is a way to iterate over slices of arrays, and should eventually be
/// incorporated into the CDMA rather than lying around here. Each
/// iteration returns an array of dimension dim, representing the last dim
/// dimensions of the input array. So for 3D data consisting of a set of 2D
/// arrays, each of the 2D arrays will be returned.
//==============================================================================
class ISliceIterator : public IObject
{
public:
	/// d-tor
	virtual ~ISliceIterator()
  {
  }
	
  /// Check if there is next slice.
  ///
  /// @return Boolean type Created on 10/11/2008
  ///
	virtual bool hasNext() = 0;
	
  /// Jump to the next slice.
  ///
  /// Created on 10/11/2008
  ///
	virtual void next() = 0;
	
  /// Get the next slice of Array.
  ///
  /// @return GDM Array
  /// @throw  Exception
  ///             Created on 10/11/2008
  ///
  virtual IArrayPtr getArrayNext() throw ( Exception ) = 0;
	
  /// Get the current slice of Array.
  ///
  /// @return GDM Array
  /// @throw  Exception
  ///             Created on 10/11/2008
  ///
  virtual IArrayPtr getArrayCurrent() throw ( Exception ) = 0;
  
  /// Get the shape of any slice that is returned. This could be used when a
  /// temporary array of the right shape needs to be created.
  ///
  /// @return dimensions of a single slice from the iterator
  /// @throw  Exception invalid range
  ///
	virtual std::vector<int> getSliceShape() throw ( Exception ) = 0;
	
  /// Get the slice position in the whole array from which this slice iterator
  /// was created.
  /// @return <code>int</code> array of the current position of the slice
  /// @note rank of the returned position is the same as the IArray shape we are slicing
  ///
	virtual std::vector<int> getSlicePosition() = 0;
 };
} //namespace CDMACore
#endif //__CDMA_ISLICEITERATOR_H__

