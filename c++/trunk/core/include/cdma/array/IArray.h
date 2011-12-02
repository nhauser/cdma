//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IARRAY_H__
#define __CDMA_IARRAY_H__

#include <string>
#include <vector>
#include <typeinfo>

#include "yat/memory/SharedPtr.h"
#include "yat/any/Any.h"

#include "cdma/IObject.h"
#include "cdma/exception/Exception.h"

namespace cdma
{

//==============================================================================
/// Array for multiple type. An Array has a <b>classType</b> which gives the
/// Class of its elements, and a <b>shape</b> which describes the number of
/// elements in each index. The <b>rank</b> is the number of indices. A
/// <b>scalar</b> Array has rank = 0. An Array may have arbitrary rank. The Array
/// <b>size</b> is the total number of elements, which must be less than 2^31
/// (about 2x10^9).
/// <p>
/// Actual data storage is done with Java 1D arrays and stride index
/// calculations. This makes our Arrays rectangular, i.e. no "ragged arrays"
/// where different elements can have different lengths as in Java
/// multidimensional arrays, which are arrays of arrays.
/// <p>
/// Each primitive Java type (bool, byte, char, short, int, long, float,
/// double) has a corresponding concrete implementation, e.g. ArrayBoolean,
/// ArrayDouble. Reference types are all implemented using the ArrayObject class,
/// with the exceptions of the reference types that correspond to the primitive
/// types, eg Double.class is mapped to double.class.
/// <p>
/// For efficiency, each Array type implementation has concrete subclasses for
/// ranks 0-7, eg ArrayDouble.D0 is a double array of rank 0, ArrayDouble.D1 is a
/// double array of rank 1, etc. These type and rank specific classes are
/// convenient to work with when you know the type and rank of the Array. Ranks
/// greater than 7 are handled by the type-specific superclass e.g. ArrayDouble.
/// The Array class itself is used for fully general handling of any type and
/// rank array. Use the Array.factory() methods to create Arrays in a general
/// way.
/// <p>
/// The stride index calculations allow <b>logical views</b> to be efficiently
/// implemented, eg subset, transpose, slice, etc. These views use the same data
/// storage as the original Array they are derived from. The index stride
/// calculations are equally efficient for any chain of logical views.
/// <p>
/// The type, shape and backing storage of an Array are immutable. The data
/// itself is read or written using an Index or an IndexIterator, which stores
/// any needed state information for efficient traversal. This makes use of
/// Arrays thread-safe (as long as you dont share the Index or IndexIterator)
/// except for the possibility of non-atomic read/write on long/doubles. If this
/// is the case, you should probably synchronize your calls. Presumably 64-bit
/// CPUs will make those operations atomic also.
//==============================================================================
class IArray : public IObject
{
public:
  // d-tor
  virtual ~IArray()
  {
  }

  /// Create a copy of this Array, copying the data so that physical order is
  /// the same as logical order.
  ///
  /// @return the new Array
  ///
  virtual yat::SharedPtr<IArray, yat::Mutex> copy() = 0;

  /// Create a copy of this Array, whether copying the data so that physical order is
  /// the same as logical order or sharing it so both IArray reference the same backing storage.
  ///
  /// @param data
  ///             if true the backing storage will be copied too else it will be shared
  /// @return the new Array
  ///
  virtual yat::SharedPtr<IArray, yat::Mutex> copy(bool data) = 0;

  /// Get an IArrayUtils that permits shape manipulations on arrays
  ///
  /// @return new IArrayUtils object
  ///
  virtual yat::SharedPtr<IArrayUtils, yat::Mutex> getArrayUtils() = 0;

  /// Get an IArrayMath that permits math calculations on arrays
  ///
  /// @return new IArrayMath object
  ///
  virtual yat::SharedPtr<IArrayMath, yat::Mutex> getArrayMath() = 0;

  /// Get the element class type of this Array.
  ///
  /// @return Class object
  ///
  virtual const std::type_info& getElementType() = 0;

  /// Get an Index object used for indexed access of this Array.
  ///
  /// @return IIndex object
  /// @see IIndex
  ///
  virtual IIndexPtr getIndex() = 0;

  /// Get Iterator to traverse the Array.
  ///
  /// @return ArrayIterator
  ///
  virtual yat::SharedPtr<IArrayIterator, yat::Mutex> getIterator() = 0;

  /// Get the number of dimensions of the array.
  ///
  /// @return number of dimensions of the array
  ///
  virtual int getRank() = 0;

  /// Get the iterator that only iterate a region of the Array. The region is
  /// described by the reference and range parameters.
  ///
  /// @param reference
  ///            java array of integer
  /// @param range
  ///            java array of integer
  /// @return ArrayIterator
  /// @throw  Exception
  ///             Created on 16/06/2008
  ///
  virtual yat::SharedPtr<IArrayIterator, yat::Mutex> getRegionIterator(std::vector<int> reference, std::vector<int> range) throw ( cdma::Exception ) = 0;

  /// Get the shape: length of array in each dimension.
  ///
  /// @return array whose length is the rank of this Array and whose elements
  ///         represent the length of each of its indices.
  ///
  virtual std::vector<int> getShape() = 0;

  /// Get the array element at the given index position
  ///
  /// @param ima
  ///            Index with current element set
  /// @return value at <code>index</code>
  ///
  virtual yat::Any get(IIndexPtr& index) = 0;
  
  /// Get the first array element
  ///
  /// @return value
  ///
  virtual yat::Any get() = 0;
  
  /// Get the total number of elements in the array.
  ///
  /// @return total number of elements in the array
  ///
  virtual long getSize() = 0;

  /// Get underlying primitive array storage. Exposed for efficiency, use at
  /// your own risk.
  ///
  /// @return any Object
  ///
  virtual void* getStorage() = 0;

  /// Set the array element at the current element of ima.
  ///
  /// @param ima
  ///            Index with current element set
  /// @param value
  ///            the new value; cast to underlying data type if necessary.
  ///
  virtual void set(const IIndexPtr& ima, const yat::Any& value) = 0;

  /// Convert the shape information to string type.
  ///
  /// @return string type Created on 20/03/2008
  ///
  virtual std::string shapeToString() = 0;

  /// Set the given index as current one for this array. Defines a viewable
  /// part of this array.
  ///
  /// @param index
  ///            of the viewable part
  ///
  virtual void setIndex(const IIndexPtr& index) = 0;

  /// Get the slice iterator with certain rank. The rank of the slice must be
  /// equal or smaller than the array itself. Otherwise throw
  /// Exception. <br>
  /// For example, for an array with the shape of [2x3x4x5]. If the rank of the
  /// slice is 1, there will be 2x3x4=24 slices. If the rank of slice is 2,
  /// there will be 2x3=6 slices. If the rank of the slice is 3, there will be
  /// 2 slices. if the rank of slice is 4, which is not recommended, there will
  /// be just 1 slices. If the rank of slice is 0, in which case it is pretty
  /// costly, there will be 120 slices.
  ///
  /// @param rank
  ///            an integer value, this will be the rank of the slice
  /// @return SliceIterator object
  /// @throw  Exception
  ///             mismatching shape
  /// @throw  Exception
  ///             invalid range Created on 23/07/2008
  ///
  virtual yat::SharedPtr<ISliceIterator, yat::Mutex> getSliceIterator(int rank) throw ( cdma::Exception ) = 0;

  /// Release the back storage of this Array. It will trigger backup routine,
  /// which saves the data into the file system that can be load back when this
  /// Array is accessed next time.
  ///
  /// Created on 04/03/2009
  ///
  /// @throw  Exception
  ///             failed to put in storage
  ///
  virtual void releaseStorage() throw ( Exception ) = 0;

  /// Get the register ID of the array.
  ///
  /// @return long value Created on 06/03/2009
  ///
  virtual long getRegisterId() = 0;

  /// Lock the array from loading data from backup storage. If the data is not
  /// backed up, this will not affecting reading out the data. Created on
  /// 05/03/2009
  ///
  virtual void lock() = 0;

  /// Release the lock of the array from loading data from backup storage.
  ///
  /// Created on 05/03/2009
  ///
  virtual void unlock() = 0;

  /// If the array has been changed since last read out from the backup
  /// storage.
  ///
  /// @return Created on 05/03/2009
  ///
  virtual bool isDirty() = 0;
};

} //namespace CDMACore
#endif //__CDMA_IARRAY_H__


