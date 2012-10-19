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

#ifndef __CDMA_IARRAY_H__
#define __CDMA_IARRAY_H__

#include <string>
#include <vector>

#include <cdma/exception/Exception.h>
#include <cdma/array/IArrayStorage.h>
#include <cdma/array/IView.h>

/// @cond clientAPI

namespace cdma
{

// Forward declaration
DECLARE_CLASS_SHARED_PTR(IArray);

#if !defined(CDMA_NO_ITERATORS)
  DECLARE_CLASS_SHARED_PTR(Slicer);
  DECLARE_CLASS_SHARED_PTR(ArrayIterator);
#endif

//==============================================================================
/// @brief Array for multiple types of data.
///
/// An Array has:
///   - a <b>type info</b> which gives the type of its elements, 
///   - a <b>shape</b> which describes the number of elements in each dimension
///   - a <b>size</b> that is the total number of elements
///   - a <b>rank</b> that is the number of dimension
/// A <b>scalar</b> Array has a rank = 0, an Array may have arbitrary rank. 
/// <p>
/// Default implementation of the data storage (IArrayStorage) is done using
/// DefaultArrayStorage that can be redefined by the plugin. This implies
/// that our Arrays are rectangular, i.e. no "ragged arrays" (where different 
/// elements can have different lengths). Array can manage any type of data
/// float, double, char...
/// <p>
/// The Array associates a View to a memory storage. This latest makes index 
/// calculations according to the given view. The stride calculations allows
/// <b>logical views</b> to be efficiently implemented, eg subset, transpose, 
/// slice, etc. These views use the same data storage as the original Array
/// they are derived from. 
/// <p>
/// The type, shape and backing storage of an Array are immutable. The data
/// itself is read or written using an iterators or indexed positions.
/// Array stores any needed state information for efficient traversal.
/// <p>
/// This interface cannot be derivated. It is intended to hide the
/// concrete implementation when we want to use only pure interface.
/// In such a case the switches NO_TEMPLATES and CDMA_NO_ITERATORS should be set
/// to prevent the inclusion of the IArray implementation definition from
/// a client code point of view
//==============================================================================
class CDMA_DECL IArray
{
//friend class ArrayIterator;

public:

  /// d-tor
  virtual ~IArray() {}

  /// Create a copy of this Array, copying the data so that physical order is
  /// the same as logical order. If the view of this array only rely a part of the
  /// whole storage, only the relevant part of the storage will be copied.
  ///
  /// @return the new Array
  /// @note be aware: can lead to out of memory 
  ///
  virtual IArrayPtr deepCopy() = 0;

  /// Get pointer to the "value" from the memory buffer according the position in the given view.
  ///
  /// @param view_ptr Shared pointer on the view to consider for the index calculation
  /// @param position into which the value will be set
  /// @return anonymous pointer to the value
  ///
  virtual void* getValue( const IViewPtr& view, std::vector<int> position ) = 0;
  
  /// Set "value" in the memory buffer according the position in the given view. The 
  /// given yat::Any will be casted into memory buffer type.
  ///
  /// @param view_ptr Shared pointer on the view to consider for the index calculation
  /// @param position into which the value will be set
  /// @param value_ptr C-style pointer to memory position to be set
  ///
  virtual void setValue(const cdma::IViewPtr& view_ptr, std::vector<int> position, void *value_ptr) = 0;
  
  /// Get the element type of this Array.
  ///
  /// @return type info
  ///
  virtual const std::type_info& getValueType() = 0;
  
  /// Get the View that describes this Array.
  ///
  /// @return Shared pointer on View object
  ///
  virtual IViewPtr getView() = 0;
  
  /// Set the View that describes this Array.
  ///
  /// @param view new View for this object
  ///
  virtual void setView(const IViewPtr& view) = 0;
  
  /// Get an iterator to traverse the Array.
  ///
  /// @return ArrayIterator
  ///
  virtual ArrayIterator begin() = 0;
  
  /// Get an iterator positioned at the end of the Array.
  ///
  /// @return ArrayIterator
  ///
  virtual ArrayIterator end() = 0;
  
  /// Returns the number of dimensions of the array.
  ///
  virtual int getRank() = 0;
  
  /// Returns a sub-part of the array defined by the given start and shape
  /// vector.
  ///
  /// @param start vector defining origin of the region in each dimension
  /// @param shape vector defining shape of the region in each dimension
  /// 
  /// @return IArrayPtr corresponding to a portion sharing same no memory
  ///
  virtual IArrayPtr getRegion(std::vector<int> start, std::vector<int> shape) throw ( cdma::Exception ) = 0;
  
  /// Get the shape: length of array in each dimension.
  ///
  /// @return array whose length is the rank of this Array and whose elements
  ///         represent the length of each of its indices.
  ///
  virtual std::vector<int> getShape() = 0;
  
  /// Return the total number of elements in the array
  ///
  virtual long getSize() = 0;
  
  /// Retruns true if the array has been changed since last read.
  ///
  virtual bool dirty() = 0;
  
  /// Get underlying array storage. Exposed for efficiency, use at your own risk.
  ///
  /// @return IArrayStorage object
  ///
  virtual const IArrayStoragePtr& getStorage() = 0;

#if !defined(CDMA_NO_TEMPLATES)

  /// Get the array element at the given index position
  ///
  /// @tparam T Data type
  /// @param view describing the array with current element set
  /// @param position vector targeting an element of the array
  ///
  /// @return value converted to the type T
  ///
  template<typename T>
  T getValue( const IViewPtr& view, std::vector<int> position );

  /// Get the array element at the given index position in the current (default) view
  ///
  /// @tparam T Data type
  /// @param position vector targeting an element of the array
  ///
  /// @return value converted to the type T
  ///
  template<typename T>
  T getValue( std::vector<int> position );

  /// Get the array element in the current (default) view
  /// Use this method in the case of a scalar value
  ///
  /// @tparam T Data type
  /// @return value converted to the type T
  ///
  template<typename T>
  T getValue( void );

  /// Set the array element at the current element given position
  ///
  /// @tparam T Data type
  /// @param view_ptr Shared pointer on the view describing the array with current element set
  /// @param position Element position where to set the value
  /// @param value the new value; cast to underlying data type if necessary.
  /// @todo  move this method in the private section
  ///
  template<typename T>
  void setValue(const IViewPtr& view_ptr, std::vector<int> position, T value);
  
  /// Set the array element at the current element given position in the current (default) view
  ///
  /// @tparam T Data type
  /// @param position Element position where to set the value
  /// @param value the new value; cast to underlying data type if necessary.
  ///
  template<typename T>
  void setValue(std::vector<int> position, T value);
  
  /// Set the array element at the current element in the current (default) view
  /// Use this method in the case of a scalar value
  ///
  /// @tparam T Data type
  /// @param value the new value; cast to underlying data type if necessary.
  ///
  template<typename T>
  void setValue(T value);

#if !defined(CDMA_NO_ITERATORS)
  /// Set the array element at the given iterator position
  ///
  /// @tparam T Data type
  /// @param it ArrayIterator containing the position of the element to affect
  /// @param value the new value; cast to underlying data type if necessary.
  ///
  template<typename T> 
  void setValue(const ArrayIterator& it, T value);
  
  /// Get the slicer of this array defined with given rank. The rank of the slicer must be
  /// equal or smaller than the array itself. Otherwise throw Exception.
  ///
  /// For example, for an array with the shape of [2x3x4x5]. If the rank of the
  /// slice is 1, there will be 2x3x4=24 slices. If the rank of slice is 2,
  /// there will be 2x3=6 slices. If the rank of the slice is 3, there will be
  /// 2 slices. if the rank of slice is 4, which is not recommended, there will
  /// be just 1 slices. If the rank of slice is 0, in which case it is pretty
  /// costly, there will be 120 slices.
  ///
  /// @param rank an integer value, this will be the rank of the slice
  /// @return Shared pointer on Slicer object
  ///
  virtual SlicerPtr getSlicer(int rank) throw ( cdma::Exception) = 0;
  
#endif // !CDMA_NO_ITERATORS

#endif // !CDMA_NO_TEMPLATES

private:
  IArray() {}

/// @cond internal

public:
  friend class Array;
  
/// @endcond
};

}

/// @endcond

#if !defined(NO_TEMPLATES)
  #include <cdma/array/impl/IArray.hpp>
#endif

#endif // __CDMA_IARRAY_H__
