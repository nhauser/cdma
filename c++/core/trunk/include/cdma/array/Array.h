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

#ifndef __CDMA_ARRAY_H__
#define __CDMA_ARRAY_H__

#include <string>
#include <vector>

#include <cdma/exception/Exception.h>
#include <cdma/array/impl/ArrayStorage.h>
#include <cdma/array/View.h>

/// @cond clientAPI

namespace cdma
{

// Forward declaration
DECLARE_CLASS_SHARED_PTR(Array);
DECLARE_CLASS_SHARED_PTR(ArrayIterator);
DECLARE_CLASS_SHARED_PTR(Slicer);

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
//==============================================================================
class CDMA_DECL Array
{
friend class ArrayIterator;

public:
  /// Copy constructor
  Array( const Array& array );

  /// c-tor
  /// @param data_ptr Shared pointer on a particular storage implementation
  /// @param view_ptr Shared pointer on a specific view
  ///
  Array( const IArrayStoragePtr& data_ptr, const ViewPtr& view_ptr );

  /// Copy constructor with view
  ///
  /// @param src Reference to copied Array object
  /// @param view_ptr Shared pointer on a specific view
  ///
  Array( const Array& src, const ViewPtr& view_ptr );

  /// Copy constructor with view
  ///
  /// @param array_ptr Shared pointer on a copied array
  /// @param view_ptr Shared pointer on a specific view
  ///
  Array( const ArrayPtr& array_ptr, const ViewPtr& view_ptr );

  /// Raw constructor
  ///
  /// @param type Data type
  /// @param shape array shape
  /// @param data_ptr anonymous c-style pointer on array data
  /// @note data is not copied
  /// @note ownership is transfered to the array
  ///
  Array( const std::type_info& type, std::vector<int> shape, void* data_ptr = NULL );

  /// Templated constructor
  ///
  /// @tparam T Data type
  /// @param shape array shape
  /// @param values_ptr typed c-style pointer on array data
  /// @note data is not copied
  /// @note ownership is transfered to the array
  ///
  template<typename T> explicit Array(std::vector<int> shape, T* values_ptr = NULL );

  /// Templated constructor for single value array (e.g. a scalar)
  ///
  /// @tparam T Data type
  /// @param value typed value
  ///
  template<typename T> explicit Array(T value );

  /// d-tor
  ~Array();

  /// Create a copy of this Array, copying the data so that physical order is
  /// the same as logical order. If the view of this array only rely a part of the
  /// whole storage, only the relevant part of the storage will be copied.
  ///
  /// @return the new Array
  /// @note be aware: can lead to out of memory 
  ///
  ArrayPtr deepCopy();
  
  /// Get the array element at the given index position
  ///
  /// @tparam T Data type
  /// @param view describing the array with current element set
  /// @param position vector targeting an element of the array
  ///
  /// @return value converted to the type T
  ///
  template<typename T>
  T getValue( const ViewPtr& view, std::vector<int> position );

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
  void setValue(const ViewPtr& view_ptr, std::vector<int> position, T value);
  
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
  
  /// Set the array element at the given iterator position
  ///
  /// @tparam T Data type
  /// @param it ArrayIterator containing the position of the element to affect
  /// @param value the new value; cast to underlying data type if necessary.
  ///
  template<typename T> void setValue(const ArrayIterator& it, T value);
  
  /// Get the element type of this Array.
  ///
  /// @return type info
  ///
  const std::type_info& getValueType();
  
  /// Get the View that describes this Array.
  ///
  /// @return Shared pointer on View object
  ///
  ViewPtr getView();
  
  /// Set the View that describes this Array.
  ///
  /// @param view new View for this object
  ///
  void setView(const ViewPtr& view);
  
  /// Get an iterator to traverse the Array.
  ///
  /// @return ArrayIterator
  ///
  ArrayIterator begin();
  
  /// Get an iterator positioned at the end of the Array.
  ///
  /// @return ArrayIterator
  ///
  ArrayIterator end();
  
  /// Returns the number of dimensions of the array.
  ///
  int getRank();
  
  /// Returns a sub-part of the array defined by the given start and shape
  /// vector.
  ///
  /// @param start vector defining origin of the region in each dimension
  /// @param shape vector defining shape of the region in each dimension
  /// 
  /// @return ArrayPtr corresponding to a portion sharing same no memory
  ///
  ArrayPtr getRegion(std::vector<int> start, std::vector<int> shape) throw ( cdma::Exception );
  
  /// Get the shape: length of array in each dimension.
  ///
  /// @return array whose length is the rank of this Array and whose elements
  ///         represent the length of each of its indices.
  ///
  std::vector<int> getShape();
  
  /// Return the total number of elements in the array
  ///
  long getSize();
  
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
  SlicerPtr getSlicer(int rank) throw ( cdma::Exception);
  
  /// Retruns true if the array has been changed since last read.
  ///
  bool dirty();
  
  /// Get underlying array storage. Exposed for efficiency, use at your own risk.
  ///
  /// @return IArrayStorage object
  ///
  const IArrayStoragePtr& getStorage() { return m_data_impl; };

private:
  IArrayStoragePtr m_data_impl; // Memory storage of the matrix
  std::vector<int> m_shape;     // Shape of the matrix
  ViewPtr          m_view_ptr;      // Viewable part of the matrix
};

}

/// @endcond

#include "Array.hpp"
#endif // __CDMA_ARRAY_H__
