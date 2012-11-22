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
#include <cdma/array/IArray.h>

/// @cond clientAPI

namespace cdma
{

// Forward declaration
#ifdef NO_ITERATORS
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
//==============================================================================
class Array: public IArray
{
friend class ArrayIterator;

public:
  /// Copy constructor
  Array( const Array& array );

  /// c-tor
  /// @param data_ptr Shared pointer on a particular storage implementation
  /// @param view_ptr Shared pointer on a specific view
  ///
  Array( const IArrayStoragePtr& data_ptr, const IViewPtr& view_ptr );

  /// Copy constructor with view
  ///
  /// @param src Reference to copied Array object
  /// @param view_ptr Shared pointer on a specific view
  ///
  Array( const Array& src, const IViewPtr& view_ptr );

  /// Copy constructor with view
  ///
  /// @param array_ptr Shared pointer on a copied array
  /// @param view_ptr Shared pointer on a specific view
  ///
  Array( const IArrayPtr& array_ptr, const IViewPtr& view_ptr );

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

  //@{ interface IArray
  virtual void* getValue( const IViewPtr& view, std::vector<int> position );
  virtual void setValue(const cdma::IViewPtr& view_ptr, std::vector<int> position, void *value_ptr);
  IArrayPtr deepCopy();
  const std::type_info& getValueType();
  IViewPtr getView();
  void setView(const IViewPtr& view);
  ArrayIterator begin();
  ArrayIterator end();
  int getRank();
  IArrayPtr getRegion(std::vector<int> start, std::vector<int> shape) throw ( cdma::Exception );
  std::vector<int> getShape();
  long getSize();
  SlicerPtr getSlicer(int rank) throw ( cdma::Exception);
  bool dirty();
  const IArrayStoragePtr& getStorage() { return m_data_impl; };
  template<typename T> T getValue( const IViewPtr& view, std::vector<int> position );
  template<typename T> T getValue( std::vector<int> position );
  template<typename T> T getValue( void );
  template<typename T> void setValue(const IViewPtr& view_ptr, std::vector<int> position, T value);
  template<typename T> void setValue(std::vector<int> position, T value);
  template<typename T> void setValue(T value);
  template<typename T> void setValue(const ArrayIterator& it, T value);
  //@}
  
private:
  IArrayStoragePtr m_data_impl; // Memory storage of the matrix
  std::vector<int> m_shape;     // Shape of the matrix
  IViewPtr         m_view_ptr;  // Viewable part of the matrix
};

}

/// @endcond

#include <cdma/array/impl/Array.hpp>

#endif // __CDMA_ARRAY_H__
