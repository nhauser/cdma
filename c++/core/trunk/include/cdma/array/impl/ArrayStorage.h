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

#ifndef __CDMA_DEFAULTARRAYSTORAGE_H__
#define __CDMA_DEFAULTARRAYSTORAGE_H__

#include <vector>
#include <stdio.h>
#include <iostream>

#include <yat/any/Any.h>
#include <cdma/array/IArrayStorage.h>
#include <cdma/array/IView.h>

/// @cond engineAPI

namespace cdma
{

//==============================================================================
/// @brief Default array storage implementation
///
/// This class implements all needed methods from IArrayStorage and provides
/// a efficient way of storing data as a continuous bloc
///
/// @tparam T data type of array elements
//==============================================================================
template<typename T> class DefaultArrayStorage : public IArrayStorage
{
public:

  /// c-tor
  /// 
  /// @param data is the memory buffer 
  /// @param shape Array shape
  ///
  DefaultArrayStorage( T* data, std::vector<int> shape );

  /// c-tor
  /// 
  /// @param data Typed C-style pointer data
  /// @param length array length
  /// @note The data is not copied
  DefaultArrayStorage( T* data, size_t length );
  
  /// d-tor
  virtual ~DefaultArrayStorage();

  //@{ IArrayStorage

    void setValue(const cdma::IViewPtr& ima, std::vector<int> position, void* value_ptr);
    const std::type_info& getType()            { return typeid(*m_data); };
    void*                 getStorage()         { return (void*) m_data; }
    bool                  dirty()              { return m_dirty; };
    void                  setDirty(bool dirty) { m_dirty = dirty; };
    void*                 getValue( const cdma::IViewPtr& view, std::vector<int> position );
    IArrayStoragePtr      deepCopy();
    IArrayStoragePtr      deepCopy(IViewPtr view);

  //@}

private:
  T*          m_data;          // pointor wearing physically the data
  size_t      m_elem_size;     // size of type T
  size_t      m_array_length;  // current number of element of type T
  bool        m_dirty;         // has the data been modified since last read
  std::string m_factory;       // name of the instanting plugin
  yat::Any    m_current;       // currently referenced cell
};

}

#include "cdma/array/impl/ArrayStorage.hpp"

/// @endcond

#endif // __CDMA_DEFAULTARRAYSTORAGE_H__
