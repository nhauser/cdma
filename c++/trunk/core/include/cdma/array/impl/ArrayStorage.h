//*****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 08/12/2011
// Author   : Rodriguez ClÃ©ment
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
//*****************************************************************************
#ifndef __CDMA_DEFAULTARRAYSTORAGE_H__
#define __CDMA_DEFAULTARRAYSTORAGE_H__

#include <vector>
#include <stdio.h>
#include <iostream>

#include <yat/any/Any.h>
#include <cdma/array/IArrayStorage.h>
#include <cdma/array/View.h>

namespace cdma
{
 
template<typename T> class DefaultArrayStorage : public IArrayStorage
{
public:
  /// Template constructor
  /// 
  /// @param data is the memory buffer 
  /// @param physical shape of the given buffer
  ///
  DefaultArrayStorage( T* data, std::vector<int> shape );
  DefaultArrayStorage( T* data, size_t length );
  
  virtual ~DefaultArrayStorage();

  //@{ IArrayStorage interface
  void set(const cdma::ViewPtr& ima, std::vector<int> position, const yat::Any& value);

  const std::type_info& getType()            { return typeid(*m_data); };
  void*                 getStorage()         { return (void*) m_data; }
  bool                  dirty()              { return m_dirty; };
  void                  setDirty(bool dirty) { m_dirty = dirty; };
  yat::Any&             get( const cdma::ViewPtr& view, std::vector<int> position );
  
  IArrayStoragePtr      deepCopy();
  IArrayStoragePtr      deepCopy(ViewPtr view);
  
  //@}
  //@{ IObject interface
  CDMAType::ModelType getModelType() const { return CDMAType::Other; };
  std::string getFactoryName() const { return m_factory; };
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

#endif // __CDMA_DEFAULTARRAYSTORAGE_H__
