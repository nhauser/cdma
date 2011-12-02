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
#ifndef __CDMA_ARRAYDATA_H__
#define __CDMA_ARRAYDATA_H__

#include <vector>
#include <stdio.h>
#include <iostream>

#include <yat/any/Any.h>

#include <cdma/array/IIndex.h>

namespace cdma
{
 
class Data
{
public:
  virtual ~Data() {};

  virtual void* getStorage() = 0;
  virtual yat::Any get( int index ) = 0;
  virtual void set(const cdma::IIndexPtr& ima, const yat::Any& value) = 0;
  virtual const std::type_info& getType() = 0;
};

template<typename T> class TypedData : public Data
{
public:
  ~TypedData();
  TypedData( T* data, std::vector<int> shape );

  void set(const cdma::IIndexPtr& ima, const yat::Any& value);

  const std::type_info& getType()      { return typeid(*m_data); };
  void*                 getStorage()   { return (void*) m_data; }
  yat::Any              get(int index) { return yat::Any(m_data[index]); };

  protected:
  T*     m_data;          // pointor wearing physically the data
  size_t m_elem_size;     // size of type T
  size_t m_array_length;  // current number of element of type T
};

}

#include "cdma/array/impl/internal/ArrayData.hpp"

#endif // __CDMA_ARRAYDATA_H__