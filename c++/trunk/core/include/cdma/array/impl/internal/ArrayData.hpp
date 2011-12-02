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
#ifndef __CDMA_ARRAYDATA_HPP__
#define __CDMA_ARRAYDATA_HPP__

#include <stdio.h>
#include <iostream>

namespace cdma
{

//----------------------------------------------------------------------------
// TypedData<T>::~TypedData
//----------------------------------------------------------------------------
template<typename T> TypedData<T>::~TypedData()
{
    delete m_data;
}

//----------------------------------------------------------------------------
// TypedData<T>::TypedData
//----------------------------------------------------------------------------
template<typename T> TypedData<T>::TypedData( T* data, std::vector<int> shape )
{
  m_data = data;
  m_elem_size = sizeof(T);
  m_array_length = 1;
  for( int i = 0; i < shape.size(); i++ )
  {
    m_array_length *= shape[i];
  }
}

//----------------------------------------------------------------------------
// TypedData<T>::set
//----------------------------------------------------------------------------
template<typename T> void TypedData<T>::set(const cdma::IIndexPtr& ima, const yat::Any& value)
{
  m_data[ima->currentElement()] = yat::any_cast<T>(value);
};

}


#endif // __CDMA_ARRAYDATA_HPP__
