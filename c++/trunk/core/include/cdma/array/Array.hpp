
// *****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 08/12/2011
// Author   : Rodriguez Clément
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// *****************************************************************************

#ifndef __CDMA_ARRAY_HPP__
#define __CDMA_ARRAY_HPP__

namespace cdma
{

//----------------------------------------------------------------------------
// Array::Array
//----------------------------------------------------------------------------
template<typename T> Array::Array(const std::string& factory, T* values, std::vector<int> shape)
{
  CDMA_FUNCTION_TRACE("Array::Array");
  m_factory = factory;
  m_data_impl = new DefaultArrayStorage<T>(values, shape);
  m_shape = shape;
  int rank = shape.size();
  int *shape_ptr = new int[rank];
  int *start_ptr = new int[rank];
  for( int i = 0; i < rank; i++ )
  {
    shape_ptr[i] = shape[i];
    start_ptr[i] = 0;
  }
  m_view = new View( rank, shape_ptr, start_ptr );
}

}

#endif // __CDMA_ARRAY_HPP__
