
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

/// @cond excluded_from_doxygen_generation

#ifndef __CDMA_ARRAY_HPP__
#define __CDMA_ARRAY_HPP__
#include <cdma/array/ArrayIterator.h>
namespace cdma
{

//---------------------------------------------------------------------------
// Array::Array
//---------------------------------------------------------------------------
template<typename T>
Array::Array(std::vector<int> shape, T* data_ptr)
{
  CDMA_FUNCTION_TRACE("template<typename T> Array::Array");
  int rank = shape.size();

  unsigned int size = 1;
  if( data_ptr == NULL )
  {
    for( int i = 0; i < rank; i++ )
    {
      size *= shape[i];
    }
    data_ptr = new T[size];
  }
  m_data_impl = new DefaultArrayStorage<T>(data_ptr, shape);

  m_shape = shape;
  int *shape_ptr = new int[rank];
  int *start_ptr = new int[rank];
  for( int i = 0; i < shape.size(); i++ )
  {
    shape_ptr[i] = shape[i];
    start_ptr[i] = 0;
  }
  m_view_ptr = new View( rank, shape_ptr, start_ptr );
}

//---------------------------------------------------------------------------
// Array::Array
//---------------------------------------------------------------------------
template<typename T>
Array::Array(T scalar_value)
{
  CDMA_FUNCTION_TRACE("template<typename T> Array::Array");

  T* data_ptr = new T[1];
  *data_ptr = scalar_value;
  std::vector<int> shape;
  shape.push_back(1);
  std::vector<int> start;
  start.push_back(0);
  m_data_impl = new DefaultArrayStorage<T>(data_ptr, shape);
  m_view_ptr = new View(shape, start);
}

//----------------------------------------------------------------------------
// Array::set
//----------------------------------------------------------------------------
template<typename T> void Array::setValue(const ArrayIterator& target, T value)
{
  m_data_impl->setValue( m_view_ptr, target.getPosition(), &value );
}

//----------------------------------------------------------------------------
// Array::set
//----------------------------------------------------------------------------
template<typename T> void Array::setValue(const ViewPtr& view, std::vector<int> position, T value)
{
  m_data_impl->setValue( view, position, &value );
}

//----------------------------------------------------------------------------
// Array::set
//----------------------------------------------------------------------------
template<typename T> void Array::setValue(std::vector<int> position, T value)
{
  m_data_impl->setValue( m_view_ptr, position, &value );
}

//----------------------------------------------------------------------------
// Array::set
//----------------------------------------------------------------------------
template<typename T> void Array::setValue(T value)
{
  m_data_impl->setValue( m_view_ptr, std::vector<int>() , &value );
}

//----------------------------------------------------------------------------
// Array::get
//----------------------------------------------------------------------------
template<typename T> T Array::getValue( std::vector<int> position )
{
  return getValue<T>( m_view_ptr, position );
}

//----------------------------------------------------------------------------
// Array::get
//----------------------------------------------------------------------------
template<typename T> T Array::getValue( void )
{
  return getValue<T>( m_view_ptr, std::vector<int>() );
}

//----------------------------------------------------------------------------
// Array::get
//----------------------------------------------------------------------------
template<typename T> T Array::getValue( const ViewPtr& view, std::vector<int> position )
{
  if( typeid(T) == m_data_impl->getType() )
    return *( (T*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(short) )
    return T( *(short*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(unsigned short) )
    return T( *(unsigned short*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(long) )
    return T( *(long*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(unsigned long) )
    return T( *(unsigned long*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(float) )
    return T( *(float*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(double) )
    return T( *(double*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(int) )
    return T( *(int*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(unsigned int) )
    return T( *(unsigned int*)( m_data_impl->getValue( view, position ) ) );

  else
    throw cdma::Exception("INVALID_TYPE", "Cannot convert data to the requested type", 
                          "Array::getValue");
}

}

#endif // __CDMA_ARRAY_HPP__

/// @endcond
