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

/// @cond excluded_from_doxygen_generation

#ifndef __CDMA_ARRAY_HPP__
#define __CDMA_ARRAY_HPP__

//#include <yat/utils/String.h>
#include <string.h>
#include <sstream>

#include <cdma/array/ArrayIterator.h>
#include <cdma/array/impl/View.h>
#include <cdma/exception/impl/ExceptionImpl.h>

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
// Array::setValue
//----------------------------------------------------------------------------
template<typename T> void Array::setValue(const ArrayIterator& target, T value)
{
  m_data_impl->setValue( m_view_ptr, target.getPosition(), &value );
}

//----------------------------------------------------------------------------
// Array::setValue
//----------------------------------------------------------------------------
template<typename T> void Array::setValue(const IViewPtr& view, std::vector<int> position, T value)
{
  m_data_impl->setValue( view, position, &value );
}

//----------------------------------------------------------------------------
// Array::setValue
//----------------------------------------------------------------------------
template<typename T> void Array::setValue(std::vector<int> position, T value)
{
  m_data_impl->setValue( m_view_ptr, position, &value );
}

//----------------------------------------------------------------------------
// Array::setValue
//----------------------------------------------------------------------------
template<typename T> void Array::setValue(T value)
{
  m_data_impl->setValue( m_view_ptr, std::vector<int>() , &value );
}

//----------------------------------------------------------------------------
// Array::getValue
//----------------------------------------------------------------------------
template<typename T> T Array::getValue( std::vector<int> position )
{
  CDMA_FUNCTION_TRACE("template<typename T> Array::getValue( std::vector<int> position )");
  return getValue<T>( m_view_ptr, position );
}

//----------------------------------------------------------------------------
// Array::getValue
//----------------------------------------------------------------------------
template<typename T> T Array::getValue( void )
{
  CDMA_FUNCTION_TRACE("template<typename T> Array::getValue( void )");
  return getValue<T>( m_view_ptr, std::vector<int>() );
}

//----------------------------------------------------------------------------
// Array::getValue<T>
//----------------------------------------------------------------------------
template<typename T> T Array::getValue( const IViewPtr& view, std::vector<int> position )
{
  CDMA_FUNCTION_TRACE("template<typename T> Array::getValue( const IViewPtr& view, std::vector<int> position )");
  if( typeid(T) == m_data_impl->getType() ||
      !strcmp(typeid(T).name(), m_data_impl->getType().name()) )
    return *( (T*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(short) || 
           !strcmp(m_data_impl->getType().name(), typeid(short).name() ) )
    return T( *(short*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(unsigned short) ||
           !strcmp(m_data_impl->getType().name(), typeid(unsigned short).name() ) )
    return T( *(unsigned short*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(long) ||
           !strcmp(m_data_impl->getType().name(), typeid(long).name() ) )
    return T( *(long*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(unsigned long) ||
           !strcmp(m_data_impl->getType().name(), typeid(unsigned long).name() ) )
    return T( *(unsigned long*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(float) ||
           !strcmp(m_data_impl->getType().name(), typeid(float).name() ) )
    return T( *(float*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(double) ||
           !strcmp(m_data_impl->getType().name(), typeid(double).name() ) )
    return T( *(double*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(int) ||
           !strcmp(m_data_impl->getType().name(), typeid(int).name() ) )
    return T( *(int*)( m_data_impl->getValue( view, position ) ) );

  else if( m_data_impl->getType() == typeid(unsigned int) ||
           !strcmp(m_data_impl->getType().name(), typeid(unsigned int).name() ) )
    return T( *(unsigned int*)( m_data_impl->getValue( view, position ) ) );

  else
    THROW_TYPE_MISMATCH("Cannot convert data to the requested type", "Array::getValue");
}

//----------------------------------------------------------------------------
// Array::getValue<std::string>
//----------------------------------------------------------------------------
template<> inline std::string Array::getValue<std::string>( const IViewPtr& view, std::vector<int> position )
{
  CDMA_FUNCTION_TRACE("template<typename T> Array::getValue<std::string>( const ViewPtr& view, std::vector<int> position )");
  if( m_data_impl->getType() == typeid(char) )
    return std::string( (char *)(m_data_impl->getValue( view, position ) ) );
  else 
  {
    std::ostringstream oss;
    
    if( m_data_impl->getType() == typeid(double) )
      oss << *(double*)( m_data_impl->getValue( view, position ) );
      //return PSZ_FMT("%g", *(double*)( m_data_impl->getValue( view, position ) ) );

    else if( m_data_impl->getType() == typeid(float) )
      oss << *(double*)( m_data_impl->getValue( view, position ) );
      //return PSZ_FMT("%g", *(float*)( m_data_impl->getValue( view, position ) ) );

    else if( m_data_impl->getType() == typeid(short) )
      oss << *(short*)( m_data_impl->getValue( view, position ) );
      //return PSZ_FMT("%hd", *(short*)( m_data_impl->getValue( view, position ) ) );

    else if( m_data_impl->getType() == typeid(unsigned short) )
      oss << *(unsigned short*)( m_data_impl->getValue( view, position ) );
      //return PSZ_FMT("%hu", *(unsigned short*)( m_data_impl->getValue( view, position ) ) );

    else if( m_data_impl->getType() == typeid(long) )
      oss << *(long*)( m_data_impl->getValue( view, position ) );
      //return PSZ_FMT("%ld", *(long*)( m_data_impl->getValue( view, position ) ) );

    else if( m_data_impl->getType() == typeid(unsigned long) )
      oss << *(unsigned long*)( m_data_impl->getValue( view, position ) );
      //return PSZ_FMT("%lu", *(unsigned long*)( m_data_impl->getValue( view, position ) ) );

    else if( m_data_impl->getType() == typeid(int) )
      oss << *(long*)( m_data_impl->getValue( view, position ) );
      //return PSZ_FMT("%d", *(long*)( m_data_impl->getValue( view, position ) ) );

    else if( m_data_impl->getType() == typeid(unsigned int) )
      oss << *(unsigned int*)( m_data_impl->getValue( view, position ) );
      //return PSZ_FMT("%u", *(unsigned int*)( m_data_impl->getValue( view, position ) ) );

    else
      THROW_TYPE_MISMATCH("Cannot convert data to the requested type", "Array::getValue");
    
    return oss.str();
  }
}

}

#endif // __CDMA_ARRAY_HPP__

/// @endcond
