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

#include <typeinfo>

#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/array/impl/Array.h>

namespace cdma
{

//---------------------------------------------------------------------------
// Array::~Array
//---------------------------------------------------------------------------
Array::~Array()
{
  CDMA_FUNCTION_TRACE("[Array::~Array");
}

//---------------------------------------------------------------------------
// Array::Array
//---------------------------------------------------------------------------
Array::Array( const yat::String& factory, const std::type_info& type, std::vector<int> shape, void* pData )
{
  CDMA_FUNCTION_TRACE("Array::Array");
  int rank = shape.size();
  m_factory = factory;
  unsigned int size = 1;
  if( pData == NULL )
  {
    for( int i = 0; i < rank; i++ )
    {
      size *= shape[i];
    }
  }

  if( type == typeid( short ) )
  {
    if( pData == NULL )
    {
      pData = new short[size];
    }
    m_data = new TypedData<short>((short*) pData, shape);
  }
  else if( type == typeid( unsigned short ) )
  {
    if( pData == NULL )
    {
      pData = new unsigned short[size];
    }
    m_data = new TypedData<unsigned short>((unsigned short*) pData, shape);
  }
  else if( type == typeid( long ) )
  {
    if( pData == NULL )
    {
      pData = new long[size];
    }
    m_data = new TypedData<long>((long*)pData, shape);
  }
  else if( type == typeid( unsigned long ) )
  {
    if( pData == NULL )
    {
      pData = new unsigned long[size];
    }
    m_data = new TypedData<unsigned long>((unsigned long*)pData, shape);
  }
  else if( type == typeid( float ) )
  {
    if( pData == NULL )
    {
      pData = new float[size];
    }
    m_data = new TypedData<float>((float*)pData, shape);
  }
  else if( type == typeid( yat::uint64 ) )
  {
    if( pData == NULL )
    {
      pData = new yat::uint64[size];
    }
    m_data = new TypedData<yat::uint64>((yat::uint64*) pData, shape);
  }
  else if( type == typeid( yat::int64 ) )
  {
    if( pData == NULL )
    {
      pData = new yat::int64[size];
    }
    m_data = new TypedData<yat::int64>((yat::int64*) pData, shape);
  }
  else if( type == typeid( double ) )
  {
    if( pData == NULL )
    {
      pData = new double[size];
    }
    m_data = new TypedData<double>((double*) pData, shape);
  }
  else
  {
    if( pData == NULL )
    {
      pData = new char[size];
    }
    m_data = new TypedData<char>((char*) pData, shape);
  }
  
  m_shape = shape;
  int *shape_ptr = new int[rank];
  int *start_ptr = new int[rank];
  for( int i = 0; i < shape.size(); i++ )
  {
    shape_ptr[i] = shape[i];
    start_ptr[i] = 0;
  }
  m_index = new Index( factory, rank, shape_ptr, start_ptr);
}

//---------------------------------------------------------------------------
// Array::Array
//---------------------------------------------------------------------------
Array::Array( const Array& array, IIndexPtr index) : m_index (index)
{
  CDMA_FUNCTION_TRACE("Array::Array");
  m_data = array.m_data;
  m_shape = index->getShape();
  m_factory = array.m_factory;
}

//---------------------------------------------------------------------------
// Array::copy
//---------------------------------------------------------------------------
cdma::IArrayPtr Array::copy()
{
  THROW_NOT_IMPLEMENTED("Array::copy");
}

//---------------------------------------------------------------------------
// Array::copy
//---------------------------------------------------------------------------
cdma::IArrayPtr Array::copy(bool data)
{
  THROW_NOT_IMPLEMENTED("Array::copy");
}

//---------------------------------------------------------------------------
// Array::getArrayUtils
//---------------------------------------------------------------------------
cdma::IArrayUtilsPtr Array::getArrayUtils()
{
  THROW_NOT_IMPLEMENTED("Array::getArrayUtils");
}

//---------------------------------------------------------------------------
// Array::getArrayMath
//---------------------------------------------------------------------------
cdma::IArrayMathPtr Array::getArrayMath()
{
  THROW_NOT_IMPLEMENTED("Array::getArrayMath");
}

//---------------------------------------------------------------------------
// Array::getElementType
//---------------------------------------------------------------------------
const std::type_info& Array::getElementType()
{
  return m_data->getType();
}

//---------------------------------------------------------------------------
// Array::getIndex
//---------------------------------------------------------------------------
cdma::IIndexPtr Array::getIndex()
{
  return cdma::IIndexPtr(m_index);
}

//---------------------------------------------------------------------------
// Array::getIterator
//---------------------------------------------------------------------------
cdma::IArrayIteratorPtr Array::getIterator()
{
  THROW_NOT_IMPLEMENTED("Array::getIterator");
}

//---------------------------------------------------------------------------
// Array::getRank
//---------------------------------------------------------------------------
int Array::getRank()
{
  return m_index->getRank();
}

//---------------------------------------------------------------------------
// Array::getRegionIterator
//---------------------------------------------------------------------------
cdma::IArrayIteratorPtr Array::getRegionIterator(std::vector<int> reference, std::vector<int> range) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("Array::getRegionIterator");
}

//---------------------------------------------------------------------------
// Array::getShape
//---------------------------------------------------------------------------
std::vector<int> Array::getShape()
{
  return m_index->getShape();
}

//---------------------------------------------------------------------------
// Array::getSize
//---------------------------------------------------------------------------
long Array::getSize()
{
  return m_index->getSize();
}


//---------------------------------------------------------------------------
// Array::shapeToString
//---------------------------------------------------------------------------
std::string Array::shapeToString()
{
  THROW_NOT_IMPLEMENTED("Array::shapeToString");
}

//---------------------------------------------------------------------------
// Array::setIndex
//---------------------------------------------------------------------------
void Array::setIndex(const cdma::IIndexPtr& index)
{

}

//---------------------------------------------------------------------------
// Array::getSliceIterator
//---------------------------------------------------------------------------
cdma::ISliceIteratorPtr Array::getSliceIterator(int rank) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("Array::getSliceIterator");
}

//---------------------------------------------------------------------------
// Array::releaseStorage
//---------------------------------------------------------------------------
void Array::releaseStorage() throw ( cdma::Exception )
{

}

//---------------------------------------------------------------------------
// Array::getRegisterId
//---------------------------------------------------------------------------
long Array::getRegisterId()
{
  THROW_NOT_IMPLEMENTED("Array::getRegisterId");
}

//---------------------------------------------------------------------------
// Array::lock
//---------------------------------------------------------------------------
void Array::lock()
{

}

//---------------------------------------------------------------------------
// Array::unlock
//---------------------------------------------------------------------------
void Array::unlock()
{

}

//---------------------------------------------------------------------------
// Array::isDirty
//---------------------------------------------------------------------------
bool Array::isDirty()
{
  THROW_NOT_IMPLEMENTED("Array::isDirty");
}

//---------------------------------------------------------------------------
// Array::get
//---------------------------------------------------------------------------
yat::Any Array::get(cdma::IIndexPtr& ind)
{
  int index = ind->currentElement();
  return m_data->get(index);
}

//---------------------------------------------------------------------------
// Array::get
//---------------------------------------------------------------------------
yat::Any Array::get()
{
  return m_data->get(0);
}

//---------------------------------------------------------------------------
// Array::set
//---------------------------------------------------------------------------
void Array::set(const cdma::IIndexPtr& ima, const yat::Any& value)
{
  m_data->set(ima, value);
}

//---------------------------------------------------------------------------
// Array::detectType
//---------------------------------------------------------------------------
/*
const type_info& Array::detectType(NexusDataType type)
{
  switch( type )
  {
    case NX_INT16:
      return typeid(short);
      break;
    case NX_UINT16:
      return typeid(unsigned short);
      break;
    case NX_UINT32:
      return typeid(unsigned long);
      break;
    case NX_INT32:
      return typeid(long);
      break;
    case NX_FLOAT32:
      return typeid(float);
      break;
    case NX_INT64:
      return typeid(yat::int64);
      break;
    case NX_UINT64:
      return typeid(unsigned long);
      break;
    case NX_FLOAT64:
      return typeid(double);
      break;
    default:  // CHAR, NX_INT8, NX_UINT8
      return typeid(char);
  }
}

//---------------------------------------------------------------------------
// Array::allocate
//---------------------------------------------------------------------------
void* Array::allocate(NexusDataType type, unsigned int length)
{
  void* data;
  switch( type )
  {
    case NX_INT16:
      data = new short[length];
      break;
    case NX_UINT16:
      data = new unsigned short[length];
      break;
    case NX_UINT32:
      data = new unsigned long[length];
      break;
    case NX_INT32:
      data = new long[length];
      break;
    case NX_FLOAT32:
      data = new float[length];
      break;
    case NX_INT64:
      data = new yat::int64[length];
      break;
    case NX_UINT64:
      data = new yat::uint64[length];
      break;
    case NX_FLOAT64:
      data = new double[length];
      break;
    default:  // CHAR, NX_INT8, NX_UINT8
      data = new char[length];
  }
  return data;
}
*/
}