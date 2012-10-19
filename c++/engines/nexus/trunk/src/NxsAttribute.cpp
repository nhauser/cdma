// ******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The cdma-core library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
// ******************************************************************************

#include <cdma/Common.h>
#include <cdma/array/IArray.h>

#include <NxsAttribute.h>
#include <TypeUtils.h>

#define MISMATCH_EXCEPTION(a,b) throw cdma::Exception("TYPE_MISMATCHING", a, b)

namespace cdma
{
namespace nexus
{

//---------------------------------------------------------------------------
// Attribute::Attribute
//---------------------------------------------------------------------------
Attribute::Attribute()
{
}

//---------------------------------------------------------------------------
// Attribute::Attribute
//---------------------------------------------------------------------------
Attribute::Attribute( const NexusFilePtr& file_ptr, const NexusAttrInfo& info )
{
  // Allocate requested memory
  int attr_bytes = info.Len() + 1;
  if( info.DataType() == NX_CHAR )
    attr_bytes += 1;

  char *buf = new char[attr_bytes];
  memset(buf, 0, attr_bytes);

  // Load corresponding data
  file_ptr->GetAttribute( info.AttrName(), &attr_bytes, buf, info.DataType() );

  // prepare shape
  std::vector<int> shape;
  if( info.DataType() != NX_CHAR )
    shape.push_back(1);
  else
    shape.push_back(attr_bytes);

  // Init Array
  cdma::Array *array_ptr = new cdma::Array( TypeUtils::toRawCType(info.DataType()),
                                            shape, buf );

  // update Array
  m_array_ptr.reset(array_ptr);

  m_name = info.AttrName();
  m_datatype = info.DataType();
}

//---------------------------------------------------------------------------
// Attribute::getName
//---------------------------------------------------------------------------
std::string Attribute::getName()
{
  return m_name;
}

//---------------------------------------------------------------------------
// Attribute::getType
//---------------------------------------------------------------------------
const std::type_info& Attribute::getType()
{
  return TypeUtils::toCType(m_datatype);
}

//---------------------------------------------------------------------------
// Attribute::isArray
//---------------------------------------------------------------------------
bool Attribute::isArray()
{
  return false;
}

//---------------------------------------------------------------------------
// Attribute::getSize
//---------------------------------------------------------------------------
int Attribute::getSize()
{
  return 1;
}

//---------------------------------------------------------------------------
// Attribute::getData
//---------------------------------------------------------------------------
cdma::IArrayPtr Attribute::getData()
{
  CDMA_FUNCTION_TRACE("cdma::nexus::Attribute::getData(vector<int> origin, vector<int> shape)");
  return m_array_ptr;
}

//---------------------------------------------------------------------------
// Attribute::setData
//---------------------------------------------------------------------------
void Attribute::setData(const cdma::IArrayPtr& array)
{
  m_array_ptr = array;
}

//---------------------------------------------------------------------------
// Attribute::setName
//---------------------------------------------------------------------------
void Attribute::setName(const std::string& name)
{
  m_name = name;
}


} // namespace nexus
} // namespace cdma
