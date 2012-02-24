// ******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
// ******************************************************************************

#include <cdma/Common.h>
#include <cdma/array/Array.h>

#include <NxsAttribute.h>
#include <TypeUtils.h>

#define MISMATCH_EXCEPTION(a,b) throw cdma::Exception("TYPE_MISMATCHING", a, b)

namespace cdma
{

//---------------------------------------------------------------------------
// NxsAttribute::NxsAttribute
//---------------------------------------------------------------------------
  NxsAttribute::NxsAttribute()
{
}

//---------------------------------------------------------------------------
// NxsAttribute::NxsAttribute
//---------------------------------------------------------------------------
NxsAttribute::NxsAttribute( const NexusFilePtr& file_ptr, const NexusAttrInfo& info )
{
  CDMA_FUNCTION_TRACE("NxsAttribute::NxsAttribute");

  // Allocate requested memory
  int attr_bytes = info.Len() + 1;
  if( info.DataType() == NX_CHAR )
    attr_bytes += 1;

  char *buf = new char[attr_bytes];
  memset(buf, 0, attr_bytes);

  // Load corresponding data
  file_ptr->GetAttribute( info.AttrName(), &attr_bytes, buf, info.DataType() );
  m_value_buf.attach( buf, attr_bytes );

  m_name = info.AttrName();
  m_datatype = info.DataType();
}

//---------------------------------------------------------------------------
// NxsAttribute::getName
//---------------------------------------------------------------------------
std::string NxsAttribute::getName()
{
  return m_name;
}

//---------------------------------------------------------------------------
// NxsAttribute::getType
//---------------------------------------------------------------------------
const std::type_info& NxsAttribute::getType()
{
  return TypeUtils::toCType(m_datatype);
}

//---------------------------------------------------------------------------
// NxsAttribute::isString
//---------------------------------------------------------------------------
bool NxsAttribute::isString()
{
  return ( m_datatype == NX_CHAR );
}

//---------------------------------------------------------------------------
// NxsAttribute::isArray
//---------------------------------------------------------------------------
bool NxsAttribute::isArray()
{
  return false;
}

//---------------------------------------------------------------------------
// NxsAttribute::getLength
//---------------------------------------------------------------------------
int NxsAttribute::getLength()
{
  return 1;
}

//---------------------------------------------------------------------------
// NxsAttribute::getStringValue
//---------------------------------------------------------------------------
std::string NxsAttribute::getStringValue()
{
  if( isString() )
  {
    return yat::String( (char*)(m_value_buf.buf()) );
  }
  MISMATCH_EXCEPTION("Requested type of result isn't valid", "NxsAttribute::getStringValue");
}

//---------------------------------------------------------------------------
// NxsAttribute::getIntValue
//---------------------------------------------------------------------------
long NxsAttribute::getIntValue()
{
  if( !isString() )
  {
    return TypeUtils::valueToType<long>( m_value_buf.buf(), getType() );
  }
  MISMATCH_EXCEPTION("Requested type of result isn't valid", "NxsAttribute::getIntValue");
}

//---------------------------------------------------------------------------
// NxsAttribute::getFloatValue
//---------------------------------------------------------------------------
double NxsAttribute::getFloatValue()
{
  if( !isString() )
  {
    return TypeUtils::valueToType<double>( m_value_buf.buf(), getType() );
  }
  MISMATCH_EXCEPTION("Requested type of result isn't valid", "NxsAttribute::getFloatValue");
}

//---------------------------------------------------------------------------
// NxsAttribute::toString
//---------------------------------------------------------------------------
std::string NxsAttribute::toString()
{
  if( this->isString() )
  {
    return getStringValue();
  }
  else
  {
    std::stringstream ss;
    ss << getFloatValue();
    return ss.str();
  }
}

//---------------------------------------------------------------------------
// NxsAttribute::setStringValue
//---------------------------------------------------------------------------
void NxsAttribute::setStringValue(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setStringValue");
}

//---------------------------------------------------------------------------
// NxsAttribute::setValue
//---------------------------------------------------------------------------
void NxsAttribute::setValue(const yat::Any&)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setValue");
}

}
