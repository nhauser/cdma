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
  m_value_buf.attach( buf, attr_bytes );

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
// Attribute::isString
//---------------------------------------------------------------------------
bool Attribute::isString()
{
  return ( m_datatype == NX_CHAR );
}

//---------------------------------------------------------------------------
// Attribute::isArray
//---------------------------------------------------------------------------
bool Attribute::isArray()
{
  return false;
}

//---------------------------------------------------------------------------
// Attribute::getLength
//---------------------------------------------------------------------------
int Attribute::getLength()
{
  return 1;
}

//---------------------------------------------------------------------------
// Attribute::getStringValue
//---------------------------------------------------------------------------
std::string Attribute::getStringValue()
{
  CDMA_FUNCTION_TRACE("cdma::nexus::Attribute::getStringValue");
  if( isString() )
  {
    return yat::String( (char*)(m_value_buf.buf()) );
  }
  MISMATCH_EXCEPTION("Requested type of result isn't valid", "cdma::nexus::Attribute::getStringValue");
}

//---------------------------------------------------------------------------
// Attribute::getIntValue
//---------------------------------------------------------------------------
long Attribute::getIntValue()
{
  if( !isString() )
  {
    return TypeUtils::valueToType<long>( m_value_buf.buf(), getType() );
  }
  MISMATCH_EXCEPTION("Requested type of result isn't valid", "cdma::nexus::Attribute::getIntValue");
}

//---------------------------------------------------------------------------
// Attribute::getFloatValue
//---------------------------------------------------------------------------
double Attribute::getFloatValue()
{
  if( !isString() )
  {
    return TypeUtils::valueToType<double>( m_value_buf.buf(), getType() );
  }
  MISMATCH_EXCEPTION("Requested type of result isn't valid", "cdma::nexus::Attribute::getFloatValue");
}

//---------------------------------------------------------------------------
// Attribute::toString
//---------------------------------------------------------------------------
std::string Attribute::toString()
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
// Attribute::setStringValue
//---------------------------------------------------------------------------
void Attribute::setStringValue(const std::string&)
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::Attribute::setStringValue");
}

//---------------------------------------------------------------------------
// Attribute::setName
//---------------------------------------------------------------------------
void Attribute::setName(const std::string& name)
{
  m_name = name;
}

//---------------------------------------------------------------------------
// Attribute::setDisplayOrder
//---------------------------------------------------------------------------
void Attribute::setIntValue(int)
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::Attribute::setIntValue");
}
  
//---------------------------------------------------------------------------
// Attribute::setDisplayOrder
//---------------------------------------------------------------------------
void Attribute::setFloatValue(float)
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::Attribute::setFloatValue");
}


} // namespace nexus
} // namespace cdma
