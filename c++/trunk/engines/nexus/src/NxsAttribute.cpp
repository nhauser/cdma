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
#include <TypeDetector.h>

#define MISMATCH_EXCEPTION(a,b) throw cdma::Exception("TYPE_MISMATCHING", a, b)

namespace cdma
{
//---------------------------------------------------------------------------
// NxsAttribute::NxsAttribute
//---------------------------------------------------------------------------
NxsAttribute::NxsAttribute( NexusFilePtr file, NexusAttrInfo* info )
{
  CDMA_FUNCTION_TRACE("NxsAttribute::NxsAttribute");
  
  m_info_ptr = info;

  // Allocate requested memory
  m_value = TypeDetector::allocate<void>( info->DataType(), info->Len() );

  // Load corresponding data
  int length = info->Len();
  file->GetAttribute( info->AttrName(), &length, m_value, info->DataType() );
}

//---------------------------------------------------------------------------
// NxsAttribute::getName
//---------------------------------------------------------------------------
std::string NxsAttribute::getName()
{
  return m_info_ptr->AttrName();
}

//---------------------------------------------------------------------------
// NxsAttribute::getType
//---------------------------------------------------------------------------
const std::type_info& NxsAttribute::getType()
{
  return TypeDetector::detectType(m_info_ptr->DataType());
}

//---------------------------------------------------------------------------
// NxsAttribute::isString
//---------------------------------------------------------------------------
bool NxsAttribute::isString()
{
  return ( m_info_ptr->DataType() == NX_CHAR );
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
  return m_info_ptr->Len();
}

//---------------------------------------------------------------------------
// NxsAttribute::getValue
//---------------------------------------------------------------------------
/*
cdma::ArrayPtr NxsAttribute::getValue()
{
}
*/

//---------------------------------------------------------------------------
// NxsAttribute::getStringValue
//---------------------------------------------------------------------------
std::string NxsAttribute::getStringValue()
{
  if( isString() )
  {
    return yat::String((char*) m_value);
  }
  MISMATCH_EXCEPTION("Requested type of result isn't valid", "NxsAttribute::getStringValue");
}

//---------------------------------------------------------------------------
// NxsAttribute::getStringValue
//---------------------------------------------------------------------------
/*std::string NxsAttribute::getStringValue(int index)
{
}
*/

//---------------------------------------------------------------------------
// NxsAttribute::getIntValue
//---------------------------------------------------------------------------
long NxsAttribute::getIntValue()
{
  if( ! isString() )
  {
    return *((long*) m_value);
  }
  MISMATCH_EXCEPTION("Requested type of result isn't valid", "NxsAttribute::getIntValue");
}

//---------------------------------------------------------------------------
// NxsAttribute::getFloatValue
//---------------------------------------------------------------------------
double NxsAttribute::getFloatValue()
{
  if( ! isString() )
  {
    return *((double*) m_value);
  }
  MISMATCH_EXCEPTION("Requested type of result isn't valid", "NxsAttribute::getFloatValue");
}

//---------------------------------------------------------------------------
// NxsAttribute::toString
//---------------------------------------------------------------------------
std::string NxsAttribute::toString()
{
  yat::String res = "";
  if( this->isString() )
  {
    res = "Attr: " + this->getName() + " = " + this->getStringValue();
  }
  else
  {
    std::stringstream ss;
    ss << "Attr: " << this->getName() << " = " <<this->getIntValue();
    res =  ss.str();
  }
  return res;
}

//---------------------------------------------------------------------------
// NxsAttribute::setStringValue
//---------------------------------------------------------------------------
void NxsAttribute::setStringValue(std::string val)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setStringValue");
}

//---------------------------------------------------------------------------
// NxsAttribute::setValue
//---------------------------------------------------------------------------
void NxsAttribute::setValue(cdma::Array& value)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setValue");
}

}
