//*****************************************************************************
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
//*****************************************************************************
#include <NxsDimension.h>

namespace cdma
{

//----------------------------------------------------------------------------
// NxsDimension::getName
//----------------------------------------------------------------------------
std::string NxsDimension::getName()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::getName");
}
//----------------------------------------------------------------------------
// NxsDimension::getLength
//----------------------------------------------------------------------------
int NxsDimension::getLength()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::getLength");
}
//----------------------------------------------------------------------------
// NxsDimension::isUnlimited
//----------------------------------------------------------------------------
bool NxsDimension::isUnlimited()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::isUnlimited");
}
//----------------------------------------------------------------------------
// NxsDimension::isVariableLength
//----------------------------------------------------------------------------
bool NxsDimension::isVariableLength()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::isVariableLength");
}

//----------------------------------------------------------------------------
// NxsDimension::isShared
//----------------------------------------------------------------------------
bool NxsDimension::isShared()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::isShared");
}
//----------------------------------------------------------------------------
// NxsDimension::getCoordinateVariable
//----------------------------------------------------------------------------
cdma::Shape NxsDimension::getCoordinateVariable()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::getCoordinateVariable");
}

//----------------------------------------------------------------------------
// NxsDimension::hashCode
//----------------------------------------------------------------------------
int NxsDimension::hashCode()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::hashCode");
}

//----------------------------------------------------------------------------
// NxsDimension::toString
//----------------------------------------------------------------------------
std::string NxsDimension::toString()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::toString");
}

//----------------------------------------------------------------------------
// NxsDimension::compareTo
//----------------------------------------------------------------------------
int NxsDimension::compareTo(const cdma::IDimensionPtr& o)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::compareTo");
}

//----------------------------------------------------------------------------
// NxsDimension::writeCDL
//----------------------------------------------------------------------------
std::string NxsDimension::writeCDL(bool strict)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::writeCDL");
}
//----------------------------------------------------------------------------
// NxsDimension::setUnlimited
//----------------------------------------------------------------------------
void NxsDimension::setUnlimited(bool b)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setUnlimited");
}

//----------------------------------------------------------------------------
// NxsDimension::setVariableLength
//----------------------------------------------------------------------------
void NxsDimension::setVariableLength(bool b)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setVariableLength");
}

//----------------------------------------------------------------------------
// NxsDimension::setShared
//----------------------------------------------------------------------------
void NxsDimension::setShared(bool b)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setShared");
}

//----------------------------------------------------------------------------
// NxsDimension::setLength
//----------------------------------------------------------------------------
void NxsDimension::setLength(int n)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setLength");
}

//----------------------------------------------------------------------------
// NxsDimension::setName
//----------------------------------------------------------------------------
void NxsDimension::setName(const std::string& name)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setName");
}

//----------------------------------------------------------------------------
// NxsDimension::setCoordinateVariable
//----------------------------------------------------------------------------
void NxsDimension::setCoordinateVariable(const cdma::ArrayPtr& array) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setCoordinateVariable");
}

}
