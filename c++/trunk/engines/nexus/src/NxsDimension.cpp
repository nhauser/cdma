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


/**
* Returns the name of this Dimension; may be null. A Dimension with a null
* name is called "anonymous" and must be private. Dimension names are
* unique within a Group.
*
* @return std::string object
*/
std::string NxsDimension::getName()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::getName");
}

/**
* Get the length of the Dimension.
*
* @return integer value
*/
int NxsDimension::getLength()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::getLength");
}

/**
* If unlimited, then the length can increase; otherwise it is immutable.
*
* @return true or false
*/
bool NxsDimension::isUnlimited()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::isUnlimited");
}

/**
* If variable length, then the length is unknown until the data is read.
*
* @return true or false
*/
bool NxsDimension::isVariableLength()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::isVariableLength");
}


/**
* If this Dimension is shared, or is private to a Variable. All Dimensions
* in NxsDimension::NetcdfFile.getDimensions() or Group.getDimensions() are shared.
* Dimensions NxsDimension::in the Variable.getDimensions() may be shared or private.
*
* @return true or false
*/
bool NxsDimension::isShared()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::isShared");
}

/**
* Get the coordinate variables or coordinate variable aliases if the
* dimension has any, else return an empty list. A coordinate variable has
* this as its single dimension, and names this Dimensions's the
* coordinates. A coordinate variable alias is the same as a coordinate
* variable, but its name must match the dimension name. If numeric,
* coordinate axis must be strictly monotonically increasing or decreasing.
*
* @return IArray containing coordinates
*/
cdma::Shape NxsDimension::getCoordinateVariable()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::getCoordinateVariable");
}


/**
* Override NxsDimension::Object.hashCode() to implement equals.
*
* @return integer value
*/
int NxsDimension::hashCode()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::hashCode");
}


/**
* std::string representation.
*
* @return std::string object
*/
std::string NxsDimension::toString()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::toString");
}


/**
* Dimensions with the same name are equal.
*
* @param o
*            compare to this Dimension
* @return 0, 1, or -1
*/
int NxsDimension::compareTo(const cdma::IDimensionPtr& o)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::compareTo");
}


/**
* std::string representation.
*
* @param strict
*            bool type
* @return std::string object
*/
std::string NxsDimension::writeCDL(bool strict)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::writeCDL");
}

/**
* Set whether this is unlimited, meaning length can increase.
*
* @param b
*            bool type
*/
void NxsDimension::setUnlimited(bool b)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setUnlimited");
}


/**
* Set whether the length is variable.
*
* @param b
*            bool type
*/
void NxsDimension::setVariableLength(bool b)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setVariableLength");
}


/**
* Set whether this is shared.
*
* @param b
*            bool type
*/
void NxsDimension::setShared(bool b)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setShared");
}


/**
* Set the Dimension length.
*
* @param n
*            integer value
*/
void NxsDimension::setLength(int n)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setLength");
}


/**
* Rename the dimension.
*
* @param name
*            std::string object
*/
void NxsDimension::setName(const std::string& name)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setName");
}


/**
* Set coordinates values for this dimension.
*
* @param array
*            with new coordinates
*/
void NxsDimension::setCoordinateVariable(const cdma::IArrayPtr& array) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setCoordinateVariable");
}

}