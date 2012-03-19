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
#include <NxsDataset.h>
#include <TypeUtils.h>

#include <cdma/navigation/IAttribute.h>

#include <nxfile.h>

namespace cdma
{
NxsDimension::NxsDimension(NxsDataItemPtr dataitem)
{
  m_item = dataitem;
  m_shared = false;
}
/*
NxsDimension::NxsDimension(NxsDataset* datasetPtr, const NexusDataSetInfo& item, const std::string& path)
{
  m_dataset_ptr = datasetPtr;
  m_item = item;
  m_path = path;
  m_path.extract_token_right('/', &m_name);
}
*/
//----------------------------------------------------------------------------
// NxsDimension::getName
//----------------------------------------------------------------------------
std::string NxsDimension::getName()
{
  return m_item->getName();
}
//----------------------------------------------------------------------------
// NxsDimension::getLength
//----------------------------------------------------------------------------
int NxsDimension::getLength()
{
  return m_item->getSize();
}
//----------------------------------------------------------------------------
// NxsDimension::isUnlimited
//----------------------------------------------------------------------------
bool NxsDimension::isUnlimited()
{
  return false;
}
//----------------------------------------------------------------------------
// NxsDimension::isVariableLength
//----------------------------------------------------------------------------
bool NxsDimension::isVariableLength()
{
  return false;
}

//----------------------------------------------------------------------------
// NxsDimension::isShared
//----------------------------------------------------------------------------
bool NxsDimension::isShared()
{
  return m_shared;
}
//----------------------------------------------------------------------------
// NxsDimension::getCoordinateVariable
//----------------------------------------------------------------------------
cdma::ArrayPtr NxsDimension::getCoordinateVariable()
{
  return m_item->getData();
}

//----------------------------------------------------------------------------
// NxsDimension::setUnlimited
//----------------------------------------------------------------------------
void NxsDimension::setUnlimited(bool)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setUnlimited");
}

//----------------------------------------------------------------------------
// NxsDimension::setVariableLength
//----------------------------------------------------------------------------
void NxsDimension::setVariableLength(bool)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setVariableLength");
}

//----------------------------------------------------------------------------
// NxsDimension::setShared
//----------------------------------------------------------------------------
void NxsDimension::setShared(bool value)
{
  m_shared = value;
}

//----------------------------------------------------------------------------
// NxsDimension::setLength
//----------------------------------------------------------------------------
void NxsDimension::setLength(int)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setLength");
}

//----------------------------------------------------------------------------
// NxsDimension::setName
//----------------------------------------------------------------------------
void NxsDimension::setName(const std::string& name)
{
  m_item->setName(name);
}

//----------------------------------------------------------------------------
// NxsDimension::setCoordinateVariable
//----------------------------------------------------------------------------
void NxsDimension::setCoordinateVariable(const cdma::ArrayPtr& array) throw ( cdma::Exception )
{
  m_item->setData(array);
}

//----------------------------------------------------------------------------
// NxsDimension::getDimensionAxis
//----------------------------------------------------------------------------
int NxsDimension::getDimensionAxis()
{
  IAttributePtr attr = m_item->getAttribute("axis");
  if( attr )
  {
    return attr->getIntValue();
  }
  else
  {
    return -1;
  }
}

//----------------------------------------------------------------------------
// NxsDimension::getDisplayOrder
//----------------------------------------------------------------------------
int NxsDimension::getDisplayOrder()
{
  IAttributePtr attr = m_item->getAttribute("primary");
  if( attr )
  {
    return attr->getIntValue();
  }
  else
  {
    return -1;
  }
}

//----------------------------------------------------------------------------
// NxsDimension::getUnitsString
//----------------------------------------------------------------------------
std::string NxsDimension::getUnitsString()
{
  return m_item->getUnitsString();
}

//---------------------------------------------------------------------------
// NxsDataItem::setDimensionAxis
//---------------------------------------------------------------------------
void NxsDimension::setDimensionAxis(int index)
{
  IAttributePtr attr = m_item->getAttribute("axis");
  if( attr )
  {
    attr->setIntValue(index);
  }
}

//---------------------------------------------------------------------------
// NxsDimension::setDisplayOrder
//---------------------------------------------------------------------------
void NxsDimension::setDisplayOrder(int order)
{
  IAttributePtr attr = m_item->getAttribute("primary");
  if( attr )
  {
    attr->setIntValue(order);
  }
}
}
