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
namespace nexus
{

//----------------------------------------------------------------------------
// Dimension::Dimension
//----------------------------------------------------------------------------
Dimension::Dimension(DataItemPtr dataitem)
{
  m_item = dataitem;
  m_shared = false;
}
/*
Dimension::Dimension(NxsDataset* datasetPtr, const NexusDataSetInfo& item, const std::string& path)
{
  m_dataset_ptr = datasetPtr;
  m_item = item;
  m_path = path;
  m_path.extract_token_right('/', &m_name);
}
*/
//----------------------------------------------------------------------------
// Dimension::getName
//----------------------------------------------------------------------------
std::string Dimension::getName()
{
  return m_item->getName();
}
//----------------------------------------------------------------------------
// Dimension::getSize
//----------------------------------------------------------------------------
int Dimension::getSize()
{
  return m_item->getSize();
}
//----------------------------------------------------------------------------
// Dimension::isUnlimited
//----------------------------------------------------------------------------
bool Dimension::isUnlimited()
{
  return false;
}
//----------------------------------------------------------------------------
// Dimension::isVariableLength
//----------------------------------------------------------------------------
bool Dimension::isVariableLength()
{
  return false;
}

//----------------------------------------------------------------------------
// Dimension::isShared
//----------------------------------------------------------------------------
bool Dimension::isShared()
{
  return m_shared;
}
//----------------------------------------------------------------------------
// Dimension::getCoordinateVariable
//----------------------------------------------------------------------------
IArrayPtr Dimension::getCoordinateVariable()
{
  return m_item->getData();
}

//----------------------------------------------------------------------------
// Dimension::setUnlimited
//----------------------------------------------------------------------------
void Dimension::setUnlimited(bool)
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Dimension::setUnlimited");
}

//----------------------------------------------------------------------------
// Dimension::setVariableLength
//----------------------------------------------------------------------------
void Dimension::setVariableLength(bool)
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Dimension::setVariableLength");
}

//----------------------------------------------------------------------------
// Dimension::setShared
//----------------------------------------------------------------------------
void Dimension::setShared(bool value)
{
  m_shared = value;
}

//----------------------------------------------------------------------------
// Dimension::setSize
//----------------------------------------------------------------------------
void Dimension::setSize(int)
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Dimension::setLength");
}

//----------------------------------------------------------------------------
// Dimension::setName
//----------------------------------------------------------------------------
void Dimension::setName(const std::string& name)
{
  m_item->setName(name);
}

//----------------------------------------------------------------------------
// Dimension::setCoordinateVariable
//----------------------------------------------------------------------------
void Dimension::setCoordinateVariable(const cdma::IArrayPtr& array) throw ( cdma::Exception )
{
  m_item->setData(array);
}

//----------------------------------------------------------------------------
// Dimension::getDimensionAxis
//----------------------------------------------------------------------------
int Dimension::getDimensionAxis()
{
  IAttributePtr attr = m_item->getAttribute("axis");
  if( attr )
  {
    return attr->getValue<long>();
  }
  else
  {
    return -1;
  }
}

//----------------------------------------------------------------------------
// Dimension::getDisplayOrder
//----------------------------------------------------------------------------
int Dimension::getDisplayOrder()
{
  IAttributePtr attr = m_item->getAttribute("primary");
  if( attr )
  {
    return attr->getValue<long>();
  }
  else
  {
    return -1;
  }
}

//----------------------------------------------------------------------------
// Dimension::getUnit
//----------------------------------------------------------------------------
std::string Dimension::getUnit()
{
  return m_item->getUnit();
}

//---------------------------------------------------------------------------
// DataItem::setDimensionAxis
//---------------------------------------------------------------------------
void Dimension::setDimensionAxis(int index)
{
  IAttributePtr attr = m_item->getAttribute("axis");
  if( attr )
  {
    attr->setValue(index);
  }
}

//---------------------------------------------------------------------------
// Dimension::setDisplayOrder
//---------------------------------------------------------------------------
void Dimension::setDisplayOrder(int order)
{
  IAttributePtr attr = m_item->getAttribute("primary");
  if( attr )
  {
    attr->setValue(order);
  }
}

}
}
