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

//-----------------------------------------------------------------------------
// DEPENDENCIES
//-----------------------------------------------------------------------------
#include <cdma/exception/Exception.h>
#include <NxsGroup.h>

namespace cdma
{

//-----------------------------------------------------------------------------
// NxsGroup::PrivEnumChildren
//-----------------------------------------------------------------------------
void NxsGroup::PrivEnumChildren()
{
  try
  {
    std::vector<std::string> datasets, groups, classes;
    m_pDataset->getHandle()->OpenGroupPath(PSZ(m_strPath), true);
    m_ptrNxFile->GetGroupChildren(&datasets, &groups, &classes);
    
    for(yat::uint16 ui=0; ui < datasets.size(); ui++)
      cdma::IDataItemPtr ptrDataItem = getDataItem(datasets[ui]);
      
    for(yat::uint16 ui=0; ui < groups.size(); ui++)
      cdma::IGroupPtr ptrGroup = getGroup(groups[ui]);

    m_bChildren = true;
    m_ptrNxFile->Close();
  }
  catch( NexusException &e )
  {
    throw cdma::Exception(e);
  }
}

//-----------------------------------------------------------------------------
// NxsGroup::NxsGroup
//-----------------------------------------------------------------------------
NxsGroup::NxsGroup(NxsDataset* pDataset)
{
  m_pDataset = pDataset;
  m_bChildren = false;
  m_pRootGroup = NULL;
  m_pParentGroup = NULL;
}
NxsGroup::NxsGroup(NxsDataset* pDataset, const yat::String& full_Path): m_strPath(full_Path)
{
  m_pDataset = pDataset;
  m_pRootGroup = NULL;
  m_pParentGroup = NULL;
  m_bChildren = false;
}
NxsGroup::NxsGroup(NxsDataset* pDataset, const yat::String& parent_path, const yat::String& name)
{
  m_pDataset = pDataset;
  m_pRootGroup = NULL;
  m_pParentGroup = NULL;
  m_bChildren = false;
  m_strPath.printf("%s/%s", PSZ(parent_path), PSZ(name));
  m_strPath.replace("//", "/");
}

//-----------------------------------------------------------------------------
// NxsGroup::~NxsGroup
//-----------------------------------------------------------------------------
NxsGroup::~NxsGroup()
{
}

//-----------------------------------------------------------------------------
// NxsGroup::addDataItem
//-----------------------------------------------------------------------------
void NxsGroup::addDataItem(const cdma::IDataItemPtr& v)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::addDataItem");
}

//-----------------------------------------------------------------------------
// NxsGroup::getParent
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::getParent()
{
  yat::String strPath = m_strPath, strTmp;
  strPath.extract_token_right('/', &strTmp);
  cdma::IGroupPtr ptrGroup = m_pDataset->getGroupFromPath(strPath);
  if( !m_pParentGroup )
    m_pParentGroup = ptrGroup.get();
  return ptrGroup;
  }

//-----------------------------------------------------------------------------
// NxsGroup::getRoot
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::getRoot()
{
  return m_pDataset->getRootGroup();
}

//-----------------------------------------------------------------------------
// NxsGroup::addOneDimension
//-----------------------------------------------------------------------------
void NxsGroup::addOneDimension(const cdma::IDimensionPtr& dimension) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsGroup::addOneDimension");
}

//-----------------------------------------------------------------------------
// NxsGroup::addOneDimension
//-----------------------------------------------------------------------------
void NxsGroup::addSubgroup(const cdma::IGroupPtr& group) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsGroup::addSubgroup");
}

//-----------------------------------------------------------------------------
// NxsGroup::getDataItem
//-----------------------------------------------------------------------------
cdma::IDataItemPtr NxsGroup::getDataItem(const std::string& shortName) throw ( cdma::Exception )
{
  if( m_ptrNxFile.is_null() )
    THROW_NO_DATA("No NeXus file", "NxsGroup::getDataItem");
  
  std::map<yat::String, cdma::IDataItemPtr>::iterator it = m_mapDataItems.find(shortName);
  if( it != m_mapDataItems.end() )
    return it->second;

  cdma::IDataItemPtr ptrDataItem = m_pDataset->getItemFromPath(m_strPath, shortName);
  m_mapDataItems[shortName] = ptrDataItem;
  return ptrDataItem;
}

//-----------------------------------------------------------------------------
// NxsGroup::getDataItemWithAttribute
//-----------------------------------------------------------------------------
cdma::IDataItemPtr NxsGroup::getDataItemWithAttribute(const std::string& name, const std::string& value)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::getDataItemWithAttribute");
}

//-----------------------------------------------------------------------------
// NxsGroup::getDimension
//-----------------------------------------------------------------------------
cdma::IDimensionPtr NxsGroup::getDimension(const std::string& name)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::getDataItemWithAttribute");
}

//-----------------------------------------------------------------------------
// NxsGroup::getGroup
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::getGroup(const std::string& shortName)
{
  if( m_ptrNxFile.is_null() )
    THROW_NO_DATA("No NeXus file", "NxsGroup::getDataItemWithAttribute");
  
  std::map<yat::String, cdma::IGroupPtr>::iterator it = m_mapGroups.find(shortName);
  if( it != m_mapGroups.end() )
    return it->second;

  cdma::IGroupPtr ptrGroup = m_pDataset->getGroupFromPath(NxsDataset::concatPath(m_strPath, shortName));
  m_mapGroups[shortName] = ptrGroup;
  return ptrGroup;
}

//-----------------------------------------------------------------------------
// NxsGroup::getGroupWithAttribute
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::getGroupWithAttribute(const std::string& attributeName, const std::string& value)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::getDataItemWithAttribute");
}

//-----------------------------------------------------------------------------
// NxsGroup::getDataItemList
//-----------------------------------------------------------------------------
std::list<cdma::IDataItemPtr> NxsGroup::getDataItemList()
{
  if( m_ptrNxFile.is_null() )
    THROW_NO_DATA("No NeXus file", "NxsGroup::getDataItemList");

  if( !m_bChildren )
    PrivEnumChildren();
  
  std::map<yat::String, cdma::IDataItemPtr>::iterator it = m_mapDataItems.begin();
  std::list<cdma::IDataItemPtr> data_items;
  for( ; it != m_mapDataItems.end(); it++ )
    data_items.push_back(it->second);
  
  return data_items;
}

//-----------------------------------------------------------------------------
// NxsGroup::getDimensionList
//-----------------------------------------------------------------------------
std::list<cdma::IDimensionPtr> NxsGroup::getDimensionList()
{
  THROW_NOT_IMPLEMENTED("NxsGroup::getDimensionList");
}

//-----------------------------------------------------------------------------
// NxsGroup::getGroupList
//-----------------------------------------------------------------------------
std::list<cdma::IGroupPtr> NxsGroup::getGroupList()
{
  if( m_ptrNxFile.is_null() )
    THROW_NO_DATA("No NeXus file", "NxsGroup::getDataItemList");

  if( !m_bChildren )
    PrivEnumChildren();
  
  std::map<yat::String, cdma::IGroupPtr>::iterator it = m_mapGroups.begin();
  std::list<cdma::IGroupPtr> groups;
  for( ; it != m_mapGroups.end(); it++ )
    groups.push_back(it->second);
  
  return groups;
}

//-----------------------------------------------------------------------------
// NxsGroup::removeDataItem
//-----------------------------------------------------------------------------
bool NxsGroup::removeDataItem(const cdma::IDataItemPtr& item)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeDataItem");
}

//-----------------------------------------------------------------------------
// NxsGroup::removeDataItem
//-----------------------------------------------------------------------------
bool NxsGroup::removeDataItem(const std::string& varName)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeDataItem");
}

//-----------------------------------------------------------------------------
// NxsGroup::removeDimension
//-----------------------------------------------------------------------------
bool NxsGroup::removeDimension(const std::string& dimName)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeDimension");
}

//-----------------------------------------------------------------------------
// NxsGroup::removeGroup
//-----------------------------------------------------------------------------
bool NxsGroup::removeGroup(const cdma::IGroupPtr& group)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeGroup");
}

//-----------------------------------------------------------------------------
// NxsGroup::removeGroup
//-----------------------------------------------------------------------------
bool NxsGroup::removeGroup(const std::string& shortName)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeGroup");
}

//-----------------------------------------------------------------------------
// NxsGroup::removeDimension
//-----------------------------------------------------------------------------
bool NxsGroup::removeDimension(const cdma::IDimensionPtr& dimension)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeDimension");
}

//-----------------------------------------------------------------------------
// NxsGroup::isRoot
//-----------------------------------------------------------------------------
bool NxsGroup::isRoot()
{
  THROW_NOT_IMPLEMENTED("NxsGroup::isRoot");
}

//-----------------------------------------------------------------------------
// NxsGroup::isEntry
//-----------------------------------------------------------------------------
bool NxsGroup::isEntry()
{
  THROW_NOT_IMPLEMENTED("NxsGroup::isEntry");
}

//-----------------------------------------------------------------------------
// NxsGroup::clone
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::clone()
{
  THROW_NOT_IMPLEMENTED("NxsGroup::clone");
}

//-----------------------------------------------------------------------------
// NxsGroup::addOneAttribute
//-----------------------------------------------------------------------------
void NxsGroup::addOneAttribute(const cdma::IAttributePtr&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::addOneAttribute");
}

//-----------------------------------------------------------------------------
// NxsGroup::addStringAttribute
//-----------------------------------------------------------------------------
void NxsGroup::addStringAttribute(const std::string&, const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::addStringAttribute");
}

//-----------------------------------------------------------------------------
// NxsGroup::getAttribute
//-----------------------------------------------------------------------------
cdma::IAttributePtr NxsGroup::getAttribute(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::getAttribute");
}

//-----------------------------------------------------------------------------
// NxsGroup::getAttributeList
//-----------------------------------------------------------------------------
std::list<cdma::IAttributePtr> NxsGroup::getAttributeList()
{
  THROW_NOT_IMPLEMENTED("NxsGroup::getAttributeList");
}

//-----------------------------------------------------------------------------
// NxsGroup::getLocation
//-----------------------------------------------------------------------------
std::string NxsGroup::getLocation()
{
  THROW_NOT_IMPLEMENTED("NxsGroup::getLocation");
}

//-----------------------------------------------------------------------------
// NxsGroup::getPath
//-----------------------------------------------------------------------------
std::string NxsGroup::getPath()
{
  return m_strPath;
}

//-----------------------------------------------------------------------------
// NxsGroup::getName
//-----------------------------------------------------------------------------
std::string NxsGroup::getName()
{
  yat::String strName, strPath = m_strPath;
  strPath.extract_token_right('/', &strName);
  return strName;
}

//-----------------------------------------------------------------------------
// NxsGroup::getShortName
//-----------------------------------------------------------------------------
std::string NxsGroup::getShortName()
{
  return getName();
}

//-----------------------------------------------------------------------------
// NxsGroup::hasAttribute
//-----------------------------------------------------------------------------
bool NxsGroup::hasAttribute(const std::string&, const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::hasAttribute");
}

//-----------------------------------------------------------------------------
// NxsGroup::removeAttribute
//-----------------------------------------------------------------------------
bool NxsGroup::removeAttribute(const cdma::IAttributePtr&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeAttribute");
}

//-----------------------------------------------------------------------------
// NxsGroup::setName
//-----------------------------------------------------------------------------
void NxsGroup::setName(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::setName");
}

//-----------------------------------------------------------------------------
// NxsGroup::setShortName
//-----------------------------------------------------------------------------
void NxsGroup::setShortName(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::setShortName");
}

//-----------------------------------------------------------------------------
// NxsGroup::setParent
//-----------------------------------------------------------------------------
void NxsGroup::setParent(const cdma::IGroupPtr&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::setParent");
}

} // namespace
