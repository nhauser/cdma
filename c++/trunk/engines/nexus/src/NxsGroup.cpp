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
    NxsDatasetPtr dataset_ptr = m_dataset_wptr.lock();

    NexusFilePtr ptrFile = dataset_ptr->getHandle();
    NexusFileAccess auto_open(ptrFile);

    std::vector<std::string> datasets, groups, classes;
    ptrFile->OpenGroupPath(PSZ(m_strPath), true);
    ptrFile->GetGroupChildren(&datasets, &groups, &classes);
    
    for(yat::uint16 ui=0; ui < datasets.size(); ui++)
      cdma::IDataItemPtr ptrDataItem = getDataItem(datasets[ui]);
      
    for(yat::uint16 ui=0; ui < groups.size(); ui++)
      cdma::IGroupPtr ptrGroup = getGroup(groups[ui]);

    m_bChildren = true;
  }
  catch( NexusException &e )
  {
    throw cdma::Exception(e);
  }
}

//-----------------------------------------------------------------------------
// NxsGroup::PrivEnumAttributes
//-----------------------------------------------------------------------------
void NxsGroup::PrivEnumAttributes()
{
  CDMA_FUNCTION_TRACE("NxsGroup::PrivEnumAttributes");
  try
  {
    NxsDatasetPtr dataset_ptr = m_dataset_wptr.lock();
    // Get handle
    NexusFilePtr ptrNxFile = dataset_ptr->getHandle();
    NexusFileAccess auto_open (ptrNxFile);
    
    // Opening path
    ptrNxFile->OpenGroupPath(PSZ(m_strPath), true);
    yat::uint16 nbAttr = ptrNxFile->AttrCount();
    
    if( nbAttr > 0 )
    {
      // First step
      NexusAttrInfo *pAttrInfo = new NexusAttrInfo();
      ptrNxFile->GetFirstAttribute(pAttrInfo);

      // Create cdma Attribute
      yat::String attrName = pAttrInfo->AttrName();
      IAttributePtr attribute( new NxsAttribute( ptrNxFile, pAttrInfo ) );
      m_attributes_map[attrName] = attribute;

      // Iterating on every following attribute
      for( yat::uint16 ui = 1; ui < nbAttr; ui++ )
      {
        // Create a new nexus attribute info
        pAttrInfo = new NexusAttrInfo();
        ptrNxFile->GetNextAttribute(pAttrInfo);

        // Create cdma Attribute
        attrName = pAttrInfo->AttrName();
        attribute.reset( new NxsAttribute( ptrNxFile, pAttrInfo ) );
        m_attributes_map[attrName] = attribute;
      }
    }
    m_attributes_loaded = true;
  }
  catch( NexusException &e )
  {
    throw cdma::Exception(e);
  }

}

//-----------------------------------------------------------------------------
// NxsGroup::NxsGroup
//-----------------------------------------------------------------------------
NxsGroup::NxsGroup(NxsDatasetWPtr dataset_wptr)
{
  m_dataset_wptr = dataset_wptr;
  m_bChildren = false;
  m_root_wptr = NULL;
  m_parent_wptr = NULL;
}
NxsGroup::NxsGroup(NxsDatasetWPtr dataset_wptr, const yat::String& full_Path): m_strPath(full_Path)
{
  m_dataset_wptr = dataset_wptr;
  m_root_wptr = NULL;
  m_parent_wptr = NULL;
  m_bChildren = false;
}
NxsGroup::NxsGroup(NxsDatasetWPtr dataset_wptr, const yat::String& parent_path, const yat::String& name)
{
  m_dataset_wptr = dataset_wptr;
  m_root_wptr = NULL;
  m_parent_wptr = NULL;
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
cdma::IDataItemPtr NxsGroup::addDataItem(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::addDataItem");
}

//-----------------------------------------------------------------------------
// NxsGroup::getParent
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::getParent() const
{
  yat::String strPath = m_strPath, strTmp;
  strPath.extract_token_right('/', &strTmp);
  cdma::IGroupPtr ptrGroup = m_dataset_wptr.lock()->getGroupFromPath(strPath);
  if( !m_parent_wptr )
    m_parent_wptr = ptrGroup;
  return ptrGroup;
  }

//-----------------------------------------------------------------------------
// NxsGroup::getRoot
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::getRoot() const
{
  return m_dataset_wptr.lock()->getRootGroup();
}

//-----------------------------------------------------------------------------
// NxsGroup::addDimension
//-----------------------------------------------------------------------------
cdma::IDimensionPtr NxsGroup::addDimension(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::addOneDimension");
}

//-----------------------------------------------------------------------------
// NxsGroup::addSubgroup
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::addSubgroup(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::addSubgroup");
}

//-----------------------------------------------------------------------------
// NxsGroup::getDataItem
//-----------------------------------------------------------------------------
cdma::IDataItemPtr NxsGroup::getDataItem(const std::string& shortName) throw ( cdma::Exception )
{
  if( !m_ptrNxFile )
    THROW_NO_DATA("No NeXus file", "NxsGroup::getDataItem");
  
  std::map<yat::String, cdma::IDataItemPtr>::iterator it = m_mapDataItems.find(shortName);
  if( it != m_mapDataItems.end() )
    return it->second;

  cdma::IDataItemPtr ptrDataItem = m_dataset_wptr.lock()->getItemFromPath(m_strPath, shortName);
  m_mapDataItems[shortName] = ptrDataItem;
  return ptrDataItem;
}

//-----------------------------------------------------------------------------
// NxsGroup::getDataItemWithAttribute
//-----------------------------------------------------------------------------
cdma::IDataItemPtr NxsGroup::getDataItemWithAttribute(const std::string&, const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::getDataItemWithAttribute");
}

//-----------------------------------------------------------------------------
// NxsGroup::getDimension
//-----------------------------------------------------------------------------
cdma::IDimensionPtr NxsGroup::getDimension(const std::string&)
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

  cdma::IGroupPtr ptrGroup = m_dataset_wptr.lock()->getGroupFromPath(NxsDataset::concatPath(m_strPath, shortName));
  m_mapGroups[shortName] = ptrGroup;
  return ptrGroup;
}

//-----------------------------------------------------------------------------
// NxsGroup::getGroupWithAttribute
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::getGroupWithAttribute(const std::string&, const std::string&)
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
bool NxsGroup::removeDataItem(const cdma::IDataItemPtr&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeDataItem");
}

//-----------------------------------------------------------------------------
// NxsGroup::removeDataItem
//-----------------------------------------------------------------------------
bool NxsGroup::removeDataItem(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeDataItem");
}

//-----------------------------------------------------------------------------
// NxsGroup::removeDimension
//-----------------------------------------------------------------------------
bool NxsGroup::removeDimension(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeDimension");
}

//-----------------------------------------------------------------------------
// NxsGroup::removeGroup
//-----------------------------------------------------------------------------
bool NxsGroup::removeGroup(const cdma::IGroupPtr&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeGroup");
}

//-----------------------------------------------------------------------------
// NxsGroup::removeGroup
//-----------------------------------------------------------------------------
bool NxsGroup::removeGroup(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeGroup");
}

//-----------------------------------------------------------------------------
// NxsGroup::removeDimension
//-----------------------------------------------------------------------------
bool NxsGroup::removeDimension(const cdma::IDimensionPtr&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::removeDimension");
}

//-----------------------------------------------------------------------------
// NxsGroup::isRoot
//-----------------------------------------------------------------------------
bool NxsGroup::isRoot() const
{
  THROW_NOT_IMPLEMENTED("NxsGroup::isRoot");
}

//-----------------------------------------------------------------------------
// NxsGroup::isEntry
//-----------------------------------------------------------------------------
bool NxsGroup::isEntry() const
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
// NxsGroup::addAttribute
//-----------------------------------------------------------------------------
cdma::IAttributePtr NxsGroup::addAttribute(const std::string&, yat::Any&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::addOneAttribute");
}

//-----------------------------------------------------------------------------
// NxsGroup::getAttribute
//-----------------------------------------------------------------------------
cdma::IAttributePtr NxsGroup::getAttribute(const std::string& name)
{
  CDMA_FUNCTION_TRACE("NxsGroup::getAttribute");

  if( !m_attributes_loaded )
    PrivEnumAttributes();

  std::map<yat::String, cdma::IAttributePtr>::iterator it = m_attributes_map.find(name);
  if( it != m_attributes_map.end() )
    return it->second;

  return IAttributePtr(NULL);
}

//-----------------------------------------------------------------------------
// NxsGroup::getAttributeList
//-----------------------------------------------------------------------------
std::list<cdma::IAttributePtr> NxsGroup::getAttributeList()
{
  CDMA_FUNCTION_TRACE("NxsGroup::getAttributeList");

  if( !m_attributes_loaded )
    PrivEnumAttributes();
  
  std::map<yat::String, cdma::IAttributePtr>::iterator it = m_attributes_map.begin();
  std::list<cdma::IAttributePtr> attributes;
  for( ; it != m_attributes_map.end(); it++ )
    attributes.push_back(it->second);
  
  return attributes;
}

//-----------------------------------------------------------------------------
// NxsGroup::getLocation
//-----------------------------------------------------------------------------
std::string NxsGroup::getLocation() const
{
  THROW_NOT_IMPLEMENTED("NxsGroup::getLocation");
}

//-----------------------------------------------------------------------------
// NxsGroup::getPath
//-----------------------------------------------------------------------------
std::string NxsGroup::getPath() const
{
  return m_strPath;
}

//-----------------------------------------------------------------------------
// NxsGroup::getName
//-----------------------------------------------------------------------------
std::string NxsGroup::getName() const
{
  yat::String strName, strPath = m_strPath;
  strPath.extract_token_right('/', &strName);
  return strName;
}

//-----------------------------------------------------------------------------
// NxsGroup::getShortName
//-----------------------------------------------------------------------------
std::string NxsGroup::getShortName() const
{
  return getName();
}

//-----------------------------------------------------------------------------
// NxsGroup::hasAttribute
//-----------------------------------------------------------------------------
bool NxsGroup::hasAttribute(const std::string& name)
{
  CDMA_FUNCTION_TRACE("NxsGroup::hasAttribute");
  
  // Get the attribute
  IAttributePtr attribute_ptr = getAttribute(name);

  // If exists check its value
  if( attribute_ptr )
  {
    return true;
  }
  return false;
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

//-----------------------------------------------------------------------------
// NxsGroup::setSelfRef
//-----------------------------------------------------------------------------
void NxsGroup::setSelfRef(const NxsGroupPtr& ptr)
{
  if( ptr.get() != this )
  {
    THROW_INVALID_POINTER("Pointer mismatch", "NxsGroup::setSelfRef");
  }

  m_self_wptr = ptr;
}

} // namespace
