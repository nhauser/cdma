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
#include <NxsDataItem.h>
#include <NxsDimension.h>

namespace cdma
{

//-----------------------------------------------------------------------------
// NxsGroup::PrivEnumChildren
//-----------------------------------------------------------------------------
void NxsGroup::PrivEnumChildren()
{
  CDMA_FUNCTION_TRACE("PrivEnumChildren");
  try
  {
    NxsDataset* dataset_ptr = m_dataset_ptr;

    NexusFilePtr ptrFile = dataset_ptr->getHandle();
    NexusFileAccess auto_open(ptrFile);

    std::vector<std::string> datasets, groups, classes;
    ptrFile->OpenGroupPath(PSZ(m_path), true);
    ptrFile->GetGroupChildren(&datasets, &groups, &classes);
    
    for(yat::uint16 ui=0; ui < datasets.size(); ui++)
    {
      cdma::IDataItemPtr ptrDataItem = dataset_ptr->getItemFromPath(m_path, datasets[ui]);
      if( ptrDataItem->hasAttribute("axis") )
      {
        m_mapDimensions[ptrDataItem->getName()] = new NxsDimension( (NxsDataItemPtr)  ptrDataItem);
      }
      else
      {
        m_mapDataItems[ptrDataItem->getName()] = ptrDataItem;
      }
    }
      
    for(yat::uint16 ui=0; ui < groups.size(); ui++)
    {
      cdma::IGroupPtr ptrGroup = dataset_ptr->getGroupFromPath( m_path + "/" + groups[ui]);
      m_mapGroups[ ptrGroup->getShortName() ] = ptrGroup;
    }

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
    // Get handle
    NexusFilePtr ptrNxFile = m_dataset_ptr->getHandle();
    NexusFileAccess auto_open (ptrNxFile);
    
    // Opening path
    ptrNxFile->OpenGroupPath(PSZ(m_path), true);
    
    if( ptrNxFile->AttrCount() > 0 )
    {
      NexusAttrInfo AttrInfo;

      // Iterating on attributes collection
      for( int rc = ptrNxFile->GetFirstAttribute(&AttrInfo); 
           NX_OK == rc; 
           rc = ptrNxFile->GetNextAttribute(&AttrInfo) )
      {
        // Create cdma Attribute
        m_attributes_map[AttrInfo.AttrName()] = IAttributePtr(new NxsAttribute( ptrNxFile, AttrInfo ) );
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
NxsGroup::NxsGroup(NxsDataset* dataset_ptr)
{
  m_dataset_ptr = dataset_ptr;
  m_bChildren = false;
}
NxsGroup::NxsGroup(NxsDataset* dataset_ptr, const yat::String& full_Path): m_path(full_Path)
{
  m_dataset_ptr = dataset_ptr;
  m_bChildren = false;
}
NxsGroup::NxsGroup(NxsDataset* dataset_ptr, const yat::String& parent_path, const yat::String& name)
{
  m_dataset_ptr = dataset_ptr;
  m_bChildren = false;
  m_path.printf("%s/%s", PSZ(parent_path), PSZ(name));
  m_path.replace("//", "/");
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
cdma::IDataItemPtr NxsGroup::addDataItem(const std::string& name)
{
  IGroupPtr parent = NULL;
  if( m_dataset_ptr != NULL )
  {
    parent = m_dataset_ptr->getGroupFromPath( m_path );
  }
  IDataItemPtr item = new NxsDataItem(m_dataset_ptr, parent, name );
  m_mapDataItems[name] = item;
  return item;
}

//-----------------------------------------------------------------------------
// NxsGroup::getParent
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::getParent() const
{
  IGroupPtr group = NULL;
  if( ! isRoot() )
  {
    yat::String strPath = m_path, strTmp;
    strPath.extract_token_right('/', &strTmp);
    group = m_dataset_ptr->getGroupFromPath(strPath);
  }
  return group;
}

//-----------------------------------------------------------------------------
// NxsGroup::getRoot
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::getRoot() const
{
  return m_dataset_ptr->getRootGroup();
}

//-----------------------------------------------------------------------------
// NxsGroup::addDimension
//-----------------------------------------------------------------------------
cdma::IDimensionPtr NxsGroup::addDimension(const cdma::IDimensionPtr& dim)
{
  m_mapDimensions[dim->getName()] = dim;
  return dim;
}

//-----------------------------------------------------------------------------
// NxsGroup::addSubgroup
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::addSubgroup(const std::string& groupName)
{
  if( !m_bChildren )
    PrivEnumChildren();
  
  IGroupPtr subGroup = new NxsGroup(m_dataset_ptr, m_path, groupName);
  m_mapGroups[subGroup->getShortName()] = subGroup;
  return subGroup;
}

//-----------------------------------------------------------------------------
// NxsGroup::getDataItem
//-----------------------------------------------------------------------------
cdma::IDataItemPtr NxsGroup::getDataItem(const std::string& shortName) throw ( cdma::Exception )
{
  if( !m_bChildren )
    PrivEnumChildren();
 
  std::map<yat::String, cdma::IDataItemPtr>::iterator it = m_mapDataItems.find(shortName);
  if( it != m_mapDataItems.end() )
  {
    return it->second;
  }

  cdma::IDataItemPtr ptrDataItem = m_dataset_ptr->getItemFromPath(m_path, shortName);
  if( ptrDataItem && !ptrDataItem->hasAttribute("axis") )
  {
    m_mapDataItems[shortName] = ptrDataItem;
  }

  return ptrDataItem;
}

//-----------------------------------------------------------------------------
// NxsGroup::getDataItemWithAttribute
//-----------------------------------------------------------------------------
cdma::IDataItemPtr NxsGroup::getDataItemWithAttribute(const std::string& name, const std::string& value)
{
  IDataItemPtr result = NULL;
  std::list<cdma::IDataItemPtr> items = getDataItemList();
  for( std::list<cdma::IDataItemPtr>::iterator iter = items.begin(); iter != items.end(); iter++ ) 
  {
    if( (*iter)->hasAttribute(name) )
    {
      IAttributePtr attr = (*iter)->getAttribute(name);
      if( attr->getStringValue() == value )
      {
        result = *iter;
        break;
      }
    }
  }
  
  return result;
}

//-----------------------------------------------------------------------------
// NxsGroup::getDimension
//-----------------------------------------------------------------------------
cdma::IDimensionPtr NxsGroup::getDimension(const std::string& name)
{
  if( ! m_bChildren )
  {
    PrivEnumChildren();
  }
  
  MapStringDimension::iterator it = m_mapDimensions.find(name);
  if( it != m_mapDimensions.end() )
  {
    return it->second;
  }
  else
  {
    return NULL;
  }
}

//-----------------------------------------------------------------------------
// NxsGroup::getGroup
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::getGroup(const std::string& shortName)
{
  CDMA_FUNCTION_TRACE("NxsGroup::getGroup");
  if( !m_bChildren )
  {
    PrivEnumChildren();
  }
  
  std::map<yat::String, cdma::IGroupPtr>::iterator it = m_mapGroups.find(shortName);
  if( it != m_mapGroups.end() )
    return it->second;

  cdma::IGroupPtr ptrGroup = m_dataset_ptr->getGroupFromPath(NxsDataset::concatPath(m_path, shortName));
  m_mapGroups[ptrGroup->getShortName()] = ptrGroup;
  return ptrGroup;
}

//-----------------------------------------------------------------------------
// NxsGroup::getGroupWithAttribute
//-----------------------------------------------------------------------------
cdma::IGroupPtr NxsGroup::getGroupWithAttribute(const std::string& name, const std::string& value)
{
  IGroupPtr result = NULL;
  std::list<cdma::IGroupPtr> groups = getGroupList();
  for( std::list<cdma::IGroupPtr>::iterator iter = groups.begin(); iter != groups.end(); iter++ ) 
  {
    if( (*iter)->hasAttribute(name) )
    {
      IAttributePtr attr = (*iter)->getAttribute(name);
      if( attr->getStringValue() == value )
      {
        result = *iter;
        break;
      }
    }
  }
  
  return result;
}

//-----------------------------------------------------------------------------
// NxsGroup::getDataItemList
//-----------------------------------------------------------------------------
std::list<cdma::IDataItemPtr> NxsGroup::getDataItemList()
{
  if( !m_bChildren )
    PrivEnumChildren();
  
  std::map<yat::String, cdma::IDataItemPtr>::iterator it = m_mapDataItems.begin();
  std::list<cdma::IDataItemPtr> data_items;
  for( ; it != m_mapDataItems.end(); it++ )
  {
    data_items.push_back(it->second);
  }
  
  return data_items;
}

//-----------------------------------------------------------------------------
// NxsGroup::getDimensionList
//-----------------------------------------------------------------------------
std::list<cdma::IDimensionPtr> NxsGroup::getDimensionList()
{
  if( ! m_bChildren )
  {
    PrivEnumChildren();
  }
  std::list<cdma::IDimensionPtr> dim_list;
  for( MapStringDimension::iterator it = m_mapDimensions.begin(); it != m_mapDimensions.end(); it++ )
  {
    dim_list.push_back(it->second);
  }
  return dim_list;
}

//-----------------------------------------------------------------------------
// NxsGroup::getGroupList
//-----------------------------------------------------------------------------
std::list<cdma::IGroupPtr> NxsGroup::getGroupList()
{
  if( !m_bChildren )
    PrivEnumChildren();
  
  std::map<yat::String, cdma::IGroupPtr>::iterator it = m_mapGroups.begin();
  std::list<cdma::IGroupPtr> groups;
  for( ; it != m_mapGroups.end(); it++ )
  {
    groups.push_back(it->second);
  }
  
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
  return (m_path == "");
}

//-----------------------------------------------------------------------------
// NxsGroup::isEntry
//-----------------------------------------------------------------------------
bool NxsGroup::isEntry() const
{
  IGroupPtr parent = getParent();
  if( parent )
  {
    return parent->isRoot();
  }
  return false;
}

//-----------------------------------------------------------------------------
// NxsGroup::addAttribute
//-----------------------------------------------------------------------------
void NxsGroup::addAttribute(const cdma::IAttributePtr& attr)
{
/*
  NxsAttribute* attr = new NxsAttribute();
  IAttributePtr attribute = attr;
  attr->setName( name );
  attr->setValue( value );
  m_attributes_map[name] = attribute;
  */
  m_attributes_map[attr->getName()] = attr;
//  return attribute;
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
  return m_path;
}

//-----------------------------------------------------------------------------
// NxsGroup::getPath
//-----------------------------------------------------------------------------
std::string NxsGroup::getPath() const
{
  return m_path;
}

//-----------------------------------------------------------------------------
// NxsGroup::getName
//-----------------------------------------------------------------------------
std::string NxsGroup::getName() const
{
  yat::String strName, strPath = m_path;
  strPath.extract_token_right('/', &strName);
  return strName;
}

//-----------------------------------------------------------------------------
// NxsGroup::getShortName
//-----------------------------------------------------------------------------
std::string NxsGroup::getShortName() const
{
  yat::String strClass, strName = getName();
  strName.extract_token_right('<', &strClass);
  
  return strName;
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
bool NxsGroup::removeAttribute(const cdma::IAttributePtr& attr)
{
  MapStringAttribute::iterator iter = m_attributes_map.find( attr->getName() );
  if( iter != m_attributes_map.end() )
  {
    m_attributes_map.erase( iter );
    return true;
  }
  return false;
}

//-----------------------------------------------------------------------------
// NxsGroup::setName
//-----------------------------------------------------------------------------
void NxsGroup::setName(const std::string& name)
{
  yat::String strName, strPath = m_path;
  strPath.extract_token_right('/', &strName);
  strPath = strPath.substr( 0, strPath.size() - strName.size() );
  strPath += name;
}

//-----------------------------------------------------------------------------
// NxsGroup::setShortName
//-----------------------------------------------------------------------------
void NxsGroup::setShortName(const std::string& shortName)
{
  yat::String strClass, strName, strPath = m_path;
  strPath.replace('(', '<');
  strPath.replace(')', '>');
  strPath.extract_token_right('/', &strName);
  strName.extract_token_right('<', &strClass);
  strClass.extract_token('>', &strClass );
  strPath = strPath.substr( 0, strPath.size() - strName.size() );
  strPath += shortName + "<" + strClass + ">";
}

//-----------------------------------------------------------------------------
// NxsGroup::setParent
//-----------------------------------------------------------------------------
void NxsGroup::setParent(const cdma::IGroupPtr&)
{
  THROW_NOT_IMPLEMENTED("NxsGroup::setParent");
}

} // namespace
