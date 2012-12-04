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
namespace nexus
{

//-----------------------------------------------------------------------------
// Group::PrivEnumChildren
//-----------------------------------------------------------------------------
void Group::PrivEnumChildren()
{
  CDMA_FUNCTION_TRACE("PrivEnumChildren");
  try
  {
    Dataset* dataset_ptr = m_dataset_ptr;

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
#ifdef CDMA_STD_SMART_PTR
        m_mapDimensions[ptrDataItem->getName()] = IDimensionPtr(new Dimension( 
                                                  std::dynamic_pointer_cast<DataItem>(ptrDataItem
                                                      )));
#else
        m_mapDimensions[ptrDataItem->getName()] = new Dimension( (DataItemPtr)ptrDataItem );
#endif
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
    RE_THROW_EXCEPTION(e);
  }
}

//-----------------------------------------------------------------------------
// Group::PrivEnumAttributes
//-----------------------------------------------------------------------------
void Group::PrivEnumAttributes()
{
  CDMA_FUNCTION_TRACE("Group::PrivEnumAttributes");
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
        m_attributes_map[AttrInfo.AttrName()] = IAttributePtr(new Attribute( ptrNxFile, AttrInfo ) );
      }
    }
    m_attributes_loaded = true;
  }
  catch( NexusException &e )
  {
    RE_THROW_EXCEPTION(e);
  }

}

//-----------------------------------------------------------------------------
// Group::Group
//-----------------------------------------------------------------------------
Group::Group(Dataset* dataset_ptr)
{
  m_dataset_ptr = dataset_ptr;
  m_bChildren = false;
}
Group::Group(Dataset* dataset_ptr, const yat::String& full_Path): m_path(full_Path)
{
  m_dataset_ptr = dataset_ptr;
  m_bChildren = false;
}
Group::Group(Dataset* dataset_ptr, const yat::String& parent_path, const yat::String& name)
{
  m_dataset_ptr = dataset_ptr;
  m_bChildren = false;
  m_path.printf("%s/%s", PSZ(parent_path), PSZ(name));
  m_path.replace("//", "/");
}

//-----------------------------------------------------------------------------
// Group::~Group
//-----------------------------------------------------------------------------
Group::~Group()
{
}

//-----------------------------------------------------------------------------
// Group::addDataItem
//-----------------------------------------------------------------------------
cdma::IDataItemPtr Group::addDataItem(const std::string& name)
{
  IGroupPtr parent = NULL;
  if( m_dataset_ptr != NULL )
  {
    parent = m_dataset_ptr->getGroupFromPath( m_path );
  }
  IDataItemPtr item(new DataItem(m_dataset_ptr, parent, name ));
  m_mapDataItems[name] = item;
  return item;
}

//-----------------------------------------------------------------------------
// Group::getParent
//-----------------------------------------------------------------------------
cdma::IGroupPtr Group::getParent() const
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
// Group::getRoot
//-----------------------------------------------------------------------------
cdma::IGroupPtr Group::getRoot() const
{
  return m_dataset_ptr->getRootGroup();
}

//-----------------------------------------------------------------------------
// Group::addDimension
//-----------------------------------------------------------------------------
cdma::IDimensionPtr Group::addDimension(const cdma::IDimensionPtr& dim)
{
  m_mapDimensions[dim->getName()] = dim;
  return dim;
}

//-----------------------------------------------------------------------------
// Group::addSubgroup
//-----------------------------------------------------------------------------
cdma::IGroupPtr Group::addSubgroup(const std::string& groupName)
{
  if( !m_bChildren )
    PrivEnumChildren();
  
  IGroupPtr subGroup (new Group(m_dataset_ptr, m_path, groupName));
  m_mapGroups[subGroup->getShortName()] = subGroup;
  return subGroup;
}

//-----------------------------------------------------------------------------
// Group::getDataItem
//-----------------------------------------------------------------------------
cdma::IDataItemPtr Group::getDataItem(const std::string& shortName) throw ( cdma::Exception )
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
// Group::getDataItemWithAttribute
//-----------------------------------------------------------------------------
cdma::IDataItemPtr Group::getDataItemWithAttribute(const std::string& name, const std::string& value)
{
  IDataItemPtr result = NULL;
  std::list<cdma::IDataItemPtr> items = getDataItemList();
  for( std::list<cdma::IDataItemPtr>::iterator iter = items.begin(); iter != items.end(); iter++ ) 
  {
    if( (*iter)->hasAttribute(name) )
    {
      IAttributePtr attr = (*iter)->getAttribute(name);
      if( attr->getValue<std::string>() == value )
      {
        result = *iter;
        break;
      }
    }
  }
  
  return result;
}

//-----------------------------------------------------------------------------
// Group::getDimension
//-----------------------------------------------------------------------------
cdma::IDimensionPtr Group::getDimension(const std::string& name)
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
// Group::getGroup
//-----------------------------------------------------------------------------
cdma::IGroupPtr Group::getGroup(const std::string& shortName)
{
  CDMA_FUNCTION_TRACE("cdma_nexus::Group::getGroup");
  if( !m_bChildren )
  {
    PrivEnumChildren();
  }
  
  std::map<yat::String, cdma::IGroupPtr>::iterator it = m_mapGroups.find(shortName);
  if( it != m_mapGroups.end() )
    return it->second;

  cdma::IGroupPtr ptrGroup = m_dataset_ptr->getGroupFromPath(Dataset::concatPath(m_path, shortName));
  m_mapGroups[ptrGroup->getShortName()] = ptrGroup;
  return ptrGroup;
}

//-----------------------------------------------------------------------------
// Group::getGroupWithAttribute
//-----------------------------------------------------------------------------
cdma::IGroupPtr Group::getGroupWithAttribute(const std::string& name, const std::string& value)
{
  IGroupPtr result = NULL;
  std::list<cdma::IGroupPtr> groups = getGroupList();
  for( std::list<cdma::IGroupPtr>::iterator iter = groups.begin(); iter != groups.end(); iter++ ) 
  {
    if( (*iter)->hasAttribute(name) )
    {
      IAttributePtr attr = (*iter)->getAttribute(name);
      if( attr->getValue<std::string>() == value )
      {
        result = *iter;
        break;
      }
    }
  }
  
  return result;
}

//-----------------------------------------------------------------------------
// Group::getDataItemList
//-----------------------------------------------------------------------------
std::list<cdma::IDataItemPtr> Group::getDataItemList()
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
// Group::getDimensionList
//-----------------------------------------------------------------------------
std::list<cdma::IDimensionPtr> Group::getDimensionList()
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
// Group::getGroupList
//-----------------------------------------------------------------------------
std::list<cdma::IGroupPtr> Group::getGroupList()
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
// Group::removeDataItem
//-----------------------------------------------------------------------------
bool Group::removeDataItem(const cdma::IDataItemPtr&)
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Group::removeDataItem");
}

//-----------------------------------------------------------------------------
// Group::removeDataItem
//-----------------------------------------------------------------------------
bool Group::removeDataItem(const std::string&)
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Group::removeDataItem");
}

//-----------------------------------------------------------------------------
// Group::removeDimension
//-----------------------------------------------------------------------------
bool Group::removeDimension(const std::string&)
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Group::removeDimension");
}

//-----------------------------------------------------------------------------
// Group::removeGroup
//-----------------------------------------------------------------------------
bool Group::removeGroup(const cdma::IGroupPtr&)
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Group::removeGroup");
}

//-----------------------------------------------------------------------------
// Group::removeGroup
//-----------------------------------------------------------------------------
bool Group::removeGroup(const std::string&)
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Group::removeGroup");
}

//-----------------------------------------------------------------------------
// Group::removeDimension
//-----------------------------------------------------------------------------
bool Group::removeDimension(const cdma::IDimensionPtr&)
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Group::removeDimension");
}

//-----------------------------------------------------------------------------
// Group::isRoot
//-----------------------------------------------------------------------------
bool Group::isRoot() const
{
  return (m_path == "");
}

//-----------------------------------------------------------------------------
// Group::isEntry
//-----------------------------------------------------------------------------
bool Group::isEntry() const
{
  IGroupPtr parent = getParent();
  if( parent )
  {
    return parent->isRoot();
  }
  return false;
}

//-----------------------------------------------------------------------------
// Group::addAttribute
//-----------------------------------------------------------------------------
void Group::addAttribute(const cdma::IAttributePtr& attr)
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
// Group::getAttribute
//-----------------------------------------------------------------------------
cdma::IAttributePtr Group::getAttribute(const std::string& name)
{
  CDMA_FUNCTION_TRACE("cdma_nexus::Group::getAttribute");

  if( !m_attributes_loaded )
    PrivEnumAttributes();

  std::map<yat::String, cdma::IAttributePtr>::iterator it = m_attributes_map.find(name);
  if( it != m_attributes_map.end() )
    return it->second;

  return IAttributePtr(NULL);
}

//-----------------------------------------------------------------------------
// Group::getAttributeList
//-----------------------------------------------------------------------------
std::list<cdma::IAttributePtr> Group::getAttributeList()
{
  CDMA_FUNCTION_TRACE("cdma_nexus::Group::getAttributeList");

  if( !m_attributes_loaded )
    PrivEnumAttributes();
  
  std::map<yat::String, cdma::IAttributePtr>::iterator it = m_attributes_map.begin();
  std::list<cdma::IAttributePtr> attributes;
  for( ; it != m_attributes_map.end(); it++ )
    attributes.push_back(it->second);
  
  return attributes;
}

//-----------------------------------------------------------------------------
// Group::getLocation
//-----------------------------------------------------------------------------
std::string Group::getLocation() const
{
  return m_path;
}

//-----------------------------------------------------------------------------
// Group::getPath
//-----------------------------------------------------------------------------
std::string Group::getPath() const
{
  return m_path;
}

//-----------------------------------------------------------------------------
// Group::getName
//-----------------------------------------------------------------------------
std::string Group::getName() const
{
  yat::String strName, strPath = m_path;
  strPath.extract_token_right('/', &strName);
  return strName;
}

//-----------------------------------------------------------------------------
// Group::getShortName
//-----------------------------------------------------------------------------
std::string Group::getShortName() const
{
  yat::String strClass, strName = getName();
  strName.extract_token_right('<', &strClass);
  
  return strName;
}

//-----------------------------------------------------------------------------
// Group::hasAttribute
//-----------------------------------------------------------------------------
bool Group::hasAttribute(const std::string& name)
{
  CDMA_FUNCTION_TRACE("cdma_nexus::Group::hasAttribute");
  
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
// Group::removeAttribute
//-----------------------------------------------------------------------------
bool Group::removeAttribute(const cdma::IAttributePtr& attr)
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
// Group::setName
//-----------------------------------------------------------------------------
void Group::setName(const std::string& name)
{
  yat::String strName, strPath = m_path;
  strPath.extract_token_right('/', &strName);
  strPath = strPath.substr( 0, strPath.size() - strName.size() );
  strPath += name;
}

//-----------------------------------------------------------------------------
// Group::setShortName
//-----------------------------------------------------------------------------
void Group::setShortName(const std::string& shortName)
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
// Group::setParent
//-----------------------------------------------------------------------------
void Group::setParent(const cdma::IGroupPtr&)
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Group::setParent");
}

} // namespace nexus
} // namespace cdma
