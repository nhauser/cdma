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

// Dependencies
#include <yat/file/FileName.h>
#include <yat/utils/String.h>

#include <cdma/Common.h>
#include <cdma/factory/Factory.h>
#include <cdma/dictionary/plugin/Dictionary.h>
#include <cdma/dictionary/ILogicalGroup.h>

#include <NxsDataset.h>
#include <NxsGroup.h>
#include <NxsDataItem.h>

namespace cdma
{
namespace nexus
{

//=============================================================================
//
// Dataset
//
//=============================================================================
//---------------------------------------------------------------------------
// Dataset::Dataset
//---------------------------------------------------------------------------
Dataset::Dataset( const yat::URI& location, IPluginFactory *factory_ptr )
{
  // Utiliser yat pour sortir un FileName et récupérer son contenu (1 ou plusieurs fichiers)
  m_location = location;
  m_factory_ptr = factory_ptr;
  m_phy_root.reset( new Group( this ) );
  CDMA_TRACE( "open file: " + m_location.get(yat::URI::PATH) );
  m_file_handle.reset( new NexusFile( PSZ( m_location.get(yat::URI::PATH) ) ) );
  // Constructor of NexusFile open the file, there is no need to let it opened
  m_file_handle->Close();
}

//---------------------------------------------------------------------------
// Dataset::Dataset
//---------------------------------------------------------------------------
Dataset::Dataset()
{
}

//---------------------------------------------------------------------------
// Dataset::Dataset
//---------------------------------------------------------------------------
Dataset::~Dataset()
{
  CDMA_TRACE("cdma::nexus::Dataset::~Dataset");
}

//---------------------------------------------------------------------------
// Dataset::fullName
//---------------------------------------------------------------------------
yat::String Dataset::concatPath(const yat::String &path, const yat::String& name)
{
  yat::String full_name = PSZ_FMT("%s/%s", PSZ(path), PSZ(name));
  full_name.replace("//", "/");
  return full_name;
}

//---------------------------------------------------------------------------
// Dataset::getItemFromPath
//---------------------------------------------------------------------------
IDataItemPtr Dataset::getItemFromPath(const std::string &fullPath)
{
  CDMA_FUNCTION_TRACE("cdma::nexus::Dataset::getItemFromPath(const std::string &)");
  yat::String strPath = fullPath, strName;
  strPath.extract_token_right('/', &strName);
  return getItemFromPath(strPath, strName);
}

IDataItemPtr Dataset::getItemFromPath(const yat::String& path, const yat::String& name)
{
  CDMA_FUNCTION_TRACE("cdma::nexus::Dataset::getItemFromPath(const yat::String&, const yat::String&)");
  CDMA_TRACE("path: " << path);

  if( m_file_handle.is_null() )
    THROW_NO_DATA("No NeXus file", "cdma_nexus::Dataset::getItemFromPath");
  
  std::map<std::string, IDataItemPtr>::iterator it = m_item_map.find(concatPath(path, name));
  if( it != m_item_map.end() )
    return it->second;

  try
  {
    NexusFileAccess auto_open(m_file_handle);
    if( m_file_handle.is_null() )
      THROW_NO_DATA("No NeXus file", "cdma_nexus::Dataset::getItemFromPath");

    m_file_handle->OpenGroupPath( PSZ(path) );
    NexusDataSetInfo* info = new NexusDataSetInfo();
    m_file_handle->GetDataSetInfo(info, PSZ(name));
    
    // Take in consideration the real path
    yat::String strPath = m_file_handle->CurrentGroupPath();
    strPath.replace('(', '<');
    strPath.replace(')', '>');
    strPath = strPath + name;
    
    // check once again the path is in map
    it = m_item_map.find(strPath);
    if( it != m_item_map.end() )
      return it->second;
      
    // Create corresponding object and stores it
    IDataItemPtr ptrItem = new DataItem(this, *info, strPath);
    m_item_map[strPath] = ptrItem;
    m_file_handle->CloseDataSet();
    return ptrItem;
  }
  catch( NexusException &e )
  {
    RE_THROW_EXCEPTION(e);
  }
}

//---------------------------------------------------------------------------
// Dataset::getGroupFromPath
//---------------------------------------------------------------------------
IGroupPtr Dataset::getGroupFromPath(const std::string &groupPath)
{
  if( m_file_handle.is_null() )
    THROW_NO_DATA("No NeXus file", "cdma_nexus::Dataset::getGroupFromPath");

  yat::String path = groupPath;
  std::map<std::string, IGroupPtr>::iterator it = m_group_map.find(path);
  if( it != m_group_map.end() )
    return it->second;

  try
  {
    // get handle and access on file
    NexusFileAccess auto_open(m_file_handle);

    if( m_file_handle.is_null() )
      THROW_NO_DATA("No NeXus file", "cdma_nexus::Dataset::getGroupFromPath");
  
    // Open the path
    m_file_handle->OpenGroupPath(PSZ(path));

    // Take in consideration the real path
    path = m_file_handle->CurrentGroupPath();
    path = path.substr( 0, path.length() - 1 );
    path.replace('(', '<');
    path.replace(')', '>');

    // check once again the path is in map
    it = m_group_map.find(path);
    if( it != m_group_map.end() )
      return it->second;

    // Create corresponding object and stores it
    IGroupPtr ptrGroup = new Group(this, path);
    m_group_map[path] = ptrGroup;
    return ptrGroup;
  }
  catch( NexusException &e )
  {
    RE_THROW_EXCEPTION(e);
  }
}

//---------------------------------------------------------------------------
// Dataset::privExtractNextPathPart
//---------------------------------------------------------------------------
void Dataset::privExtractNextPathPart(yat::String* path_p, yat::String* name_p, yat::String* class_p)
{
  yat::String group;
  path_p->extract_token('/', &group);

  group.extract_token_right('<', '>', class_p);
  if( class_p->empty() )
    // 2nd chance !
    group.extract_token_right('(', ')', class_p);
  if( class_p->empty() )
    // last chance !
    group.extract_token_right('{', '}', class_p);

  (*name_p) = group;

  (*name_p).trim();
  (*class_p).trim();
}

//---------------------------------------------------------------------------
// Dataset::privMatchingNodes
//---------------------------------------------------------------------------
NexusItemInfoList Dataset::privMatchingNodes(const yat::String& current_path, 
                                             const yat::String& name_pattern, 
                                             const yat::String& class_name)
{
  CDMA_FUNCTION_TRACE("cdma::nexus::Dataset::privMatchingNodes");
  CDMA_TRACE("Path: " << current_path << "  pattern: " << name_pattern << "  class: " << class_name);

  // Get the children list at current level in the NeXus file
  NexusItemInfoList listItemInfo;

  // First look in cache
  NexusItemInfoListCache::const_iterator cit = m_node_cache.find(m_file_handle->CurrentGroupPath());
  if( cit != m_node_cache.end() )
    listItemInfo = cit->second;
  else
    listItemInfo = m_file_handle->GetGroupChildren();

  NexusItemInfoList listItemInfoMatch;

  for( NexusItemInfoList::const_iterator cit = listItemInfo.begin();
       cit != listItemInfo.end(); ++cit )
  {
    bool match = false;
    if( (*cit)->IsGroup() )
    {
      if( ( class_name.empty() && !name_pattern.empty() && name_pattern.match( (*cit)->ItemName() ) ) || 
          ( name_pattern.empty() && class_name.is_equal_no_case( (*cit)->ClassName() ) ) ||
          ( !name_pattern.empty() && name_pattern.match( (*cit)->ItemName() ) &&
            !class_name.empty() && class_name.is_equal_no_case( (*cit)->ClassName() ) )
        )
      {
        match = true;
        CDMA_TRACE("Group matching: " << (*cit)->ItemName() << '(' << (*cit)->ClassName() << ')');
      }
    }
    else if( (*cit)->IsDataSet() )
    {
      if( !name_pattern.empty() && name_pattern.match( (*cit)->ItemName() ) && class_name.empty() )
      {
        match = true;
        CDMA_TRACE("Dataset matching: " << (*cit)->ItemName());
      }
    }

    if( match )
    {
      listItemInfoMatch.push_back( (*cit) );
    }
  }

  m_node_cache[current_path] = listItemInfo;

  return listItemInfoMatch;
}

//---------------------------------------------------------------------------
// Dataset::privFindContainer
//---------------------------------------------------------------------------
IContainerPtrList Dataset::privFindContainer(const std::string& input_path, bool first_only)
{
  CDMA_FUNCTION_TRACE("cdma::nexus::Dataset::privFindContainer");

  yat::String item_name, class_name, path = input_path;

  CDMA_TRACE("input_path :" << input_path);

  IContainerPtrList listContainer;

  // we extract the name and the class's name of the Item.
  privExtractNextPathPart(&path, &item_name, &class_name);

  // get all children that match item & class name
  NexusItemInfoList listItemInfo = privMatchingNodes(path, item_name, class_name);

  // open each of them and continue with next level
  for( NexusItemInfoList::const_iterator cit = listItemInfo.begin();
       cit != listItemInfo.end(); ++cit )
  {
    if( path.empty() )
    {
      if( (*cit)->IsDataSet() )
      {
        CDMA_TRACE("Add matching dataset :" << (*cit)->ItemName());
        IDataItemPtr ptrItem = new DataItem(this, m_file_handle->CurrentGroupPath() + (*cit)->ItemName());
        listContainer.push_back(ptrItem);
      }

      if( (*cit)->IsGroup() )
      {
        CDMA_TRACE("Add matching group :" << (*cit)->ItemName());
        IGroupPtr ptrGroup = new Group(this, m_file_handle->CurrentGroupPath(), (*cit)->ItemName());
        listContainer.push_back(ptrGroup);
      }
    }

    else if( (*cit)->IsGroup() )
    {
      m_file_handle->OpenGroup( (*cit)->ItemName(), (*cit)->ClassName(), true);

      // Recursivaly continue with the remaining content of the initial path
      // Merge the founded containers with the current list
      IContainerPtrList tmp_list = privFindContainer(path, first_only);
      listContainer.splice( listContainer.begin(), tmp_list );

      // Up one level
      m_file_handle->CloseGroup();

      if( first_only && listContainer.size() )
        // Return the first container found
        return listContainer;
    }
  }
  return listContainer;
}

//---------------------------------------------------------------------------
// Dataset::findContainerByPath
//---------------------------------------------------------------------------
IContainerPtr Dataset::findContainerByPath(const std::string& input_path)
{
  CDMA_FUNCTION_TRACE("cdma::nexus::Dataset::findContainerByPath");
  IContainerPtrList containers = findAllContainerByPath(input_path, true);
  if( containers.begin() != containers.end() )
    return *(containers.begin());
  return NULL;
}

//---------------------------------------------------------------------------
// Dataset::findAllContainerByPath
//---------------------------------------------------------------------------
IContainerPtrList Dataset::findAllContainerByPath(const std::string& input_path, bool first_only)
{
  CDMA_FUNCTION_TRACE("cdma::nexus::Dataset::findAllContainerByPath");

  if( m_file_handle.is_null() )
    THROW_NO_DATA("No NeXus file", "cdma_nexus::Dataset::getGroupFromPath");

  // Remove the first character if it's a '/' 
  yat::String path = input_path;
  if( path.start_with("/") )
    path = path.substr(1);

  // Look in cache
  ContainerCache::iterator it = m_container_map.find(path);
  if( it != m_container_map.end() )
    return it->second;

  NexusFileAccess auto_open(m_file_handle);
/*
  if( !getInsidePath().empty() )
    // Go to the dataset's root group
    m_file_handle->OpenGroupPath(PSZ(getInsidePath()));
*/
  IContainerPtrList listContainer = privFindContainer(path, first_only);

  // Add in cache
  m_container_map[path] = listContainer;

  return listContainer;
}

//---------------------------------------------------------------------------
// Dataset::getRootGroup
//---------------------------------------------------------------------------
IGroupPtr Dataset::getRootGroup()
{
  CDMA_FUNCTION_TRACE("cdma::nexus::Dataset::getRootGroup");
  return m_phy_root;
}

//---------------------------------------------------------------------------
// Dataset::getLogicalRoot
//---------------------------------------------------------------------------
ILogicalGroupPtr Dataset::getLogicalRoot()
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::Dataset::getLogicalRoot");
}

//---------------------------------------------------------------------------
// Dataset::getLocation
//---------------------------------------------------------------------------
std::string Dataset::getLocation()
{
  return m_location.get();
}

//---------------------------------------------------------------------------
// Dataset::getTitle
//---------------------------------------------------------------------------
std::string Dataset::getTitle()
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::Dataset::getTitle");
}

//---------------------------------------------------------------------------
// Dataset::setLocation
//---------------------------------------------------------------------------
void Dataset::setLocation(const std::string& location)
{
  close();
  m_location.set(location);
}

//---------------------------------------------------------------------------
// Dataset::setLocation
//---------------------------------------------------------------------------
void Dataset::setLocation(const yat::URI& location)
{
  close();
  m_location = location;
}

//---------------------------------------------------------------------------
// Dataset::setTitle
//---------------------------------------------------------------------------
void Dataset::setTitle(const std::string&)
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::Dataset::setTitle");
}


//---------------------------------------------------------------------------
// Dataset::sync
//---------------------------------------------------------------------------
bool Dataset::sync() throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::Dataset::sync");
}


//---------------------------------------------------------------------------
// Dataset::save
//---------------------------------------------------------------------------
void Dataset::save() throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::Dataset::save");
}


//---------------------------------------------------------------------------
// Dataset::saveTo
//---------------------------------------------------------------------------
void Dataset::saveTo(const std::string&) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::Dataset::saveTo");
}


//---------------------------------------------------------------------------
// Dataset::save
//---------------------------------------------------------------------------
void Dataset::save(const IContainer&) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::Dataset::save");
}


//---------------------------------------------------------------------------
// Dataset::save
//---------------------------------------------------------------------------
void Dataset::save(const std::string&, const IAttributePtr&) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("cdma::nexus::Dataset::save");
}

//---------------------------------------------------------------------------
// Dataset::close
//---------------------------------------------------------------------------
void Dataset::close()
{
  if( m_file_handle )
  {
    m_file_handle->Close();
    m_file_handle.reset();
  }
}

//---------------------------------------------------------------------------
// NexusFileAccess::NexusFileAccess
//---------------------------------------------------------------------------
NexusFileAccess::NexusFileAccess( const NexusFilePtr& handle )
{
  m_file_handle = handle;
  if( m_file_handle )
  { 
    m_file_handle->OpenRead(NULL);
  }
}

//---------------------------------------------------------------------------
// NexusFileAccess::~NexusFileAccess
//---------------------------------------------------------------------------
NexusFileAccess::~NexusFileAccess()
{
  if( m_file_handle )
  { 
    m_file_handle->Close();
  }
}

} // namespace nexus
} // namespace cdma
