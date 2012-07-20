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
#include <cdma/Factory.h>
#include <cdma/dictionary/Dictionary.h>
#include <cdma/dictionary/LogicalGroup.h>

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
Dataset::Dataset( const yat::URI& location, IFactory *factory_ptr )
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
  CDMA_TRACE("cdma_nexus::Dataset::~Dataset");
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
  CDMA_FUNCTION_TRACE("cdma_nexus::Dataset::getItemFromPath(const std::string &)");
  yat::String strPath = fullPath, strName;
  strPath.extract_token_right('/', &strName);
  return getItemFromPath(strPath, strName);
}

IDataItemPtr Dataset::getItemFromPath(const yat::String& path, const yat::String& name)
{
  CDMA_FUNCTION_TRACE("cdma_nexus::Dataset::getItemFromPath(const yat::String&, const yat::String&)");

  if( m_file_handle.is_null() )
    THROW_NO_DATA("No NeXus file", "cdma_nexus::Dataset::getItemFromPath");
  
  std::map<yat::String, IDataItemPtr>::iterator it = m_item_map.find(concatPath(path, name));
  if( it != m_item_map.end() )
    return it->second;

  try
  {
    NexusFileAccess auto_open(m_file_handle);
    if( m_file_handle.is_null() )
      THROW_NO_DATA("No NeXus file", "cdma_nexus::Dataset::getItemFromPath");
  
    m_file_handle->OpenGroupPath(PSZ(path));
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
    throw Exception(e);
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
  std::map<yat::String, IGroupPtr>::iterator it = m_group_map.find(path);
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
    throw Exception(e);
  }
}

//---------------------------------------------------------------------------
// Dataset::getRootGroup
//---------------------------------------------------------------------------
IGroupPtr Dataset::getRootGroup()
{
CDMA_FUNCTION_TRACE("cdma_nexus::Dataset::getRootGroup");
  return m_phy_root;
}

//---------------------------------------------------------------------------
// Dataset::getLogicalRoot
//---------------------------------------------------------------------------
LogicalGroupPtr Dataset::getLogicalRoot()
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Dataset::getLogicalRoot");
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
  THROW_NOT_IMPLEMENTED("cdma_nexus::Dataset::getTitle");
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
  THROW_NOT_IMPLEMENTED("cdma_nexus::Dataset::setTitle");
}


//---------------------------------------------------------------------------
// Dataset::sync
//---------------------------------------------------------------------------
bool Dataset::sync() throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Dataset::sync");
}


//---------------------------------------------------------------------------
// Dataset::save
//---------------------------------------------------------------------------
void Dataset::save() throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Dataset::save");
}


//---------------------------------------------------------------------------
// Dataset::saveTo
//---------------------------------------------------------------------------
void Dataset::saveTo(const std::string&) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Dataset::saveTo");
}


//---------------------------------------------------------------------------
// Dataset::save
//---------------------------------------------------------------------------
void Dataset::save(const IContainer&) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Dataset::save");
}


//---------------------------------------------------------------------------
// Dataset::save
//---------------------------------------------------------------------------
void Dataset::save(const std::string&, const IAttributePtr&) throw ( Exception )
{
  THROW_NOT_IMPLEMENTED("cdma_nexus::Dataset::save");
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
