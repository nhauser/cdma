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

//=============================================================================
//
// NxsDataset
//
//=============================================================================
//---------------------------------------------------------------------------
// NxsDataset::NxsDataset
//---------------------------------------------------------------------------
NxsDataset::NxsDataset( const yat::URI& location, IFactory *factory_ptr )
{
  // Utiliser yat pour sortir un FileName et récupérer son contenu (1 ou plusieurs fichiers)
  m_location = location;
  m_factory_ptr = factory_ptr;
  m_phy_root.reset( new NxsGroup( this ) );
  CDMA_TRACE( "open file: " + m_location.get(yat::URI::PATH) );
  m_file_handle.reset( new NexusFile( PSZ( m_location.get(yat::URI::PATH) ) ) );
}

//---------------------------------------------------------------------------
// NxsDataset::NxsDataset
//---------------------------------------------------------------------------
NxsDataset::NxsDataset()
{
}

//---------------------------------------------------------------------------
// NxsDataset::NxsDataset
//---------------------------------------------------------------------------
NxsDataset::~NxsDataset()
{
  CDMA_TRACE("NxsDataset::~NxsDataset");
}

//---------------------------------------------------------------------------
// NxsDataset::fullName
//---------------------------------------------------------------------------
yat::String NxsDataset::concatPath(const yat::String &path, const yat::String& name)
{
  yat::String full_name = PSZ_FMT("%s/%s", PSZ(path), PSZ(name));
  full_name.replace("//", "/");
  return full_name;
}

//---------------------------------------------------------------------------
// NxsDataset::getItemFromPath
//---------------------------------------------------------------------------
cdma::IDataItemPtr NxsDataset::getItemFromPath(const std::string &fullPath)
{
  CDMA_FUNCTION_TRACE("NxsDataset::getItemFromPath(const std::string &)");
  yat::String strPath = fullPath, strName;
  strPath.extract_token_right('/', &strName);
  return getItemFromPath(strPath, strName);
}

cdma::IDataItemPtr NxsDataset::getItemFromPath(const yat::String& path, const yat::String& name)
{
  CDMA_FUNCTION_TRACE("NxsDataset::getItemFromPath(const yat::String&, const yat::String&)");

  if( m_file_handle.is_null() )
    THROW_NO_DATA("No NeXus file", "NxsGroup::getItemFromPath");
  
  std::map<yat::String, cdma::IDataItemPtr>::iterator it = m_item_map.find(concatPath(path, name));
  if( it != m_item_map.end() )
    return it->second;

  try
  {
    NexusFileAccess auto_open(m_file_handle);
    if( m_file_handle.is_null() )
      THROW_NO_DATA("No NeXus file", "NxsDataset::getItemFromPath");
  
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
    cdma::IDataItemPtr ptrItem = new NxsDataItem(this, *info, strPath);
    m_item_map[strPath] = ptrItem;
    m_file_handle->CloseDataSet();
    return ptrItem;
  }
  catch( NexusException &e )
  {
    throw cdma::Exception(e);
  }
}

//---------------------------------------------------------------------------
// NxsDataset::getGroupFromPath
//---------------------------------------------------------------------------
cdma::IGroupPtr NxsDataset::getGroupFromPath(const std::string &groupPath)
{
  if( m_file_handle.is_null() )
    THROW_NO_DATA("No NeXus file", "NxsGroup::getGroupFromPath");

  yat::String path = groupPath;
  std::map<yat::String, cdma::IGroupPtr>::iterator it = m_group_map.find(path);
  if( it != m_group_map.end() )
    return it->second;

  try
  {
    // get handle and access on file
    NexusFileAccess auto_open(m_file_handle);

    if( m_file_handle.is_null() )
      THROW_NO_DATA("No NeXus file", "NxsGroup::getGroupFromPath");
  
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
    cdma::IGroupPtr ptrGroup = new NxsGroup(this, path);
    m_group_map[path] = ptrGroup;
    return ptrGroup;
  }
  catch( NexusException &e )
  {
    throw cdma::Exception(e);
  }
}

//---------------------------------------------------------------------------
// NxsDataset::getRootGroup
//---------------------------------------------------------------------------
cdma::IGroupPtr NxsDataset::getRootGroup()
{
CDMA_FUNCTION_TRACE("NxsDataset::getRootGroup");
  return m_phy_root;
}

//---------------------------------------------------------------------------
// NxsDataset::getLogicalRoot
//---------------------------------------------------------------------------
cdma::LogicalGroupPtr NxsDataset::getLogicalRoot()
{
  THROW_NOT_IMPLEMENTED("NxsDataset::getLogicalRoot");
}

//---------------------------------------------------------------------------
// NxsDataset::getLocation
//---------------------------------------------------------------------------
std::string NxsDataset::getLocation()
{
  return m_location.get();
}

//---------------------------------------------------------------------------
// NxsDataset::getTitle
//---------------------------------------------------------------------------
std::string NxsDataset::getTitle()
{
  THROW_NOT_IMPLEMENTED("NxsDimension::getTitle");
}

//---------------------------------------------------------------------------
// NxsDataset::setLocation
//---------------------------------------------------------------------------
void NxsDataset::setLocation(const std::string& location)
{
  close();
  m_location.set(location);
}

//---------------------------------------------------------------------------
// NxsDataset::setLocation
//---------------------------------------------------------------------------
void NxsDataset::setLocation(const yat::URI& location)
{
  close();
  m_location = location;
}

//---------------------------------------------------------------------------
// NxsDataset::setTitle
//---------------------------------------------------------------------------
void NxsDataset::setTitle(const std::string&)
{
  THROW_NOT_IMPLEMENTED("NxsDimension::setTitle");
}


//---------------------------------------------------------------------------
// NxsDataset::sync
//---------------------------------------------------------------------------
bool NxsDataset::sync() throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDimension::sync");
}


//---------------------------------------------------------------------------
// NxsDataset::save
//---------------------------------------------------------------------------
void NxsDataset::save() throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDataset::save");
}


//---------------------------------------------------------------------------
// NxsDataset::saveTo
//---------------------------------------------------------------------------
void NxsDataset::saveTo(const std::string&) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDataset::saveTo");
}


//---------------------------------------------------------------------------
// NxsDataset::save
//---------------------------------------------------------------------------
void NxsDataset::save(const cdma::IContainer&) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDataset::save");
}


//---------------------------------------------------------------------------
// NxsDataset::save
//---------------------------------------------------------------------------
void NxsDataset::save(const std::string&, const cdma::IAttributePtr&) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDataset::save");
}

//---------------------------------------------------------------------------
// NxsDataset::close
//---------------------------------------------------------------------------
void NxsDataset::close()
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


}
