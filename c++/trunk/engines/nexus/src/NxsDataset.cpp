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
NxsDataset::NxsDataset( const std::string& filename )
{
  // Utiliser yat pour sortir un FileName et récupérer son contenu (1 ou plusieurs fichiers)
  m_uri = filename;
  m_ptrNxFile.reset(new NexusFile(m_uri.data()));
  m_detector = DictionaryDetector(m_ptrNxFile);
  //##m_log_root.reset( new NxsLogicalGroup() );
}

//---------------------------------------------------------------------------
// NxsDataset::setSelfRef
//---------------------------------------------------------------------------
void NxsDataset::setSelfRef(const NxsDatasetPtr& ptr)
{
  if( ptr.get() != this )
  {
    THROW_INVALID_POINTER("Pointer mismatch", "NxsDataset::setSelfRef");
  }

  m_self_wptr = ptr;
  m_phy_root.reset( new NxsGroup( m_self_wptr ) );
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
  yat::String strPath = fullPath, strName;
  strPath.extract_token_right('/', &strName);
  return getItemFromPath(strPath, strName);
}

cdma::IDataItemPtr NxsDataset::getItemFromPath(const yat::String &path, const yat::String& name)
{
  if( m_ptrNxFile.is_null() )
    THROW_NO_DATA("No NeXus file", "NxsGroup::getItemFromPath");
  
  std::map<yat::String, cdma::IDataItemPtr>::iterator it = m_item_map.find(concatPath(path, name));
  if( it != m_item_map.end() )
    return it->second;

  try
  {
    NexusFileAccess auto_open(m_ptrNxFile);
    if( m_ptrNxFile.is_null() )
      THROW_NO_DATA("No NeXus file", "NxsDataset::getItemFromPath");
  
    m_ptrNxFile->OpenGroupPath(PSZ(path));
    NexusDataSetInfo* info = new NexusDataSetInfo();
    m_ptrNxFile->GetDataSetInfo(info, PSZ(name));
    
    // Take in consideration the real path
    yat::String strPath = m_ptrNxFile->CurrentGroupPath();
    strPath.replace('(', '<');
    strPath.replace(')', '>');
    strPath = strPath + name;
    
    // check once again the path is in map
    it = m_item_map.find(strPath);
    if( it != m_item_map.end() )
      return it->second;
      
    // Create corresponding object and stores it
    cdma::IDataItemPtr ptrItem = new NxsDataItem(m_self_wptr, *info, strPath);
    m_item_map[strPath] = ptrItem;
    m_ptrNxFile->CloseDataSet();
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
  //## TODO debug
  if( m_ptrNxFile.is_null() )
    THROW_NO_DATA("No NeXus file", "NxsGroup::getGroupFromPath");

  yat::String path = groupPath;
  std::map<yat::String, cdma::IGroupPtr>::iterator it = m_group_map.find(path);
  if( it != m_group_map.end() )
    return it->second;

  try
  {
    // get handle and access on file
    NexusFileAccess auto_open(m_ptrNxFile);

    if( m_ptrNxFile.is_null() )
      THROW_NO_DATA("No NeXus file", "NxsGroup::getGroupFromPath");
  
    // Open the path
    m_ptrNxFile->OpenGroupPath(PSZ(path));

    // Take in consideration the real path
    path = m_ptrNxFile->CurrentGroupPath();
    path = path.substr( 0, path.length() - 1 );
    path.replace('(', '<');
    path.replace(')', '>');

    // check once again the path is in map
    it = m_group_map.find(path);
    if( it != m_group_map.end() )
      return it->second;

    // Create corresponding object and stores it
    cdma::IGroupPtr ptrGroup = new NxsGroup(m_self_wptr, path);
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
  return m_phy_root;
}

//---------------------------------------------------------------------------
// NxsDataset::getLogicalRoot
//---------------------------------------------------------------------------
cdma::LogicalGroupPtr NxsDataset::getLogicalRoot()
{
  CDMA_FUNCTION_TRACE("NxsDataset::getLogicalRoot");
  if( m_log_root.is_null() )
  {
    CDMA_TRACE("Getting key file");
    yat::String keyFile = Factory::getKeyDictionaryPath();

    CDMA_TRACE("Creating Dictionary detector");
    DictionaryDetector detector ( m_ptrNxFile );
    CDMA_TRACE("Getting mapping file");
//    yat::String file = Factory::getMappingDictionaryFolder( new NxsFactory() ) + detector.getDictionaryName();
    yat::FileName file( Factory::getDictionariesFolder() + "/" + "NxsFactory" + "/" + detector.getDictionaryName());

    yat::FileName mapFile ( file );

    CDMA_TRACE("Creating dictionary");
    DictionaryPtr dictionary ( new Dictionary( ) );
    dictionary->setKeyFilePath( keyFile );
    dictionary->setMappingFilePath( mapFile.full_name() );

    CDMA_TRACE("Read the dictionary");
    dictionary->readEntries();

    CDMA_TRACE("Creating logical root");
    LogicalGroup* ptrRoot = new LogicalGroup( this, NULL, KeyPtr(NULL), dictionary );
    m_log_root.reset( ptrRoot );
  }
  return m_log_root;
}

//---------------------------------------------------------------------------
// NxsDataset::getLocation
//---------------------------------------------------------------------------
std::string NxsDataset::getLocation()
{
  return m_uri;
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
  if( m_ptrNxFile != NULL )
  {
    m_ptrNxFile->Close();
  }
  m_uri = location;
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
  THROW_NOT_IMPLEMENTED("NxsDimension::save");
}


//---------------------------------------------------------------------------
// NxsDataset::saveTo
//---------------------------------------------------------------------------
void NxsDataset::saveTo(const std::string&) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDimension::saveTo");
}


//---------------------------------------------------------------------------
// NxsDataset::save
//---------------------------------------------------------------------------
void NxsDataset::save(const cdma::IContainer&) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDimension::save");
}


//---------------------------------------------------------------------------
// NxsDataset::save
//---------------------------------------------------------------------------
void NxsDataset::save(const std::string&, const cdma::IAttributePtr&) throw ( cdma::Exception )
{
  THROW_NOT_IMPLEMENTED("NxsDimension::save");
}


//---------------------------------------------------------------------------
// NxsDataset::getMappingFileName
//---------------------------------------------------------------------------
const std::string& NxsDataset::getMappingFileName()
{
  return m_detector.getDictionaryName();
}

//---------------------------------------------------------------------------
// NexusFileAccess::NexusFileAccess
//---------------------------------------------------------------------------
NexusFileAccess::NexusFileAccess( const NexusFilePtr& ptrFile )
{
  CDMA_FUNCTION_TRACE("NexusFileAccess::NexusFileAccess"); 
  m_ptrNxFile = ptrFile;
  if( m_ptrNxFile )
  { 
    m_ptrNxFile->OpenRead(NULL);
  }
}

//---------------------------------------------------------------------------
// NexusFileAccess::~NexusFileAccess
//---------------------------------------------------------------------------
NexusFileAccess::~NexusFileAccess()
{
  CDMA_FUNCTION_TRACE("NexusFileAccess::~NexusFileAccess"); 
  if( m_ptrNxFile ) 
  {
    m_ptrNxFile->Close();
  }
}


}
