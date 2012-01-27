// ****************************************************************************
// Synchrotron SOLEIL
//
// Creation : 08/12/2011
// Author   : See AUTHORS file
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; version 2 of the License.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// ****************************************************************************

// Yat
#include <yat/plugin/PlugInSymbols.h>
#include <yat/file/FileName.h>
#include <yat/utils/URI.h>

// CDMA core
#include <cdma/dictionary/Key.h>
#include <cdma/dictionary/Dictionary.h>
#include <cdma/dictionary/LogicalGroup.h>

// Soleil plugin
#include <SoleilNxsFactory.h>
#include <SoleilNxsDataSource.h>
#include <SoleilNxsDataset.h>

namespace cdma
{

//---------------------------------------------------------------------------
// SoleilNxsDataset::SoleilNxsDataset
//---------------------------------------------------------------------------
SoleilNxsDataset::SoleilNxsDataset(const yat::URI& location)
: NxsDataset(location)
{
}

//---------------------------------------------------------------------------
// SoleilNxsDataset::SoleilNxsDataset
//---------------------------------------------------------------------------
SoleilNxsDataset::SoleilNxsDataset()
: NxsDataset()
{
}

//---------------------------------------------------------------------------
// SoleilNxsDataset::getDataset
//---------------------------------------------------------------------------
NxsDatasetPtr SoleilNxsDataset::getDataset(const yat::URI& location)
{
  CDMA_STATIC_FUNCTION_TRACE("SoleilNxsFactory::getDataset");
  NxsDatasetPtr ptr( new SoleilNxsDataset(location) );
  ptr->setSelfRef(ptr);
  return ptr;
}

//---------------------------------------------------------------------------
// SoleilNxsDataset::newDataset
//---------------------------------------------------------------------------
NxsDatasetPtr SoleilNxsDataset::newDataset()
{
  CDMA_STATIC_FUNCTION_TRACE("SoleilNxsFactory::newDataset");
  NxsDatasetPtr ptr( new SoleilNxsDataset() );
  ptr->setSelfRef(ptr);
  return ptr;
}

//---------------------------------------------------------------------------
// SoleilNxsDataset::getLogicalRoot
//---------------------------------------------------------------------------
LogicalGroupPtr SoleilNxsDataset::getLogicalRoot()
{
  CDMA_FUNCTION_TRACE("SoleilNxsDataset::getLogicalRoot");

  if( m_log_root.is_null() )
  {
    CDMA_TRACE("Getting key file");
    yat::String keyFile = cdma::Factory::getKeyDictionaryPath();

    CDMA_TRACE("Creating Dictionary detector");
    DictionaryDetector detector ( m_file_handle );
    CDMA_TRACE("Getting mapping file");
    yat::FileName file( cdma::Factory::getDictionariesFolder() + "/" +\
                        PlugInID + "/" + detector.getDictionaryName());

    yat::FileName mapFile ( file );

    CDMA_TRACE("Creating dictionary");
    DictionaryPtr dictionary ( new cdma::Dictionary( ) );
    dictionary->setKeyFilePath( keyFile );
    dictionary->setMappingFilePath( mapFile.full_name() );

    CDMA_TRACE("Read the dictionary");
    dictionary->readEntries();

    CDMA_TRACE("Creating logical root");
    LogicalGroup* ptrRoot = new cdma::LogicalGroup( this, NULL, KeyPtr(NULL), dictionary );
    m_log_root.reset( ptrRoot );
  }
  return m_log_root;
}

} // namespace cdma
