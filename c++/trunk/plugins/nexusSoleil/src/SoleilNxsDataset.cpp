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
SoleilNxsDataset::SoleilNxsDataset(const std::string& filepath) : NxsDataset(filepath)
{
  m_detector = DictionaryDetector(m_ptrNxFile);
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
    DictionaryDetector detector ( m_ptrNxFile );
    CDMA_TRACE("Getting mapping file");
    yat::FileName file( cdma::Factory::getDictionariesFolder() + "/" + PlugInID + "/" + detector.getDictionaryName());

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

//---------------------------------------------------------------------------
// SoleilNxsDataset::getMappingFileName
//---------------------------------------------------------------------------
const std::string& SoleilNxsDataset::getMappingFileName()
{
  return m_detector.getDictionaryName();
}


} // namespace cdma
