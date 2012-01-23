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
#include <string>
#include <vector>

#include <cdma/exception/Exception.h>
#include <DictionaryDetector.h>
#include <NxsDataset.h>

namespace cdma
{
  
  //-----------------------------------------------------------------------------
  // DictionaryDetector::DictionaryDetector
  //-----------------------------------------------------------------------------
  DictionaryDetector::DictionaryDetector() { }
  DictionaryDetector::DictionaryDetector(const NexusFilePtr& handle, const yat::String& uri)
  {
    m_ptrNxFile = handle;
    m_uri = uri;
  }

  //-----------------------------------------------------------------------------
  // DictionaryDetector::~DictionaryDetector
  //-----------------------------------------------------------------------------
  DictionaryDetector::~DictionaryDetector() { }

  //-----------------------------------------------------------------------------
  // DictionaryDetector::getDictionaryName
  //-----------------------------------------------------------------------------
  yat::String DictionaryDetector::getDictionaryName() throw ( cdma::Exception )
  {
    if( m_beamline == "" || m_beamline == "UNKNOWN" ) {
      detectBeamline();
    }
    if( m_model == "" || m_model == "UNKNOWN"  ) {
      detectDataModel();
    }
    yat::String file (m_beamline + "_" + m_model + ".xml");
    file.to_lower();
    return file;
  }

  //-----------------------------------------------------------------------------
  // DictionaryDetector::detectBeamline
  //-----------------------------------------------------------------------------
  void DictionaryDetector::detectBeamline()
  {
    yat::String path = "/<NXentry>/<NXinstrument>";
    NexusFileAccess auto_open ( m_ptrNxFile, m_uri );
    if( m_ptrNxFile->OpenGroupPath(PSZ(path), false) )
    {
      CDMA_TRACE(std::string(m_ptrNxFile->CurrentGroupName()).c_str());
      CDMA_TRACE(std::string(m_ptrNxFile->CurrentGroupClass()).c_str());
      m_beamline = std::string(m_ptrNxFile->CurrentGroupName());
    }
    else
    {
      m_beamline = "UNKNOWN";
    }
  }

  //-----------------------------------------------------------------------------
  // DictionaryDetector::detectDataModel
  //-----------------------------------------------------------------------------
  void DictionaryDetector::detectDataModel()
  {
    m_model = "UNKNOWN";
    if( m_beamline != "UNKNOWN" )
    {
      if( isScanServer() )
      {
        m_model = "SCANSERVER";
      }
      else if( isFlyScan() )
      {
        m_model = "FLYSCAN";
      }
      else
      {
        m_model = "PASSERELLE";
      }
    }
  }

  //-----------------------------------------------------------------------------
  // DictionaryDetector::isFlyScan
  //-----------------------------------------------------------------------------
  bool DictionaryDetector::isFlyScan()
  {
    bool result = false;
    yat::String pathGrp = "/<NXentry>/";
    yat::String testClass = "NXdata";
    yat::String testName = "scan_data";
    std::vector<std::string> res;
    NexusFileAccess auto_open ( m_ptrNxFile, m_uri );
    if( m_ptrNxFile->SearchGroup(PSZ(testName), PSZ(testClass), &res, PSZ(pathGrp) ) == NX_OK )
    {
      CDMA_TRACE( std::string(m_ptrNxFile->CurrentGroupName()).c_str() );
      if( ! m_ptrNxFile->OpenDataSet( "time_1", false ) )
      {
        result = true;
      }
    }
    return result;
  }
  
  //-----------------------------------------------------------------------------
  // DictionaryDetector::detectDataModel
  //-----------------------------------------------------------------------------
  bool DictionaryDetector::isScanServer()
  {
    bool result = false;
    yat::String pathGrp = "/<NXentry>/";
    yat::String testClass = "NXdata";
    yat::String testName = "scan_data";
    std::vector<std::string> res;
    NexusFileAccess auto_open ( m_ptrNxFile, m_uri );
    if( m_ptrNxFile->SearchGroup(PSZ(testName), PSZ(testClass), &res, PSZ(pathGrp) ) == NX_OK )
    {
      CDMA_TRACE( std::string(m_ptrNxFile->CurrentGroupName()).c_str() );
      if( m_ptrNxFile->OpenDataSet( "time_1", false ) )
      {
        result = true;
      }
    }
    return result;
  }

} // namespace
