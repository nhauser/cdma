//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The cdma-core library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************


#include <string>
#include <iostream>
#include <stdio.h>

// yat
#include <yat/utils/String.h>
#include <yat/file/FileName.h>

// CDMA core
#include <cdma/navigation/IGroup.h>

// CDMA engine
#include <NxsAttribute.h>

// CDMA plugin
#include <SoleilNxsDataSource.h>


namespace cdma
{
namespace soleil
{
namespace nexus
{

std::string CREATOR = "Synchrotron SOLEIL";
int         NB_BEAMLINES = 27;
std::string BEAMLINES [] = {"contacq", "CONTACQ", "AILES", "ANTARES", "CASSIOPEE", "CRISTAL", "DIFFABS", "DEIMOS", "DESIRS", "DISCO", "GALAXIES", "LUCIA", "MARS", "METROLOGIE", "NANOSCOPIUM", "ODE", "PLEIADES", "PROXIMA1", "PROXIMA2", "PSICHE", "SAMBA", "SEXTANTS", "SIRIUS", "SIXS", "SMIS", "TEMPO", "SWING"};

//----------------------------------------------------------------------------
// DataSource::isReadable
//----------------------------------------------------------------------------
bool DataSource::isReadable(const std::string& dataset_uri) const
{
  yat::URI dataset_location(dataset_uri);
  FUNCTION_TRACE("DataSource::isReadable");
  CDMA_TRACE("dataset_location: " << dataset_location.get());
  CDMA_TRACE("path: " << dataset_location.get( yat::URI::PATH ));
  // Get the path from URI
  yat::String path = dataset_location.get( yat::URI::PATH );
  
  // Check file exists and is has a NeXus extension
  yat::FileName file ( path );
  
  if( file.file_exist() )
  {
    // Will try to open the file and close it
    m_factory_ptr->openDataset( dataset_location.get(yat::URI::SCHEME) + ':' + path );
    return true;
  }
  return false; 
}

//----------------------------------------------------------------------------
// DataSource::isBrowsable
//----------------------------------------------------------------------------
bool DataSource::isBrowsable(const std::string&) const
{
  return false;
}

//----------------------------------------------------------------------------
// DataSource::isProducer
//----------------------------------------------------------------------------
bool DataSource::isProducer(const std::string& dataset_uri) const
{
  FUNCTION_TRACE("DataSource::isProducer");
  CDMA_TRACE("dataset_location: " << dataset_uri);
  bool result = false;
  if( isReadable( dataset_uri ) )
  {
    yat::URI dataset_location(dataset_uri);
    // Get the path from URI
    yat::String path = dataset_location.get(yat::URI::PATH);
    Factory plug_factory;
    cdma::IDatasetPtr dataset = plug_factory.openDataset( dataset_location.get(yat::URI::SCHEME) + ':' + path );
    // seek at root for 'creator' attribute
    cdma::IGroupPtr group = dataset->getRootGroup();
    if( group->hasAttribute("creator") && group->getAttribute("creator")->getValue<std::string>() == CREATOR )
    {
      result = true;
    }
    // Check the beamline is one used at Soleil
    else 
    {
      group = group->getGroup("<NXentry>");
      if( group ) 
      {
        group = group->getGroup("<NXinstrument>");
        if( group ) 
        {
          std::string node = group->getShortName();
          for( int i = 0; i < NB_BEAMLINES; i++ ) 
          {
            if( node == BEAMLINES[i]  ) 
            {
              result = true;
              break;
            }
          }
        }
      }
    }
    
  }
  return result;
}

//----------------------------------------------------------------------------
// DataSource::isExperiment
//----------------------------------------------------------------------------
bool DataSource::isExperiment(const std::string& dataset_uri) const
{
  return isProducer(dataset_uri);
}

} // namespace nexus
} // namespace soleil
} // namespace cdma
