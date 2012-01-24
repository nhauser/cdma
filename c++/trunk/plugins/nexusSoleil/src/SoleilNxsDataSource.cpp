//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
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
#include <NxsDataset.h>
#include <NxsAttribute.h>

// CDMA plugin
#include <SoleilNxsDataSource.h>


namespace cdma
{

std::string CREATOR = "Synchrotron SOLEIL";
int          NB_BEAMLINES = 26;
std::string BEAMLINES [] = {"contacq", "AILES", "ANTARES", "CASSIOPEE", "CRISTAL", "DIFFABS", "DEIMOS", "DESIRS", "DISCO", "GALAXIES", "LUCIA", "MARS", "METROLOGIE", "NANOSCOPIUM", "ODE", "PLEIADES", "PROXIMA1", "PROXIMA2", "PSICHE", "SAMBA", "SEXTANTS", "SIRIUS", "SIXS", "SMIS", "TEMPO", "SWING"};


//----------------------------------------------------------------------------
// SoleilNxsDataSource::isReadable
//----------------------------------------------------------------------------
bool SoleilNxsDataSource::isReadable(const yat::URI& destination) const
{
  bool result = false;
  // Get the path from URI
  yat::String path = destination.get( yat::URI::PATH );
  
  // Check file exists and is has a NeXus extension
  yat::FileName file ( path );
  
  if( file.file_exist() && file.ext() == "nxs" )
  {
    result = true;
  }
  
  return result;
}

//----------------------------------------------------------------------------
// SoleilNxsDataSource::isBrowsable
//----------------------------------------------------------------------------
bool SoleilNxsDataSource::isBrowsable(const yat::URI& destination) const
{
  return isReadable( destination );
}

//----------------------------------------------------------------------------
// SoleilNxsDataSource::isProducer
//----------------------------------------------------------------------------
bool SoleilNxsDataSource::isProducer(const yat::URI& destination) const
{
  bool result = false;
  if( isReadable( destination) )
  {
    // Get the path from URI
    yat::String path = destination.get( yat::URI::PATH );
    NxsDataset dataset ( path );
    
    // seek at root for 'creator' attribute
    IGroupPtr group = dataset.getRootGroup();
    if( group->hasAttribute("creator") && group->getAttribute("creator")->toString() == CREATOR )
    {
      result = true;
    }
    // Check the beamline is one used at Soleil
    else 
    {
      group = group->getGroup("<NXentry>");
      if( ! group.is_null() ) 
      {
        group = group->getGroup("<NXinstrument>");
      }
    
      if( ! group.is_null() ) 
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
  return result;
}

//----------------------------------------------------------------------------
// SoleilNxsDataSource::isExperiment
//----------------------------------------------------------------------------
bool SoleilNxsDataSource::isExperiment(const yat::URI& destination) const
{
  bool result;
  return result;
}

} // namespace cdma
