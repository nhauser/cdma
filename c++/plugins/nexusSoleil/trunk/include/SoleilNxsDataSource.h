//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************

#ifndef __CDMA_SOLEIL_NXSDATASOURCE_H__
#define __CDMA_SOLEIL_NXSDATASOURCE_H__

// STD
#include <string>

// yat
#include <yat/utils/URI.h>

// CDMA core
#include <cdma/IDataSource.h>

// Soleil NeXus plugin
#include <SoleilNxsFactory.h>


namespace cdma
{

//==============================================================================
/// IDataSource implementation
//==============================================================================
class SoleilNxsDataSource : public IDataSource 
{
friend class SoleilNxsFactory;
public:
  ~SoleilNxsDataSource() {};

  //@{ IDataSource methods ------------

  bool isReadable(const yat::URI& dataset_location) const;
  bool isBrowsable(const yat::URI& dataset_location) const;
  bool isProducer(const yat::URI& dataset_location) const;
  bool isExperiment(const yat::URI& dataset_location) const;
  
  //@}

private:
  SoleilNxsDataSource(SoleilNxsFactory *factory_ptr): m_factory_ptr(factory_ptr) {};

  SoleilNxsFactory *m_factory_ptr;
};

} //namespace CDMACore
#endif //__CDMA_NXSDATASOURCE_H__

