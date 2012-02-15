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
public:
  SoleilNxsDataSource()  {};
  ~SoleilNxsDataSource() {};

  //@{ IDataSource methods ------------

  bool   isReadable(const yat::URI& destination) const;
  bool  isBrowsable(const yat::URI& destination) const;
  bool   isProducer(const yat::URI& destination) const;
  bool isExperiment(const yat::URI& destination) const;
  
  //@}

  //@{ IObject methods ----------------

  std::string       getFactoryName() const { return NXS_FACTORY_NAME; }
  CDMAType::ModelType getModelType() const { return CDMAType::Other; }

  //@}

};

} //namespace CDMACore
#endif //__CDMA_NXSDATASOURCE_H__

