//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************

#ifndef __CDMA_NXSDATASOURCE_H__
#define __CDMA_NXSDATASOURCE_H__

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
  SoleilNxsDataSource() {};
  ~SoleilNxsDataSource() {};

  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param URI: the URI asked for
  ///
  /// @return true of false
  ///
  bool isReadable(const yat::URI& destination) const;
  
  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param URI the URI asked for
  ///
  /// @return true of false
  ///
  bool isBrowsable(const yat::URI& destination) const;

  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param URI: the URI asked for
  ///
  /// @return true of false
  ///
  bool isProducer(const yat::URI& destination) const;
  
  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param URI: the URI asked for
  ///
  /// @return true of false
  ///
  bool isExperiment(const yat::URI& destination) const;
  
  std::string getFactoryName() const { return NXS_FACTORY_NAME; };
  CDMAType::ModelType getModelType() const { return CDMAType::Other; };
  
private: 
  static std::string  CREATOR;
	static std::string* BEAMLINES;
	static int          NB_BEAMLINES;

};
} //namespace CDMACore
#endif //__CDMA_NXSDATASOURCE_H__

