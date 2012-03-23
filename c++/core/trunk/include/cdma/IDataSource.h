//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IDATASOURCE_H__
#define __CDMA_IDATASOURCE_H__

// Standard includes
#include <list>
#include <vector>
#include <typeinfo>

#include <yat/utils/String.h>
#include <yat/threading/Mutex.h>
#include <yat/memory/SharedPtr.h>

// CDMA includes
#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IContainer.h>

/// @cond pluginAPI

namespace cdma
{

//==============================================================================
/// @brief Abstraction of the plugin dataset recognition
///
/// The CDMA asks each plug-ins through this interface to find out which one
/// is the best choice for accessing the data referenced by the given URI
//==============================================================================
class CDMA_DECL IDataSource
{
public:

  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param dataset_location location of the dataset asked for
  ///
  /// @return true of false
  ///
  virtual bool isReadable(const yat::URI& dataset_location) const = 0;
  
  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param dataset_location location of the dataset asked for
  ///
  /// @return true of false
  ///
  virtual bool isBrowsable(const yat::URI& dataset_location) const = 0;

  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param dataset_location location of the dataset asked for
  ///
  /// @return true of false
  ///
  virtual bool isProducer(const yat::URI& dataset_location) const = 0;
  
  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param dataset_location location of the dataset asked for
  ///
  /// @return true of false
  ///
  virtual bool isExperiment(const yat::URI& dataset_location) const = 0;
 };

DECLARE_SHARED_PTR(IDataSource);

} //namespace cdma

/// @endcond pluginAPI

#endif //__CDMA_IDATASOURCE_H__

