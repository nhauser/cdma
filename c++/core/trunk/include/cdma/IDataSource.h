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

#include <yat/utils/URI.h>

// CDMA includes
#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IContainer.h>

/// @cond pluginAPI

namespace cdma
{

//==============================================================================
/// @brief Abstraction of the plugin data source information
///
/// The CDMA asks each plug-ins through this interface to find out which one
/// is the best choice for accessing the data referenced by the given URI
//==============================================================================
class CDMA_DECL IDataSource
{
public:

  /// Ask the plugin if the location corresponds to a readable dataset
  /// using at least the standard navigation interface.
  ///
  /// It's a necessary condition for reading the dataset but not
  /// sufficient for decide to apply the dictionary on it.
  ///
  /// @see isExperiment
  ///
  /// @param dataset_location location of the dataset asked for
  /// @return true of false
  ///
  virtual bool isReadable(const yat::URI& dataset_location) const = 0;
  
  /// When browsing a file system, ask the plugin if the location corresponds
  /// to a file containing one or more datasets, at this plug-in sense.
  ///
  /// If the answer is true, the client should open the file (through 
  /// a yet to be developed CDMA service) the same way the directory
  /// containing it.
  ///
  /// @param dataset_location location of the dataset asked for
  /// @return true or false
  ///
  virtual bool isBrowsable(const yat::URI& dataset_location) const = 0;

  /// Ask the plug-in if the location corresponds to a file produced by the 
  /// institute's editor of this plug-in.
  ///
  /// It make sense for a dataset that is a directory containing several
  /// files: each file is effectively produced by the institute who provide 
  /// the plug-in, but is not a dataset, just a part.
  ///
  /// @param dataset_location location of the dataset asked for
  /// @return true of false
  ///
  virtual bool isProducer(const yat::URI& dataset_location) const = 0;
  
  /// Ask the plugin if the location corresponds to a dataset produced
  /// by the same institute providing this plug-in.
  ///
  /// @param dataset_location location of the dataset asked for
  /// @return true of false
  ///
  virtual bool isExperiment(const yat::URI& dataset_location) const = 0;
 };

DECLARE_SHARED_PTR(IDataSource);

} //namespace cdma

/// @endcond pluginAPI

#endif //__CDMA_IDATASOURCE_H__

