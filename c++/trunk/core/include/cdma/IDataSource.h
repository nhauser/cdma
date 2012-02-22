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

namespace cdma
{

//==============================================================================
/// IDataSource Interface
/// A DataItem is a logical container for data. It has a DataType, a set of
/// Dimensions that define its array shape, and optionally a set of Attributes.
//==============================================================================
class CDMA_DECL IDataSource
{
public:

  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param URI: the URI asked for
  ///
  /// @return true of false
  ///
  virtual bool isReadable(const yat::URI& destination) const =0;
  
  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param URI the URI asked for
  ///
  /// @return true of false
  ///
  virtual bool isBrowsable(const yat::URI& destination) const =0;

  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param URI: the URI asked for
  ///
  /// @return true of false
  ///
  virtual bool isProducer(const yat::URI& destination) const =0;
  
  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param URI: the URI asked for
  ///
  /// @return true of false
  ///
  virtual bool isExperiment(const yat::URI& destination) const =0;

 };

DECLARE_SHARED_PTR(IDataSource);

} //namespace cdma

#endif //__CDMA_IDATASOURCE_H__

