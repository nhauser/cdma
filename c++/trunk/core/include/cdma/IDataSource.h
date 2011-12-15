//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IDATAITEM_H__
#define __CDMA_IDATAITEM_H__

// Standard includes
#include <list>
#include <vector>
#include <typeinfo>

#include <yat/utils/String.h>
#include <yat/threading/Mutex.h>
#include <yat/memory/SharedPtr.h>

// CDMA includes
#include <cdma/exception/Exception.h>
#include <cdma/navigation/IContainer.h>
#include <cdma/IObject.h>

namespace cdma
{

//==============================================================================
/// IDataSource Interface
/// A DataItem is a logical container for data. It has a DataType, a set of
/// Dimensions that define its array shape, and optionally a set of Attributes.
//==============================================================================
class CDMA_DECL IDataSource : public IObject
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
  bool isReadable(/*yat::URI*/)=0;
  
  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param URI the URI asked for
  ///
  /// @return true of false
  ///
  bool isBrowsable(/*yat::URI*/)=0;

  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param URI: the URI asked for
  ///
  /// @return true of false
  ///
  bool isProducer(/*yat::URI*/)=0;
  
  /// Ask the plugin if the URI corresponds to a strictly readable dataset
  /// Only the basic navigation is applicable in such a dataset
  /// The dictionnary mechanism is not relevant for this kind of dataset
  ///
  /// @param URI: the URI asked for
  ///
  /// @return true of false
  ///
  bool isExperiment(/*yat::URI*/)=0;

 };
 
} //namespace cdma

#endif //__CDMA_IDATAITEM_H__

