//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IPATHPARAMETER_H__
#define __CDMA_IPATHPARAMETER_H__

#include <string>
#include <yat/any/Any.h>
#include <yat/memory/SharedPtr.h>
#include <yat/threading/Mutex.h>

#include <cdma/IObject.h>

namespace cdma
{
  
//=============================================================================
/// A IPathParameter represents conditions that permits identifying a specific
/// node using the extended dictionary mechanism.
/// When according to a given IPath several IContainer can be returned, the path
/// parameter will make possible how to find which one is relevant.
/// <p>
/// The parameter can consist in a regular expression on a name, an attribute or 
/// whatever that should be relevant to formally identify a specific node while 
/// several are possible according to the path.
//=============================================================================
class PathParameter : public IObject 
{
public:
  //Virtual destructor
  virtual ~PathParameter() {};

  /// Get the filter's kind
  /// 
  /// @return filter's kind
  ///
  virtual CDMAType::ParameterType getType() = 0;

  /// Get the filter's value
  /// 
  /// @return filter's value
  ///
  virtual void* getValue() = 0;

  /// Get the filter's name
  /// 
  /// @return name of the filter
  ///
  virtual std::string getName() = 0;

  /// Set the filter's value
  /// 
  /// @param value of the filter
  ///
  virtual void setValue(const yat::Any& value) = 0;

  /// Equality test
  /// 
  /// @return true if both KeyFilter have same kind and value
  ///

  virtual bool equals(const PathParameterPtr& keyfilter) = 0;

  /// To string method
  /// 
  /// @return a string representation of the KeyFilter
  ///
  virtual std::string toString() = 0;

  /// Clone this IKeyFilter
  /// @return a copy of this
  ///
  virtual PathParameterPtr clone() = 0;
};

} //namespace CDMACore
#endif //__CDMA_IPATHPARAMETER_H__


