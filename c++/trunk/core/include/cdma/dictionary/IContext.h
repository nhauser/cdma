//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_ICONTEXT_H__
#define __CDMA_ICONTEXT_H__

#include <yat/memory/SharedPtr.h>

// Include CDMA
#include <cdma/navigation/IContainer.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/dictionary/Key.h>
#include <cdma/IObject.h>

namespace cdma
{

//==============================================================================
/// This interface is used when invoking an external method.
/// It should contain all required information so the called method,
/// can work properly as if it were in the CDM.
/// The context is compound of the dataset we are working on,
/// the caller of the method, the key used to call that method (that can
/// have some parameters), the path (with parameters set) and some
/// parameters that are set by the institute's plug-in.
//==============================================================================
class IContext : public IObject
{
public:
  // d-tor
  virtual ~IContext()
  {
  }
  
  /// Permits to get the IDataset we want to work on.
  ///
  virtual yat::SharedPtr<IDataset, yat::Mutex> getDataset() = 0;
  virtual void setDataset(IDataset dataset) = 0;

  /// Permits to get the IContainer that instantiated the context.
  ///
  virtual yat::SharedPtr<IContainer, yat::Mutex>  getCaller() = 0;
  virtual void setCaller(IContainer caller) = 0;

  /// Permits to get the IKey that lead to this instantiation.
  ///
  virtual yat::SharedPtr<IKey, yat::Mutex> getKey() = 0;
  virtual void setKey(IKey key) = 0;

  /// Permits to get the IPath corresponding to the IKey
  ///
  virtual yat::SharedPtr<IPath, yat::Mutex> getPath() = 0;
  virtual void setPath(IPath path) = 0;

  /// Permits to have some parameters that are defined by the instantiating plug-in
  /// and that can be useful for the method using this context.
  ///  
  /// @return array of object
  ///
  virtual void* getParams() = 0;
  virtual void setParams(void* params) = 0;
};

} //namespace CDMACore
#endif //__CDMA_ICONTEXT_H__

