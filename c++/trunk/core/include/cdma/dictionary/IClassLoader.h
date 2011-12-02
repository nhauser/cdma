//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_ICLASSLOADER_H__
#define __CDMA_ICLASSLOADER_H__

// Include STL
#include <string>

// Include CDMA
#include <cdma/IObject.h>
#include <cdma/exception/Exception.h>
#include <cdma/dictionary/IContext.h>

namespace cdma
{
/*
//==============================================================================
/// The IDictionaryMethod aims to provide a mechanism permitting to call
/// methods that a specified in the xml mapping document from the
/// dictionary mechanism
/// This interface have to be implemented into the 'institute plugin'
//==============================================================================
class IDictionaryMethod : public IObject
{
public:
  // d-tor
  virtual ~IDictionaryMethod()
  {
  }

  /// Execute the method that is given using it's namespace. The corresponding
  /// class will be searched, loaded and instantiated, so the method can be called.
  ///
  /// @param methodNameSpace full namespace of the method (package + class + method name)
  /// @param context of the CDM status while invoking the so called method
  /// @return List of IObject that have been created using the called method.
  /// @throw  Exception in case of any trouble
  ///
  /// @note the method's namespace must be that form: my.package.if.any.MyClass.MyMethod
  ///
  virtual void* invoke( const std::string& method_name, IContext& context ) throw (Exception) = 0;
};
*/
} //namespace CDMACore
#endif //__CDMA_ICLASSLOADER_H__
