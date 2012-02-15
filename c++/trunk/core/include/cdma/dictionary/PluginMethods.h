//******************************************************************************
// Copyright (c) 2011-2012 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_PLUGIN_METHODS_H__
#define __CDMA_PLUGIN_METHODS_H__

// Include CDMA
#include <cdma/IObject.h>
#include <cdma/exception/Exception.h>
#include <cdma/dictionary/Context.h>

namespace cdma
{

#define EXPORT_PLUGIN_METHOD(m) \
  extern "C" CDMA_DECL cdma::IPluginMethod* get ## m ## Class(void) { return new m(); }

//==============================================================================
/// The IPluginMethod aims to provide a mechanism permitting to call
/// methods that are specified in the xml mapping document from the
/// dictionary mechanism
/// This interface have to be implemented into the 'institute plugin'
/// For each such methods
//==============================================================================
class CDMA_DECL IPluginMethod
{
public:

  // d-tor
  virtual ~IPluginMethod()  {  }

  /// Execute the method
  ///
  /// @param context input/ouput context (see Context class definition)
  /// @throw  Exception in case of any trouble
  ///
  virtual void execute(Context& context) throw (cdma::Exception) = 0;
};

} //namespace CDMACore

#endif //__CDMA_PLUGIN_METHODS_H__
