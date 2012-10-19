//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
//
// This file is part of cdma-core library.
//
// The cdma-core library is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License as published by the Free 
// Software Foundation; either version 2 of the License, or (at your option) 
// any later version.
//
// The CDMA library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cmda-python.  If not, see <http://www.gnu.org/licenses/>.
//
// Contributors :
// See AUTHORS file 
//******************************************************************************

#ifndef __CDMA_PLUGIN_METHODS_H__
#define __CDMA_PLUGIN_METHODS_H__

// Include CDMA
#include <cdma/Common.h>
#include <cdma/exception/Exception.h>

/// @cond pluginAPI

namespace cdma
{

/// Macro helper aimed to declare dictionary methods
#define EXPORT_PLUGIN_METHOD(m) \
  extern "C" CDMA_DECL_EXPORT cdma::IPluginMethod* get ## m ## Class(void) { return new m(); }

//==============================================================================
/// @brief Abstract interface of the "dictionary methods" implemented by the plugins
///
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

  /// Executes the method
  ///
  /// @param context input/ouput context (see Context class definition)
  /// @throw  Exception in case of any trouble
  ///
  virtual void execute(class Context& context) throw (cdma::Exception) = 0;
};

/// Declaration of shared pointer IPluginMethodPtr 
DECLARE_SHARED_PTR(IPluginMethod);

} //namespace CDMACore

/// @endcond pluginAPI

#endif //__CDMA_PLUGIN_METHODS_H__
