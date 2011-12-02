//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IPATHPARAMRESOLVER_H__
#define __CDMA_IPATHPARAMRESOLVER_H__

#include "yat/memory/SharedPtr.h"

#include "cdma/IObject.h"

namespace cdma
{
//==============================================================================
/// IPathParamResolver is used internally by the extended dictionary mechanism. 
/// Its aim is to determine according to a given path how to reach a specific IContainer
/// without any ambiguity by returning the IPathParameter that corresponds to the IContainer.
/// <p>
/// The IPathParaResolver is associated to a IPath that is stored in the dictionary.
/// It's the comparison between the associated path and the IContainer's physical real path,
/// that will define the IPathParameter.
//==============================================================================
class IPathParamResolver : public IObject
{
public:
  // d-tor
  virtual ~IPathParamResolver()
  {
  }

  /// It will returns
  /// @param node 
  /// @return
  ///
  virtual yat::SharedPtr<IPathParameter, yat::Mutex> resolvePathParameter(IContainer& node) = 0;
};
} //namespace CDMACore
#endif //__CDMA_IPATHPARAMRESOLVER_H__
