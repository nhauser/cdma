//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_IPATHMETHOD_H__
#define __CDMA_IPATHMETHOD_H__


#include <list>
#include <string>

namespace cdma
{

//=============================================================================
/// This interface is only used by the extended dictionary mechanism. Its aim
/// is to allow mapping dictionary to specify how to get a IDataItem that don't
/// only rely on a specific path. The IPathMethod permits to specify a method that
/// must be executed associated to a IPath. The more often it's called because we have
/// to pre/post process data to fit a particular need. 
/// <p>
/// The method will be called using an already implemented mechanism. The call will
/// be done only when the IPath, carrying it, will be resolved by the ILogicalGroup.
/// <p>
/// @example In case of a stack of spectrums that are split onto various IDataItem
/// if we want to see them as only one IDataItem the only solution will be to use a
/// specific method for the request.
//=============================================================================
class IPathMethod 
{
public:
  //Virtual destructor
  virtual ~IPathMethod() {};

  /// Returns name of the method that will be called (using it's package name)
  /// return String
  ///
  virtual std::string getName() = 0;

  /// Sets the name of the method that will be called (using it's package name)
  /// return String
  ///
  virtual void setName(std::string method) = 0;

  /// Return parameters Object that are used by this method
  /// @return Object array
  ///
  virtual std::list<void > getParam() = 0;

  /// Set a parameter value that will be used by this method
  /// @return Object array
  /// @note works as a FIFO
  ///
  virtual void pushParam(void* param) = 0;

  /// Set a parameter value that will be used by this method
  /// @return Object array
  /// @note works as a FIFO
  ///
  virtual void* popParam() = 0;

  /// Tells whether or not the method is already contained by the plug-in or if it 
  /// will be dynamically loaded from the external folder specific to the plug-in.
  /// @return bool
  ///
  virtual bool isExternalCall() = 0;

  /// Set whether or not the method is already contained by the plug-in or if it 
  /// will be dynamically loaded from the external folder specific to the plug-in.
  /// @return bool
  /// @see LogicalGroup.resolveMethod
  /// @see org.gumtree.data.dictionary.IClassLoader
  ///
  virtual void isExternal(bool external) = 0;
};

} //namespace CDMACore
#endif //__CDMA_IPATHMETHOD_H__
