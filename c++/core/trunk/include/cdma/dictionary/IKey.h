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

#ifndef __CDMA_IKEY_H__
#define __CDMA_IKEY_H__

#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/dictionary/plugin/PluginMethods.h>

/// @cond dictAPI

namespace cdma
{

//=============================================================================
/// @brief Key object holds keywords realatives to LogicalGroup or IDataItem
///
/// Key objects are used by group to query the dictionary. Indeed the key's
/// name corresponds to an entry in the dictionary. This entry targets a path 
/// in the currently explored dataset or a dictionary methods.
///
/// @todo supporting filters in order to help group to decide which node is relevant.
/// The filters can specify an order index to open a particular type of node, an
/// attribute, a part of the name...
///
//=============================================================================
class CDMA_DECL IKey
{
public:

  /// key type: determines if the key is related to a IDataItem or a ILogicalGroup
  enum Type
  {
    UNDEFINED = 0,
    ITEM,
    GROUP
  };
  
public:

  /// Get the entry name in the dictionary that will be
  /// searched when using this key.
  ///
  /// @return the name of this key
  ///
  virtual const std::string& getName() const = 0;
 
  /// Set the entry name in the dictionary that will be
  /// searched when using this key.
  ///
  /// @param name of this key
  ///
  virtual void setName(const std::string& name) = 0;
 
  /// Get the key related notion: LogicalGroup or DataItem
  ///
  /// @return Key::Type value
  ///
  virtual const Type& getType() const = 0;

  /// Set the key related notion: LogicalGroup or DataItem
  /// 
  /// @param type of this key
  ///
  virtual void setType(Type type) = 0;

private:
  // This interface is not derivable in client applications code
  IKey() {}
  
/// @cond internal

public:
  // Implementation
  friend class Key;
  
/// @endcond internal

};

/// Declaration of shared pointer IKeyPtr
DECLARE_CLASS_SHARED_PTR(IKey);

} //namespace

/// @endcond dictAPI

#endif //__CDMA_IKEY_H__
