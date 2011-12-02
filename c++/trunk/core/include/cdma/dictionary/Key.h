//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************
#ifndef __CDMA_KEY_H__
#define __CDMA_KEY_H__

#include <list>
#include <string>

#include <yat/memory/SharedPtr.h>
#include <yat/threading/Mutex.h>

#include <cdma/IObject.h>

namespace cdma
{

//=============================================================================
/// Key
///
/// The IKey is used by group to interrogate the dictionary. Indeed the key's
/// name corresponds to an entry in the dictionary. This entry targets a path in the
/// currently explored document. The group will open it.
///
/// The IKey can carry some filters to help group to decide which node is relevant.
/// The filters can specify an order index to open a particular type of node, an
/// attribute, a part of the name...
///
//=============================================================================
class Key
{
public:

  /// KeyType
  /// Kind of Key: will the key correspond to DataItem, a Group, or ...
  enum Type
  {
    Undefined = 0,
    Item,
    Group
  };
  
private:
  std::string m_name;
  Type m_type;
  
public:
  // constructor
  Key(const std::string& name, Type type = Undefined);

//destructor
  ~Key();
  
  /// Get the entry name in the dictionary that will be
  /// searched when using this key.
  ///
  /// @return the name of this key
  ///
  std::string getName();
 
  /// Set the entry name in the dictionary that will be
  /// searched when using this key.
  ///
  /// @param name of this key
  ///
  void setName(const std::string& name);
 
  /// Return true if both key have similar names.
  /// Filters are not compared.
  /// @param key to compare
  /// @return true if both keys have same name
  ///
  bool isEqual(const KeyPtr& ptrKey);
 
  /// Get the list of filters that will be applied when using this key.
  ///
  /// @return list of IKeyFilter
  ///
  std::list<PathParameterPtr> getParameterList();
 
  /// Add a IKeyFilter to this IKey that will be used when
  /// searching an object with this key. .
  ///
  /// @param filter to be applied
  /// @note work as a FILO
  ///
  void pushParameter(const PathParameterPtr& filter);
 
  /// Remove a IKeyFilter to this IKey that will be used when
  /// searching an object with this key.
  ///
  /// @return filter that won't be applied anymore
  /// @note work as a FILO
  ///
  PathParameterPtr popParameter();
 
  /// Copy entirely the key : name and filters are cloned
  /// @return a copy of this key
  ///
  KeyPtr clone();

  Type getType();
  void setType(Type type);
};
 
} //namespace CDMACore

#endif //__CDMA_KEY_H__
