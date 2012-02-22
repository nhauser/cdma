//******************************************************************************
// Copyright (c) 2011-2012 Synchrotron Soleil.
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

#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/dictionary/PluginMethods.h>

namespace cdma
{

// forward declaration
class Context;

//=============================================================================
/// Key
///
/// The IKey is used by group to interrogate the dictionary. Indeed the key's
/// name corresponds to an entry in the dictionary. This entry targets a path in the
/// currently explored document. The group will open it.
///
/// The Key can carry some filters to help group to decide which node is relevant.
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
    UNDEFINED = 0,
    ITEM,
    GROUP
  };
  
private:
  std::string m_name;
  Type m_type;
  
public:
  /// constructor
  Key(const std::string& name, Type type = UNDEFINED)
    : m_name(name), m_type(type) {}

  //@{ Accessors -----------------

  /// Get the entry name in the dictionary that will be
  /// searched when using this key.
  ///
  /// @return the name of this key
  ///
  const std::string& getName() const { return m_name; }
 
  /// Set the entry name in the dictionary that will be
  /// searched when using this key.
  ///
  /// @param name of this key
  ///
  void setName(const std::string& name) { m_name = name; }
 
 
  /// Get the key related notion: LogicalGroup or DataItem
  ///
  /// @return Key::Type value
  ///
  const Type& getType() const { return m_type; }

  /// Set the key related notion: LogicalGroup or DataItem
  /// 
  /// @param type of this key
  ///
  void setType(Type type) { m_type = type; }
};

DECLARE_CLASS_SHARED_PTR(Key);

//==============================================================================
/// IKeySolver
//==============================================================================
class IKeySolver
{
public:
  // d-tor
  virtual ~IKeySolver()  {  }

  /// Solve the key
  ///
  /// @param context input/ouput context (see Context class definition)
  /// @throw  Exception in case of any trouble
  ///
  virtual void solve(Context& context) throw (cdma::Exception) = 0;
};

DECLARE_CLASS_SHARED_PTR(IKeySolver);
typedef std::list<IKeySolverPtr> SolverList;

//==============================================================================
/// KeyPath
//==============================================================================
class KeyPath : public IKeySolver
{
private:
  std::string m_path;

public:

  /// c-tor
  KeyPath(const std::string &path) : m_path(path)
  {
  }

  /// IKeyResolver
  void solve(Context& context) throw (cdma::Exception);
};

//==============================================================================
/// KeyMethod
//==============================================================================
class KeyMethod : public IKeySolver
{
private:
  std::string      m_method_name;
  IPluginMethodPtr m_method_ptr;

public:

  /// c-tor
  KeyMethod(const std::string& method_name, const IPluginMethodPtr& method_ptr)
  : m_method_name(method_name), m_method_ptr(method_ptr)
  {
  }

  /// IKeyResolver
  void solve(Context& context) throw (cdma::Exception);
};

// Smart pointers
DECLARE_CLASS_SHARED_PTR(KeyPath);
DECLARE_CLASS_SHARED_PTR(KeyMethod);

} //namespace CDMACore

#endif //__CDMA_KEY_H__
