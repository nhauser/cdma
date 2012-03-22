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

/// @cond dictAPI

namespace cdma
{

// forward declaration
class CDMA_DECL Context;

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
class CDMA_DECL Key
{
public:

  /// key type: determines if the key is related to a IDataItem or a LogicalGroup
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

/// Declaration of shared pointer KeyPtr
DECLARE_CLASS_SHARED_PTR(Key);

/// @cond internal
// !! IKeySolver, KeyPath, KeyMethod classes are strictly for internal purpose !!

//==============================================================================
/// Internal class
//==============================================================================
class CDMA_DECL IKeySolver
{
public:
  // d-tor
  virtual ~IKeySolver()  {  }

  /// @internal Solve the key
  /// @param context input/ouput context (see Context class definition)
  /// @throw  Exception in case of any trouble
  ///
  virtual void solve(Context& context) throw (cdma::Exception) = 0;
};

/// internal declaration
DECLARE_CLASS_SHARED_PTR(IKeySolver);
/// internal declaration
typedef std::list<IKeySolverPtr> SolverList;


//==============================================================================
/// Internal class
//==============================================================================
class CDMA_DECL KeyPath : public IKeySolver
{
private:
  std::string m_path;

public:

  /// @internal c-tor
  KeyPath(const std::string &path) : m_path(path)
  {
  }

  /// @internal IKeyResolver
  void solve(Context& context) throw (cdma::Exception);
};

//==============================================================================
/// Internal class
//==============================================================================
class CDMA_DECL KeyMethod : public IKeySolver
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

  /// @internal IKeyResolver
  void solve(Context& context) throw (cdma::Exception);
};

/// for internal purpose
DECLARE_SHARED_PTR(KeyPath);
/// for internal purpose
DECLARE_SHARED_PTR(KeyMethod);

/// @endcond internal

} //namespace

/// @endcond dictAPI

#endif //__CDMA_KEY_H__
