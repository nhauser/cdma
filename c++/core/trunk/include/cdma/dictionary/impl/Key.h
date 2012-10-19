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

#ifndef __CDMA_KEY_H__
#define __CDMA_KEY_H__

#include <cdma/Common.h>
#include <cdma/exception/Exception.h>
#include <cdma/dictionary/plugin/PluginMethods.h>
#include <cdma/dictionary/IKey.h>

/// @cond internal

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
class Key: public cdma::IKey
{
public:
  
private:
  std::string m_name;
  IKey::Type m_type;
  
public:
  // c-tor
  Key(const std::string& name, IKey::Type type = UNDEFINED)
    : m_name(name), m_type(type) {}

  // Interface IKey
  const std::string& getName() const { return m_name; }
  void setName(const std::string& name) { m_name = name; }
  const IKey::Type& getType() const { return m_type; }
  void setType(IKey::Type type) { m_type = type; }
};

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

} //namespace

/// @endcond internal

#endif //__CDMA_KEY_H__
