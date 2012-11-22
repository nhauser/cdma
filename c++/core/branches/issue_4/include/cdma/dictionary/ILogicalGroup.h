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

#ifndef __CDMA_ILOGICALGROUP_H__
#define __CDMA_ILOGICALGROUP_H__

#include <cdma/Common.h>
#include <cdma/navigation/IContainer.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/dictionary/IKey.h>

/// @cond clientAPI

namespace cdma
{

//==============================================================================
/// @brief Group notion used in dictionary mode of accessing data
///
/// The LogicalGroup is an object that is purely @e logical. Its existence is
/// correlated to the Dictionary.
/// 
/// A standard CDMA dictionary make a link between a key and a path. Now let's imagine
/// a dictionary with keys having a tree structure. This structure hierarchically organized
/// might now have a meaning regardless their physical organization.
/// So the keys are now simple notions that can have a human friendly meaning.
/// 
/// The LogicalGroup permits to browse simply through those different levels
/// of key. More over the key used can be filtered according to some criteria.
/// The aim is to find a really specific node by doing a search that get narrower
/// while iterating over queries.
//==============================================================================
class CDMA_DECL ILogicalGroup
{
public:

  /// d-tor
  virtual ~ILogicalGroup() {}

  /// Find the IDataItem corresponding to the given key in the dictionary.
  ///
  /// @param key_ptr Shared pointer on the keyword description
  ///
  /// @return the first encountered DataItem that match the key, else null
  ///
  virtual IDataItemPtr getDataItem(const IKeyPtr& key_ptr) = 0;

  /// Find the DataItem corresponding to the given key in the dictionary.
  ///
  /// @param keyPath keywords path (keyword1:keyword2...)
  ///
  /// @return the first encountered DataItem that match the key, else null
  /// @note keyPath can contain several keys concatenated with a plug-in's separator
  ///
  virtual IDataItemPtr getDataItem(const std::string& keyPath) = 0;

  /// Find all IDataItems corresponding to the given key in the dictionary.
  ///
  /// @param key_ptr Shared pointer on the keyword description
  ///
  /// @return a std::list of DataItem that match the key
  ///
  virtual std::list<IDataItemPtr> getDataItemList(const IKeyPtr& key_ptr) = 0;

  /// Find all IDataItems corresponding to the given path of key in the dictionary.
  ///
  /// @param keyPath keywords path (keyword1::keyword2...)
  ///
  /// @return a std::list of DataItem that match the key
  ///
  virtual std::list<IDataItemPtr> getDataItemList(const std::string& keyPath) = 0;

  /// Find the Group corresponding to the given key in the dictionary.
  ///
  /// @param key_ptr Shared pointer on the keyword description
  /// @return the first encountered LogicalGroup that matches the key, else null
  ///
  virtual ILogicalGroupPtr getGroup(const IKeyPtr& key_ptr) = 0;

  /// Find the Group corresponding to the given key in the dictionary.
  ///
  /// @param keyPath keywords path (keyword1::keyword2...)
  /// @return the first encountered LogicalGroup that matches the key, else null
  ///
  virtual ILogicalGroupPtr getGroup(const std::string& keyPath) = 0;

  /// Return the std::list of key that match the given model type.
  ///
  /// @param type Which kind of keys (ie: DATAITEM or GROUP)
  ///
  /// @return List of type Group; may be empty, not null.
  ///
  virtual std::list<std::string> getKeyNames(IContainer::Type type) = 0;

  /// Return a list of available keys for this LogicalGroup
  ///
  /// @return List of keys that can be asked
  ///
  virtual std::list<IKeyPtr> getKeys() = 0;

  /// Bind the given key with the given name, so the key can be accessed by the bind
  ///
  /// @param bind value with which we can get the key
  /// @param key_ptr Shared pointer on the keyword description
  /// @return the given key
  ///
  virtual IKeyPtr bindKey(const std::string& bind, const IKeyPtr& key_ptr) = 0;

  /// Get the parent of this logical group
  /// @return group LogicalGroup
  ///
  virtual ILogicalGroupPtr getParent() const = 0;

  /// Return the logical location
  virtual std::string getLocation() const = 0;

  /// Return the logical name
  virtual std::string getName() const = 0;

  /// Return the logical short name
  virtual std::string getShortName() const = 0;

private:
  ILogicalGroup() {};
  
/// @cond internal

public:
  // implementation
  friend class LogicalGroup;
  
/// @endcond internal
};

/// @endcond clientAPI

} //namespace cdma
#endif //__CDMA_LOGICALGROUP_H__
