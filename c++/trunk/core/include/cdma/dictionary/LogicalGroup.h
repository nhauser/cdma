/////***************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
/////***************************************************************************
#ifndef __CDMA_LOGICALGROUP_H__
#define __CDMA_LOGICALGROUP_H__

#include <list>
#include <string>
#include <map>

#include <yat/memory/SharedPtr.h>

#include <cdma/Common.h>
#include <cdma/navigation/IContainer.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/dictionary/Dictionary.h>

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
class CDMA_DECL LogicalGroup
{
private:
  IDataset*                              m_dataset_ptr;    // C-style pointer to the parent dataset
  LogicalGroup*                          m_parent_ptr;     // C-style pointer to the parent group (NULL if root)
  KeyPtr                                 m_key_ptr;        // Key from which this logical group was intantiated (is_null() = true if root)
  DictionaryPtr                          m_dictionary_ptr; // Dictionary this logical group match to
  std::map<std::string, LogicalGroupPtr> m_child_groups;   // List of child groups
  StringListPtr                          m_listkey_ptr;

  // Retreive the data associated with a keyword
  void PrivSolveKey(Context *context_ptr);

public:
  /// c-tor
  ///
  /// @param dataset_ptr C-style pointer to the parent dataset
  /// @param parent_ptr C-style pointer to the parent Logical group
  /// @param key_ptr Shared pointer on the corresponding GROUP-type key
  /// @param dictionary_ptr Shared pointer on the keywords dictionary object
  ///
  LogicalGroup( IDataset* dataset_ptr, LogicalGroup* parent_ptr, 
                const KeyPtr& key_ptr, const DictionaryPtr& dictionary_ptr );

  /// d-tor
  virtual ~LogicalGroup();

  /// Find the IDataItem corresponding to the given key in the dictionary.
  ///
  /// @param key_ptr Shared pointer on the keyword description
  ///
  /// @return the first encountered DataItem that match the key, else null
  ///
  IDataItemPtr getDataItem(const KeyPtr& key_ptr);

  /// Find the DataItem corresponding to the given key in the dictionary.
  ///
  /// @param keyPath keywords path (keyword1::keyword2...)
  ///
  /// @return the first encountered DataItem that match the key, else null
  /// @note keyPath can contain several keys concatenated with a plug-in's separator
  ///
  IDataItemPtr getDataItem(const std::string& keyPath);

  /// Find all IDataItems corresponding to the given key in the dictionary.
  ///
  /// @param key_ptr Shared pointer on the keyword description
  ///
  /// @return a std::list of DataItem that match the key
  ///
  std::list<IDataItemPtr> getDataItemList(const KeyPtr& key_ptr);

  /// Find all IDataItems corresponding to the given path of key in the dictionary.
  ///
  /// @param keyPath keywords path (keyword1::keyword2...)
  ///
  /// @return a std::list of DataItem that match the key
  ///
  std::list<IDataItemPtr> getDataItemList(const std::string& keyPath);

  /// Find the Group corresponding to the given key in the dictionary.
  ///
  /// @param key_ptr Shared pointer on the keyword description
  /// @return the first encountered LogicalGroup that matches the key, else null
  ///
  LogicalGroupPtr getGroup(const KeyPtr& key_ptr);

  /// Find the Group corresponding to the given key in the dictionary.
  ///
  /// @param keyPath keywords path (keyword1::keyword2...)
  /// @return the first encountered LogicalGroup that matches the key, else null
  ///
  LogicalGroupPtr getGroup(const std::string& keyPath);

  /// Return the std::list of key that match the given model type.
  ///
  /// @param type Which kind of keys (ie: DATAITEM or GROUP)
  ///
  /// @return List of type Group; may be empty, not null.
  ///
  std::list<std::string> getKeyNames(IContainer::Type type);

  /// Return a list of available keys for this LogicalGroup
  ///
  /// @return List of keys that can be asked
  ///
  std::list<KeyPtr> getKeys();

  /// Bind the given key with the given name, so the key can be accessed by the bind
  ///
  /// @param bind value with which we can get the key
  /// @param key_ptr Shared pointer on the keyword description
  /// @return the given key
  ///
  KeyPtr bindKey(const std::string& bind, const KeyPtr& key_ptr);

  /// Get the parent of this logical group
  /// @return group LogicalGroup
  ///
  LogicalGroupPtr getParent() const;

  /// Set the given logical group as parent of this logical group
  /// @param group_ptr LogicalGroup
  ///
  void setParent(LogicalGroup* group_ptr);

  /// Return the logical location
  std::string getLocation() const;

  /// Return the logical name
  std::string getName() const;

  /// Return the logical short name
  std::string getShortName() const;
};

/// @endcond dictAPI

} //namespace cdma
#endif //__CDMA_LOGICALGROUP_H__
