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

#ifndef __CDMA_LOGICALGROUP_H__
#define __CDMA_LOGICALGROUP_H__

#include <cdma/Common.h>
#include <cdma/navigation/IContainer.h>
#include <cdma/navigation/IDataset.h>
#include <cdma/dictionary/plugin/Dictionary.h>
#include <cdma/dictionary/ILogicalGroup.h>

// !! LogicalGroup is a internal class !!
/// @cond internal

namespace cdma
{

//==============================================================================
// ILogicalGroup implementation
//==============================================================================
class LogicalGroup: public ILogicalGroup
{
private:
  IDataset*                               m_dataset_ptr;    // C-style pointer to the parent dataset
  std::string                             m_parent_path;    // Path of parent logical group
  IKeyPtr                                 m_key_ptr;        // Key from which this logical group was intantiated (is_null() = true if root)
  DictionaryPtr                           m_dictionary_ptr; // Dictionary this logical group match to
  std::map<std::string, ILogicalGroupPtr> m_child_groups;   // List of child groups
  StringListPtr                           m_listkey_ptr;

  // Retreive the data associated with a keyword
  void PrivSolveKey(Context *context_ptr);

public:
  // c-tor
  //
  // @param dataset_ptr C-style pointer to the parent dataset
  // @param parent_ptr C-style pointer to the parent Logical group
  // @param key_ptr Shared pointer on the corresponding GROUP-type key
  // @param dictionary_ptr Shared pointer on the keywords dictionary object
  //
  LogicalGroup( IDataset* dataset_ptr, LogicalGroup* parent_ptr, 
                const IKeyPtr& key_ptr, const DictionaryPtr& dictionary_ptr );

  // d-tor
  ~LogicalGroup();
  
  //@{ ILogicalGroup interface -------------
  
  IDataItemPtr getDataItem(const IKeyPtr& key_ptr);
  IDataItemPtr getDataItem(const std::string& keyPath);
  std::list<IDataItemPtr> getDataItemList(const IKeyPtr& key_ptr);
  std::list<IDataItemPtr> getDataItemList(const std::string& keyPath);
  ILogicalGroupPtr getGroup(const IKeyPtr& key_ptr);
  ILogicalGroupPtr getGroup(const std::string& keyPath);
  std::list<std::string> getKeyNames(IContainer::Type type);
  std::list<IKeyPtr> getKeys();
  IKeyPtr bindKey(const std::string& bind, const IKeyPtr& key_ptr);
  ILogicalGroupPtr getParent() const;
  void setParent(LogicalGroup* group_ptr);
  std::string getLocation() const;
  std::string getName() const;
  std::string getShortName() const;
  
  //@} 
};

/// @endcond dictAPI

} //namespace cdma
#endif //__CDMA_ILOGICALGROUP_H__
