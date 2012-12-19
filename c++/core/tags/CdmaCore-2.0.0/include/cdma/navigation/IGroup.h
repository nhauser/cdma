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

#ifndef __CDMA_IGROUP_H__
#define __CDMA_IGROUP_H__

#include <cdma/exception/Exception.h>
#include <cdma/navigation/IContainer.h>
#include <cdma/navigation/IDataItem.h>
#include <cdma/navigation/IDimension.h>

namespace cdma
{

//==============================================================================
/// @brief An IGroup is the abstraction of a collection of IDataItem(s).
///
/// The IGroup type objects in a Dataset form a hierarchical tree, like
/// directories on a disk. 
/// A IGroup has a name, contains one or mode IDataItem object and optionally
/// a set of IAttribute objects containing its metadata.
///
/// @note It must be a base class of data format engine data group class concretization
/// which can be overrided, if needed, by plug-ins based on same engine
//==============================================================================
class CDMA_DECL IGroup : public IContainer
{
public:
  virtual ~IGroup() {};

  /// Check if this is the root group.
  ///
  /// @return true or false
  //
  virtual bool isRoot() const = 0;

  /// Check if this is an entry group. Entries are immediate sub-group of the
  /// root group.
  ///
  /// @return true or false
  //
  virtual bool isEntry() const = 0;

  /// Get its parent Group, or null if its the root group.
  ///
  /// @return group object
  ///
  virtual IGroupPtr getParent() const = 0;
  
  /// Get the root group of the tree that holds the current Group.
  ///
  /// @return the root group
  ///
  virtual IGroupPtr getRoot() const = 0;

  //@{ Read-oriented methods

    /// Find the DataItem with the specified (short) name in this group.
    /// @param shortName Short name of DataItem within this group.
    /// @return a shared valid pointeur on the data item (may be a null pointer)
    ///
    virtual IDataItemPtr getDataItem(const std::string& shortName) throw ( cdma::Exception ) = 0;

    /// Find the DataItem that has the specific attribute, with the name and
    /// value given.
    ///
    /// @param name
    ///            in string type
    /// @param value
    ///            in string type
    /// @return DataItem object
    //
    virtual IDataItemPtr getDataItemWithAttribute(const std::string& name, const std::string& value) = 0;

    /// Retrieve a Dimension using its (short) name. If it does not exist in this
    /// group, recursively look in parent groups.
    ///
    /// @param name
    ///            Dimension name.
    /// @return the Dimension, or null if not found
    //
    virtual IDimensionPtr getDimension(const std::string& name) = 0;

    /// Retrieve the Group with the specified (short) name as a sub-group of the
    /// current group.
    ///
    /// @param shortName
    ///            short name of the nested group you are looking for.
    /// @return the Group, or null if not found
    //
    virtual IGroupPtr getGroup(const std::string& shortName) = 0;

    /// Find the sub-Group that has the specific attribute, with the name and
    /// value given.
    ///
    /// @param attributeName
    ///            string object
    /// @param value
    ///            in string type
    /// @return Group object
    //
    virtual IGroupPtr getGroupWithAttribute(const std::string& attributeName, const std::string& value) = 0;

    /// Get the Variables contained directly in this group.
    ///
    /// @return list of type Variable; may be empty, not null.
    //
    virtual std::list<IDataItemPtr> getDataItemList() = 0;

    /// Get the Dimensions contained directly in this group.
    ///
    /// @return list of type Dimension; may be empty, not null.
    //
    virtual std::list<IDimensionPtr > getDimensionList() = 0;

    /// Get the Groups contained directly in this Group.
    ///
    /// @return list of type Group; may be empty, not null.
    //
    virtual std::list<IGroupPtr> getGroupList() = 0;

  //@} Read-oriented methods

  //@{ Write-oriented methods

    /// Add a new data item to the group.
    ///
    virtual IDataItemPtr addDataItem(const std::string& shortName) = 0;

    /// Add a shared Dimension.
    ///
    virtual IDimensionPtr addDimension(const cdma::IDimensionPtr& dim) = 0;

    /// Add a nested Group.
    ///
    virtual IGroupPtr addSubgroup(const std::string& shortName) = 0;

    /// Remove a DataItem from the DataItem list.
    ///
    /// @param item
    ///            GDM DataItem
    /// @return bool type
    //
    virtual bool removeDataItem(const IDataItemPtr& item) = 0;

    /// remove a Variable using its (short) name, in this group only.
    ///
    /// @param varName
    ///            Variable name.
    /// @return true if Variable found and removed
    //
    virtual bool removeDataItem(const std::string & varName) = 0;

    /// remove a Dimension using its name, in this group only.
    ///
    /// @param dimName
    ///            Dimension name
    /// @return true if dimension found and removed
    //
    virtual bool removeDimension(const std::string & dimName) = 0;

    /// Remove a Group from the sub Group list.
    ///
    /// @param group
    ///            GDM Group
    /// @return bool type
    //
    virtual bool removeGroup(const IGroupPtr& group) = 0;

    /// Remove the Group with a certain name in the sub Group list.
    ///
    /// @param shortName
    ///            in string type
    /// @return bool type
    //
    virtual bool removeGroup(const std::string & shortName) = 0;

    /// Remove a Dimension from the Dimension list.
    ///
    /// @param dimension IDimension
    /// @return IDimension type 
    //
    virtual bool removeDimension(const IDimensionPtr& dimension) = 0;

 //@} Write-oriented methods
    
};
 

  
} //namespace cdma
#endif //__CDMA_IGROUP_H__

