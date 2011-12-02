//******************************************************************************
// Copyright (c) 2011 Synchrotron Soleil.
// The CDMA library is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your option)
// any later version.
// Contributors :
// See AUTHORS file
//******************************************************************************

#ifndef __CDMA_IGROUP_H__
#define __CDMA_IGROUP_H__

#include <list>
#include <map>
#include <string>

#include <yat/memory/SharedPtr.h>
#include <yat/threading/Mutex.h>

#include <cdma/exception/Exception.h>
#include <cdma/navigation/IContainer.h>

namespace cdma
{
//==============================================================================
/// IGroup
/// A Group is a collection of DataItems. The Groups in a Dataset form a
/// hierarchical tree, like directories on a disk. A Group has a name and
/// optionally a set of Attributes.
//==============================================================================
class IGroup : public IContainer, public IObject
{
  public:
    virtual ~IGroup() {};

    /// Add a data item to the group.
    ///
    virtual void addDataItem(const IDataItemPtr& v) = 0;

    /// Add a shared Dimension.
    ///
    virtual void addOneDimension(const IDimensionPtr& dimension) = 0;

    /// Add a nested Group.
    ///
    virtual void addSubgroup(const IGroupPtr& group) = 0;

    /// Find the DataItem with the specified (short) name in this group.
    /// @param shortName Short name of DataItem within this group.
    /// @return a shared valid pointeur on the data item (may be a null pointer)
    ///
    virtual IDataItemPtr getDataItem(const std::string& shortName) = 0;

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

    /// Check if this is the root group.
    ///
    /// @return true or false
    //
    virtual bool isRoot() = 0;

    /// Check if this is an entry group. Entries are immediate sub-group of the
    /// root group.
    ///
    /// @return true or false
    //
    virtual bool isEntry() = 0;

    /// Return a clone of this Group object. The tree structure is new. However
    /// the data items are shallow copies that share the same storages with the
    /// original ones.
    ///
    /// @return new Group GDM group object
    //
    virtual IGroupPtr clone() = 0;

    /// Get its parent Group, or null if its the root group.
    ///
    /// @return group object
    ///
    virtual IGroupPtr getParent() = 0;
    
    /// Get the root group of the tree that holds the current Group.
    ///
    /// @return the root group
    ///
    virtual IGroupPtr getRoot() = 0;
    
};
  
  
} //namespace cdma
#endif //__CDMA_IGROUP_H__

