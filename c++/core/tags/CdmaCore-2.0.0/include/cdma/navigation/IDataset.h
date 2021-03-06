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

#ifndef __CDMA_IDATASET_H__
#define __CDMA_IDATASET_H__

#include <cdma/exception/Exception.h>
#include <cdma/navigation/IGroup.h>
#include <cdma/navigation/IDataItem.h>
#include <cdma/dictionary/ILogicalGroup.h>

namespace cdma
{

//==============================================================================
/// @brief Handler on a physical storage of a data source.
///
/// A dataset holds a reference of a root group, which is the root of a
/// tree of Groups.
/// IDataset is the entry point to have an access to data structure it represents.
/// In case of a data file container, IDataset should refer to the file handle
//==============================================================================
class CDMA_DECL IDataset
{
public:
  //Virtual destructor
  virtual ~IDataset() {};

  /// Return the root group of the dataset.
  /// @return CDMA Group type Created on 16/06/2008
  ///
  virtual IGroupPtr getRootGroup() = 0;

  /// Return the the logical root of the dataset.
  /// @return CDMA Group type
  ///
  virtual ILogicalGroupPtr getLogicalRoot() = 0;

  /// Return the location of the dataset. If it's a file, return the path. 
  /// @return string type 
  ///
  virtual std::string getLocation() = 0;

  /// Return the title of the dataset.
  /// @return string type 
  ///
  virtual std::string getTitle() = 0;

  /// Set the location (URI) field of the dataset.
  /// @param location URI
  ///
  virtual void setLocation(const std::string& location) = 0;

  /// Set the location field of the dataset.
  /// @param location as yat::URI object
  ///
  //virtual void setLocation(const yat::URI& location) = 0;

  /// Set the title for the Dataset.
  /// @param title a string object 
  ///
  virtual void setTitle(const std::string& title) = 0;

  /// Synchronize the dataset with the file reference.
  /// @return true or false
  ///
  virtual bool sync() throw ( Exception ) = 0;

  /// Save the contents / changes of the dataset to the file.
  ///
  virtual void save() throw ( Exception ) = 0;

  /// Save the contents of the dataset to a new file.
  ///
  virtual void saveTo(const std::string& location) throw ( Exception ) = 0;

  /// Save the specific contents / changes of the dataset to the file.
  ///
  virtual void save(const IContainer& container) throw ( Exception ) = 0;

  /// Save the attribute to the specific path of the file.
  ///
  virtual void save(const std::string& parentPath, const IAttributePtr& attribute)
               throw ( Exception ) = 0;

  /// Open the node defined by the path (inside the dataset) and
  /// returns the IGroup that corresponds
  /// @param path String representation of the IGroup's path
  /// @note the given path must be absolute
  ///
  virtual IGroupPtr getGroupFromPath(const std::string &path) = 0;

  /// Open the node defined by the path (inside the dataset) and
  /// returns the IDataItem that corresponds
  /// @param path String representation of the IDataItem's path
  /// @note the given path must be absolute
  ///
  virtual IDataItemPtr getItemFromPath(const std::string &path) = 0;

  /// Find a group or a data item by its path
  ///
  /// @param path path of the element relative to this object
  ///
  virtual IContainerPtr findContainerByPath(const std::string& path) = 0;

  /// Find a list of groups and/or data items giving a path
  ///
  /// @param path path of the element relative to this object
  /// @param first_only if true returns only the first container found that match the input path
  ///
  virtual IContainerPtrList findAllContainerByPath(const std::string& path, bool first_only=false) = 0;

};


} //namespace cdma
#endif //__CDMA_IDATASET_H__

